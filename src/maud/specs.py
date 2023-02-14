import abc
import collections
import dataclasses
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple, Union
import warnings

import datasets

from maud import data, utils


def _convert_records_to_single_dict(records: List[dict]) -> Dict[str, List]:
    """Convert List[dict] (records) to Dict[str, list]

    The return value is a suitable input for datasets.from_dict().
    """
    data_dict = collections.defaultdict(list)
    for d in records:
        data_dict["contract_name"].append(d["contract_name"])
        data_dict["text"].append(d["text"])
        data_dict["answer"].append(d["answer"])
        data_dict["label"].append(d["label"])
    return data_dict


def _encode_records_as_dataset(records: List[dict], tokenizer, *, dry_run=False):
    assert tokenizer is not None
    data_dicts = _convert_records_to_single_dict(records)
    ds = datasets.Dataset.from_dict(data_dicts)
    if dry_run:
        return ds

    encoded_ds = ds.map(
        lambda examples: tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
        ),
        desc="tokenizing",
        batched=True,
        num_proc=4,
    )

    encoded_ds.set_format(
        type='torch',
        columns=[*data.MODEL_INPUT_KEYS, "label"],
    )
    print(encoded_ds)
    return encoded_ds


@dataclasses.dataclass
class BaseDatasetSpec(abc.ABC):
    id: str
    context_key: str
    answer_key: str

    IGNORED_ANSWER_SET: ClassVar[List[str]] = [
        "(None entered)",
        "..",  # Not the same as normal period dot.
        "No limitation + reasonable best efforts standard"
    ]

    def __post_init__(self):
        assert self.context_key != self.answer_key
        headers = data.get_headers()
        assert self.context_key in headers
        assert self.answer_key in headers

    @property
    def sub_question_key(self) -> str:
        return "<NONE>"

    def _load_answer_set(self) -> Set[str]:
        result = set()
        ## Main data
        contracts = data.load_contracts(self.context_key)
        for d in contracts:
            result.add(d[self.answer_key])
        for bad_answer in self.IGNORED_ANSWER_SET:
            result.discard(bad_answer)
        if len(result) <= 1:
            self.id = "SKIPME: one or zero types of test label"
            return result

        ## Additional data
        out = data.load_bonus_context(self.answer_key)
        if out is not None:
            _, _, add_answer = out
            for answer in add_answer:
                if answer in self.IGNORED_ANSWER_SET:
                    continue
                if not answer in result:
                    # Should match main data labels
                    raise RuntimeError(f"{answer} not in {result}")

        ## Synth data
        out_s = data.load_synth_data(self.answer_key)
        if out_s is not None:
            contexts, answers = out_s
            for answer in answers:
                if answer in self.IGNORED_ANSWER_SET:
                    continue
                if not answer in result:
                    # Should match main data labels
                    raise RuntimeError(f"{answer} not in {result}")

        return result

    @property
    @abc.abstractmethod
    def n_classes(self) -> int:
        """The number of possible integer label values or equivalently the number of possible string answer values."""

    @abc.abstractmethod
    def answer_to_label(self, label: str) -> int:
        pass

    @abc.abstractmethod
    def label_to_answer(self, num: int) -> str:
        pass

    @property
    def ordered_answers(self) -> List[str]:
        return [self.label_to_answer(i) for i in range(self.n_classes)]

    def records_post_process(self, records: List[dict]) -> List[dict]:
        # Ignore BAD_CONTRACT.
        BAD_CONTRACTS = {
            'Soliton, Inc._AbbVie Inc..pdf',  # Duplicate contract
        }
        result = [rec for rec in records
                  if rec["contract_name"] != 'Soliton, Inc._AbbVie Inc..pdf']

        for rec in result:
            assert rec["contract_name"] != 'Soliton, Inc._AbbVie Inc..pdf'
        return result

    def _load_data_records(self) -> List[dict]:
        """Parse from the data files a list of dictionaries"""
        contracts = data.load_contracts(self.context_key)
        data_records: List[dict] = []
        for d in contracts:
            contract_name = d["Filename"]
            if (context := d.get(self.context_key)) is None:
                raise KeyError(f"Missing context key '{context}' in {contract_name}.")
            if (answer := d.get(self.answer_key)) is None:
                raise KeyError(f"Missing answer key '{answer}' in {contract_name}.")

            if answer in self.IGNORED_ANSWER_SET:
                continue

            int_label = self.answer_to_label(answer)
            if int_label is None:  # Invalid answer or ignored answer.
                raise ValueError(f"Unexpected answer {answer}")

            data_records.append(
                dict(
                    data_type="main",
                    contract_name=contract_name,
                    text=context,
                    answer=answer,
                    label=int_label,
                    question=self.answer_key,
                    subquestion="<NONE>",
                    text_type=self.context_key,
                    id=self.id,
                )
            )
        proc_data_records = self.records_post_process(data_records)
        for rec in proc_data_records:
            assert rec["contract_name"] != 'Soliton, Inc._AbbVie Inc..pdf'
        return proc_data_records

    def _load_additional_data_records(self) -> Optional[List[Dict]]:
        out = data.load_bonus_context(self.answer_key)
        if out is None:
            return None
        add_contract_names, add_context, add_answer = out
        assert len(add_contract_names) == len(add_context)
        assert len(add_answer) == len(add_context)

        records = []
        for i in range(len(add_contract_names)):
            context = add_context[i]
            contract_name = add_contract_names[i]
            answer = add_answer[i]
            if answer in self.IGNORED_ANSWER_SET:
                continue

            try:
                int_label = self.answer_to_label(answer)
            except ValueError:
                warnings.warn(f"Unknown answer '{answer}' in additional dataset that did not appear in main dataset. "
                              "Implement logic for handling this case please!")
                continue
            records.append(dict(
                data_type="abridged",
                text=context,
                contract_name=contract_name,
                answer=answer,
                label=int_label,
                question=self.answer_key,
                subquestion="<NONE>",
                text_type=self.context_key,
                id=self.id,
            ))
            assert len(records) > 0
        records = self.records_post_process(records)
        return records

    def _load_synth_data_records(self) -> Optional[List[Dict]]:
        out_s = data.load_synth_data(self.answer_key)
        if out_s is None:
            return None
        else:
            records = []
            contexts, answers = out_s
            assert len(contexts) == len(answers)
            assert len(contexts) > 0
            for context, answer in utils.strict_zip(contexts, answers):
                if answer in self.IGNORED_ANSWER_SET:
                    continue

                int_label = self.answer_to_label(answer)
                records.append(dict(
                    data_type="rare_answers",
                    contract_name="<RARE_ANSWERS>",
                    text=context,
                    answer=answer,
                    label=int_label,
                    question=self.answer_key,
                    subquestion="<NONE>",
                    text_type=self.context_key,
                    id=self.id,
                ))
            records = self.records_post_process(records)
            return records

    def has_synth_data(self) -> bool:
        return self._load_synth_data_records() is not None

    def answer_counter(self) -> collections.Counter:
        return collections.Counter(d["answer"] for d in self._load_data_records())

    def answer_counter_synth(self) -> collections.Counter:
        if not self.has_synth_data():
            return collections.Counter()
        return collections.Counter(d["answer"] for d in self._load_synth_data_records())

    def answer_counter_all(self) -> collections.Counter:
        main_ct = self.answer_counter()
        if self.has_synth_data():
            new = collections.Counter(d["answer"] for d in self._load_synth_data_records())
            for d in self._load_synth_data_records():
                self.answer_to_label(d["answer"])  # Check for answer compabitility
            main_ct.update(new)
        if (add_records := self._load_additional_data_records()) is not None:
            new = collections.Counter(d["answer"] for d in add_records)
            main_ct.update(new)

        # Account for comma delimited labels in multi-label
        result = collections.Counter()
        for possibly_comma_delimited, count in main_ct.items():
            if ", " not in possibly_comma_delimited:
                assert "," not in possibly_comma_delimited
            for x in possibly_comma_delimited.split(", "):  # singleton if delimiter not present
                result[x] += count

        return result

    def to_dataset_args(self, args):
        use_synth = args.use_synth
        use_add = args.use_add
        return self.to_dataset(
            add_synth=use_synth, verbose=True, add_add=use_add,
        )

    def to_dataset(self, *, max_len=None, add_synth=False, add_add=False,
                   verbose: bool = False,
                   ) -> Tuple[List[dict], List[dict], List[dict]]:
        """Handy function for returning all sorts of records at once."""
        records = []
        records.extend(self._load_data_records())

        synth = []
        if add_synth and (synth_records := self._load_synth_data_records()) is not None:
            synth.extend(synth_records)
            print(f"Loaded {len(synth_records)} synthetic records.")

        add = []
        if add_add and (add_records := self._load_additional_data_records()) is not None:
            add.extend(add_records)
            print(f"Loaded {len(add_records)} additional records.")

        # Cool trick: slicing with max_len=None is a no-op.
        records = records[:max_len]
        return records, add, synth

    def to_train_test_split(  # Currently superceded by args version, for analysis reasons, but should still work.
            self, tokenizer, add_synth: bool, add_add: bool,
            valid_prop: float,
            synth_capped: bool,
            verbose: bool = True,
    ) -> Tuple[datasets.Dataset, datasets.Dataset]:
        # data for normal training + evaluation
        ds, add_ds, synth_ds = self.to_dataset(add_synth=add_synth, verbose=True, add_add=add_add)

        if verbose:
            print(f"{len(ds)} records loaded from main dataset ({self.n_classes} classes)")

        train_ds, test_ds = data.build_balanced_split_and_tokenize(
            ds,
            tokenizer,
            valid_prop,
            verbose=verbose,
            add_ds=add_ds,
            synth_ds=synth_ds,
            synth_capped=synth_capped,
        )
        if verbose:
            print(f"{len(train_ds)} examples put in train_ds")
            print(f"{len(test_ds)} examples put in test_ds")
        return train_ds, test_ds

    def to_additional_dataset(self, tokenizer=None, max_len=None) -> Union[None, Sequence[dict], datasets.Dataset]:
        records = self._load_additional_data_records()
        if records is None:
            return None
        records = records[:max_len]
        if tokenizer is not None:
            return _encode_records_as_dataset(records, tokenizer)
        else:
            return records

    def to_synth_dataset(self, tokenizer=None, max_len=None) -> Union[None, Sequence[dict], datasets.Dataset]:
        records = self._load_synth_data_records()
        if records is None:
            return None
        records = records[:max_len]
        if tokenizer is not None:
            return _encode_records_as_dataset(records, tokenizer)
        else:
            return records

    def __str__(self):
        return self.to_str()

    def to_str(self, verbosity: int = 1) -> str:
        """Stub function that allows utility scripts to describe this Spec with different
        levels of verbosity."""
        return super().__str__()


@dataclasses.dataclass
class MultipleChoiceDatasetSpec(BaseDatasetSpec):

    def __post_init__(self):
        contracts = data.load_contracts(self.context_key)
        for contract in contracts:
            # We should have removed these rows already.
            assert "(None entered)" not in contract[self.context_key]
        answer_choices = self._load_answer_set()
        self._answer_choices_ordered = tuple(sorted(answer_choices))
        assert len(set(self.answer_choices_ordered)) == len(self.answer_choices_ordered), "duplicate"
        assert len(set(self.answer_choices_ordered)) == len(self.answer_choices_ordered), "duplicate"
        super().__post_init__()

    def __repr__(self):
        return super().__repr__() + f" _answer_choices_ordered: {self._answer_choices_ordered}"

    @property
    def answer_choices_ordered(self) -> Tuple[str, ...]:
        return self._answer_choices_ordered

    @property
    def n_classes(self) -> int:
        return len(self.answer_choices_ordered)

    def answer_to_label(self, label: str) -> int:
        return self.answer_choices_ordered.index(label)

    def label_to_answer(self, num: int) -> str:
        return self.answer_choices_ordered[num]


@dataclasses.dataclass
class MultiBinaryDatasetSubQuestionSpec(BaseDatasetSpec):
    """A spec for inducing a single binary Dataset from a multi-binary answer column."""
    positive_answer: str

    # NOTE: Setting type as ClassVar prevents dataclass from parsing these annotations as (optional) parameters.
    NO_LABEL: ClassVar[int] = 0
    YES_LABEL: ClassVar[int] = 1
    OTHER_ANSWER: ClassVar[str] = "<OTHER>"

    def __post_init__(self):
        assert self.positive_answer != self.OTHER_ANSWER
        super().__post_init__()

    @property
    def sub_question_key(self) -> str:
        return self.positive_answer

    @property
    def n_classes(self) -> int:
        return 2

    def _convert_ans(self, ans):
        if self.positive_answer in [a.strip() for a in ans.split(", ")]:
            return self.positive_answer
        else:
            return self.OTHER_ANSWER

    def answer_counter(self) -> collections.Counter:
        orig_count = super().answer_counter()
        result = collections.Counter()
        for ans, count in orig_count.items():
            result[self._convert_ans(ans)] += count
        return result

    def answer_counter_all(self) -> collections.Counter:
        orig_count = super().answer_counter_all()
        result = collections.Counter()
        for ans, count in orig_count.items():
            result[self._convert_ans(ans)] += count
        return result

    def answer_to_label(self, label: str) -> int:
        if self.positive_answer in label.split(", "):
            return self.YES_LABEL
        return self.NO_LABEL

    def label_to_answer(self, num: int) -> str:
        if num == self.YES_LABEL:
            return self.positive_answer
        elif num == self.NO_LABEL:
            return self.OTHER_ANSWER
        else:
            raise ValueError(num)

    def records_post_process(self, records: List[dict]) -> List[dict]:
        records = super().records_post_process(records)
        for rec in records:
            # answer = rec["answer"]
            # rec["answer"] = self._convert_ans(answer)
            rec["subquestion"] = self.positive_answer
        return records


@dataclasses.dataclass
class MultiBinaryDatasetSpec(BaseDatasetSpec):
    """Unimplemented placeholder spec for loading all multibinary answers at once."""
    def __init__(self, id, context_key, answer_key):
        super().__init__(id=id, context_key=context_key, answer_key=answer_key)
        self._raw_answer_set = set()

        contracts = data.load_contracts(self.context_key)
        for d in contracts:
            contract_name = d["Filename"]
            if (answer := d.get(self.answer_key)) is None:
                raise KeyError(f"Missing answer key '{answer}' in {contract_name}.")
            self._raw_answer_set.add(answer)

        _binary_answers_set = set()

        # Go through all of the answer values in the answer column and split each by ", "
        # to recover the possible choices for the multibinary question.
        for answer in self._raw_answer_set:
            split_answers = answer.split(", ")
            assert len(split_answers) == len(set(split_answers)), split_answers
            split_answers_set = set(a.strip() for a in split_answers)
            for bad_answer in self.IGNORED_ANSWER_SET:
                split_answers_set.discard(bad_answer)
            _binary_answers_set.update(split_answers_set)
        self._sorted_binary_answers = sorted(_binary_answers_set)

    @property
    def answer_choices_ordered(self) -> Tuple[str, ...]:
        return tuple(self._sorted_binary_answers)

    def to_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def answer_counter(self) -> collections.Counter:
        return collections.Counter(["fake_counter"])

    @property
    def n_classes(self) -> int:
        return len(self.answer_choices_ordered)

    def answer_to_label(self, answer: str) -> int:
        return self._sorted_binary_answers.index(answer)

    def label_to_answer(self, num: int) -> str:
        return self._sorted_binary_answers[num]

    def make_subquestions(self) -> List[MultiBinaryDatasetSubQuestionSpec]:
        subquestions = []
        for i, curr_answer in enumerate(self._sorted_binary_answers):
            new_id = f"{self.id}-{i}"
            sub_q = MultiBinaryDatasetSubQuestionSpec(
                id=new_id,
                context_key=self.context_key,
                answer_key=self.answer_key,
                positive_answer=curr_answer,
            )
            subquestions.append(sub_q)
        return subquestions

    def has_synth_data(self) -> bool:
        return False
