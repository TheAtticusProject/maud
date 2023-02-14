import collections
import csv
import random
import regex
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import datasets
from datasets.utils import disable_progress_bar
import numpy as np
import torch.utils.data as th_data

from maud import specs, utils


# Disable tqdm progress bar for tokenizing, because it overwrites other logs when multiprocessed..
disable_progress_bar()

MODEL_INPUT_KEYS = ['input_ids', 'attention_mask']  # Use these keys for model inference.


_contracts_cached: Optional[Tuple[Dict[str, str]]] = None


def _load_contracts() -> Tuple[Dict[str, str], ...]:
    global _contracts_cached
    if _contracts_cached is None:
        with open('data/raw/main.csv') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        _contracts_cached = tuple(rows)
    return _contracts_cached


def get_headers() -> List[str]:
    contracts = _load_contracts()
    return list(contracts[0].keys())


def load_contracts(context_key=None) -> Tuple[dict, ...]:
    # Add context_key to drop rows where the extracted context is (none entered)
    if context_key is None:
        return _load_contracts()

    assert context_key in get_headers()
    contracts = _load_contracts()
    result = []

    for row in contracts:
        context: str = row[context_key]
        new_context = utils.handle_none_entered_context(context)
        if new_context is None:
            continue
        elif new_context != context:
            row = dict(**row)  # Copy record
            row[context_key] = new_context   # Replace with new_context
        result.append(row)
    assert len(result) >= 0
    return tuple(result)


def oversample(dataset: datasets.Dataset) -> th_data.Subset:
    labels = np.array([_item_if_torch(d["label"]) for d in dataset])
    label_counts = collections.Counter(labels)
    max_count = max(label_counts.values())
    n_required = {label: max_count - count for label, count in label_counts.items()}
    ind = list(range(len(dataset)))
    for i in set(labels):
        if label_counts[i] == 0:
            continue

        quot = n_required[i] // label_counts[i]
        rem = n_required[i] % label_counts[i]

        matching_label_ind = np.argwhere(labels == i).flatten().tolist()
        residual = matching_label_ind[:rem]
        ind.extend(matching_label_ind*quot + residual)
    return th_data.Subset(dataset, ind)


def strip_page_info(s: str, strict=True) -> str:
    match = regex.search(r"\(Page.*\)$", s)
    if match is None:
        if strict:
            raise ValueError(f"page not found in {s} during strict mode")
    else:
        start_idx = match.start()
        s = s[:start_idx]
    return s


def regex_match_header(pattern: str) -> str:
    headers = get_headers()
    answer_key_idx = regex_index_of(headers, pattern)
    assert answer_key_idx is not None
    answer_key = headers[answer_key_idx]
    return answer_key


def regex_index_of(strings: Sequence[str], pattern: str, strict: bool = True) -> Optional[int]:
    matches = [bool(pattern == s or regex.fullmatch(pattern, s)) for s in strings]
    if sum(matches) == 0:
        return None
    elif sum(matches) == 1:
        return matches.index(True)
    else:
        if strict:
            # To prevent unexpected silent errors, explicitly error when multiple regexes match.
            print(strings, pattern, matches)
            raise ValueError(f"Multiple answers matched {pattern}!")
        return None


def _item_if_torch(x) -> int:
    if isinstance(x, int):
        return x
    else:
        # Assume torch int
        return x.item()


def make_label_counter(ds, labels: Optional[Sequence[str]] = None) -> collections.Counter:
    def auto_key(k: int):
        if labels is None:
            return k
        else:
            return labels[k]

    counter = collections.Counter()
    if labels is not None:
        for label in labels:
            # Prepopulate labels to force an ordering and prevent missing labels later.
            counter[label] = 0

    for x in ds:
        k = auto_key(_item_if_torch(x["label"]))
        counter[k] += 1

    # Counter Insertion ordered as of python 3.7. This assertion will fail if python < 3.7!
    if labels is not None:
        assert list(labels) == list(counter.keys()), "Ordering failed!"
    return counter


def count_labels(ds, labels: Optional[Sequence[str]] = None) -> List[Tuple[Any, int]]:
    counter = make_label_counter(ds, labels)

    result: List[Tuple[Any, int]] = []
    for k in sorted(counter.keys()):
        val = counter[k]
        assert k not in set(r[0] for r in result), "Duplicate key"
        result.append((k, val))
    return result


def synth_capped_subset(
        synth_records: Sequence[dict],
        max_label_count: int,
) -> List[dict]:
    """Remove records so that the count of each label is at most `max_label_count`."""
    if max_label_count < 1:
        raise ValueError(max_label_count)

    synth_label_counts = make_label_counter(synth_records)
    synth_int_labels = set(_item_if_torch(label) for label in synth_label_counts.keys())

    synth_label_to_records: Dict[int, List[dict]] = {label: [] for label in synth_int_labels}
    for rec in synth_records:
        idx = _item_if_torch(rec["label"])
        synth_label_to_records[idx].append(rec)

    result = []
    for label, count in synth_label_counts.items():
        if count <= max_label_count:
            # Can use as is
            subset = synth_label_to_records[label]
        else:
            # select a subset.
            subset = random.sample(synth_label_to_records[label], max_label_count)
        result.extend(subset)
    return result


def keep_data_part(ds: Sequence[dict], keep_prop: float = 1.0) -> Sequence[dict]:
    """Drop examples from the dataset so that we only keep `keep_prop` of each label class,
    while ensuring that we have at least one label of each label class in the remaining dataset."""
    assert 0.0 < keep_prop <= 1.0
    _, valid_ds = _split_data_balanced(ds, valid_prop=keep_prop, error_on_only_one_sample=False)
    # The valid_ds return value is guaranteed to have at least one label of each label class.
    return valid_ds


def _split_data_balanced(ds: Sequence[dict], valid_prop: float,
                         verbose=False,
                         error_on_only_one_sample=True,
                         seed: int = None,
                         ) -> Tuple[Sequence[dict], Sequence[dict]]:
    label_set = set()
    for x in ds:
        label_set.add(x["label"])
    subsets = []
    for label in label_set:
        idx = [x["label"] == label for x in ds]
        subset = np.array(ds)[idx]
        assert len(subset) > 0
        subsets.append(subset)

    def safe_split(ds_: datasets.Dataset):
        from sklearn.model_selection import train_test_split
        train_count = int(len(ds_) * (1-valid_prop))
        if train_count == 0:
            # Dataset.train_test_split errors out if its arguments would result in a train split with 0 examples.
            # Return the closest valid test split instead, with 1 training example.
            if error_on_only_one_sample:
                # Note that train_test_split will automatically error out if there is only one sample in the dataset.
                return train_test_split(ds_, train_size=1, random_state=seed)
            else:  # We are okay with having one sample in the "validation set", but none in the training set.
                # print("NOTE: splitting by label where there is only one example!")
                return [], ds_
        return train_test_split(ds_, test_size=valid_prop, random_state=seed)

    splits: List[dict] = [safe_split(subset) for subset in subsets]
    test_ds = []
    train_ds = []
    for split in splits:
        test_ds.extend(split[1])
        train_ds.extend(split[0])
    if verbose:
        print("total:", count_labels(ds))
        print("train:", count_labels(train_ds))
        print("test:", count_labels(test_ds))
    return train_ds, test_ds


def build_balanced_split_only(
        ds: Optional[Sequence[dict]],  # Probably doesn't need to be Optional.
        valid_prop: float = 0.2,
        verbose: bool = False,
        add_ds: Sequence[dict] = (),
        synth_ds: Sequence[dict] = (),
        *,
        synth_capped: bool = False,
        strict: bool = True,
        seed: int = None,
):
    if ds is not None:
        train_ds, test_ds = _split_data_balanced(ds, valid_prop=valid_prop, verbose=verbose,
                                                 error_on_only_one_sample=False,
                                                 seed=seed)
    else:
        assert not strict
        train_ds = []
        test_ds = []

    # Adding in additional data. Prevent data leakage by putting data
    #   with the same contract name on the same side of the split.
    DUMMY_CONTRACT_NAMES = ['<RARE_ANSWERS>']
    accept_add_data = set([a["contract_name"] for a in train_ds])
    accept_add_data_test = set([a["contract_name"] for a in test_ds])
    intersection = accept_add_data.intersection(accept_add_data_test)
    for contract_name in DUMMY_CONTRACT_NAMES:
        intersection.discard(contract_name)
    assert len(intersection) == 0
    set_labels = set([a["label"] for a in train_ds])
    temp = len(train_ds)
    for rec in add_ds:
        ## Add compatible records to test set.
        if rec["contract_name"] in accept_add_data_test:
            test_ds.append(rec)
        else:
            train_ds.append(rec)

    if synth_capped:
        label_counts = make_label_counter(ds)
        max_answer_count = max(label_counts.values())
        synth_ds = synth_capped_subset(synth_ds, max_answer_count)
    for rec in synth_ds:
        train_ds.append(rec)  # no testset contamination
        # If we want to include synth in testset,
        # then we will add it to `splits` above instead of doing this.

    # Sanity check
    train_counter = _count_labels(train_ds)
    test_counter = _count_labels(test_ds)
    if train_counter.keys() != test_counter.keys():
        raise ValueError(f"Mismatching keys: train={train_counter} and test={test_counter}")
    return train_ds, test_ds


def _count_labels(records: dict) -> collections.Counter:
    result = collections.Counter()
    for rec in records:
        result[rec["label"]] += 1
    return result


def build_balanced_split_and_tokenize(
        ds: Sequence[dict],
        tokenizer: Optional["Tokenizer"] = None,
        valid_prop: float = 0.2,
        verbose: bool = False,
        add_ds: Sequence[dict] = (),
        synth_ds: Sequence[dict] = (),
        *,
        synth_capped: bool = False,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    train_ds, test_ds = build_balanced_split_only(
        ds, valid_prop, verbose,
        add_ds, synth_ds, synth_capped=synth_capped)

    if tokenizer is not None:
        train_ds = specs._encode_records_as_dataset(train_ds, tokenizer)
        test_ds = specs._encode_records_as_dataset(test_ds, tokenizer)

    return train_ds, test_ds


_add_rows = None
def _load_add_rows():
    global _add_rows
    if _add_rows is None:
        with open('data/raw/abridged.csv') as f:
            reader = csv.reader(f)
            _add_rows = list(reader)
    return _add_rows


def load_bonus_context(answer_key, *, verbose=False):
    rows = _load_add_rows()
    bonus_preamble_rows, bonus_content_rows = rows[0:3], rows[3:]

    answer_column = -1
    for e, a in enumerate(bonus_preamble_rows[2]):
        if a == answer_key:
            answer_column = e
    if answer_column == -1:
        return None
    context_column = -1

    for e, a in enumerate(bonus_preamble_rows[1][answer_column::-1]):
        if a != '':
            # The assumption here is that the context column has non empty cells at rows 1 and 2,
            # usually to reference columns from the original dataset.
            context_column = answer_column - e
            break
    contexts = []
    answers = []
    contract_names = []
    for a in bonus_content_rows:
        contexts.append(a[context_column])
        answers.append(a[answer_column])
        contract_names.append(a[0])

    result_contract_names = []
    result_answers = []
    result_contexts = []
    for i in range(len(contexts)):
        curr_context = contexts[i]
        matches = list(regex.finditer(r"\(Page.*\)", curr_context))
        if len(matches) == 0:
            # warnings.warn(f'Broken additional context cell: "{curr_context}"')
            # This is fine.
            pass
        elif len(matches) != 2:
            # This function currently parse additional context cells with 2 contexts in it (we didn't know
            # that you could have 4 contexts in one cell).

            # Idea: Parse between matches[j].end() and matches[j].start()?
            #  No this is too hard, in fact the data format is irregular.
            #  And we already have more additional records than main records.
            PARSE_ALL_MATCHES = False
            if PARSE_ALL_MATCHES:
                raise NotImplementedError
            else:
                # We are just going to skip cells whose format we don't understand for now.
                # Heterogeneous data format for this spreadsheet.
                # print(f"Skipped additional cell with {len(matches)}!=2 examples.")
                # print(curr_context)
                pass
        else:
            split_idx = matches[0].end()
            assert matches[1].end() == len(curr_context), "context should end in (Page.*) string"

            context1 = curr_context[:split_idx].strip()
            context2 = curr_context[split_idx:].strip()
            assert len(context1) >= 5  # Would be surprising if context were less than 5 characters.
            assert len(context2) >= 5

            # result_contexts.append(strip_page_info(context1.strip()))
            result_contexts.append(context1)
            result_answers.append(answers[i])
            result_contract_names.append(contract_names[i])

            # result_contexts.append(strip_page_info(context2.strip()))
            result_contexts.append(context2)
            result_answers.append(answers[i])
            result_contract_names.append(contract_names[i])

    if verbose:
        print(str(len(result_contexts)) + " ADDITIONAL DATA ADDED")
    return result_contract_names, result_contexts, result_answers


_synth_data_rows_cached: Optional[List[List[str]]] = None
def _load_synthetic_data() -> List[List[str]]:
    """Load (huge) CSV from synthetic_data.csv."""
    global _synth_data_rows_cached
    if _synth_data_rows_cached is not None:
        return _synth_data_rows_cached
    with open('data/raw/counterfactual.csv') as f:
        reader = csv.reader(f)
        rows = list(reader)
    _synth_data_rows_cached = rows
    return rows


def _ignore_synth_context(context: str) -> bool:
    if context.lower().strip() in ["", "(none entered)"]:
        return True
    return False


def load_synth_data(answer_key: str) -> Optional[Tuple[List[str], List[str]]]:
    # Structure of synthetic data rows:
    #
    # Special columns that should be ignored when parsing data:
    # Column 0: Names of the rows
    # Column -1: CONTRACT NAME  (only apply to some cells, so we aren't using this)
    #
    # Row 0: Annotator Information, unused by this script.
    # Row 1: 🎯 Column number (reference to main data)
    # Row 2: Answer column name in main data.
    # Row 3: Answer values (e.g. Yes or No) of all the synthetic contexts in this column.
    # Row 4: Answer count column
    # Row >=5: data rows.

    # New structure of synth data:
    #
    # Row 0: Answer key   (deal point question)
    # Row 1: Answer value   (deal point answer)
    # Row 2+: Counterfactual context (deal point text)
    synth_rows = _load_synthetic_data()
    synth_answer_keys = synth_rows[0]
    synth_answer_values = synth_rows[1]
    synth_contexts = synth_rows[2:]
    matching_cols: List[int] = []

    debug_mapping: Dict[str, List[str]] = collections.defaultdict(list)
    for key, val in zip(synth_answer_keys, synth_answer_values):
        assert val not in debug_mapping[key], f"Duplicate question/answer pair {key}, {val}"
        debug_mapping[key].append(val)

    for col, _answer_key in enumerate(synth_answer_keys):
        if answer_key.lower() == _answer_key.lower():
            matching_cols.append(col)

    if len(matching_cols) == 0:
        return None

    contexts = []
    answers = []
    for col in matching_cols:
        answer_value = synth_answer_values[col]
        for row in synth_contexts:
            context = row[col]
            if _ignore_synth_context(context):
                continue
            contexts.append(context)
            answers.append(answer_value)
    assert len(contexts) > 0
    assert len(contexts) == len(answers)
    return contexts, answers