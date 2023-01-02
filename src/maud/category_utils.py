import collections
import pathlib
import pickle
from typing import Dict, List

from matplotlib import pyplot as plt
import pandas as pd
import tqdm


from maud import auto_load_data, specs


def load_all_records(add_main: bool = False, add_synth: bool = False, add_add: bool = False) -> List[dict]:
    records = []
    for spec in auto_load_data.get_all_valid_specs():
        if add_main:
            main_data = spec._load_data_records()
            records.extend(main_data)
        if add_synth:
            synth_data = spec.to_synth_dataset() or []
            records.extend(synth_data)
        if add_add:
            add_data = spec.to_additional_dataset() or []
            records.extend(add_data)
    return records


def load_all_contexts(**kwargs):
    records = load_all_records(**kwargs)
    contexts = []
    for rec in records:
        contexts.append(rec["text"])
    return contexts


def build_spec_id_to_context_set(
        add_main: bool = False, add_synth: bool = False, add_add: bool = False,
) -> Dict[str, set]:
    result = collections.defaultdict(set)
    for spec in auto_load_data.get_all_valid_specs():
        records = []
        if add_main:
            main_data = spec._load_data_records()
            records.extend(main_data)
        if add_synth:
            synth_data = spec.to_synth_dataset() or []
            records.extend(synth_data)
        if add_add:
            add_data = spec.to_additional_dataset() or []
            records.extend(add_data)

        for rec in records:
            result[spec.id].add(rec["text"])
    return result


def get_spec_id_to_max_tokens(spec_id: str, tokenizer = None):
    if tokenizer is None:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/bigbird-roberta-base")
    spec_id_to_context_set = build_spec_id_to_context_set(
        add_main=True,
        add_add=True,
        add_synth=True)

    context_set = spec_id_to_context_set[spec_id]
    n_tokens_list = []
    for context in context_set:
        tokens = tokenizer.tokenize(context)
        n_tokens_list.append(len(tokens))
    max_n_tokens = max(n_tokens_list)
    return max_n_tokens


def dict_spec_id_to_max_tokens(tokenizer=None, dirty_cache: bool = False) -> Dict[str, int]:
    """Builds a dict from valid_specs (e.g. with id 10.2) to max num tokens."""
    CACHE_PATH = pathlib.Path("scrap/spec_id_to_max_tokens.pkl")
    if not dirty_cache and CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            result = pickle.load(f)
        print(f"Loaded from cache {str(CACHE_PATH)}")
        return result

    if tokenizer is None:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "google/bigbird-roberta-base")
    result = {}
    for spec in tqdm.tqdm(valid_specs):
        spec_id = spec.id
        result[spec_id] = get_spec_id_to_max_tokens(spec_id, tokenizer=tokenizer)

    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result


def get_valid_spec_with_id(spec_id: str) -> specs.BaseDatasetSpec:
    for spec in valid_specs:
        if spec.id == spec_id:
            return spec
    raise KeyError(spec_id)


def get_bigbird_spec_ids_by_gpu_name() -> Dict[str, List[str]]:
    spec_id_to_max_tokens = dict_spec_id_to_max_tokens()
    rows = []
    for spec_id, max_tokens in spec_id_to_max_tokens.items():
        row = dict(spec_id=spec_id, max_tokens=max_tokens)
        rows.append(row)
    df = pd.DataFrame.from_records(rows)

    THRESHOLDS = [0, 700, 1200, 2100, 8000]  # vanilla
    # THRESHOLDS = [0, 750, 1200, 2500, 8000]  # w/ amp
    spec_id_lists = []
    gpu_name_to_spec_ids: Dict[str, List[str]] = {}
    names = ["small", "mid", "balrog", "saruman"]
    for gpu_name, (thresh_lo, thresh_hi) in zip(names, zip(THRESHOLDS[:-1], THRESHOLDS[1:])):
        mask1 = thresh_lo < df["max_tokens"]
        mask2 = df["max_tokens"] <= thresh_hi
        mask = mask1 & mask2
        spec_lst = list(df[mask]["spec_id"])
        spec_id_lists.append(spec_lst)
        gpu_name_to_spec_ids[gpu_name] = spec_lst
    assert sum(len(x) for x in gpu_name_to_spec_ids.values()) == len(valid_specs)

    # Validate
    _all_specs = set()
    for spec_ids in gpu_name_to_spec_ids.values():
        for sid in spec_ids:
            assert sid not in _all_specs
            _all_specs.add(sid)
    assert len(_all_specs) == len(valid_specs)

    return gpu_name_to_spec_ids


MAIN_CONTEXTS_SET = set(load_all_contexts(add_main=True))
spec_id_to_context_set: Dict[str, set]


categories_to_contexts = {
    "General Information": ["Type of Consideration"],
    "Conditions to Closing": [
        "Accuracy of Target R&W Closing Condition",
        "Compliance with Covenant Closing Condition",
        "Absence of Litigation Closing Condition",
    ],
    "Material Adverse Effect": ["MAE Definition"],
    "Knowledge": ["Knowledge Definition"],
    "Operating and Efforts Covenant":
        ["Ordinary course covenant", "Negative interim operating covenant", "General Antitrust Efforts Standard"],
    "Deal Protection and Related Provisions": [
      'No-Shop',
      'Fiduciary exception:  Board determination (no-shop)',
      'Fiduciary exception to COR covenant',
      'Agreement provides for matching rights in connection with COR',
      'Superior Offer Definition',
      'Intervening Event Definition',
      'FTR Triggers',  # Unfortunately only one example of a second label class.
      'Limitations on FTR Exercise',
      'Agreement provides for matching rights in connection with FTR',
      'Tail Period & Acquisition Proposal Details',
      # '"General" Breach (not specific to No-Shop or Meeting Covenant)',  # Only one example.
      'Breach of No Shop',
      'Breach of Meeting Covenant',
      # 'Limitations on Antitrust Efforts',   # Double context required. We now manually block.
    ],
    "Remedies": ["Specific Performance",],
}

CATEGORIES = list(categories_to_contexts.keys())

# Includes MultiBinaryDatasetSpec, does not include children.
all_specs = auto_load_data.generate_specs()
# Includes MultiBinaryDatasetSubquestionSpec, does not include parent
valid_specs = auto_load_data.get_all_valid_specs()

context_key_to_specs = collections.defaultdict(list)
answer_key_to_valid_specs: Dict[str, List[specs.BaseDatasetSpec]] = collections.defaultdict(list)

context_keys_set = set()
for spec in all_specs:
    if isinstance(spec, specs.MultiBinaryDatasetSubQuestionSpec):
        # Probably skip.
        # Rather, we will describe the parent spec.
        continue
    context_key_to_specs[spec.context_key].append(spec)
    context_keys_set.add(spec.context_key)

for spec in valid_specs:
    answer_key_to_valid_specs[spec.answer_key].append(spec)

context_to_category = dict()
cat_context_set = set()
for category, contexts in categories_to_contexts.items():
    for context in contexts:
        assert context in context_key_to_specs.keys(), context
        assert context not in cat_context_set, f"Repeated context {context}"
        cat_context_set.add(context)
        context_to_category[context] = category

if cat_context_set != context_keys_set:
    raise ValueError("{}".format(cat_context_set.symmetric_difference(context_keys_set)))

# WARNING: All of these spec IDS will be based on the base questions. The IDs will contain no "."
categories_to_spec_ids = collections.defaultdict(list)
spec_id_to_category = dict()
spec_id_to_spec = dict()
spec_id_to_question = dict()
question_to_category = dict()
for category, contexts in categories_to_contexts.items():
    for context in contexts:
        for spec in context_key_to_specs[context]:
            spec: specs.BaseDatasetSpec
            categories_to_spec_ids[category].append(spec.id)
            spec_id_to_category[spec.id] = category
            spec_id_to_spec[spec.id] = spec
            spec_id_to_question[spec.id] = spec.answer_key
            if (cat := question_to_category.get(spec.answer_key)) is not None:
                assert cat == category
            else:
                question_to_category[spec.answer_key] = category

assert cat_context_set == context_keys_set


def get_all_context_keys() -> List[str]:
    return list(context_keys_set)


def print_md_like_category_table():
    for category, contexts in categories_to_contexts.items():
        print(f"## CATEGORY: {category}")
        for context in contexts:
            matching_specs = context_key_to_specs[context]
            print_context_specs(context, matching_specs)


def make_df() -> pd.DataFrame():
    rows = []
    for spec in all_specs:
        if isinstance(spec, specs.MultiBinaryDatasetSubQuestionSpec):
            continue
        category = context_to_category[spec.context_key]
        yn_mapping = {True: "Y", False: "N"}
        row = {
            "Category": category,
            "Type": spec.context_key,
            "Question": spec.answer_key,
            "# Answers": spec.n_classes,
            "Multilabel": yn_mapping[isinstance(spec, specs.MultiBinaryDatasetSpec)],
        }
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    return df


BIG_DF = make_df()


def write_latex(path="scrap/dataset.tex"):
    df = make_df()
    print(df)
    latex_str = df.to_latex(index=False)
    with open(path, "w") as f:
        f.write(latex_str)
    print(f"Wrote to {path}.")


def write_csv(path="scrap/dataset.csv"):
    df = make_df()
    print(df)
    latex_str = df.to_csv(index=False)
    with open(path, "w") as f:
        f.write(latex_str)
    print(f"Wrote to {path}.")


def make_label_dist_plot():
    df = make_df()["# Answers"]
    print(df)
    plt.hist(df)
    plt.show()


def print_context_specs(context_key, matching_specs):
    print(f"TEXT_TYPE: {context_key}")
    for i, spec in enumerate(matching_specs):
        if isinstance(spec, specs.MultiBinaryDatasetSpec):
            suffix = " [MULTILABEL]"
        else:
            suffix = ""
        print(f"\tQUESTION {i+1}: {spec.answer_key}{suffix}")
        for j, answer in enumerate(spec.answer_choices_ordered):
            print(f"\t\tANSWER {j+1}: {answer}")


if __name__ == "__main__":
    print_md_like_category_table()
    # write_csv()

