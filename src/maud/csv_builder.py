import csv
import pathlib
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from maud import auto_load_data, category_utils, data, data_utils


home = pathlib.Path("/tmp")
# home = pathlib.Path(".")
train_csv_path = home / "MAUD_train.csv"
dev_csv_path = home / "MAUD_dev.csv"
test_csv_path = home / "MAUD_test.csv"
contract_splits_path = home / "_MAUD_contract_splits.csv"
# train_pkl_path = home / "MAUD_train.pkl"
# test_pkl_path = home / "MAUD_test.pkl"


# IDEA: Also return the mask used to drop bad rows. We can use this for
#   reconstructing results.
def drop_bad_add_rows(df: pd.DataFrame, all_df: pd.DataFrame, *, key: str = None) -> pd.DataFrame:
    """
    Return df, but with certain duplicate rows dropped.
    If key is provided, then save the mask used to decide with rows to keep to disk,
        in `home` as `home / "MAUD_keep_mask_{key}.csv"`.
    """
    df_orig = df
    main_texts = all_df[all_df.data_type=="main"]["text"]
    main_texts_set = set(main_texts)
    dup_mask = df.duplicated(keep="first")
    add_mask = df.data_type == "abridged"
    match_main_text_mask = df["text"].isin(main_texts_set)

    dup_add_mask: pd.Series = (add_mask & dup_mask) | (add_mask & match_main_text_mask)
    # print(np.sum(add_mask & dup_add_mask))
    # print(np.sum(add_mask & match_main_text_mask))
    # print(np.sum(dup_add_mask))
    # print(np.sum(match_main_text_mask))
    df_no_dup_add = df[~dup_add_mask]

    if key is not None:
        save_path = home / f"MAUD_keep_mask_{key}.csv"
        keep_mask = ~dup_add_mask
        keep_mask.to_csv(save_path, index=False)
        print(f"Saved keep mask (size={len(keep_mask)}) to {save_path}")

    print(f"dropped {len(df_orig) - len(df_no_dup_add)} dup rows")
    # print("This should be zero", np.sum(df_no_dup_add.duplicated()))
    return df_no_dup_add


def post_process(df: pd.DataFrame) -> pd.DataFrame:
    mapper = dict(context="text")
    df = df.rename(mapper)
    return df


def get_all_contract_names() -> Set[str]:
    contract_names = set()
    for spec in auto_load_data.get_all_valid_specs():
        main_ds = spec._load_data_records()
        synth_ds = spec.to_synth_dataset() or []
        add_ds = spec.to_additional_dataset() or []
        for ds in main_ds, synth_ds, add_ds:
            for rec in ds:
                name = rec["contract_name"]
                contract_names.add(name)
    return contract_names


CONTRACT_NAMES_SET = get_all_contract_names()


def _anonymize_contract_name(rec: dict) -> dict:
    rec = dict(rec)
    name = rec["contract_name"]
    rec["contract_name"] = ANON_CONTRACT_MAPPING[name]
    return rec


def anonymize_contracts(recs: List[dict]) -> List[dict]:
    return list(map(_anonymize_contract_name, recs))


def splitting_hares(valid_prop=0.18):
    print(f"There are {len(CONTRACT_NAMES_SET)} unique and recognized contract names.")
    train_records = []
    dev_records = []
    test_records = []
    for spec in auto_load_data.get_all_valid_specs():
        main_ds = anonymize_contracts(spec._load_data_records())
        synth_ds = anonymize_contracts(spec.to_synth_dataset() or [])
        add_ds = anonymize_contracts(spec.to_additional_dataset() or [])
        train_dev_ds, test_ds = data.build_balanced_split_only(
            ds=main_ds,
            verbose=False,
            valid_prop=valid_prop,
            add_ds=add_ds,
            synth_ds=synth_ds,
        )
        assert len(train_dev_ds) > len(test_ds)

        data_types = ["main", "abridged", "rare_answers"]
        grouped_train_dev_data: Dict[str, List[dict]] = {dt: [] for dt in data_types}
        for rec in train_dev_ds:
            assert rec["data_type"] in data_types
            grouped_train_dev_data[rec["data_type"]].append(rec)

        spec_train_records, spec_dev_records = data_utils.split_train_dev(
            main_ds=grouped_train_dev_data["main"],
            add_ds=grouped_train_dev_data["abridged"],
            cf_ds=grouped_train_dev_data["rare_answers"],
        )
        assert len(spec_train_records) + len(spec_dev_records) == len(train_dev_ds)
        train_records.extend(spec_train_records)
        dev_records.extend(spec_dev_records)
        test_records.extend(test_ds)

    for rec in [*train_records, *dev_records, *test_records]:
        rec["category"] = category_utils.question_to_category[rec["question"]]

    rec_splits = dict(train=train_records, dev=dev_records, test=test_records)
    df_splits_pre = {k: pd.DataFrame.from_records(recs) for k, recs in rec_splits.items()}
    full_df_pre = pd.concat(list(df_splits_pre.values()))

    df_splits = {k: post_process(df) for k, df in df_splits_pre.items()}
    df_all = pd.concat(df_splits.values())
    df_splits_no_dup = {k: drop_bad_add_rows(df, df_all, key=k) for k, df in df_splits.items()}

    df_train, df_dev, df_test = [df_splits_no_dup[k] for k in ["train", "dev", "test"]]

    assert set(df_dev["question"].unique()) == set(df_test["question"].unique())
    assert set(df_dev["subquestion"].unique()) == set(df_test["subquestion"].unique())
    print(df_test["data_type"].value_counts())

    df_train.to_csv(train_csv_path, index=False)
    df_dev.to_csv(dev_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)
    print(f"Wrote {len(df_train)} records to {train_csv_path}")
    print(f"Wrote {len(df_dev)} records to {dev_csv_path}")
    print(f"Wrote {len(df_test)} records to {test_csv_path}")

    sanity_checks(df_train, df_dev, df_test)

    df_loaded = pd.read_csv(train_csv_path)
    val = df_loaded.values == df_train.values
    assert np.all(val)


def df_to_grouped_records(
        df: pd.DataFrame,
        question: str,
        subquestion: str = "<NONE>",
) -> Dict[str, List[dict]]:
    mask = (df["question"] == question) & (df["subquestion"] == subquestion)
    assert np.sum(mask) > 0
    df = df[mask]
    result = {}
    for data_type in ["main", "abridged", "rare_answers"]:
        mask2 = df["data_type"] == data_type
        result[data_type] = df[mask2].to_dict("records")
    return result


def make_anon_contract_mapping() -> Dict[str, str]:
    contracts_set = CONTRACT_NAMES_SET

    # This is a mapping from:
    #    ABC.pdf\n\nABCAmendment.pdf => ABC.pdf\n\nABCAmendment.pdf
    #    XYZ.pdf => XYZ.pdf  if there is no amendment augmented contract.
    #    ABC.pdf => ABC.pdf\n\nABCAmendment.pdf  if there is the amendment augmented contract.
    contract_name_to_merged_name = {contract: contract for contract in contracts_set}

    for name in contracts_set:
        if "\n\n" in name:
            assert "Amendment" in name
            prefix = name.split("\n\n")[0]
            if "Kindred" not in prefix and "Aegion" not in prefix and "Capstead" not in prefix:
                # e.g. prefix=ABC.pdf  and name="ABC.pdf\n\nABC_Amendments.pdf"
                assert prefix in contracts_set
                contract_name_to_merged_name[prefix] = name
            else:
                assert prefix not in contracts_set

    # merged_contract_names is a list of all unique contract names,
    #    excluding contract names that have a corresponding amendment-augmented
    #    contract. E.g. we would include "ABC.pdf\n\nABC_Amendments.pdf" but not "ABC.pdf".
    merged_contract_names = set(contract_name_to_merged_name.values())
    # Should be merged into augmented name
    assert "Cantel Medical Corp._STERIS plc.pdf" not in merged_contract_names

    SPECIAL_CONTRACT = "<RARE_ANSWERS>"  # This is not a contract. Do not allocate contract ID.
    merged_contract_names.discard(SPECIAL_CONTRACT)
    merged_contract_name_to_generic_name = {SPECIAL_CONTRACT: SPECIAL_CONTRACT}

    for i, contract in enumerate(sorted(merged_contract_names)):
        merged_contract_name_to_generic_name[contract] = f"contract_{i}"

    merged_contract_name_to_generic_name[SPECIAL_CONTRACT] = SPECIAL_CONTRACT

    # Write contract ID mapping to file.
    with open("/tmp/contracts.txt", "w") as f:
        for i, contract in enumerate(sorted(merged_contract_names)):
            f.write(f"{i}. {contract}\n")
            # print(i, contract)

    direct_mapping = {}
    A, B = contract_name_to_merged_name, merged_contract_name_to_generic_name
    for k in A.keys():
        direct_mapping[k] = B[A[k]]

    assert CONTRACT_NAMES_SET == set(direct_mapping.keys())
    return direct_mapping


ANON_CONTRACT_MAPPING = make_anon_contract_mapping()


def sanity_checks(df_train, df_dev, df_test):
    df_train_marked = df_train.copy()
    df_dev_marked = df_dev.copy()
    df_test_marked = df_test.copy()
    df_train_marked["split"] = "train"
    df_dev_marked["split"] = "dev"
    df_test_marked["split"] = "test"

    df_all_post = pd.concat([df_train_marked, df_dev_marked, df_test_marked])
    print(df_all_post.groupby("data_type").size())
    print(df_all_post.groupby("split").size())
    # print(df_all_post.groupby(["data_type", "category"]).size())
    # print(df_all_post.columns)  # want spec id or something that can split contracts
    contract_splits_df = df_all_post.groupby(["question", "subquestion", "split", "contract_name"]).size()
    contract_splits_df.to_csv(contract_splits_path)
    print(f"Wrote contract split info to {contract_splits_path}")

    # Now I'm going to check whether XYZ is ok.
    # I mean, whether there is any overlap between the contracts, unit test style.
    # Or I could move on to the next step? Here's the idea. For every question, subquestion combination, assign an ID.
    question_tup_set = set()

    for i, row in df_all_post.iterrows():
        q = row["question"]
        sq = row["subquestion"]
        question_tup_set.add((q, sq))

    map_question_tup_to_id = {}
    for i, q_tup in enumerate(sorted(question_tup_set)):
        map_question_tup_to_id[q_tup] = i
    print(len(question_tup_set))
    # print(map_question_tup_to_id)

    # Next, we add the q_id to the DataFrame.
    # The point of all this coding is check whether there is overlap between the contracts.
    # I don't need this ID right now. THis is a waste of time.
    # I can do the same in a naive way --- I can iterate over pairs of question and subquestion.

    for q, sq in question_tup_set:
        mask = (df_all_post["question"] == q) & (df_all_post["subquestion"] == sq)
        df_isolate = df_all_post[mask]
        # OK, now we want to check for overlap between contracts.
        # We have three splits df_isolate["split"] \in {"train", "dev", "test"}
        contract_sets = []
        for split_name in ["train", "dev", "test"]:
            mask = df_isolate["split"] == split_name
            df_isolate_single_split = df_isolate[mask]
            contracts = set(df_isolate_single_split["contract_name"])
            contracts.discard("<RARE_ANSWERS>")
            contract_sets.append(contracts)

        # Sanity check: Ensure that there is no overlap between the sets.
        for i in range(len(contract_sets)):
            for j in range(i+1, len(contract_sets)):
                set_i = contract_sets[i]
                set_j = contract_sets[j]
                assert set_i.isdisjoint(set_j)


if __name__ == "__main__":
    SEED = 50
    np.random.seed(SEED)
    splitting_hares()
