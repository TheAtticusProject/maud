import collections
import functools
import itertools
import json
import pathlib
import pickle
import numpy as np
import tqdm
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

from matplotlib import pyplot as plt
import pandas as pd

from maud import category_utils, pr_curves, utils


FAST = False
DIRTY_CACHE = False


# Values for q_keep_mode argument:
KEEP_ALL_POSSIBLE = None
# Calculate with all available main/additional records for every
# question.
# Pro: Uses all records.
# Con: Different test question set for main and add.

KEEP_ADD_SUPPORT_ONLY = "keep_add"
# Keep only records that have any additional records.
# Pro: Same test question set for main and add.
# Con: Some questions (ones that don't have additional examples) will be excluded.


data_types_paths = [
    pathlib.Path("scrap/eval_utils/test_dl_data_types3.json"),
    pathlib.Path("scrap/eval_utils/test_dl_data_types.json"),
    pathlib.Path("scrap/eval_utils/test_dl_data_types_backup.json"),
]


def _get_matching_data_types_list(record):
    ok = False
    data_types_list = None
    logits, labels, spec_id = record["logits"], record["labels"], record["spec_id"]
    for path in data_types_paths:
        with open(path) as f:
            sid_to_data_types_list: Dict[str, List[str]] = json.load(f)
            # Example: {"10.1": ["main", "main", ... "additional"], ...}
            # The data types list tells us which indices of the logits
            # are associated with the "main" subset xor the "additional" subset.
        data_types_list = sid_to_data_types_list[record["spec_id"]]
        if len(data_types_list) == len(logits):
            ok = True
            break
    if not ok:
        # No match.
        raise ValueError("I need to regenerate test_dl_data_types.json because "
                         "the dataset has been regenerated.")

    assert len(data_types_list) == len(logits)
    assert np.sum(np.array(data_types_list) == "main") > 0
    return data_types_list


def has_additional_samples(record):
    data_types_list = _get_matching_data_types_list(record)
    return collections.Counter(data_types_list)["abridged"] > 0


def get_filtered_labels_and_logits(record, data_type_filter):
    """Refactor out new_labels and new_logits generator so that we can continue."""
    assert data_type_filter in {"main", "abridged"}
    logits: np.ndarray = record["logits"]
    labels: np.ndarray = record["labels"]
    data_types_list = _get_matching_data_types_list(record)
    dtype_mask: np.ndarray = np.array(data_types_list) == data_type_filter
    new_logits = logits[dtype_mask]
    new_labels = labels[dtype_mask]
    assert len(new_labels.shape) == 1
    return new_logits, new_labels


def build_full_df(records: List[dict], *, save_dir: pathlib.Path = None,
                  q_keep_mode: Optional[str] = KEEP_ALL_POSSIBLE,
                  data_type_filter: Optional[str] = None,
                  proc_epoch: bool = False,
                  ) -> pd.DataFrame:
    assert isinstance(records[0], dict)
    if q_keep_mode is KEEP_ALL_POSSIBLE and data_type_filter == "abridged":
        raise ValueError("Doesn't make sense use q_keep_mode is None when data_type_filter='additional'"
                         "Consider using q_keep_mode='keep_add' instead."
                         )

    df_rows = []
    for record in tqdm.tqdm(records, desc="full df from records"):
        _, n_classes = record["logits"].shape

        if q_keep_mode is not None:
            if q_keep_mode == KEEP_ADD_SUPPORT_ONLY:
                # If there aren't enough test additional samples for this question
                # to have at least one sample of each class, then skip this question.
                assert record["args"].eval_split == "test"
                _, _add_labels = get_filtered_labels_and_logits(record, "abridged")
                if len(_add_labels) == 0:
                    continue
                if len(set(_add_labels)) != n_classes:
                    continue
            else:
                raise NotImplementedError(q_keep_mode)

        if data_type_filter is not None:
            # Replace `record` with a copy that only keeps examples of type `data_type_filter`.
            assert data_type_filter in {"main", "abridged"}
            new_logits, new_labels = get_filtered_labels_and_logits(record, data_type_filter)

            if len(new_logits) == 0:
                continue
            record = dict(record)
            record["logits"] = new_logits
            record["labels"] = new_labels
            # We expect that there is always at least one example of each class.
            # If filtering on the data type would cause us to lose all examples for a class,
            # (only possible for "additional" filtering)
            # then we should have already skipped this example in the `q_keep_mode` filtering
            # step.
            assert len(set(new_labels)) > 1

        curves = pr_curves.MAUDPrecRecallCurve.from_tune_record_binarized(record)
        for c in curves:
            n_classes = record["logits"].shape[1]
            spec_id = record["spec_id"]
            category = category_utils.spec_id_to_category[spec_id.split("-")[0]]
            if np.isnan(c.auprc):
                raise ValueError(c)
            args = record["args"]
            if not hasattr(args, "partial_data") or args.partial_data is None:
                partial = 1.0
            else:
                partial = args.partial_data
            row = dict(
                aupr=c.auprc,
                category=category,
                spec_id=record["spec_id"],
                run_num=record["run_num"],
                lr=args.learning_rate,
                partial=partial,
                model_name=args.model,
                update_num=record["batch_num"],
                n_classes=n_classes,
            )
            if proc_epoch:
                row["epoch_num"] = record["epoch"]
            df_rows.append(row)
    df = pd.DataFrame.from_records(df_rows)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "full_df.csv"
        print(f"Saved full dataframe to {path}.")
        df.to_csv(path)
    return df


def iter_masked_dfs(df: pd.DataFrame, col_names: Collection[str]) -> Iterable[Tuple[Dict[str, Any], pd.DataFrame]]:
    """Iterate through every combination of unique values for each column.

    Yields a dict with the unique key-value combinations used to mask the DF, and the masked DF.
    """
    assert len(set(col_names)) == len(col_names)
    for col in col_names:
        assert col in df.columns
    col_unique_vals = [sorted(set(df[col_name])) for col_name in col_names]
    for col_vals in tqdm.tqdm(list(itertools.product(*col_unique_vals)), desc="masked dfs"):
        mask = np.ones(len(df), dtype=bool)
        for col_name, col_val in zip(col_names, col_vals):
            mask &= (df[col_name] == col_val)
        if np.sum(mask) == 0:
            continue
        mask_dict = dict(zip(col_names, col_vals))
        masked_df = df[mask]
        yield mask_dict, masked_df


def reduce_aupr_df_mean(df, *, query_col_names, drop_col_names=(), weight_col_name=None, strict=True) -> pd.DataFrame:
    """Builds a Dataframe containing aupr over `query_col_names`, while dropping columns in `drop_col_names`
    and keeping all remaining columns constant.

    If strict is True, then ensure that all columns not in `col_names` and not in `drop_col_names` are
    have the same value.

    If weight_col is provided, then weight the mean using values in the weight_col, which should be nonnegative.
    The weight_column is dropped from the final dataframe.
    """
    df = df.copy(deep=False)
    if weight_col_name is not None:
        assert not np.any(df[weight_col_name] < 0)
        assert not np.any(np.isnan(df[weight_col_name]))
    assert len(query_col_names) == len(set(query_col_names))
    assert len(set(drop_col_names)) == len(drop_col_names)

    query_col_names = set(query_col_names)
    drop_col_names = set(drop_col_names)
    all_col_names = set(df.columns)
    assert query_col_names <= all_col_names
    assert drop_col_names <= all_col_names
    assert len(query_col_names.intersection(drop_col_names)) == 0

    const_cols = all_col_names.difference(query_col_names).difference(drop_col_names).difference({"aupr"})
    if weight_col_name:
        assert weight_col_name in drop_col_names
        const_cols.discard(weight_col_name)
    rows = []
    for mask_dict, masked_df in iter_masked_dfs(df, query_col_names):
        const_dict = {}
        for const_col in const_cols:
            if strict:
                assert len(set(masked_df[const_col])) == 1, (const_col, set(masked_df[const_col]))
            const_dict[const_col] = masked_df.iloc[0][const_col]
        if not weight_col_name:
            mean_aupr = masked_df["aupr"].mean()
        else:
            weights = masked_df[weight_col_name]
            assert np.sum(weights) > 0
            mean_aupr = np.sum(masked_df["aupr"] * weights) / np.sum(weights)
            assert mean_aupr.shape == ()
        row = {**mask_dict, "aupr": mean_aupr, **const_dict}
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def reduce_aupr_df_max(df, *, query_col_names, reduce_col_names=(), strict=True) -> pd.DataFrame:
    """Builds a Dataframe containing the max aupr for each setting `query_col_names`,
    while keeping in each row the values associated the best aupr for each column in `reduce_col_names`,
    and keeping all remaining columns constant.

    If strict is True, then ensure that all columns not in `col_names` and not in `reduce_col_names` are
    required to have the same value.
    """
    assert len(query_col_names) == len(set(query_col_names))
    assert len(set(reduce_col_names)) == len(reduce_col_names)

    query_col_names = set(query_col_names)
    reduce_col_names = set(reduce_col_names)
    all_col_names = set(df.columns)
    assert query_col_names <= all_col_names
    assert reduce_col_names <= all_col_names
    assert len(query_col_names.intersection(reduce_col_names)) == 0

    const_cols = all_col_names.difference(query_col_names).difference(reduce_col_names).difference({"aupr"})
    rows = []
    for mask_dict, masked_df in iter_masked_dfs(df, query_col_names):
        const_dict = {}
        for const_col in const_cols:
            if strict:
                assert len(set(masked_df[const_col])) == 1, (const_col, set(masked_df[const_col]))
            const_dict[const_col] = masked_df.iloc[0][const_col]
        max_aupr = masked_df["aupr"].max()
        max_idx_aupr = masked_df["aupr"].argmax()
        reduced_dict = {k: masked_df.iloc[max_idx_aupr][k] for k in reduce_col_names}
        row = {**mask_dict, **reduced_dict, "aupr": max_aupr, **const_dict}
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def restrict_num_updates(full_df: pd.DataFrame, max_update_num: int) -> pd.DataFrame:
    mask = full_df["update_num"] <= max_update_num
    return full_df[mask]


def restrict_epoch_num(full_df: pd.DataFrame, epoch_num: int) -> pd.DataFrame:
    mask = full_df["epoch_num"] == epoch_num
    return full_df[mask]


def restrict_learning_rates(full_df: pd.DataFrame, lr_set: Collection[float]) -> pd.DataFrame:
    mask = full_df["lr"].isin(lr_set)
    return full_df[mask]


def build_best_hps_df(
        full_df: pd.DataFrame,
        save_dir: pathlib.Path = None,
        *,
        extra_queries: list = None,
) -> pd.DataFrame:
    if extra_queries is None:
        extra_queries = []
    # (1) Reduce mean over run_num to get the mean score for each *(hyperparameter, spec_id) setting.
    #   This result will be sufficient for selecting the best hyperparameters for each spec_id in the next step.
    df_mean_group = reduce_aupr_df_mean(
        full_df,
        query_col_names=["spec_id", "model_name", "partial", "lr", "update_num"] + extra_queries,
        drop_col_names=["run_num"],
    )
    """===>
          update_num       lr spec_id       model_name      aupr               category  n_classes
    0            100  0.00003    10.0  bert-base-cased  0.399709  Conditions to Closing          2
    144          100  0.00010    10.0  bert-base-cased  0.295518  Conditions to Closing          2
    288          200  0.00003    10.0  bert-base-cased  0.455360  Conditions to Closing          2
    432          200  0.00010    10.0  bert-base-cased  0.421332  Conditions to Closing          2
    576          300  0.00003    10.0  bert-base-cased  0.477265  Conditions to Closing          2
    """
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "mean_scores_by_hp.csv"
        df_mean_group.to_csv(save_path)
        print(f"Saved best hps to {save_path}")

    # (3) Reduce max over (update_num, lr) to get the best score and hps for each (model_name, group_id) pair.
    df_best_hps_spec_id = reduce_aupr_df_max(
        df_mean_group,
        query_col_names=["model_name", "partial", "spec_id"] + extra_queries,
        reduce_col_names=["update_num", "lr"],
    )
    """ ==>
              model_name spec_id       lr  update_num      aupr                                category  n_classes
    0    bert-base-cased    10.0  0.00010         300  0.520828                   Conditions to Closing          2
    1    bert-base-cased    10.1  0.00003         300  0.275990                   Conditions to Closing          2
    2    bert-base-cased   10.10  0.00003         800  0.610740                   Conditions to Closing          2
    3    bert-base-cased   10.11  0.00003        1000  0.761451                   Conditions to Closing          2
    """
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        df_best_hps_spec_id.to_csv(save_dir / "df_best_hps_spec_id.csv")
        print(f"Saved best hps to {save_dir}/df_best_hps_spec_id.csv")

    return df_best_hps_spec_id


def get_model_scores(df_best_hps: pd.DataFrame, save_dir: pathlib.Path = None,
                     extra_queries: list = None,
                     ) -> pd.DataFrame:
    if extra_queries is None:
        extra_queries = []
    # (4) Get a final mean AUPR score for each (model), dropping all columns except model_name and aupr.
    # Details: The mean AUPR score must be weighted by `n_classes`, because our AUPR score is based on
    #  the mean minority AUPR score over every (question, answer) pair (IE the group_id), and we have
    #  already reduced away the group_id in a previous step.
    df_single_scores = reduce_aupr_df_mean(
        df_best_hps,
        query_col_names=["model_name", "partial"] + extra_queries,
        drop_col_names=["question_id", "category"],
        # weight_col_name="n_classes",
    )
    """ ===>
        model_name      aupr
0  bert-base-cased  0.659455
    """
    if save_dir:
        print(df_best_hps)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "df_model_scores.csv"
        print(df_single_scores)
        df_single_scores.to_csv(save_path)
        print(f"Saved best hps to {save_path}")
    return df_single_scores


def find_the_best_hps_for_each_hp(records, save_dir=None, *,
                                  q_keep_mode: str,
                                  data_type_filter: str,
                                  allowed_lrs=None,
                                  max_update_num: int = None,
                                  exact_epoch_num: int = None,
                                  proc_epoch: bool = False,
                                  grouping_type: str = "experimental",
                                  verbose: bool = False,
                                  ):
    if exact_epoch_num is not None and not proc_epoch:
        print(f"Automatically activating proc_epoch flag because exact_epoch_num={exact_epoch_num} was set.")
        proc_epoch = True
    full_df = build_full_df(records, save_dir=save_dir, proc_epoch=proc_epoch, q_keep_mode=q_keep_mode,
                            data_type_filter=data_type_filter)
    if max_update_num is not None:
        full_df = restrict_num_updates(full_df, max_update_num=max_update_num)
    if exact_epoch_num is not None:
        full_df = restrict_epoch_num(full_df, epoch_num=exact_epoch_num)
    if allowed_lrs is not None:
        full_df = restrict_learning_rates(full_df, allowed_lrs)

    if proc_epoch:
        assert proc_epoch >= 0
        mod_proc_epoch = ["epoch_num"]
    else:
        mod_proc_epoch = []

    # (1-3) Reduce mean over run_num to get the mean score for each (*hyperparameter, spec_id) setting.
    df_best_hps_spec_id = build_best_hps_df(full_df, save_dir=save_dir, extra_queries=mod_proc_epoch)

    def map_spec_id_to_key(spec_id):
        if grouping_type == "classic":
            return spec_id
        elif grouping_type == "experimental":
            return spec_id.split("-")[0]
        else:
            raise ValueError(grouping_type)

    df_best_hps_grouped = df_best_hps_spec_id.copy()
    df_best_hps_grouped["question_id"] = df_best_hps_grouped["spec_id"].map(map_spec_id_to_key)
    df_best_hps_grouped_meaned = reduce_aupr_df_mean(
        df_best_hps_grouped,
        query_col_names=["model_name", "question_id", "partial", "category"] + mod_proc_epoch,
        drop_col_names=["spec_id", "n_classes", "update_num", "lr"],
    ).sort_values(by=["model_name"] + mod_proc_epoch)

    # (4)
    df_single_scores = get_model_scores(df_best_hps_grouped_meaned, save_dir=save_dir, extra_queries=mod_proc_epoch).sort_values(by=["model_name"] + mod_proc_epoch)
    print(df_single_scores)

    # (5) Get final mean AUPR scores by category
    df_single_scores_by_category = reduce_aupr_df_mean(
        df_best_hps_grouped_meaned,
        query_col_names=["model_name", "partial", "category"] + mod_proc_epoch,
        drop_col_names=["question_id"],
    ).sort_values(by=["model_name", "category"])
    """ ===>
                                 category       model_name      aupr
0                   Conditions to Closing  bert-base-cased  0.557691
1  Deal Protection and Related Provisions  bert-base-cased  0.668816
2                     General Information  bert-base-cased  0.953165
3                               Knowledge  bert-base-cased  0.902921
4                 Material Adverse Effect  bert-base-cased  0.629237
5          Operating and Efforts Covenant  bert-base-cased  0.921402
6                                Remedies  bert-base-cased  1.000000
    """
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "df_scores_by_category.csv"
        df_single_scores_by_category.to_csv(save_path)
        print(f"Saved scores by category to {save_path}")
    if verbose:
        print(df_single_scores_by_category)
