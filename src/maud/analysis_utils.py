import collections
import functools
import pathlib
import pickle
from typing import Callable, List, Optional, Sequence, Tuple, Generator
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
import tqdm

from maud import category_utils


def _macro_mean_processed(
        metric_fn: Callable,
        true_labels: Sequence[int],
        pred_labels: Sequence[int],
) -> Tuple[float, float]:
    """

    Args:
        metric_fn: One of sklearn.{f1,recall,precision}_score
        true_labels: A list of ground truth labels.
        pred_labels: A list of predictions on the ground truth labels.

    Returns:
        A macro averaged score for the metric_fn, and a macro-without-plurality score for the metric_fn.
    """
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    assert pred_labels.shape == true_labels.shape

    assert len(set(true_labels)) >= 2
    # BIG WARNING: Need to fix this later -- sometimes we have test_ds that has 0 labels!

    # NOTE: f1_* here is a stand in for any one of f1_score, precision_score, or recall_score.
    f1_labels = list(set(true_labels))
    f1_macro = metric_fn(true_labels, pred_labels, labels=f1_labels, average="macro")

    # Calculate F1 macro score without plurality class.
    f1_unaveraged = metric_fn(true_labels, pred_labels, labels=f1_labels, average=None)
    assert np.mean(f1_unaveraged) - f1_macro < 1e-6

    #   Find the index of the most common label so that we can remove
    #   it and get the f1 macro without the plurality label. (Equivalent to minority class f1 score in the
    #   case of a binary question).
    true_label_counter = collections.Counter(true_labels)
    most_common_true_label = true_label_counter.most_common(1)[0][0]
    plurality_index = f1_labels.index(most_common_true_label)

    f1_unaveraged_rm_plurality = list(f1_unaveraged)
    del f1_unaveraged_rm_plurality[plurality_index]
    f1_macro_remove_plurality = np.mean(f1_unaveraged_rm_plurality)
    return float(f1_macro), float(f1_macro_remove_plurality)


f1_macro_processed = functools.partial(_macro_mean_processed, metrics.f1_score)
precision_macro_processed = functools.partial(_macro_mean_processed, metrics.precision_score)
recall_macro_processed = functools.partial(_macro_mean_processed, metrics.recall_score)


def make_augmentation_flags_str(args) -> str:
    result = ""
    if args.oversample:
        result += "o"
    if args.use_add:
        result += "a"
    if args.use_synth:
        result += "s"
    if args.synth_capped:
        result += "c"
    return result


def _make_row(record: dict) -> dict:
    row = dict()
    row["epoch"] = record["epoch"]
    row["batch_num"] = record["batch_num"]
    row["augmentation_flags"] = make_augmentation_flags_str(record["args"])
    row["model"] = record["args"].model
    row["spec_id"] = record["spec_id"]
    return row


def build_hyperparam_sweep_df(
    records: Sequence[dict],
    sweep_name: Optional[str] = None,
) -> pd.DataFrame:
    df_rows: List[dict] = []
    for record in tqdm.tqdm(records):
        try:
            true_labels: np.ndarray = np.array(record["labels"])
            pred_labels: np.ndarray = get_preds_from_logits(record["logits"])
        except KeyError:
            warnings.warn("Malformed record, skipping")
            continue

        if len(set(true_labels)) < 2:
            print(f"Skipped spec id={record['spec_id']} due to labelling problem")
            continue

        f1_macro, f1_macro_remove_plurality = f1_macro_processed(true_labels, pred_labels)
        row = _make_row(record)
        row["f1_macro"] = f1_macro
        row["f1_macro_remove_plurality"] = f1_macro_remove_plurality
        spec_id = record["spec_id"]
        if "." in spec_id:
            spec_id_key = spec_id.split("-")[0]
        else:
            spec_id_key = spec_id

        if spec_id_key in category_utils.spec_id_to_category:
            row["category"] = category_utils.spec_id_to_category[spec_id_key]
        else:
            warnings.warn(f"Unknown spec_id {spec_id}, setting category to UNKNOWN")
            row["category"] = "UNKNOWN"

        df_rows.append(row)

    df = pd.DataFrame.from_records(df_rows)
    df["sweep_name"] = sweep_name

    verbose = True
    if verbose:
        # Some interesting stats.
        num_decrease = np.sum(np.mean(df["f1_macro"] > df["f1_macro_remove_plurality"]))
        mean_decrease = np.mean(df["f1_macro"] - df["f1_macro_remove_plurality"])
        print(f"Num  F1 decrease after rm plur: {num_decrease*100:.2f}% (n={len(df)})")
        print(f"Mean F1 decrease after rm plur: {mean_decrease:.3f}")
    return df


def build_best_hyperparams_df(
        records: List[dict],
        sweep_name: Optional[str] = None,
        sort_by_no_plur: bool = True,
        hp_sweep_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Produce a Dataframe that finds the best combination of augmentations flags, and epoch number
    (equivalent to finding best batch number in runs using --num_update) for each question, by taking the
    mean over F1_remove_plurality_macro scores for each question, augmentation pair, and then taking the
    max over these mean F1 scores.

    Args:
        records: List of results pickles from tune.py, all generated by a hyperparameter sweep on the same model
            with fixed learning rate. (To see which hyperparams are allowed to vary, see the definition of `row` below).
        sort_by_no_plur: If true, for each spec id, keep the row (corresponding to a hyperparam choice)
            that has the highest
            f1_score excluding plurality.
        Otherwise, keep the row that has the highest f1_score.
    """
    if hp_sweep_df is None:
        df = build_hyperparam_sweep_df(records, sweep_name)
    else:
        df = hp_sweep_df
    # means_df = df.groupby(["spec_id", "augmentation_flags", "epoch"]).max().add_suffix("_MEAN").reset_index()

    stats_df = df.groupby(["spec_id", "augmentation_flags", "epoch", "category"]
                          ).agg(
        {'f1_macro': ['mean', 'std', 'sem', 'var'], 'f1_macro_remove_plurality': ['mean', 'std', 'sem', 'var']})

    if sort_by_no_plur:
        dff = stats_df["f1_macro_remove_plurality"]
    else:
        dff = stats_df["f1_macro"]
    max_df_idx = dff.groupby("spec_id")["mean"].idxmax()

    best_df = stats_df.loc[max_df_idx]
    best_df = best_df.sort_values(by="spec_id")
    return best_df


def get_preds_from_logits(logits: np.ndarray) -> np.ndarray:
    assert len(logits.shape) == 2
    n_batch, n_classes = logits.shape
    result = np.argmax(logits, axis=1)
    assert result.shape == (n_batch,)
    return result


def f1_baseline_score(positive_proportion: float) -> float:
    """Get an F1 baseline score (baseline: always predict positive).
    This can be shown to be the best baseline in all cases."""
    assert 0 < positive_proportion <= 1
    r = positive_proportion
    return 2*r / (1 + r)


def f1_baseline_score_macro_avg(positive_proportions: List[float], *, remove_plurality: bool = False, sanity: bool = True,
                                ) -> List[float]:
    """Get the macro f1 baseline score."""
    props = sorted(positive_proportions)
    if sanity:  # Sanity checks for a paranoid coder.
        assert len(props) >= 2
        assert props[-1] >= props[0], "checking that I understand sorted()..."
        assert np.sum(props) - 1.0 < 1e-4, np.sum(props)
    if remove_plurality:
        props = props[:-1]
    return [f1_baseline_score(p) for p in props]


def filter_records(records: List[dict], match_dict: dict) -> List[dict]:
    result = []
    for rec in records:
        match = True
        for k, v in match_dict.items():
            if rec[k] != v:
                match = False
                break
        if match:
            result.append(rec)
    return result


def recursive_load_record_scrap(record_path: pathlib.Path, *, fast: bool = False) -> List[dict]:
    """Either load the record at record_path, or recursively load all records in the directory at record_path."""
    def inner(path: pathlib.Path) -> Generator[dict, None, None]:
        assert path.exists(), path
        if path.is_dir():
            count = 0
            for child in path.iterdir():
                yield from inner(child)
                count += 1
                if count >= 10 and fast:
                    break
        else:
            if str(path).endswith(".pkl"):
                # Only load files that have pickle suffix.
                with open(path, "rb") as f:
                    record = pickle.load(f)
                yield record
    result = list(inner(record_path))
    print(f"Loaded {len(result)} record(s).")
    return result


def restrict_epochs(records: List[dict], epoch: int) -> List[dict]:
    assert epoch > 0
    new_records = []
    for rec in records:
        if rec["epoch"] == epoch:
            new_records.append(rec)
    print(f"Restricting to epoch={epoch}. "
          f"Kept {len(new_records)}/{len(records)} records.")
    assert len(new_records) > 0
    return new_records


def keep_records_with_hyperparams(
        records: Sequence[dict],
        spec_id: str,
        augmentation_flags: str,
        epoch: int,
) -> List[dict]:
    result = []
    for rec in records:
        row = _make_row(rec)
        if row["spec_id"] != spec_id:
            continue
        if row["augmentation_flags"] != augmentation_flags:
            continue
        if row["epoch"] != epoch:
            continue
        result.append(rec)
    return result


def keep_best_records(records: Sequence[dict], best_df: pd.DataFrame) -> List[dict]:
    best_records = []
    for (spec_id, augmentation_flags, epoch, cat), row in best_df.iterrows():
        grouped_records = keep_records_with_hyperparams(
            records=records,
            spec_id=spec_id,
            augmentation_flags=augmentation_flags,
            epoch=epoch,
        )
        assert len(grouped_records) > 0
        best_records.extend(grouped_records)
    return best_records
