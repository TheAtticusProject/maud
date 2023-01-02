import functools
import pathlib
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas
import pandas as pd

from maud import data


@functools.lru_cache()
def load_df(split: str, data_dir: Union[str, pathlib.Path] = "data") -> pandas.DataFrame:
    # train => false means we load the test path.
    assert split in ["train", "dev", "test"]
    path = pathlib.Path(data_dir, f"MAUD_{split}.csv")
    df_loaded = pandas.read_csv(path)
    return df_loaded


def df_filter_by_question(
        df: pandas.DataFrame,
        question: str,
        subquestion: str = "<NONE>",
) -> pd.DataFrame:
    mask = (df["question"] == question) & (df["subquestion"] == subquestion)
    assert np.sum(mask) > 0
    df = df[mask]
    return df


def df_to_grouped_records(
        df: pandas.DataFrame,
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


VALID_PROP = 0.20
def split_train_dev(main_ds, add_ds, cf_ds) -> Tuple[List[dict], List[dict]]:
    train_ds, dev_ds = data.build_balanced_split_only(
        ds=(main_ds + cf_ds),  # both sides
        valid_prop=VALID_PROP,
        verbose=False,
        add_ds=add_ds,  # grouped by contract name.
        synth_ds=[],
        seed=420,
    )
    return train_ds, dev_ds



def df_to_records(
        df: pandas.DataFrame,
        question: str,
        subquestion: str = "<NONE>",
) -> List[dict]:
    mask = (df["question"] == question) & (df["subquestion"] == subquestion)
    assert np.sum(mask) > 0
    df = df[mask]
    return df.to_dict("records")
