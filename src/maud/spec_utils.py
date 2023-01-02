from typing import List

from maud import specs, data


# This is not actually very useful because of inconsistencies in DatasetSpec involving synthetic data.
#  In particular, we sometimes have different numbers of X. Therefore, let's directly load the labels instead.
def get_data_proportions(rec: dict, split: str = "test") -> List[float]:
    """
    Split should be 'train' or 'test'.

    rec is an analysis record.

    We automatically add synthetic data and additioanl data to the calculation by looking at rec["args"].use_synth
    and .use_add.
    """
    assert split in ["train", "test"]
    args = rec["args"]
    spec: specs.BaseDatasetSpec = rec["spec"]
    main_ds, add_ds, synth_ds = spec.to_dataset_args(args)

    train_ds, test_ds = data.build_balanced_split_and_tokenize(
        ds=main_ds,
        tokenizer=None,
        valid_prop=args.valid_prop,
        add_ds=add_ds,
        synth_ds=synth_ds,
    )

    if split == "train":
        assert args.valid_prop == 0.2  # Temporary sanity check -- remove it later.
        counted_ds = train_ds
    else:
        assert split == "test"
        counted_ds = test_ds

    count_tuples = data.count_labels(counted_ds, spec._answer_choices_ordered)
    answers = [t[0] for t in count_tuples]
    assert len(set(answers)) == len(answers)
    assert answers == list(spec._answer_choices_ordered)

    counts = [x[1] for x in count_tuples]
    total = len(counted_ds)
    assert total == sum(counts)
    proportions = [c / total for c in counts]
    assert len(proportions) == len(spec._answer_choices_ordered)
    return proportions


# Prefer direct usage of _macro_mean_processed instead.
def scores_remove_plurality(props: List[float], scores: List[float]) -> List[float]:
    """
    Given a list of data proportions and list of scores (matching indices correspond between the lists),
    return a copy of scores where the score with the highest proportion is removed.

    Useful for calculating minority or plurality-removed F1 scores.
    """
    assert len(props) == len(scores)
    assert len(props) >= 2
    sortable = [(p, s) for p, s in zip(props, scores)]
    sortable.sort()
    sortable_remove_plurality = sortable[:-1]
    scores_rm_plurality = [score for (_, score) in sortable_remove_plurality]
    return scores_rm_plurality
