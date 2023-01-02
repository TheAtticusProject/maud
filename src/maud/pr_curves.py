import collections
import dataclasses
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import matplotlib.axes
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics


def avg_prec_from_pr(precisions, recalls):
    # Adapted from _binary_uniterpolated_average_precision in sklearn.metrics._ranking.
    #
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    return -np.sum(np.diff(recalls) * np.array(precisions)[:-1])


def y_true_as_one_hots(y_true: Sequence[int], n_classes: int) -> np.ndarray:
    y_true_one_hot = []
    y_true = np.array(y_true)
    (n_batch,) = y_true.shape

    assert np.all(y_true >= 0)
    assert np.all(y_true < n_classes)
    for i in range(n_classes):
        y_true_one_hot.append(y_true == i)
    y_true_one_hot = np.stack(y_true_one_hot).astype(dtype=int).T
    assert y_true_one_hot.shape == (n_batch, n_classes)
    return y_true_one_hot


def pr_curves_average(all_precisions: Sequence[np.ndarray], all_recalls: Sequence[np.ndarray]
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an average PR curve from multiple PR curves produced by `scipy.metrics.precision_recall_curve`.

    For simplicity (and resolving conceptual ambiguity of averaging in this case) when there are multiple precisions
    for each recall on a PR curve, we keep only the largest precision, omitting the "downward vertical spike."

    This is equivalent to keeping only the precision associated with the lowest threshold at each recall value,
    and ensures that when using the Average Precision method to calculate the AUPR score, the AUPR score of the
    mean curve is the same as the mean AUPR score of the individual curves.

    Args:
        all_precisions: An (n_batch,)-length list of precision arrays.
        all_recalls: An (n_batch,)-length list of recall arrays, with threshold values corresponding to
            those in all_precisions.
    """
    n_batch = len(all_precisions)
    assert n_batch > 0
    assert len(all_recalls) == n_batch
    # To make the mean PR curve, we need to first align the X values of the PR curves (recalls)
    #   with each other through interpolation. Afterwards we can start averaging.
    #
    # Returns the averaged precisions (the Y values) and the union over all recalls (the X values).

    def _is_sorted(x: Sequence[float]) -> bool:
        if len(x) <= 1:
            return True
        for a, b in zip(x[:-1], x[1:]):
            if not (a <= b):
                return False
        return True

    def _sanity_check_r(recalls):
        assert not np.any(np.isnan(recalls))
        assert _is_sorted(recalls[::-1])  # recalls should be sorted in descending order.
        assert recalls[-1] == 0
        assert recalls.ndim == 1

    def _sanity_check_p(precisions):
        assert not np.any(np.isnan(precisions))
        assert precisions.ndim == 1
        assert precisions[-1] == 1

    for precisions, recalls in zip(all_precisions, all_recalls):
        assert len(precisions) == len(recalls)
        _sanity_check_p(precisions)
        _sanity_check_r(recalls)

    recalls_uniq_set = set()
    for recalls in all_recalls:
        recalls_uniq_set = recalls_uniq_set.union(set(recalls))
    recalls_uniq = sorted(recalls_uniq_set, reverse=True)

    # On every curve, there can be multiple precision values for every recall value. This is because it is
    #   possible to add false positives when increasing the threshold, thus decreasing the precision while
    #   keeping the recall constant.
    #
    # For simplicity, in each PR curve at every recall, we will only keep the precision corresponding to the
    # the smallest threshold. This is equivalent to picking the largest precision at each recall. This also
    # ensures that the Average Precision method of calculating the AUPR score
    # gives the same number when we take the AUPR of the meaned curve, and the mean of each curve's AUPR.
    #
    # If another curve has a recall value r_new that this curve doesn't have, then we "interpolate" by repeating
    # the most recent precision value at r_new (will be a precision value corresponding to a higher threshold).
    new_precisions_by_curve = []
    for precisions, recalls in zip(all_precisions, all_recalls):
        recall_to_prec: Dict[float, float] = {}
        j = 0  # Index for recalls_uniq
        for i in range(len(recalls) - 1):
            r1 = recalls[i]
            r2 = recalls[i+1]
            if r1 != r2:
                assert r1 > r2  # Guaranteed since recalls is reverse sorted.
                # The threshold is decreasing as we advance i.
                # precisions[i] gives the smallest threshold precision (also the max precision) associated with
                #   (r1 := recalls[i]). We interpolate the
                assert recalls_uniq[j] == r1
                while recalls_uniq[j] > r2:
                    r, p = recalls_uniq[j], precisions[i]
                    recall_to_prec[r] = p
                    j += 1

        # Edge case. Save the lowest threshold precision for recall=0, which is always 1.
        assert recalls[-1] == 0
        assert precisions[-1] == 1
        recall_to_prec[0] = 1

        assert set(recall_to_prec.keys()) == recalls_uniq_set
        flat_and_interp_precisions = [recall_to_prec[r] for r in recalls_uniq]
        new_precisions_by_curve.append(flat_and_interp_precisions)

    new_precisions = np.mean(new_precisions_by_curve, axis=0)
    new_recalls = np.array(recalls_uniq)

    # Confirm that format of precision, recall arrays confirms to scipy.metrics.precision_recall_curve
    # return format.
    assert len(new_precisions) == len(new_recalls)
    _sanity_check_r(new_recalls)
    _sanity_check_p(new_precisions)

    # Confirm that mean AUPR of individual curves is close to AUPR of meaned curve.
    solo_auprs = []
    for precisions, recalls in zip(all_precisions, all_recalls):
        solo_auprs.append(avg_prec_from_pr(precisions=precisions, recalls=recalls))
    mean_aupr_from_solo = np.mean(solo_auprs)
    mean_aupr_from_mean = avg_prec_from_pr(precisions=new_precisions, recalls=new_recalls)
    within_tolerance = mean_aupr_from_solo - mean_aupr_from_mean < 1e-6
    assert within_tolerance

    return new_precisions, new_recalls


@dataclasses.dataclass
class MAUDPrecRecallCurve:
    precisions_macro: np.ndarray  # Doesn't actually need to be macro.
    recalls_macro: np.ndarray
    random_baseline: float
    class_to_precisions: Optional[Dict[int, np.ndarray]] = None  # Becomes invalidated if we do averages (None)
    class_to_recalls: Optional[Dict[int, np.ndarray]] = None
    key: Optional[Hashable] = None  # The key used for aggregating a list of curves into a list of mean curves.

    def __post_init__(self):
        # A bunch of sanity checks on instance variables.
        if self.class_to_recalls is not None:
            assert self.class_to_precisions is not None, "Must provide both, or None at all"
            assert self.class_to_recalls.keys() == self.class_to_precisions.keys()
            for k, v in self.class_to_recalls.items():
                assert v.ndim == 1
                assert v.shape == self.class_to_precisions[k].shape
        assert self.precisions_macro.ndim == 1
        assert self.precisions_macro.shape == self.recalls_macro.shape
        assert 0 <= self.random_baseline <= 1

    @property
    def auprc(self):
        """Equivalent to average macro-reduced precision."""
        return avg_prec_from_pr(precisions=self.precisions_macro, recalls=self.recalls_macro)

    def drop_class_curves(self) -> "MAUDPrecRecallCurve":
        return dataclasses.replace(self, class_to_recalls=None, class_to_precisions=None)

    @classmethod
    def from_results_binarized(cls, y_true, logits, pos_label, *, auto_minority=True) -> "MAUDPrecRecallCurve":
        """Build a MAUDPrecRecallCurve for an individual label class."""
        assert len(set(y_true)) >= 2

        n_batch, n_classes = logits.shape
        # Is this even binarized?
        y_scores = 2*logits[:, pos_label] - np.sum(logits, axis=1)
        if n_classes == 2:
            old_calc = logits[:, pos_label] - logits[:, 1-pos_label]
            assert np.max(y_scores - old_calc) < 1e-5
        assert y_scores.shape == (n_batch,)
        del logits
        assert len(y_true) == n_batch

        y_true_binarized: np.ndarray = (y_true == pos_label).astype(int)
        if auto_minority and np.sum(y_true_binarized) > n_batch / 2:
            y_true_binarized = 1 - y_true_binarized
            y_scores = -y_scores
        del y_true
        assert len(y_true_binarized.shape) == 1
        assert y_true_binarized.shape == y_scores.shape
        positive_prop = np.sum(y_true_binarized) / len(y_true_binarized)

        assert positive_prop > 0, positive_prop  # Caller should be filtering out labels with no test examples

        class_to_precisions: Dict[int, np.ndarray] = dict()
        class_to_recalls: Dict[int, np.ndarray] = dict()
        precs, recalls, _ = metrics.precision_recall_curve(y_true_binarized, y_scores)

        # Precision at recall=1 should be equal to the true label proportion.
        #   sklearn.metrics.precision_recall_curve() somehow sets the precision at recalls
        #   to impossible values like 1 when not all samples are of the class.
        #   (Maybe an error due to limits or something like that).
        # assert recalls[0] == 1
        # precs[0] = positive_prop
        return cls(
            precisions_macro=precs,
            recalls_macro=recalls,
            random_baseline=positive_prop,
            class_to_recalls=class_to_recalls,
            class_to_precisions=class_to_precisions,
        )

    @classmethod
    def from_results(  # Preferred method of initialization this cuve.
            cls,
            y_true: Sequence[int],
            logits: np.ndarray,
            rm_plurality: bool = False,
    ) -> "MAUDPrecRecallCurve":
        """Generate a macro precision-recall curve.

        Args:
            y_true: A listlike of integer classes.
            logits: A 2D array of shape (n_batch, n_shape) with scores
                (e.g. logits or probas) for each label.
            rm_plurality: If True, then only plot and calculate the mean after
                removing curve associated with the most common true label.
        """
        assert len(set(y_true)) >= 2
        n_batch, n_classes = logits.shape
        assert len(y_true) == n_batch
        y_true_one_hot = y_true_as_one_hots(y_true, n_classes=n_classes)
        assert y_true_one_hot.shape == logits.shape

        true_label_counter = collections.Counter(y_true)
        assert sum(true_label_counter.values()) == n_batch
        true_label_proportions = {k: v / n_batch for k, v in true_label_counter.items()}

        class_to_precisions: Dict[int, np.ndarray] = dict()
        class_to_recalls: Dict[int, np.ndarray] = dict()

        for i in range(n_classes):
            if true_label_counter[i] == 0:
                continue
            y_true_binary_i = y_true_one_hot[:, i]
            y_scores = logits[:, i]
            precs, recalls, _ = metrics.precision_recall_curve(y_true_binary_i, y_scores)
            class_to_precisions[i] = precs
            class_to_recalls[i] = recalls
            
        #### DON"T EDIT ME 😢

        if rm_plurality:
            most_common_true_label = true_label_counter.most_common(1)[0][0]
            del class_to_precisions[most_common_true_label]
            del class_to_recalls[most_common_true_label]
            del true_label_proportions[most_common_true_label]

        # Precisions should go to the true label proportion.
        #   metrics.precision_recall_curve() somehow sets the precision at recalls
        #   to impossible values like 1 when not all samples are of the class.
        #   (Maybe an error due to limits or something like that).
        for i in class_to_recalls.keys():
            recalls = class_to_recalls[i]
            precs = class_to_precisions[i]
            assert recalls[0] == 1
            precs[0] = true_label_proportions[i]

        random_baseline = float(np.mean(list(true_label_proportions.values())))
        average_precisions_by_pos_class = metrics.average_precision_score(y_true_one_hot, logits, average=None)
        precs_macro, recalls_macro = pr_curves_average(
            list(class_to_precisions.values()),
            list(class_to_recalls.values()),
        )

        return cls(
            precisions_macro=precs_macro,
            recalls_macro=recalls_macro,
            random_baseline=random_baseline,
            class_to_recalls=class_to_recalls,
            class_to_precisions=class_to_precisions,
        )

    @classmethod
    def from_tune_record_binarized(cls, record: dict, key_type: str = "experimental") -> List["MAUDPrecRecallCurve"]:
        y_true = record["labels"]
        logits = record["logits"]
        return cls.from_labels_and_logits(y_true=y_true, logits=logits, spec_id=record["spec_id"], key_type=key_type)

    @classmethod
    def from_labels_and_logits(
            cls, y_true: np.ndarray, logits: np.ndarray, spec_id: str, *, key_type: str,
    ) -> List["MAUDPrecRecallCurve"]:
        assert key_type in ("classic", "experimental")
        """key_type determines how to group binary subquestions created from multilabel deal point questions.
        
        classic: We treat binary subquestions like any other question. (Average over 144 questions and subquestions).
             This is the simplest treatment, but upweights the scores of multilabel questions:
            (1, 2, 3, 10.1, 10.2, 10.3)  => (1, 2, 3, 10.1, 10.2, 10.3)
        experimental: We group all binary subquestions from the same multilabel question together (average over 92
            true questions). This the approach described in paper.
            (1, 2, 3, 10.1, 10.2, 10.3)  => (1, 2, 3, 10)
        """
        _, n_classes = logits.shape
        assert n_classes >= 2
        for i in range(n_classes):
            if np.sum(y_true == i) in (0, len(y_true)):
                raise ValueError(f"Bad {y_true} with all or no {i} examples.")

        result = []
        for i in range(n_classes):
            curve = cls.from_results_binarized(y_true=y_true, logits=logits, pos_label=i)
            if key_type == "classic":
                curve.key = spec_id
            if key_type == "experimental":
                curve.key = spec_id.split("-")[0]
            result.append(curve)
        return result

    @classmethod
    def from_reduced_means_by_group(
        cls,
        curves: Sequence["MAUDPrecRecallCurve"],
        strict: bool = True,
    ) -> Dict[Any, "MAUDPrecRecallCurve"]:
        key_to_curves = collections.defaultdict(list)
        for curve in curves:
            if strict:
                assert curve.key is not None
            key_to_curves[curve.key].append(curve)

        result = dict()
        for key, grouped_curves in key_to_curves.items():
            reduced_curve = cls.from_totally_reduced_mean(grouped_curves)
            reduced_curve.key = key
            result[key] = reduced_curve
        return result

    @classmethod
    def from_totally_reduced_mean(cls, curves: Sequence["MAUDPrecRecallCurve"]) -> "MAUDPrecRecallCurve":
        all_precs_macro = [curve.precisions_macro for curve in curves]
        all_recalls_macro = [curve.recalls_macro for curve in curves]
        all_rand_baselines = [curve.random_baseline for curve in curves]
        reduced_precs_macro, reduced_recalls_macro = pr_curves_average(all_precs_macro, all_recalls_macro)
        reduced_rand_baseline = float(np.mean(all_rand_baselines))
        return cls(
            precisions_macro=reduced_precs_macro,
            recalls_macro=reduced_recalls_macro,
            random_baseline=reduced_rand_baseline,
        )

    def make_fig(
            self,
            title="Default MAUD PR Plot",
            curve_name: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = make_default_fig_and_ax(title=title)
        self.plot_to_ax(ax, curve_name=curve_name)
        return fig, ax

    def plot_to_ax(self, ax: matplotlib.axes.Axes, curve_name: Optional[str] = None,
                   include_avg_prec: bool = True,
                   color: Optional[str] = None,
                   zorder: Optional[int] = None,
                   ) -> None:
        if self.class_to_recalls is not None:
            classes = list(self.class_to_recalls.keys())
            for i in classes:
                precs = self.class_to_precisions[i]
                recalls = self.class_to_recalls[i]
                disp = metrics.PrecisionRecallDisplay(
                    self.class_to_precisions[i],
                    self.class_to_recalls[i],
                    average_precision=(metrics.auc(recalls, precs) if include_avg_prec else None),
                )
                disp.plot(ax=ax, name=f"Class {i}", linestyle="--", drawstyle="default",
                          lw=1, color=color)

        disp = metrics.PrecisionRecallDisplay(
            self.precisions_macro,
            self.recalls_macro,
            average_precision=(self.auprc if include_avg_prec else None),
        )
        disp.plot(ax=ax, name=curve_name, lw=LW, color=color, zorder=zorder)

    def plot_baseline(self, ax, include_avg_prec=False, color: Optional[str] = None):
        if include_avg_prec:
            label = f"random baseline (AP = {self.random_baseline:0.3f})"
        else:
            label = f"Random"
        ax.axhline(y=self.random_baseline, linestyle="--", label=label, color=color, lw=LW)
        # ax.set_ylim(bottom=self.random_baseline - 0.05)
        ax.set_ylim(bottom=0.0)


LW = 2

def make_default_fig_and_ax(title="Default MAUD PR Plot") -> Tuple[plt.Figure, plt.Axes]:
    # fig, ax = plt.subplots(figsize=(4*1.2, 3*1.2))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(ls='dashed')
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # ax.xaxis.set_ticks_position('none')
    if title:
        ax.set_title(title)
    return fig, ax


