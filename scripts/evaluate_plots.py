import argparse
import collections
import pathlib
import pickle
from typing import Any, Dict, List, Tuple

import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from maud import analysis_utils, category_utils, eval_utils, pr_curves, utils


FILE_WRITE_GLOBAL = False
def log(text: str = "", newline="\n", stdout=True, file_write: bool = None):
    if file_write is None:
        file_write = FILE_WRITE_GLOBAL
    if file_write:
        with open("analysis.txt", "a") as f:
            f.write(str(text) + newline)
    if stdout:
        print(text)


CSV_COLLECTION_DIR = pathlib.Path("scrap", "best_augment_best_epochs", "new_format")


def make_curves(records: List[dict]) -> List[pr_curves.MAUDPrecRecallCurve]:
    curves = []
    for rec in records:
        # Creates one curve for each label in the rec. They are grouped by answer (unique .group_id) so that
        #  when we average curves, we weight each unique question answer pair equally.
        binarized_curves = pr_curves.MAUDPrecRecallCurve.from_tune_record_binarized(rec)
        curves.extend(binarized_curves)
    return curves


def make_mean_curve(records, category: str):
    curves = make_curves(records)
    grouped_curves = pr_curves.MAUDPrecRecallCurve.from_reduced_means_by_group(
        curves, category=category)
    # ...
    # (Pdb) p grouped_curves[('10.0', 0)].auprc
    # 0.6037858161351344
    print(f"CATEGORY = {category}")
    print(f"Produced {len(grouped_curves)} curve groups.")
    auprs_before = [v.auprc for v in grouped_curves.values()]
    manual_aupr_mean = np.mean(auprs_before)
    mean_curve = pr_curves.MAUDPrecRecallCurve.from_totally_reduced_mean(list(grouped_curves.values()))
    diff = manual_aupr_mean - mean_curve.auprc
    assert diff < 1e-5, diff
    return mean_curve


def worker(sweep_dir: pathlib.Path, args) -> Dict[str, Dict[str, pr_curves.MAUDPrecRecallCurve]]:
    records = analysis_utils.recursive_load_record_scrap(sweep_dir)
    for rec in records:
        # Post processing
        rec["model_name"] = rec["args"].model
        rec["lr"] = rec["args"].learning_rate
        rec["update_num"] = rec["batch_num"]
    assert isinstance(records, list)

    CACHE_DIRTY = True
    cache_path = pathlib.Path("/tmp", str(sweep_dir).replace("/", "_") + ".pkl")

    if CACHE_DIRTY or not cache_path.is_file():
        full_df = eval_utils.build_full_df(records)
        best_hps_df_specs, best_hps_df_group_id = eval_utils.build_best_hps_df(full_df)
        with open(cache_path, "wb") as f:
            pickle.dump((best_hps_df_specs, best_hps_df_group_id), f)
    else:
        with open(cache_path, "rb") as f:
            best_hps_df_specs, best_hps_df_group_id = pickle.load(f)

    best_hps_df = best_hps_df_group_id
    # best_hps_df = best_hps_df_specs

    # (1) I need to get the best-hp curves for each model, and these are generated from the best records.
    # So I need to make a Dict[str, List[dict]] from model_name to list of best records for each model.
    # I can use filter_dicts to pick up the right models.
    model_name_to_best_records: Dict[str, List[dict]] = collections.defaultdict(list)
    # model_name_to_curves: Dict[str, List[pr_curves.MAUDPrecRecallCurve]] = collections.defaultdict(list)
    n_rows = len(best_hps_df)
    for i in tqdm.tqdm(range(n_rows), desc="Filter records"):
        row = best_hps_df.iloc[i]
        model_name = row["model_name"]
        filter_dict = dict(
            model_name=model_name,
            spec_id=row["spec_id"],
            lr=row["lr"],
            update_num=row["update_num"],
        )
        matching_recs = analysis_utils.filter_records(records, filter_dict)
        model_name_to_best_records[model_name].extend(matching_recs)

    model_to_cat_to_mean_curve: Dict[str, Dict[str, pr_curves.MAUDPrecRecallCurve]] = collections.defaultdict(dict)
    categories = ["ALL"] + list(category_utils.categories_to_spec_ids)
    for model_name, best_records in model_name_to_best_records.items():
        print("****MODEL:", model_name)
        for category in categories:
            if category == "ALL":
                kept_records = best_records
                best_rec_spec_id_set = set(rec["spec_id"] for rec in best_records)
                if len(best_rec_spec_id_set) != 144:
                    print(f"{model_name} only had {len(best_rec_spec_id_set)} best hp records.")
            else:
                kept_records = []
                allowed_spec_ids = category_utils.categories_to_spec_ids[category]
                for rec in best_records:
                    if rec["spec_id"].split(".")[0] in allowed_spec_ids:
                        kept_records.append(rec)
            cat_to_mean_curve = model_to_cat_to_mean_curve[model_name]
            cat_to_mean_curve[category] = make_mean_curve(kept_records, category=category)
    return model_to_cat_to_mean_curve


FORCE_MAPPING = True


def _correct_sweep_name(sweep_name: str) -> str:
    if "partial" in sweep_name:
        return sweep_name
    if "longformer" in sweep_name:
        return sweep_name
    mapping = dict(
        deberta="DeBERTa",
        roberta="RoBERTa",
        bert="BERT",
    )
    for prefix, prefix_replacement in mapping.items():
        # Correct capitalization of roberta to RoBERTa
        sweep_name: str
        if "bigbird" in sweep_name:
            sweep_name = "BigBird"
        elif prefix in sweep_name:
            sweep_name = prefix_replacement
            break
    return sweep_name


def plot_curves(
        model_to_mean_curve: Dict[str, pr_curves.MAUDPrecRecallCurve],
        category: str,
        plot_dir: pathlib.Path,
        show_avg_prec: bool,
):
    fig, ax = pr_curves.make_default_fig_and_ax()
    colors = ['royalblue', 'limegreen', '#F15757', 'orange']
    # colors = ['royalblue', 'limegreen', '#F15757', 'purple']
    n_sweeps = len(model_to_mean_curve)
    order = ['bigbird', 'deberta', 'roberta', 'bert']

    ordered = [None] * len(order)
    others = []


    for sweep_name, curve in model_to_mean_curve.items():
        if FORCE_MAPPING:
            sweep_name = _correct_sweep_name(sweep_name)
        if sweep_name.lower() in order:
            ordered[order.index(sweep_name.lower())] = (sweep_name, curve)
        else:
            others.append((sweep_name, curve))

    ordered_curve_tups = [tup for tup in ordered if tup is not None] + others

    for i, (sweep_name, curve) in enumerate(ordered_curve_tups):
        curve: pr_curves.MAUDPrecRecallCurve
        curve.plot_to_ax(ax, sweep_name, include_avg_prec=show_avg_prec,
                         color=(colors[i] if i < len(colors) else None),
                         zorder=n_sweeps - i)
    # Plot baseline last so that it shows up in legend last.
    curve.plot_baseline(ax, color='black', include_avg_prec=show_avg_prec)

    leg = plt.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)

    if category.lower() == "all":
        title = f'MAUD Precision Recall Curve'
    else:
        title = f'MAUD "{category}" Precision Recall Curve'
    plt.title(title)
    ax.set_xlim(right=1.002)

    filename = category.replace('/', '_')
    plot_path = plot_dir / f"{filename}.pdf"
    fig.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved a figure to {str(plot_path)}")


def _flatten(x: List[list]) -> List[Any]:
    result = []
    for l in x:
        result.extend(l)
    return result


def main_all_sweeps_combined(results_dir, args) -> None:
    model_to_category_to_mean_curve = worker(results_dir, args)

    # Print random baseline results for posterity lol
    for model, cat_to_mean_curve in model_to_category_to_mean_curve.items():
        log(f"%%%% RANDOM BASELINES FOR {model}")
        for category, mean_curve in sorted(cat_to_mean_curve.items()):
            log(f"{category}: {mean_curve.random_baseline:0.4f}")

    for model, cat_to_mean_curve in model_to_category_to_mean_curve.items():
        log(f"%%%% AUPR scores FOR {model}")
        for category, mean_curve in sorted(cat_to_mean_curve.items()):
            log(f"{category}: {mean_curve.auprc:0.4f}")

    # Do some data reordering for the plotting... AMAZING
    category_to_model_to_mean_curve = collections.defaultdict(dict)
    for model, cat_to_mean_curve in model_to_category_to_mean_curve.items():
        for category, mean_curve in cat_to_mean_curve.items():
            model_to_mean_curve: dict = category_to_model_to_mean_curve[category]
            model_to_mean_curve[model] = mean_curve

    plot_dir = pathlib.Path("scrap", "eval_plots", utils.make_unique_filename())
    plot_dir.mkdir(parents=True)
    for category, model_to_mean_curve in category_to_model_to_mean_curve.items():
        plot_curves(model_to_mean_curve=model_to_mean_curve, category=category, plot_dir=plot_dir,
                    show_avg_prec=args.show_aupr)

    if not args.no_show:
        plt.show()


FAST = False

PLOT_DIR = pathlib.Path("figures/pr_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def console_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--dummy", action="store_true", help="Use dumb parallelism")
    parser.add_argument("--log", "-L", action="store_true", help="Log to analysis.txt")
    parser.add_argument("--no_show", "-N", action="store_true", help="Don't pop up plots")
    parser.add_argument("--show_aupr", action="store_true", help="Add AUPR to legend")
    args = parser.parse_args()
    if args.log:
        global FILE_WRITE_GLOBAL
        FILE_WRITE_GLOBAL = True

    assert args.results_dir.is_dir()
    main_all_sweeps_combined(args.results_dir, args)


if __name__ == "__main__":
    # main_all_sweeps_combined()
    console_main()
