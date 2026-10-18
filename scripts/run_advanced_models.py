"""CLI script to execute 5-Fold Stratified Cross-Validation on Gradient Boosting and Stacking."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_config  # noqa: E402
from src.models_advanced import run_advanced_cv_single  # noqa: E402

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for advanced model evaluations."""
    parser = argparse.ArgumentParser(
        description="Run 5-Fold Stratified CV on Gradient Boosting and StackingClassifier models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--experiment_type",
        choices=["all", "trees", "stacking"],
        default="all",
        help="Category of advanced experiment to run.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["light_normalized"],
        help="Preprocessed text pipeline variants to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional filter to subset evaluated models.",
    )
    return parser.parse_args()


def get_advanced_grid(experiment_type: str) -> list[tuple[str, str, str, str]]:
    """Define tuples of (representation, dim_reduction, model_type, model_name)."""
    grid_trees = [
        ("sparse_fusion", "svd_300", "tree", "LightGBM"),
        ("sparse_fusion", "svd_300", "tree", "XGBoost"),
        ("sparse_fusion", "svd_300", "tree", "HistGradientBoosting"),
    ]

    grid_stacking = [
        ("sparse_fusion", "none", "stacking", "StackingEnsemble"),
    ]

    if experiment_type == "trees":
        return grid_trees
    if experiment_type == "stacking":
        return grid_stacking
    return grid_trees + grid_stacking


def main() -> None:
    """Execute advanced model evaluations and log results."""
    args = parse_args()
    cfg = load_config(args.config)

    adv_exp_cfg = cfg.get("advanced_experiments", {})
    results_csv_rel = adv_exp_cfg.get("results_csv", "results/results_advanced.csv")
    results_csv = PROJECT_ROOT / results_csv_rel

    grid = get_advanced_grid(args.experiment_type)
    if args.models:
        grid = [item for item in grid if item[3] in args.models]

    total_runs = len(args.pipelines) * len(grid)

    print("=" * 85)
    print("PHASE 5: GRADIENT BOOSTING TREES & STACKING ENSEMBLE BENCHMARK")
    print(f"Experiment Group: {args.experiment_type.upper()}")
    print(f"Total Combinations: {total_runs} (5 folds each -> {total_runs * 5} folds)")
    print(f"Results CSV: {results_csv}")
    print("=" * 85)

    summary_records: list[dict[str, Any]] = []
    run_counter = 0

    for pipe_name in args.pipelines:
        for rep_name, dim_red, model_type, model_name in grid:
            run_counter += 1
            combo_name = (
                f"[{run_counter}/{total_runs}] {model_name} | {rep_name} | "
                f"SVD: {dim_red} | {pipe_name}"
            )
            print(f"\n---> Starting: {combo_name}")

            fold_rows = run_advanced_cv_single(
                pipeline_name=pipe_name,
                representation=rep_name,
                model_type=model_type,
                model_name=model_name,
                dim_reduction=dim_red,
                config=cfg,
                project_root=PROJECT_ROOT,
                results_csv=results_csv,
            )

            accs = [r["accuracy"] for r in fold_rows]
            precs = [r["macro_precision"] for r in fold_rows]
            recs = [r["macro_recall"] for r in fold_rows]
            f1s = [r["macro_f1"] for r in fold_rows]
            times = [r["train_time_sec"] for r in fold_rows]
            lats = [r["inference_ms_per_doc"] for r in fold_rows]

            print(
                f"     Macro-F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | "
                f"Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f} | "
                f"Avg Train: {np.mean(times):.1f}s | "
                f"Latency: {np.mean(lats):.3f}ms/doc"
            )

            summary_records.append(
                {
                    "Model": model_name,
                    "Representation": rep_name,
                    "Dim_Reduction": dim_red,
                    "Preprocessing": pipe_name,
                    "Accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
                    "Macro-Precision": f"{np.mean(precs):.4f} +/- {np.std(precs):.4f}",
                    "Macro-Recall": f"{np.mean(recs):.4f} +/- {np.std(recs):.4f}",
                    "Macro-F1": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
                    "Mean_Train_Sec": f"{np.mean(times):.2f}",
                    "Mean_Infer_ms": f"{np.mean(lats):.3f}",
                }
            )

    print("\n" + "=" * 95)
    print("ALL ADVANCED MODEL EXPERIMENTS COMPLETE - SUMMARY OF 5-FOLD CV ON DEV")
    print("=" * 95)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print(f"\nDetailed individual fold results saved to: {results_csv}")


if __name__ == "__main__":
    main()
