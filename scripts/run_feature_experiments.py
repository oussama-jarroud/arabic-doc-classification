"""CLI script to run 5-Fold Stratified Cross-Validation for advanced feature engineering."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_config  # noqa: E402
from src.models_classical import run_cross_validation_single  # noqa: E402

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for advanced feature engineering experiments."""
    parser = argparse.ArgumentParser(
        description="Run 5-Fold Stratified CV on advanced feature representations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--experiment_type",
        choices=["all", "ngrams", "fusion", "svd"],
        default="all",
        help="Category of feature experiment to execute.",
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
        help="Optional filter to subset models.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=None,
        help="Optional filter to subset representations.",
    )
    return parser.parse_args()


def get_experiment_grid(experiment_type: str) -> list[tuple[str, str, str]]:
    """Define tuples of (representation, dim_reduction, model_name) for evaluation."""
    grid_ngrams = [
        ("tfidf", "none", "LinearSVC"),
        ("tfidf_word_1_2", "none", "LinearSVC"),
        ("tfidf_word_1_3", "none", "LinearSVC"),
        ("tfidf", "none", "LogisticRegression"),
        ("tfidf_word_1_2", "none", "LogisticRegression"),
        ("tfidf_word_1_3", "none", "LogisticRegression"),
    ]

    grid_fusion = [
        ("sparse_fusion", "none", "LinearSVC"),
        ("sparse_fusion", "none", "LogisticRegression"),
        ("sparse_fusion", "none", "SGDClassifier"),
        ("sparse_fusion", "none", "MultinomialNB"),
    ]

    grid_svd = [
        ("tfidf", "svd_300", "LinearSVC"),
        ("tfidf", "svd_1000", "LinearSVC"),
        ("tfidf", "svd_300", "LogisticRegression"),
        ("tfidf", "svd_1000", "LogisticRegression"),
        ("sparse_fusion", "svd_300", "LinearSVC"),
        ("sparse_fusion", "svd_1000", "LinearSVC"),
    ]

    if experiment_type == "ngrams":
        return grid_ngrams
    if experiment_type == "fusion":
        return grid_fusion
    if experiment_type == "svd":
        return grid_svd
    return grid_ngrams + grid_fusion + grid_svd


def main() -> None:
    """Execute feature engineering experiments across specified grids."""
    args = parse_args()
    cfg = load_config(args.config)

    feature_exp_cfg = cfg.get("feature_experiments", {})
    results_csv_rel = feature_exp_cfg.get("results_csv", "results/results_features.csv")
    results_csv = PROJECT_ROOT / results_csv_rel

    grid = get_experiment_grid(args.experiment_type)
    if args.models:
        grid = [item for item in grid if item[2] in args.models]
    if args.representations:
        grid = [item for item in grid if item[0] in args.representations]

    total_runs = len(args.pipelines) * len(grid)

    print("=" * 85)
    print("PHASE 4: ADVANCED FEATURE ENGINEERING & REPRESENTATIONS BENCHMARK")
    print(f"Experiment Group: {args.experiment_type.upper()}")
    print(f"Total Combinations: {total_runs} (5 folds each -> {total_runs * 5} folds)")
    print(f"Results CSV: {results_csv}")
    print("=" * 85)

    summary_records: list[dict[str, Any]] = []
    run_counter = 0

    for pipe_name in args.pipelines:
        for rep_name, dim_red, model_name in grid:
            run_counter += 1
            combo_name = (
                f"[{run_counter}/{total_runs}] {model_name} | {rep_name} | "
                f"SVD: {dim_red} | {pipe_name}"
            )
            print(f"\n---> Starting: {combo_name}")

            fold_rows = run_cross_validation_single(
                pipeline_name=pipe_name,
                representation=rep_name,
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
    print("ALL FEATURE EXPERIMENTS COMPLETE - SUMMARY OF 5-FOLD CV ON DEV")
    print("=" * 95)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print(f"\nDetailed individual fold results saved to: {results_csv}")


if __name__ == "__main__":
    main()
