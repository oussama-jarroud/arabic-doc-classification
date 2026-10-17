"""CLI entry point to execute 5-Fold Stratified Cross-Validation on classical baselines."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_config  # noqa: E402
from src.models_classical import MODEL_REGISTRY, run_cross_validation_single  # noqa: E402

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line flags for classical baseline experiments."""
    parser = argparse.ArgumentParser(
        description="Run 5-Fold Stratified Cross-Validation for classical ML baselines."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["light_normalized", "paper_heavy_stemming"],
        help="Preprocessed text variants to evaluate.",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["bow", "tfidf", "tfidf_char_wb"],
        help="Feature extraction representations to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="List of classical models to benchmark.",
    )
    return parser.parse_args()


def run_experiments(
    pipelines: list[str],
    representations: list[str],
    models: list[str],
    config_path: str,
) -> None:
    """Run all specified cross-validation grid combinations and print summary tables."""
    cfg = load_config(config_path)
    total_runs = len(pipelines) * len(representations) * len(models)

    print("=" * 80)
    print("CLASSICAL ML BASELINE BENCHMARK: 5-FOLD STRATIFIED CV ON DEV")
    print(f"Total Model Combinations: {total_runs} (5 folds each -> {total_runs * 5} folds)")
    print(f"Pipelines: {pipelines}")
    print(f"Representations: {representations}")
    print(f"Models: {models}")
    print("=" * 80)

    summary_records: list[dict[str, Any]] = []
    run_counter = 0

    for pipe_name in pipelines:
        for rep_name in representations:
            for model_name in models:
                run_counter += 1
                combo_name = f"[{run_counter}/{total_runs}] {model_name} | {rep_name} | {pipe_name}"
                print(f"\n---> Starting: {combo_name}")

                fold_rows = run_cross_validation_single(
                    pipeline_name=pipe_name,
                    representation=rep_name,
                    model_name=model_name,
                    config=cfg,
                    project_root=PROJECT_ROOT,
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
                        "Preprocessing": pipe_name,
                        "Accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
                        "Macro-Precision": f"{np.mean(precs):.4f} +/- {np.std(precs):.4f}",
                        "Macro-Recall": f"{np.mean(recs):.4f} +/- {np.std(recs):.4f}",
                        "Macro-F1": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
                        "Mean_Train_Sec": f"{np.mean(times):.2f}",
                        "Mean_Infer_ms": f"{np.mean(lats):.3f}",
                    }
                )

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTS COMPLETE - SUMMARY OF 5-FOLD CV ON DEV")
    print("=" * 90)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    results_csv = PROJECT_ROOT / cfg.get("classical_experiments", {}).get(
        "results_csv", "results/results_classical.csv"
    )
    print(f"\nDetailed individual fold results saved to: {results_csv}")


def main() -> None:
    args = parse_args()
    run_experiments(
        pipelines=args.pipelines,
        representations=args.representations,
        models=args.models,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
