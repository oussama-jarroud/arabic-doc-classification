"""CLI and Google Colab script to train PyTorch Arabic 1D-CNN with GPU acceleration."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

# Ensure project root is in sys.path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_config, load_processed_split  # noqa: E402
from src.models_dl import run_dl_training  # noqa: E402

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse command line flags for deep learning training."""
    parser = argparse.ArgumentParser(
        description="Train Arabic 1D-CNN on Google Colab or local GPU."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment configuration YAML.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="light_normalized",
        help="Preprocessed data variant to load (e.g., light_normalized).",
    )
    parser.add_argument(
        "--mode",
        choices=["single_split", "cv5"],
        default="single_split",
        help="Validation protocol: 'single_split' (fast 80/20 on dev) or 'cv5' (5-fold CV).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio when mode='single_split'.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute PyTorch deep learning training and log metrics."""
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 80)
    print("PHASE 6: DEEP LEARNING (1D-CNN) ARABIC TEXT CLASSIFICATION")
    print(f"Device: {device} ({device_name})")
    print(f"Dataset Pipeline: {args.pipeline}")
    print(f"Validation Mode: {args.mode}")
    print("=" * 80)

    # 1. Load dev split
    texts, labels = load_processed_split(
        pipeline_name=args.pipeline,
        split_name="dev",
        config=cfg,
        project_root=PROJECT_ROOT,
    )
    print(f"Loaded {len(texts)} articles from dev partition.")

    seed = int(cfg["dataset"]["seed"])

    if args.mode == "single_split":
        # Deterministic stratified 80/20 train/val split inside dev
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=args.val_ratio,
            random_state=seed,
            stratify=labels,
        )
        print(
            f"Dev partitioned into {len(train_texts)} train and "
            f"{len(val_texts)} validation articles."
        )

        row = run_dl_training(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            config=cfg,
            device=device,
            split_name="dev_val_split",
            project_root=PROJECT_ROOT,
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - MEASURED VALIDATION METRICS")
        print("=" * 80)
        print(f"Accuracy:        {row['accuracy']:.4f}")
        print(f"Macro-Precision: {row['macro_precision']:.4f}")
        print(f"Macro-Recall:    {row['macro_recall']:.4f}")
        print(f"Macro-F1:        {row['macro_f1']:.4f}")
        print(f"Training Time:   {row['train_time_sec']:.2f} seconds")
        print(f"Latency:         {row['inference_ms_per_doc']:.3f} ms/doc")

    elif args.mode == "cv5":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        texts_arr = np.asarray(texts, dtype=object)
        labels_arr = np.asarray(labels, dtype=int)
        fold_rows: list[dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
            print(f"\n---> Starting Fold {fold_idx + 1}/5")
            row = run_dl_training(
                train_texts=texts_arr[train_idx].tolist(),
                train_labels=labels_arr[train_idx].tolist(),
                val_texts=texts_arr[val_idx].tolist(),
                val_labels=labels_arr[val_idx].tolist(),
                config=cfg,
                device=device,
                split_name=f"cv_fold_{fold_idx}",
                project_root=PROJECT_ROOT,
            )
            fold_rows.append(row)
            print(f"     Fold {fold_idx + 1} Macro-F1: {row['macro_f1']:.4f}")

        accs = [r["accuracy"] for r in fold_rows]
        f1s = [r["macro_f1"] for r in fold_rows]
        times = [r["train_time_sec"] for r in fold_rows]
        lats = [r["inference_ms_per_doc"] for r in fold_rows]

        print("\n" + "=" * 80)
        print("5-FOLD CV COMPLETE - SUMMARY METRICS")
        print("=" * 80)
        print(f"Mean Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"Mean Macro-F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"Avg Train Time: {np.mean(times):.1f}s | Latency: {np.mean(lats):.3f} ms/doc")

    results_csv_rel = cfg.get("deep_learning", {}).get("results_csv", "results/results_dl.csv")
    print(f"\nResults appended to: {PROJECT_ROOT / results_csv_rel}")


if __name__ == "__main__":
    main()
