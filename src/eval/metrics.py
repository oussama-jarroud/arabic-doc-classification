"""Metric computation, latency measurement, and strict CSV result logging."""

import csv
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RESULTS_COLUMNS: list[str] = [
    "run_id",
    "model",
    "representation",
    "dim_reduction",
    "preprocessing",
    "noise_level",
    "seed",
    "split",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "train_time_sec",
    "inference_ms_per_doc",
]


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Compute standard classification evaluation metrics via scikit-learn.

    Uses macro averaging and sets zero_division=0 to guarantee mathematically valid,
    unbiased multi-class evaluation without NaN edge cases.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def measure_inference_latency_ms_per_doc(
    model: Any,
    texts: Sequence[str],
    repetitions: int = 3,
    sample_size: int = 500,
) -> float:
    """Measure average per-document inference latency in milliseconds.

    Evaluates throughput on a representative validation subsample to record deployment
    feasibility metrics alongside classification accuracy.
    """
    if not texts:
        return 0.0

    eval_texts = list(texts[:sample_size])
    n_docs = len(eval_texts)

    # Warmup pass
    _ = model.predict(eval_texts[: min(10, n_docs)])

    durations: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        _ = model.predict(eval_texts)
        durations.append(time.perf_counter() - start)

    # Use minimum duration to minimize OS scheduling jitter
    best_time_sec = min(durations)
    return float((best_time_sec / n_docs) * 1000.0)


def log_result(
    csv_path: Path | str,
    row: dict[str, Any],
) -> None:
    """Append a validated experiment record to the results CSV file.

    Guarantees strict schema conformity across the 14 mandatory columns without
    permitting missing fields or schema drift.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [col for col in RESULTS_COLUMNS if col not in row]
    if missing:
        raise ValueError(f"Missing mandatory column(s) in result row: {missing}")

    file_exists = path.is_file() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row[col] for col in RESULTS_COLUMNS})
