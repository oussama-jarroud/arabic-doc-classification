"""Classical machine learning baseline estimators and 5-fold cross-validation engine."""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src.data import get_cv_folds, load_config, load_processed_split
from src.eval.metrics import (
    compute_classification_metrics,
    log_result,
    measure_inference_latency_ms_per_doc,
)
from src.features import build_dim_reduction, build_vectorizer

MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "MultinomialNB": MultinomialNB,
    "BernoulliNB": BernoulliNB,
    "LogisticRegression": LogisticRegression,
    "SGDClassifier": SGDClassifier,
    "LinearSVC": LinearSVC,
    "SVC": SVC,
}


def build_classical_model(
    name: str,
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> BaseEstimator:
    """Instantiate a classical scikit-learn classification estimator configured from YAML.

    Builds deterministic baselines from Muaad et al. (2022) to enable empirical benchmarking
    against modern feature representations.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported classical model '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    model_cfg: dict[str, Any] = cfg.get("classical_models", {}).get(name, {})

    params = dict(model_cfg)
    params.update(kwargs)

    estimator_cls = MODEL_REGISTRY[name]
    return estimator_cls(**params)


def build_classical_pipeline(
    representation: str,
    model_name: str,
    config: dict[str, Any] | Path | str | None = None,
    vec_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    dim_reduction: str | None = None,
    dim_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    """Compose feature extraction, optional dimension reduction, and classifier into a pipeline.

    Encapsulates text transformation and classification into a unified estimator to prevent
    information leakage during cross-validation.
    """
    vec = build_vectorizer(representation, config, **(vec_kwargs or {}))
    steps: list[tuple[str, Any]] = [("vectorizer", vec)]

    if dim_reduction and dim_reduction.lower() != "none":
        dim_trans = build_dim_reduction(dim_reduction, config, **(dim_kwargs or {}))
        if dim_trans is not None:
            steps.append(("dim_reduction", dim_trans))

    clf = build_classical_model(model_name, config, **(model_kwargs or {}))
    steps.append(("clf", clf))

    return Pipeline(steps)


def run_cross_validation_single(
    pipeline_name: str,
    representation: str,
    model_name: str,
    dim_reduction: str = "none",
    config: dict[str, Any] | Path | str | None = None,
    project_root: Path | None = None,
    save_best_model: bool = True,
    results_csv: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Execute leak-free 5-Fold Stratified Cross-Validation strictly within the dev partition.

    Trains the complete vectorizer, optional dimension reduction, and classifier pipeline per fold,
    benchmarks metrics and inference latency, logs records to CSV, and saves the top pipeline.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    exp_cfg = cfg.get("classical_experiments", {})
    if results_csv is None:
        results_csv_rel = exp_cfg.get("results_csv", "results/results_classical.csv")
        target_csv = root / results_csv_rel
    else:
        target_csv = Path(results_csv) if Path(results_csv).is_absolute() else root / results_csv

    models_dir_rel = exp_cfg.get("models_dir", "models_saved")
    models_dir = root / models_dir_rel

    seed = int(cfg["dataset"]["seed"])

    texts, labels = load_processed_split(
        pipeline_name=pipeline_name,
        split_name="dev",
        config=cfg,
        project_root=root,
    )

    texts_arr = np.asarray(texts, dtype=object)
    labels_arr = np.asarray(labels, dtype=int)

    fold_results: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_macro_f1 = -1.0
    best_fold_idx = -1

    for fold_idx, (train_idx, val_idx) in enumerate(get_cv_folds(labels_arr, cfg)):
        fold_train_texts: Sequence[str] = texts_arr[train_idx].tolist()
        fold_train_labels: Sequence[int] = labels_arr[train_idx].tolist()
        fold_val_texts: Sequence[str] = texts_arr[val_idx].tolist()
        fold_val_labels: Sequence[int] = labels_arr[val_idx].tolist()

        # Build clean un-fitted pipeline strictly inside fold
        pipeline = build_classical_pipeline(
            representation=representation,
            model_name=model_name,
            dim_reduction=dim_reduction,
            config=cfg,
        )

        start_train = time.perf_counter()
        pipeline.fit(fold_train_texts, fold_train_labels)
        train_time_sec = time.perf_counter() - start_train

        val_preds = pipeline.predict(fold_val_texts)
        metrics = compute_classification_metrics(fold_val_labels, val_preds)
        inference_ms = measure_inference_latency_ms_per_doc(pipeline, fold_val_texts)

        run_id = f"{model_name}_{representation}_{dim_reduction}_{pipeline_name}_fold{fold_idx}"

        row: dict[str, Any] = {
            "run_id": run_id,
            "model": model_name,
            "representation": representation,
            "dim_reduction": dim_reduction,
            "preprocessing": pipeline_name,
            "noise_level": 0.0,
            "seed": seed,
            "split": f"cv_fold_{fold_idx}",
            "accuracy": round(metrics["accuracy"], 6),
            "macro_precision": round(metrics["macro_precision"], 6),
            "macro_recall": round(metrics["macro_recall"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "train_time_sec": round(train_time_sec, 4),
            "inference_ms_per_doc": round(inference_ms, 4),
        }

        log_result(target_csv, row)
        fold_results.append(row)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_pipeline = pipeline
            best_fold_idx = fold_idx

    if save_best_model and best_pipeline is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"{model_name}_{representation}_{dim_reduction}_{pipeline_name}.joblib"
        model_save_path = models_dir / model_filename
        joblib.dump(
            {
                "pipeline": best_pipeline,
                "model_name": model_name,
                "representation": representation,
                "dim_reduction": dim_reduction,
                "preprocessing": pipeline_name,
                "best_fold": best_fold_idx,
                "best_macro_f1": best_macro_f1,
                "config": cfg,
            },
            model_save_path,
        )

    return fold_results
