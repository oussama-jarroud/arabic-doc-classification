"""Advanced Gradient Boosting models and Stacking Classifier ensemble architecture."""

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline

from src.data import get_cv_folds, load_config, load_processed_split
from src.eval.metrics import (
    compute_classification_metrics,
    log_result,
    measure_inference_latency_ms_per_doc,
)
from src.features import build_dim_reduction, build_feature_extractor
from src.models_classical import build_classical_model

logger = logging.getLogger(__name__)


def build_tree_model(
    name: str,
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> BaseEstimator:
    """Instantiate a gradient boosted decision tree classifier with fallback support.

    Constructs LightGBM, XGBoost, or HistGradientBoosting estimators tuned for dense
    latent semantic representations.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    adv_cfg: dict[str, Any] = cfg.get("advanced_models", {})

    if name not in adv_cfg:
        supported = list(adv_cfg.keys())
        raise ValueError(f"Unsupported tree model '{name}'. Available: {supported}")

    params = dict(adv_cfg[name])
    params.update(kwargs)

    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(**params)
        except ImportError:
            logger.warning("lightgbm not available; falling back to HistGradientBoostingClassifier")
            fallback_params = {
                "max_depth": params.get("max_depth", 7),
                "learning_rate": params.get("learning_rate", 0.1),
                "random_state": params.get("random_state", 42),
            }
            return HistGradientBoostingClassifier(**fallback_params)

    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(**params)
        except ImportError:
            logger.warning("xgboost not available; falling back to HistGradientBoostingClassifier")
            fallback_params = {
                "max_depth": params.get("max_depth", 7),
                "learning_rate": params.get("learning_rate", 0.1),
                "random_state": params.get("random_state", 42),
            }
            return HistGradientBoostingClassifier(**fallback_params)

    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(**params)

    raise ValueError(f"Unrecognized gradient boosting model key: {name}")


def build_stacking_ensemble(
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> StackingClassifier:
    """Build a leak-free StackingClassifier combining diverse classical base estimators.

    Uses internal out-of-fold cross-validation (cv=5) on the training set to construct
    unbiased meta-features for the final logistic regression meta-learner.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    stack_cfg: dict[str, Any] = dict(cfg.get("stacking_ensemble", {}))
    stack_cfg.update(kwargs)

    base_names: list[str] = stack_cfg.get(
        "base_estimators", ["LinearSVC", "LogisticRegression", "MultinomialNB"]
    )
    estimators: list[tuple[str, BaseEstimator]] = [
        (name.lower(), build_classical_model(name, config=cfg)) for name in base_names
    ]

    meta_name: str = stack_cfg.get("meta_learner", "LogisticRegression")
    meta_params: dict[str, Any] = stack_cfg.get("meta_learner_params", {})
    final_estimator = build_classical_model(meta_name, config=cfg, **meta_params)

    cv = int(stack_cfg.get("cv", 5))
    n_jobs = int(stack_cfg.get("n_jobs", -1))

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=n_jobs,
        passthrough=False,
    )


def build_advanced_pipeline(
    representation: str,
    model_type: str,
    model_name: str,
    dim_reduction: str | None = None,
    config: dict[str, Any] | Path | str | None = None,
    vec_kwargs: dict[str, Any] | None = None,
    dim_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    """Compose complete text feature extraction, reduction, and advanced estimator pipeline."""
    steps: list[tuple[str, Any]] = [
        ("vectorizer", build_feature_extractor(representation, config=config, **(vec_kwargs or {})))
    ]

    if dim_reduction and dim_reduction.lower() != "none":
        dim_trans = build_dim_reduction(dim_reduction, config=config, **(dim_kwargs or {}))
        if dim_trans is not None:
            steps.append(("dim_reduction", dim_trans))

    if model_type == "tree":
        clf = build_tree_model(model_name, config=config, **(model_kwargs or {}))
    elif model_type == "stacking":
        clf = build_stacking_ensemble(config=config, **(model_kwargs or {}))
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Expected 'tree' or 'stacking'.")

    steps.append(("clf", clf))
    return Pipeline(steps)


def run_advanced_cv_single(
    pipeline_name: str,
    representation: str,
    model_type: str,
    model_name: str,
    dim_reduction: str = "none",
    config: dict[str, Any] | Path | str | None = None,
    project_root: Path | None = None,
    save_best_model: bool = True,
    results_csv: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Execute 5-Fold Stratified Cross-Validation strictly within dev for advanced architectures."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    exp_cfg = cfg.get("advanced_experiments", {})
    if results_csv is None:
        results_csv_rel = exp_cfg.get("results_csv", "results/results_advanced.csv")
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

        pipeline = build_advanced_pipeline(
            representation=representation,
            model_type=model_type,
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
