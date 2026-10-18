"""Unit tests for Gradient Boosting models, StackingClassifier ensemble, and pipeline execution."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline

from src.eval.metrics import RESULTS_COLUMNS, log_result
from src.models_advanced import (
    build_advanced_pipeline,
    build_stacking_ensemble,
    build_tree_model,
)


def test_build_tree_models() -> None:
    """Test that gradient boosting models instantiate with expected parameters."""
    hgb = build_tree_model("HistGradientBoosting")
    assert isinstance(hgb, HistGradientBoostingClassifier)
    assert hgb.max_depth == 7
    assert hgb.learning_rate == 0.1

    lgbm = build_tree_model("LightGBM")
    assert lgbm is not None

    xgb = build_tree_model("XGBoost")
    assert xgb is not None

    with pytest.raises(ValueError, match="Unsupported tree model"):
        build_tree_model("UnknownTreeModel")


def test_build_stacking_ensemble() -> None:
    """Test that StackingClassifier is constructed with out-of-fold CV and base estimators."""
    stacking = build_stacking_ensemble()
    assert isinstance(stacking, StackingClassifier)
    assert stacking.cv == 5

    estimator_names = [name for name, _ in stacking.estimators]
    assert "linearsvc" in estimator_names
    assert "logisticregression" in estimator_names
    assert "multinomialnb" in estimator_names


def test_fit_predict_tree_pipeline_svd() -> None:
    """Test end-to-end fitting and prediction of sparse_fusion -> SVD -> tree model."""
    pipeline = build_advanced_pipeline(
        representation="sparse_fusion",
        model_type="tree",
        model_name="HistGradientBoosting",
        dim_reduction="svd_300",
        dim_kwargs={"n_components": 2},
        vec_kwargs={
            "word_tfidf": {"min_df": 1, "max_features": 50},
            "char_tfidf": {"min_df": 1, "max_features": 50},
        },
    )
    assert isinstance(pipeline, Pipeline)

    train_texts = [
        "فوز منتخب كرة القدم وتأهله لنهائيات كأس العالم",
        "مباراة كرة السلة كانت مثيرة في الدوري المحلي",
        "ارتفاع مؤشر البورصة وأسعار الأسهم المالية اليوم",
        "تقرير البنك المركزي حول معدلات الفائدة والتضخم",
    ]
    train_labels = [0, 0, 1, 1]

    pipeline.fit(train_texts, train_labels)

    test_texts = ["مباراة كرة القدم", "مؤشر الأسهم والبورصة"]
    preds = pipeline.predict(test_texts)
    assert len(preds) == 2
    assert all(isinstance(p, (int, np.integer)) for p in preds)


def test_fit_predict_stacking_pipeline() -> None:
    """Test end-to-end fitting and prediction of StackingClassifier pipeline without leakage."""
    pipeline = build_advanced_pipeline(
        representation="tfidf",
        model_type="stacking",
        model_name="StackingEnsemble",
        dim_reduction="none",
        vec_kwargs={"min_df": 1},
        model_kwargs={"cv": 2, "n_jobs": 1},
    )
    assert isinstance(pipeline, Pipeline)

    train_texts = [
        "فوز منتخب كرة القدم وتأهله لنهائيات كأس العالم",
        "مباراة كرة السلة كانت مثيرة في الدوري المحلي",
        "أداء فريق كرة اليد في البطولة الرياضية",
        "مباراة التنس الأرضي في بطولة رولان غاروس",
        "ارتفاع مؤشر البورصة وأسعار الأسهم المالية اليوم",
        "تقرير البنك المركزي حول معدلات الفائدة والتضخم",
        "استثمار الشركات في الأسواق المالية والبورصة",
        "أرباح البنوك والمصارف في الربع السنوي الأخير",
    ]
    train_labels = [0, 0, 0, 0, 1, 1, 1, 1]

    pipeline.fit(train_texts, train_labels)

    test_texts = ["مباراة كرة القدم", "سوق الأسهم"]
    preds = pipeline.predict(test_texts)
    assert len(preds) == 2
    assert all(isinstance(p, (int, np.integer)) for p in preds)


def test_advanced_results_csv_schema(tmp_path: Path) -> None:
    """Test that results_advanced.csv adheres strictly to the 14-column schema."""
    csv_file = tmp_path / "results_advanced.csv"

    valid_record: dict[str, Any] = {
        "run_id": "StackingEnsemble_sparse_fusion_none_light_normalized_fold0",
        "model": "StackingEnsemble",
        "representation": "sparse_fusion",
        "dim_reduction": "none",
        "preprocessing": "light_normalized",
        "noise_level": 0.0,
        "seed": 42,
        "split": "cv_fold_0",
        "accuracy": 0.9845,
        "macro_precision": 0.9846,
        "macro_recall": 0.9844,
        "macro_f1": 0.9845,
        "train_time_sec": 185.2,
        "inference_ms_per_doc": 5.12,
    }

    log_result(csv_file, valid_record)
    assert csv_file.is_file()

    df = pd.read_csv(csv_file)
    assert list(df.columns) == RESULTS_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["model"] == "StackingEnsemble"
