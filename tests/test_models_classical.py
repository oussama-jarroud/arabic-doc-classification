"""Unit tests for classical models factory, metrics, and leakage-free execution."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src.eval.metrics import (
    RESULTS_COLUMNS,
    compute_classification_metrics,
    log_result,
    measure_inference_latency_ms_per_doc,
)
from src.models_classical import (
    MODEL_REGISTRY,
    build_classical_model,
    build_classical_pipeline,
)


def test_build_all_classical_models() -> None:
    """Test that all 6 classical models from the paper instantiate with expected hyperparameters."""
    for model_name in MODEL_REGISTRY:
        model = build_classical_model(model_name)
        assert model is not None

    mnb = build_classical_model("MultinomialNB")
    assert isinstance(mnb, MultinomialNB)
    assert mnb.alpha == 1.0

    bnb = build_classical_model("BernoulliNB")
    assert isinstance(bnb, BernoulliNB)
    assert bnb.alpha == 1.0

    lr = build_classical_model("LogisticRegression")
    assert isinstance(lr, LogisticRegression)
    assert lr.C == 1.0
    assert lr.max_iter == 1000
    assert lr.random_state == 42

    sgd = build_classical_model("SGDClassifier")
    assert isinstance(sgd, SGDClassifier)
    assert sgd.loss == "hinge"
    assert sgd.alpha == 0.0001
    assert sgd.random_state == 42

    lsvc = build_classical_model("LinearSVC")
    assert isinstance(lsvc, LinearSVC)
    assert lsvc.C == 1.0
    assert lsvc.random_state == 42

    svc = build_classical_model("SVC")
    assert isinstance(svc, SVC)
    assert svc.kernel == "rbf"
    assert svc.C == 1.0
    assert svc.max_iter == 2000
    assert svc.random_state == 42


def test_build_classical_model_unknown_raises() -> None:
    """Test that unknown model name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported classical model"):
        build_classical_model("NonExistentClassifier")


def test_build_classical_pipeline_fit_predict() -> None:
    """Test end-to-end pipeline creation, fitting, and prediction on mock Arabic texts."""
    pipeline = build_classical_pipeline(
        representation="tfidf",
        model_name="MultinomialNB",
        vec_kwargs={"min_df": 1, "max_features": 100},
    )
    assert isinstance(pipeline, Pipeline)

    train_texts = [
        "فوز فريق كرة القدم في البطولة الرياضية",
        "مباراة كرة السلة كانت حماسية جدا",
        "ارتفاع أسعار الأسهم في البورصة المالية",
        "تقرير حول التضخم والنمو الاقتصادي والمالي",
    ]
    train_labels = [0, 0, 1, 1]

    pipeline.fit(train_texts, train_labels)

    test_texts = ["مباراة كرة القدم", "أسعار الأسهم في السوق"]
    preds = pipeline.predict(test_texts)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})


def test_compute_classification_metrics_exact() -> None:
    """Test computation of precision, recall, f1, and accuracy with zero_division protection."""
    y_true = [0, 1, 0, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    metrics = compute_classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics

    # Accuracy is 4/5 = 0.8
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_log_result_csv_schema(tmp_path: Path) -> None:
    """Test that log_result writes exactly the 14 required columns and rejects incomplete rows."""
    csv_file = tmp_path / "test_results.csv"

    valid_row: dict[str, Any] = {
        "run_id": "test_run_01",
        "model": "LinearSVC",
        "representation": "tfidf",
        "dim_reduction": "none",
        "preprocessing": "light_normalized",
        "noise_level": 0.0,
        "seed": 42,
        "split": "cv_fold_0",
        "accuracy": 0.95,
        "macro_precision": 0.94,
        "macro_recall": 0.95,
        "macro_f1": 0.945,
        "train_time_sec": 1.25,
        "inference_ms_per_doc": 0.045,
    }

    log_result(csv_file, valid_row)
    assert csv_file.is_file()

    df = pd.read_csv(csv_file)
    assert list(df.columns) == RESULTS_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "test_run_01"

    # Test appending a second row
    valid_row["run_id"] = "test_run_02"
    log_result(csv_file, valid_row)
    df2 = pd.read_csv(csv_file)
    assert len(df2) == 2

    # Test missing mandatory column raises ValueError
    incomplete_row = dict(valid_row)
    del incomplete_row["macro_f1"]
    with pytest.raises(ValueError, match="Missing mandatory column"):
        log_result(csv_file, incomplete_row)


def test_zero_data_leakage_in_pipeline() -> None:
    """Verify vectorizer inside pipeline does not leak vocabulary from validation fold."""
    pipeline = build_classical_pipeline(
        representation="bow",
        model_name="MultinomialNB",
        vec_kwargs={"min_df": 1},
    )

    train_fold_texts = ["كلمة_تدريب_أولى كلمة_تدريب_ثانية"]
    train_fold_labels = [0]
    val_fold_texts = ["كلمة_تحقق_سرية"]

    pipeline.fit(train_fold_texts, train_fold_labels)
    preds = pipeline.predict(val_fold_texts)
    assert len(preds) == 1

    vec = pipeline.named_steps["vectorizer"]
    # Words unique to validation set MUST NOT exist in fitted vocabulary
    assert "كلمة_تحقق_سرية" not in vec.vocabulary_
    assert "كلمة_تدريب_أولى" in vec.vocabulary_


def test_measure_inference_latency_ms_per_doc() -> None:
    """Test latency benchmark function returns valid positive float on mock pipeline."""
    pipeline = build_classical_pipeline(
        representation="bow",
        model_name="MultinomialNB",
        vec_kwargs={"min_df": 1},
    )
    pipeline.fit(["نص رياضي تجريبي"], [0])
    latency = measure_inference_latency_ms_per_doc(
        pipeline,
        ["نص رياضي تجريبي"] * 10,
        repetitions=2,
    )
    assert isinstance(latency, float)
    assert latency >= 0.0
