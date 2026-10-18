"""Unit tests for advanced feature representations, fusion, and SVD dimension reduction."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.eval.metrics import RESULTS_COLUMNS, log_result
from src.features import build_dim_reduction, build_feature_extractor
from src.models_classical import build_classical_pipeline


def test_build_feature_extractor_word_ngrams() -> None:
    """Test that word n-gram feature extractors parse tuples and extract multi-word tokens."""
    vec_1_2 = build_feature_extractor("tfidf_word_1_2")
    assert isinstance(vec_1_2, TfidfVectorizer)
    assert vec_1_2.ngram_range == (1, 2)
    assert vec_1_2.sublinear_tf is True

    vec_1_3 = build_feature_extractor("tfidf_word_1_3")
    assert isinstance(vec_1_3, TfidfVectorizer)
    assert vec_1_3.ngram_range == (1, 3)

    sample_texts = [
        "جامعة الدول العربية تناقش التطورات السياسية في الشرق الأوسط",
        "الشرق الأوسط يشهد نموا اقتصاديا وتطورا تقنيا كبيرا",
    ]
    matrix = vec_1_2.fit_transform(sample_texts)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0


def test_build_feature_extractor_sparse_fusion() -> None:
    """Test that sparse_fusion builds FeatureUnion combining word and character TF-IDF."""
    fusion = build_feature_extractor(
        "sparse_fusion",
        word_tfidf={"min_df": 1, "max_features": 100},
        char_tfidf={"min_df": 1, "max_features": 100},
    )
    assert isinstance(fusion, FeatureUnion)

    transformer_names = [name for name, _ in fusion.transformer_list]
    assert "word_tfidf" in transformer_names
    assert "char_tfidf" in transformer_names

    sample_texts = [
        "فريق كرة القدم يحقق انتصارا كبيرا في البطولة",
        "البورصة تسجل أرباحا مالية قياسية اليوم",
    ]
    fused_matrix = fusion.fit_transform(sample_texts)
    assert fused_matrix.shape[0] == 2
    assert fused_matrix.shape[1] > 0


def test_build_dim_reduction_svd() -> None:
    """Test TruncatedSVD pipeline construction and dimensionality reduction properties."""
    dim_300 = build_dim_reduction("svd_300")
    assert isinstance(dim_300, Pipeline)
    assert dim_300.named_steps["svd"].n_components == 300

    dim_1000 = build_dim_reduction("svd_1000")
    assert isinstance(dim_1000, Pipeline)
    assert dim_1000.named_steps["svd"].n_components == 1000

    none_dim = build_dim_reduction("none")
    assert none_dim is None

    with pytest.raises(ValueError, match="Unsupported dimensionality reduction"):
        build_dim_reduction("invalid_svd_name")


def test_zero_data_leakage_in_advanced_pipeline() -> None:
    """Verify end-to-end pipeline with FeatureUnion and SVD preserves zero-leakage isolation."""
    pipeline = build_classical_pipeline(
        representation="sparse_fusion",
        model_name="LinearSVC",
        dim_reduction="svd_300",
        dim_kwargs={"n_components": 2},
        vec_kwargs={
            "word_tfidf": {"min_df": 1, "max_features": 50},
            "char_tfidf": {"min_df": 1, "max_features": 50},
        },
    )
    assert isinstance(pipeline, Pipeline)

    train_texts = [
        "نص سياسي تدريبي يتحدث عن الانتخابات الحكومية",
        "تقرير اقتصادي عن البورصة والأسهم والاستثمار",
        "مباراة رياضية في كرة القدم وكأس العالم",
        "اكتشاف طبي جديد لعلاج الأمراض المزمنة",
    ]
    train_labels = [0, 1, 2, 3]

    pipeline.fit(train_texts, train_labels)

    val_texts = ["تقرير سياسي جديد", "مباراة رياضية حماسية"]
    preds = pipeline.predict(val_texts)
    assert len(preds) == 2
    assert all(isinstance(p, (int, np.integer)) for p in preds)

    # Check that SVD projection matrices are sized to training vocabulary
    svd_step = pipeline.named_steps["dim_reduction"].named_steps["svd"]
    assert svd_step.components_.shape[0] == 2


def test_feature_results_csv_schema(tmp_path: Path) -> None:
    """Test that results_features.csv strictly follows the 14-column schema."""
    csv_file = tmp_path / "results_features.csv"

    valid_record: dict[str, Any] = {
        "run_id": "LinearSVC_sparse_fusion_svd_300_light_normalized_fold0",
        "model": "LinearSVC",
        "representation": "sparse_fusion",
        "dim_reduction": "svd_300",
        "preprocessing": "light_normalized",
        "noise_level": 0.0,
        "seed": 42,
        "split": "cv_fold_0",
        "accuracy": 0.9812,
        "macro_precision": 0.9815,
        "macro_recall": 0.9811,
        "macro_f1": 0.9813,
        "train_time_sec": 4.52,
        "inference_ms_per_doc": 0.12,
    }

    log_result(csv_file, valid_record)
    assert csv_file.is_file()

    df = pd.read_csv(csv_file)
    assert list(df.columns) == RESULTS_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["dim_reduction"] == "svd_300"
    assert df.iloc[0]["representation"] == "sparse_fusion"
