"""Unit tests for feature extraction factory and vectorizer configurations."""

import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.features import build_vectorizer


def test_build_vectorizer_bow() -> None:
    """Test that bow builds CountVectorizer with expected configuration and transforms Arabic."""
    vec = build_vectorizer("bow")
    assert isinstance(vec, CountVectorizer)
    assert vec.max_features == 50000
    assert vec.min_df == 2

    # Verify fit-transform on mock Arabic documents
    sample_texts = [
        "هذا نص تجريبي للاختبار في الرياضة",
        "نص تجريبي آخر يتحدث عن الرياضة والثقافة",
        "الثقافة والفنون في العالم العربي",
    ]
    matrix = vec.fit_transform(sample_texts)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0


def test_build_vectorizer_tfidf() -> None:
    """Test that tfidf builds TfidfVectorizer with sublinear_tf enabled."""
    vec = build_vectorizer("tfidf")
    assert isinstance(vec, TfidfVectorizer)
    assert vec.max_features == 50000
    assert vec.sublinear_tf is True


def test_build_vectorizer_tfidf_char_wb() -> None:
    """Test that tfidf_char_wb builds character n-gram TfidfVectorizer."""
    vec = build_vectorizer("tfidf_char_wb")
    assert isinstance(vec, TfidfVectorizer)
    assert vec.analyzer == "char_wb"
    assert vec.ngram_range == (3, 5)
    assert vec.max_features == 75000
    assert vec.min_df == 3
    assert vec.sublinear_tf is True


def test_build_vectorizer_unsupported_raises() -> None:
    """Test that requesting an unknown vectorizer name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported feature representation"):
        build_vectorizer("unknown_rep")


def test_build_vectorizer_override_kwargs() -> None:
    """Test that kwargs properly override default config settings."""
    vec = build_vectorizer("bow", max_features=100, min_df=1)
    assert vec.max_features == 100
    assert vec.min_df == 1
