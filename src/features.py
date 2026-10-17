"""Feature extraction factory functions for text representation pipelines."""

from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.data import load_config


def build_vectorizer(
    name: str,
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> BaseEstimator:
    """Build a deterministic scikit-learn text vectorizer according to config specifications.

    Extracts word-level or character n-gram bag-of-words or TF-IDF features to benchmark
    sparse lexical representations against dense and deep learning baselines.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    features_cfg: dict[str, Any] = cfg.get("features", {})

    if name not in features_cfg:
        supported = list(features_cfg.keys())
        raise ValueError(
            f"Unsupported feature representation '{name}'. Available options: {supported}"
        )

    params: dict[str, Any] = dict(features_cfg[name])
    params.update(kwargs)

    # Convert list representations of ngram_range from YAML into tuples
    if "ngram_range" in params and isinstance(params["ngram_range"], list):
        params["ngram_range"] = tuple(params["ngram_range"])

    if name == "bow":
        return CountVectorizer(**params)
    if name in ("tfidf", "tfidf_char_wb"):
        return TfidfVectorizer(**params)

    raise ValueError(f"No estimator mapping defined for vectorizer key: '{name}'")
