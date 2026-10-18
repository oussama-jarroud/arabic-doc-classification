"""Feature extraction and dimensionality reduction factory functions for text representation."""

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_config


def _clean_ngram_range(params: dict[str, Any]) -> dict[str, Any]:
    """Ensure ngram_range from YAML list format is converted to tuple."""
    params_copy = dict(params)
    if "ngram_range" in params_copy and isinstance(params_copy["ngram_range"], list):
        params_copy["ngram_range"] = tuple(params_copy["ngram_range"])
    return params_copy


def build_vectorizer(
    name: str,
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> BaseEstimator:
    """Build a deterministic scikit-learn text vectorizer or feature union from config.

    Extracts word, character n-gram, or multi-representation hybrid features to benchmark
    sparse lexical representations against dense baselines. Uses float32 precision for
    halved memory footprint and enhanced vectorization throughput.
    """
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    features_cfg: dict[str, Any] = cfg.get("features", {})

    if name not in features_cfg:
        supported = list(features_cfg.keys())
        raise ValueError(
            f"Unsupported feature representation '{name}'. Available options: {supported}"
        )

    if name == "sparse_fusion":
        fusion_cfg: dict[str, Any] = features_cfg["sparse_fusion"]
        word_params = _clean_ngram_range(fusion_cfg.get("word_tfidf", {}))
        char_params = _clean_ngram_range(fusion_cfg.get("char_tfidf", {}))

        # Default to float32 to conserve memory during large sparse matrix slicing
        word_params.setdefault("dtype", np.float32)
        char_params.setdefault("dtype", np.float32)

        # Allow kwargs to override sub-transformer parameters if provided
        if "word_tfidf" in kwargs:
            word_params.update(_clean_ngram_range(kwargs["word_tfidf"]))
        if "char_tfidf" in kwargs:
            char_params.update(_clean_ngram_range(kwargs["char_tfidf"]))

        word_vec = TfidfVectorizer(**word_params)
        char_vec = TfidfVectorizer(**char_params)
        return FeatureUnion([("word_tfidf", word_vec), ("char_tfidf", char_vec)])

    params: dict[str, Any] = _clean_ngram_range(dict(features_cfg[name]))
    params.update(_clean_ngram_range(kwargs))

    if name == "bow":
        return CountVectorizer(**params)
    if name in ("tfidf", "tfidf_char_wb", "tfidf_word_1_2", "tfidf_word_1_3"):
        params.setdefault("dtype", np.float32)
        return TfidfVectorizer(**params)

    raise ValueError(f"No estimator mapping defined for vectorizer key: '{name}'")


def build_feature_extractor(
    name: str,
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> BaseEstimator:
    """Alias for build_vectorizer supporting composite and multi-representation extractors."""
    return build_vectorizer(name, config=config, **kwargs)


def build_dim_reduction(
    name: str | None,
    config: dict[str, Any] | Path | str | None = None,
    **kwargs: Any,
) -> Pipeline | None:
    """Build dimensionality reduction pipeline (e.g., TruncatedSVD + StandardScaler).

    Projects high-dimensional sparse text vectors into dense latent semantic subspaces
    while fitting solely on training folds to prevent target and data leakage.
    """
    if name is None or name.lower() == "none":
        return None

    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    dim_cfg: dict[str, Any] = cfg.get("dim_reduction", {})

    if name not in dim_cfg:
        supported = list(dim_cfg.keys()) + ["none"]
        raise ValueError(
            f"Unsupported dimensionality reduction '{name}'. Available options: {supported}"
        )

    params: dict[str, Any] = dict(dim_cfg[name])
    params.update(kwargs)

    n_components = int(params.get("n_components", 300))
    random_state = int(params.get("random_state", 42))

    return Pipeline([
        ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
        ("scaler", StandardScaler(with_mean=True)),
    ])
