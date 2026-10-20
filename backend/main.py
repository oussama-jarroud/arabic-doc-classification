"""FastAPI backend for Arabic Document Classification using .joblib pipelines."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import re
import time
from typing import Any, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODELS_DIR = Path("/Users/ahmedyassine/Desktop/NLP_Project/models")
PREFERRED_DEFAULTS = [
    "StackingEnsemble_sparse_fusion_none_light_normalized.joblib",
    "MultinomialNB_tfidf_light_normalized.joblib",
]

# Standard 7-class mapping from model config['dataset']['categories']
DEFAULT_CATEGORIES = [
    "Culture",
    "Finance",
    "Medical",
    "Politics",
    "Religion",
    "Sports",
    "Tech",
]

CATEGORY_LABELS_AR = {
    "Culture": "ثقافة",
    "Finance": "اقتصاد",
    "Medical": "صحة",
    "Politics": "سياسة",
    "Religion": "دين",
    "Sports": "رياضة",
    "Tech": "تكنولوجيا",
}

# Standard Arabic Stopwords for light_normalized
ARABIC_STOPWORDS = {
    "في", "من", "إلى", "الى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "الذي", "التي", "الذين", "اللواتي", "وهو", "وهي", "هو", "هي", "هم", "هن",
    "كان", "كانت", "يكون", "تكون", "أن", "ان", "إن", "ان", "ما", "لا", "لم",
    "لن", "قد", "كل", "بعض", "أو", "او", "ثم", "بل", "حتى", "بين", "بعد",
    "قبل", "عند", "منذ", "حيث", "كيف", "متى", "أين", "اين", "لماذا", "كم",
    "نحن", "أنا", "انا", "أنت", "انت", "أنتم", "انتم", "له", "لها", "لهم",
    "به", "بها", "بهم", "فيها", "فيه", "عليه", "عليها", "منه", "منها", "وقد",
    "فقد", "كما", "أي", "اي", "غير", "دون", "حول", "خلال", "ضد", "نحو", "لدى",
}

# Compiled regex patterns for Arabic light_normalized preprocessing
TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670]")
TATWEEL_RE = re.compile(r"\u0640+")
ALEF_VARIANTS_RE = re.compile(r"[إأآٱ]")
NON_ARABIC_PUNCT_RE = re.compile(r"[^\u0600-\u06FF\s]")
WHITESPACE_RE = re.compile(r"\s+")

# In-memory model registry: { filename: {"pipeline": ..., "meta": ...} }
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
DEFAULT_MODEL_NAME: Optional[str] = None


def light_normalized(
    text: str,
    remove_tashkeel: bool = True,
    normalize_alef: bool = True,
    remove_stopwords: bool = True,
) -> str:
    """Applies the Arabic `light_normalized` preprocessing pipeline."""
    if not text:
        return ""

    # 1. Remove Tatweel (kashida)
    cleaned = TATWEEL_RE.sub("", text)

    # 2. Strip Tashkeel / diacritics
    if remove_tashkeel:
        cleaned = TASHKEEL_RE.sub("", cleaned)

    # 3. Normalize Alef variants (أ, إ, آ, ٱ -> ا) and Ya / Alef Maqsura (ى -> ي)
    if normalize_alef:
        cleaned = ALEF_VARIANTS_RE.sub("ا", cleaned)
        cleaned = cleaned.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")

    # 4. Remove non-Arabic punctuation/digits and collapse whitespace
    cleaned = NON_ARABIC_PUNCT_RE.sub(" ", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    # 5. Remove Arabic stopwords
    if remove_stopwords:
        tokens = []
        for tok in cleaned.split():
            norm_tok = ALEF_VARIANTS_RE.sub("ا", tok)
            if tok not in ARABIC_STOPWORDS and norm_tok not in ARABIC_STOPWORDS:
                tokens.append(tok)
        cleaned = " ".join(tokens)

    return cleaned or text.strip()


def format_display_name(filename: str, meta: dict[str, Any]) -> str:
    """Generates a clean human-readable display name for a .joblib model."""
    stem = Path(filename).stem
    model_name = meta.get("model_name") or stem.split("_")[0]
    rep = meta.get("representation") or (
        "Sparse Fusion (Word + Char TF-IDF)"
        if "sparse_fusion" in stem
        else "TF-IDF"
    )
    f1 = meta.get("best_macro_f1")
    if isinstance(f1, (int, float)):
        return f"{model_name} ({rep}) — F1: {f1 * 100:.1f}%"
    return f"{model_name} ({rep})"


def extract_pipeline_and_meta(raw_obj: Any, filename: str) -> dict[str, Any]:
    """Unpacks a .joblib artifact whether it is a dict bundle or a sklearn Pipeline."""
    if isinstance(raw_obj, dict):
        pipeline = (
            raw_obj.get("pipeline")
            or raw_obj.get("model")
            or raw_obj.get("estimator")
        )
        config = raw_obj.get("config", {})
        categories = (
            config.get("dataset", {}).get("categories") or DEFAULT_CATEGORIES
        )
        meta = {
            "model_name": raw_obj.get("model_name"),
            "representation": raw_obj.get("representation"),
            "preprocessing": raw_obj.get("preprocessing", "light_normalized"),
            "best_macro_f1": raw_obj.get("best_macro_f1"),
            "categories": categories,
        }
        return {"pipeline": pipeline, "raw": raw_obj, "meta": meta}

    return {
        "pipeline": raw_obj,
        "raw": raw_obj,
        "meta": {
            "model_name": Path(filename).stem.split("_")[0],
            "representation": "tfidf",
            "preprocessing": "light_normalized",
            "best_macro_f1": None,
            "categories": DEFAULT_CATEGORIES,
        },
    }


def load_all_models() -> None:
    """Loads all .joblib models from MODELS_DIR into memory on startup."""
    global DEFAULT_MODEL_NAME
    MODEL_REGISTRY.clear()

    if not MODELS_DIR.exists():
        raise RuntimeError(f"Models directory does not exist: {MODELS_DIR}")

    joblib_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not joblib_files:
        raise RuntimeError(f"No .joblib files found in {MODELS_DIR}")

    for path in joblib_files:
        raw_obj = joblib.load(path)
        entry = extract_pipeline_and_meta(raw_obj, path.name)
        if entry["pipeline"] is None:
            raise RuntimeError(f"Could not find estimator/pipeline in {path.name}")
        MODEL_REGISTRY[path.name] = entry

    # Set default model according to priority list
    DEFAULT_MODEL_NAME = next(
        (name for name in PREFERRED_DEFAULTS if name in MODEL_REGISTRY),
        next(iter(MODEL_REGISTRY.keys())),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield
    MODEL_REGISTRY.clear()


app = FastAPI(
    title="Arabic Document Classification API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Arabic text to classify")
    model_name: Optional[str] = Field(
        default=None, description="Filename of the .joblib model to use"
    )
    options: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional preprocessing flags (remove_tashkeel, normalize_alef, remove_stopwords)",
    )


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable Softmax for converting decision_function margins to probabilities."""
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def map_class_label(raw_label: Any, categories: list[str]) -> str:
    """Maps integer or English category names to Arabic display labels (with English key)."""
    eng_label = str(raw_label)
    if isinstance(raw_label, (int, np.integer)) and 0 <= int(raw_label) < len(categories):
        eng_label = categories[int(raw_label)]
    elif eng_label.isdigit() and 0 <= int(eng_label) < len(categories):
        eng_label = categories[int(eng_label)]

    ar_label = CATEGORY_LABELS_AR.get(eng_label)
    return f"{ar_label} ({eng_label})" if ar_label else eng_label


def extract_top_keywords(pipeline: Any, processed_text: str, top_k: int = 5) -> list[str]:
    """Extracts the top K most influential word-level Arabic TF-IDF tokens in the input text."""
    vectorizer = None
    if hasattr(pipeline, "named_steps"):
        vectorizer = pipeline.named_steps.get("vectorizer") or pipeline.named_steps.get("tfidf")
        if vectorizer is None and len(pipeline.steps) > 0:
            vectorizer = pipeline.steps[0][1]

    word_vectorizer = None
    if vectorizer is not None:
        # Handle FeatureUnion (e.g., word_tfidf + char_tfidf in StackingEnsemble)
        if hasattr(vectorizer, "transformer_list"):
            for name, trans in vectorizer.transformer_list:
                if "word" in name or getattr(trans, "analyzer", None) == "word":
                    word_vectorizer = trans
                    break
            if word_vectorizer is None and vectorizer.transformer_list:
                word_vectorizer = vectorizer.transformer_list[0][1]
        elif hasattr(vectorizer, "transform") and hasattr(vectorizer, "get_feature_names_out"):
            word_vectorizer = vectorizer

    if word_vectorizer is not None:
        try:
            tfidf_vec = word_vectorizer.transform([processed_text])
            feature_names = word_vectorizer.get_feature_names_out()
            row = tfidf_vec.tocoo()
            scored_tokens = [
                (str(feature_names[idx]).strip(), float(val))
                for idx, val in zip(row.col, row.data)
                if len(str(feature_names[idx]).strip()) > 2
            ]
            scored_tokens.sort(key=lambda item: item[1], reverse=True)
            if scored_tokens:
                return [tok for tok, _ in scored_tokens[:top_k]]
        except Exception:
            pass

    # Fallback: top unique Arabic tokens from the processed text
    seen: list[str] = []
    for token in processed_text.split():
        if len(token) > 2 and token not in seen and token not in ARABIC_STOPWORDS:
            seen.append(token)
        if len(seen) >= top_k:
            break
    return seen


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL_NAME,
        "loaded_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models")
def list_models() -> dict[str, Any]:
    models_info = []
    for filename, entry in MODEL_REGISTRY.items():
        meta = entry["meta"]
        models_info.append(
            {
                "id": filename,
                "name": filename,
                "display_name": format_display_name(filename, meta),
                "model_type": meta.get("model_name"),
                "representation": meta.get("representation"),
                "preprocessing": meta.get("preprocessing"),
                "best_macro_f1": meta.get("best_macro_f1"),
                "is_default": filename == DEFAULT_MODEL_NAME,
            }
        )
    return {
        "default_model": DEFAULT_MODEL_NAME,
        "models": models_info,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not MODEL_REGISTRY:
        raise HTTPException(status_code=503, detail="No models are loaded in memory.")

    selected_model = request.model_name or DEFAULT_MODEL_NAME
    if selected_model not in MODEL_REGISTRY:
        # Allow matching by stem or partial name
        matched = next(
            (k for k in MODEL_REGISTRY if k.startswith(selected_model)),
            None,
        )
        if not matched:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{selected_model}' not found. Available: {list(MODEL_REGISTRY.keys())}",
            )
        selected_model = matched

    entry = MODEL_REGISTRY[selected_model]
    pipeline = entry["pipeline"]
    categories = entry["meta"].get("categories") or DEFAULT_CATEGORIES

    # Parse preprocessing options (supports both backend and frontend flag names)
    opts = request.options or {}
    remove_tashkeel = bool(opts.get("remove_tashkeel", opts.get("tashkeel", True)))
    normalize_alef = bool(opts.get("normalize_alef", opts.get("alef", True)))
    remove_stopwords = bool(opts.get("remove_stopwords", opts.get("stop", True)))

    t0 = time.perf_counter()

    processed_text = light_normalized(
        request.text,
        remove_tashkeel=remove_tashkeel,
        normalize_alef=normalize_alef,
        remove_stopwords=remove_stopwords,
    )

    try:
        classes = getattr(pipeline, "classes_", None)
        if classes is None and hasattr(pipeline, "steps"):
            classes = getattr(pipeline.steps[-1][1], "classes_", None)
        if classes is None:
            classes = np.arange(len(categories))

        if hasattr(pipeline, "predict_proba"):
            probs = np.asarray(pipeline.predict_proba([processed_text])[0], dtype=float)
        elif hasattr(pipeline, "decision_function"):
            margins = np.asarray(pipeline.decision_function([processed_text])[0], dtype=float)
            if margins.ndim == 0:
                probs = np.array([1.0 - 1.0 / (1.0 + np.exp(-margins)), 1.0 / (1.0 + np.exp(-margins))])
            else:
                probs = stable_softmax(margins)
        else:
            pred_raw = pipeline.predict([processed_text])[0]
            probs = np.array([1.0 if c == pred_raw else 0.0 for c in classes], dtype=float)

        top_indices = np.argsort(probs)[::-1]
        best_idx = int(top_indices[0])
        primary_category = map_class_label(classes[best_idx], categories)
        confidence = round(float(probs[best_idx]) * 100.0, 2)

        top_probabilities = [
            {
                "category": map_class_label(classes[int(idx)], categories),
                "score": round(float(probs[int(idx)]) * 100.0, 2),
            }
            for idx in top_indices[:4]
        ]

        keywords = extract_top_keywords(pipeline, processed_text, top_k=5)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction error with {selected_model}: {exc}"
        ) from exc

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "category": primary_category,
        "confidence": confidence,
        "probabilities": top_probabilities,
        "keywords": keywords,
        "latency_ms": latency_ms,
        "model_used": selected_model,
    }
