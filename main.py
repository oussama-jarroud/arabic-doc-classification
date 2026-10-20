"""API FastAPI pour la classification de documents arabes."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).with_name("modele_classification_arabe.pkl")
classifier: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage de l'API."""
    global classifier
    try:
        classifier = joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}") from exc
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger le modèle : {exc}") from exc
    yield
    classifier = None


app = FastAPI(
    title="Classification de documents arabes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte arabe à classifier")


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Retourne la catégorie prédite et un score de confiance simulé."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")

    try:
        category = classifier.predict([request.text])[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant la prédiction : {exc}"
        ) from exc

    return {
        "category": str(category),
        # Score fixe et simulé : le modèle n'expose pas nécessairement predict_proba.
        "confidence": 0.95,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
