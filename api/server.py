"""
API REST — Prédiction de durée de trajet taxi NYC.

Pattern : Synchronous (Web Single)
Ref : https://github.com/mercari/ml-system-design-pattern/tree/master/Serving-patterns

Lancement :
    python -m api.main

Endpoints :
    GET  /health              — statut du service
    GET  /models              — modèles disponibles avec metadata
    POST /predict             — prédiction single  (?model=nyc_taxi)
    POST /predict/batch       — prédiction batch   (?model=nyc_taxi)
    GET  /docs                — documentation interactive (Swagger UI)
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query

from api.logger import logger_prediction
from api.registry import DEFAULT_MODEL, ModelRegistry
from data.postprocessing import postprocesser
from data.preprocessing import preparer_inference
from data.schema import (
    BatchPredictInput,
    BatchPredictOutput,
    ModelInfo,
    PredictInput,
    PredictResponse,
)

# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Taxi Trip Duration",
    description="Prédit la durée d'un trajet taxi new-yorkais à partir des coordonnées et de l'heure.",
    version="1.0.0",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _predire(req: PredictInput, version: str) -> PredictResponse:
    """Exécute le pipeline complet pour un seul trajet."""
    try:
        artefact, info = ModelRegistry.get(version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        X     = preparer_inference(
            req,
            artefact["kmeans"],
            artefact["paire_stats"],
            artefact["mediane_globale"],
        )
        y_log  = artefact["modele"].predict(X.values)[0]
        output = postprocesser(y_log, req.pickup_lat, req.pickup_lon,
                               req.dropoff_lat, req.dropoff_lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        **output.model_dump(),
        model_version=info.version,
        predicted_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_disponibles": ModelRegistry.versions()}


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    """Liste les modèles disponibles avec leur metadata."""
    result = []
    for version in ModelRegistry.versions():
        _, info = ModelRegistry.get(version)
        result.append(info)
    return result


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictInput,
    model: str = Query(default=DEFAULT_MODEL, description="Version du modèle à utiliser"),
):
    """Prédit la durée d'un trajet unique."""
    response = _predire(req, model)
    logger_prediction(req, response)
    return response


@app.post("/predict/batch", response_model=BatchPredictOutput)
def predict_batch(
    req: BatchPredictInput,
    model: str = Query(default=DEFAULT_MODEL, description="Version du modèle à utiliser"),
):
    """Prédit la durée pour une liste de trajets (max 500)."""
    predicted_at = datetime.now(timezone.utc).isoformat()
    predictions  = []

    for item in req.items:
        response = _predire(item, model)
        logger_prediction(item, response)
        predictions.append(response)

    return BatchPredictOutput(
        predictions=predictions,
        model_version=model,
        predicted_at=predicted_at,
        count=len(predictions),
    )
