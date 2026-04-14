"""
API REST — Prédiction de durée de trajet taxi NYC.

Pattern : Synchronous (Web Single)
Ref : https://github.com/mercari/ml-system-design-pattern/tree/master/Serving-patterns

Lancement :
    conda activate nyc-taxi
    uvicorn api.server:app --reload

Endpoints :
    GET  /health          — statut du service et version du modèle
    POST /predict         — prédiction pour un trajet
    GET  /docs            — documentation interactive (Swagger UI)
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from data.preprocessing import preparer_inference

# ── Chargement du modèle ──────────────────────────────────────────────────────

MODEL_PATH = Path("models/nyc_taxi.model")

def _charger_modele():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}. Lancez d'abord python -m model.train")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artefact        = _charger_modele()
modele          = artefact["modele"]
kmeans          = artefact["kmeans"]
paire_stats     = artefact["paire_stats"]
mediane_globale = artefact["mediane_globale"]
model_features  = artefact["features"]

# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Taxi Trip Duration",
    description="Prédit la durée d'un trajet taxi new-yorkais à partir des coordonnées et de l'heure.",
    version="1.0.0",
)

# ── Schémas ───────────────────────────────────────────────────────────────────

NYC_LON = (-74.3, -73.6)
NYC_LAT = (40.4,  41.0)

class PredictRequest(BaseModel):
    pickup_lat:      float = Field(..., ge=NYC_LAT[0], le=NYC_LAT[1], example=40.7580)
    pickup_lon:      float = Field(..., ge=NYC_LON[0], le=NYC_LON[1], example=-73.9855)
    dropoff_lat:     float = Field(..., ge=NYC_LAT[0], le=NYC_LAT[1], example=40.6413)
    dropoff_lon:     float = Field(..., ge=NYC_LON[0], le=NYC_LON[1], example=-73.7781)
    pickup_datetime: datetime = Field(..., example="2016-06-15T17:30:00")

    @model_validator(mode="after")
    def coords_differentes(self):
        if self.pickup_lat == self.dropoff_lat and self.pickup_lon == self.dropoff_lon:
            raise ValueError("Les coordonnées de départ et d'arrivée sont identiques.")
        return self


class PredictResponse(BaseModel):
    trip_duration_sec: float
    trip_duration_min: float
    distance_km:       float


class HealthResponse(BaseModel):
    status:   str
    features: list[str]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", features=model_features)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = preparer_inference(
            req.pickup_lat, req.pickup_lon,
            req.dropoff_lat, req.dropoff_lon,
            req.pickup_datetime,
            kmeans, paire_stats, mediane_globale,
        )
        duree_sec = float(np.expm1(modele.predict(X.values)[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from haversine import haversine, Unit
    dist_km = haversine(
        (req.pickup_lat, req.pickup_lon),
        (req.dropoff_lat, req.dropoff_lon),
        unit=Unit.KILOMETERS,
    )

    return PredictResponse(
        trip_duration_sec=round(duree_sec, 1),
        trip_duration_min=round(duree_sec / 60, 2),
        distance_km=round(dist_km, 3),
    )
