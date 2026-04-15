"""
Schémas de données du projet NYC Taxi Trip Duration.

TripRaw          — enregistrement brut tel que présent dans train.csv / test.csv
PredictInput     — données minimales nécessaires à l'inférence
PredictOutput    — sortie brute du postprocessing (sans metadata)
PredictResponse  — réponse API complète (sortie + metadata d'inférence)
ModelInfo        — metadata du modèle (version, path, date de création, features)
BatchPredictInput / BatchPredictOutput — inférence en lot
"""

from datetime import datetime
from typing import Literal

from haversine import haversine, Unit
from pydantic import BaseModel, Field, model_validator

from config import CFG

# ── Bornes géographiques NYC ──────────────────────────────────────────────────

LON_MIN, LON_MAX = CFG.geo.lon_min, CFG.geo.lon_max
LAT_MIN, LAT_MAX = CFG.geo.lat_min, CFG.geo.lat_max
MIN_DISTANCE_M   = CFG.api.min_distance_m


# ── Données brutes (CSV) ──────────────────────────────────────────────────────

class TripRaw(BaseModel):
    """
    Enregistrement brut issu de train.csv ou test.csv.
    Les champs dropoff_datetime et trip_duration sont absents du jeu test.
    """
    id:                 str
    vendor_id:          Literal[1, 2]
    pickup_datetime:    datetime
    passenger_count:    int   = Field(ge=1, le=6)
    pickup_longitude:   float = Field(ge=LON_MIN, le=LON_MAX)
    pickup_latitude:    float = Field(ge=LAT_MIN, le=LAT_MAX)
    dropoff_longitude:  float = Field(ge=LON_MIN, le=LON_MAX)
    dropoff_latitude:   float = Field(ge=LAT_MIN, le=LAT_MAX)
    store_and_fwd_flag: Literal["Y", "N"]

    dropoff_datetime:   datetime | None = None
    trip_duration:      int | None      = Field(default=None, ge=1)


# ── Inférence — entrée ────────────────────────────────────────────────────────

class PredictInput(BaseModel):
    """
    Données minimales requises pour obtenir une prédiction.

    Validations :
    - Coordonnées dans la bounding box NYC
    - Distance pickup→dropoff ≥ 50 m
    """
    pickup_lat:      float    = Field(..., ge=LAT_MIN, le=LAT_MAX, examples=[40.7580])
    pickup_lon:      float    = Field(..., ge=LON_MIN, le=LON_MAX, examples=[-73.9855])
    dropoff_lat:     float    = Field(..., ge=LAT_MIN, le=LAT_MAX, examples=[40.6413])
    dropoff_lon:     float    = Field(..., ge=LON_MIN, le=LON_MAX, examples=[-73.7781])
    pickup_datetime: datetime = Field(..., examples=["2016-06-15T17:30:00"])

    @model_validator(mode="after")
    def valider_distance_minimale(self):
        dist_m = haversine(
            (self.pickup_lat, self.pickup_lon),
            (self.dropoff_lat, self.dropoff_lon),
            unit=Unit.METERS,
        )
        if dist_m < MIN_DISTANCE_M:
            raise ValueError(
                f"Distance pickup→dropoff trop faible ({dist_m:.0f} m). "
                f"Minimum requis : {MIN_DISTANCE_M:.0f} m."
            )
        return self


# ── Inférence — sortie ────────────────────────────────────────────────────────

class PredictOutput(BaseModel):
    """Sortie brute du postprocessing (sans metadata d'inférence)."""
    trip_duration_sec: float = Field(description="Durée estimée en secondes")
    trip_duration_min: float = Field(description="Durée estimée en minutes")
    distance_km:       float = Field(description="Distance à vol d'oiseau en kilomètres")


class PredictResponse(PredictOutput):
    """Réponse API complète : sortie du modèle + metadata d'inférence."""
    model_version: str = Field(description="Version du modèle utilisé")
    predicted_at:  str = Field(description="Timestamp UTC de la prédiction (ISO 8601)")


# ── Modèle — metadata ─────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Metadata d'un modèle chargé."""
    version:    str       = Field(description="Nom de version (stem du fichier .model)")
    path:       str       = Field(description="Chemin du fichier modèle")
    created_at: str       = Field(description="Date de création du fichier (UTC ISO 8601)")
    features:   list[str] = Field(description="Liste des features utilisées")
    n_features: int       = Field(description="Nombre de features")


# ── Batch ─────────────────────────────────────────────────────────────────────

class BatchPredictInput(BaseModel):
    """Liste de trajets à prédire en une seule requête."""
    items: list[PredictInput] = Field(..., min_length=1, max_length=500)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        # Times Square → JFK Airport (heure de pointe vendredi soir)
                        {
                            "pickup_lat": 40.7580, "pickup_lon": -73.9855,
                            "dropoff_lat": 40.6413, "dropoff_lon": -73.7781,
                            "pickup_datetime": "2016-06-17T17:30:00",
                        },
                        # Penn Station → Grand Central (matin semaine)
                        {
                            "pickup_lat": 40.7506, "pickup_lon": -73.9971,
                            "dropoff_lat": 40.7527, "dropoff_lon": -73.9772,
                            "pickup_datetime": "2016-06-15T08:15:00",
                        },
                        # LaGuardia Airport → Wall Street (mi-journée)
                        {
                            "pickup_lat": 40.7769, "pickup_lon": -73.8740,
                            "dropoff_lat": 40.7074, "dropoff_lon": -74.0113,
                            "pickup_datetime": "2016-06-15T12:00:00",
                        },
                        # Brooklyn Bridge → Central Park (week-end après-midi)
                        {
                            "pickup_lat": 40.7061, "pickup_lon": -73.9969,
                            "dropoff_lat": 40.7851, "dropoff_lon": -73.9683,
                            "pickup_datetime": "2016-06-18T14:45:00",
                        },
                        # Empire State Building → Statue of Liberty ferry (nuit)
                        {
                            "pickup_lat": 40.7484, "pickup_lon": -73.9857,
                            "dropoff_lat": 40.6892, "dropoff_lon": -74.0445,
                            "pickup_datetime": "2016-06-16T23:00:00",
                        },
                    ]
                }
            ]
        }
    }


class BatchPredictOutput(BaseModel):
    """Résultats d'une prédiction batch."""
    predictions:   list[PredictResponse]
    model_version: str
    predicted_at:  str
    count:         int
