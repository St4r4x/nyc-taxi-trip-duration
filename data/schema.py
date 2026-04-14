"""
Schémas de données du projet NYC Taxi Trip Duration.

TripRaw       — enregistrement brut tel que présent dans train.csv / test.csv
PredictInput  — données minimales nécessaires à l'inférence (coordonnées + datetime)
PredictOutput — réponse du modèle
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ── Bornes géographiques NYC ──────────────────────────────────────────────────

LON_MIN, LON_MAX = -74.3, -73.6
LAT_MIN, LAT_MAX =  40.4,  41.0


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

    # Champs présents uniquement dans train.csv
    dropoff_datetime:   datetime | None = None
    trip_duration:      int | None      = Field(default=None, ge=1)


# ── Inférence ─────────────────────────────────────────────────────────────────

class PredictInput(BaseModel):
    """Données minimales requises pour obtenir une prédiction."""
    pickup_lat:      float    = Field(..., ge=LAT_MIN, le=LAT_MAX, examples=[40.7580])
    pickup_lon:      float    = Field(..., ge=LON_MIN, le=LON_MAX, examples=[-73.9855])
    dropoff_lat:     float    = Field(..., ge=LAT_MIN, le=LAT_MAX, examples=[40.6413])
    dropoff_lon:     float    = Field(..., ge=LON_MIN, le=LON_MAX, examples=[-73.7781])
    pickup_datetime: datetime = Field(..., examples=["2016-06-15T17:30:00"])

    @model_validator(mode="after")
    def coords_differentes(self):
        if self.pickup_lat == self.dropoff_lat and self.pickup_lon == self.dropoff_lon:
            raise ValueError("Les coordonnées de départ et d'arrivée sont identiques.")
        return self


class PredictOutput(BaseModel):
    """Réponse du modèle."""
    trip_duration_sec: float = Field(description="Durée estimée en secondes")
    trip_duration_min: float = Field(description="Durée estimée en minutes")
    distance_km:       float = Field(description="Distance à vol d'oiseau en kilomètres")
