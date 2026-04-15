"""
Logging des prédictions en base SQLite.

Chaque appel à /predict est enregistré dans la table `predictions` de
data/processed/nyc_taxi.db pour permettre le suivi de la qualité du modèle
en production (data drift, distribution des durées prédites, etc.).

Fonction exportée :
    logger_prediction(req, response)
"""

import sqlite3
from pathlib import Path

from data.schema import PredictInput, PredictResponse

DB_PATH = Path("data/processed/nyc_taxi.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at          TEXT    NOT NULL,
    model_version      TEXT    NOT NULL,
    pickup_lat         REAL    NOT NULL,
    pickup_lon         REAL    NOT NULL,
    dropoff_lat        REAL    NOT NULL,
    dropoff_lon        REAL    NOT NULL,
    pickup_datetime    TEXT    NOT NULL,
    trip_duration_sec  REAL    NOT NULL,
    trip_duration_min  REAL    NOT NULL,
    distance_km        REAL    NOT NULL
)
"""

_INSERT = """
INSERT INTO predictions
    (logged_at, model_version, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
     pickup_datetime, trip_duration_sec, trip_duration_min, distance_km)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _init_db(con: sqlite3.Connection) -> None:
    con.execute(_CREATE_TABLE)
    con.commit()


def logger_prediction(req: PredictInput, response: PredictResponse) -> None:
    """Enregistre une prédiction dans la table predictions."""
    with sqlite3.connect(DB_PATH) as con:
        _init_db(con)
        con.execute(_INSERT, (
            response.predicted_at,
            response.model_version,
            req.pickup_lat,
            req.pickup_lon,
            req.dropoff_lat,
            req.dropoff_lon,
            req.pickup_datetime.isoformat(),
            response.trip_duration_sec,
            response.trip_duration_min,
            response.distance_km,
        ))
