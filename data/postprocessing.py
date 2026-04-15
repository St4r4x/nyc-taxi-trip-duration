"""
Postprocessing des prédictions du modèle.

Le modèle prédit log1p(trip_duration). Le postprocessing :
  1. Inverse la transformation : expm1(y_log) → secondes
  2. Calcule la distance haversine entre pickup et dropoff
  3. Construit le PredictOutput final

Fonction exportée :
    postprocesser(y_log, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
"""

import numpy as np
from haversine import haversine, Unit

from data.schema import PredictOutput


def postprocesser(
    y_log: float,
    pickup_lat: float,
    pickup_lon: float,
    dropoff_lat: float,
    dropoff_lon: float,
) -> PredictOutput:
    """
    Convertit la sortie brute du modèle en PredictOutput.

    Args:
        y_log      : prédiction du modèle (espace log1p)
        pickup_*   : coordonnées du point de départ
        dropoff_*  : coordonnées du point d'arrivée

    Returns:
        PredictOutput avec durée en secondes, minutes et distance en km
    """
    duree_sec = float(np.expm1(y_log))
    dist_km   = haversine(
        (pickup_lat, pickup_lon),
        (dropoff_lat, dropoff_lon),
        unit=Unit.KILOMETERS,
    )
    return PredictOutput(
        trip_duration_sec=round(duree_sec, 1),
        trip_duration_min=round(duree_sec / 60, 2),
        distance_km=round(dist_km, 3),
    )
