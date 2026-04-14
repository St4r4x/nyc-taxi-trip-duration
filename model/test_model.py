"""
Charge le modèle sérialisé et effectue des prédictions sur un échantillon aléatoire du jeu de test.

Usage :
    python -m model.test_model

Prérequis :
    models/nyc_taxi.model  (généré par python -m model.train)
    data/processed/nyc_taxi.db  (table test)
"""

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit

DB_PATH    = Path("data/processed/nyc_taxi.db")
MODEL_PATH = Path("models/nyc_taxi.model")

KM_PAR_DEGRE = 111.0


def ajouter_features(df: pd.DataFrame, kmeans) -> pd.DataFrame:
    df = df.copy()
    dep = list(zip(df["pickup_latitude"],  df["pickup_longitude"]))
    arr = list(zip(df["dropoff_latitude"], df["dropoff_longitude"]))
    df["dist_haversine_km"] = haversine_vector(dep, arr, unit=Unit.KILOMETERS)
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing_deg             = (np.degrees(np.arctan2(x, y)) + 360) % 360
    df["bearing_deg"]       = bearing_deg
    df["bearing_sin"]       = np.sin(np.radians(bearing_deg))
    df["bearing_cos"]       = np.cos(np.radians(bearing_deg))
    df["dist_manhattan_km"] = (
        np.abs(df["dropoff_latitude"]  - df["pickup_latitude"])  * KM_PAR_DEGRE
        + np.abs(df["dropoff_longitude"] - df["pickup_longitude"]) * KM_PAR_DEGRE * np.cos(lat1)
    )
    dt = pd.to_datetime(df["pickup_datetime"])
    df["heure"]        = dt.dt.hour
    df["jour_semaine"] = dt.dt.dayofweek
    df["mois"]         = dt.dt.month
    df["jour_annee"]   = dt.dt.dayofyear
    df["is_weekend"]   = (df["jour_semaine"] >= 5).astype(int)
    df["is_nuit"]      = (dt.dt.hour.between(22, 23) | dt.dt.hour.between(0, 5)).astype(int)
    df["is_rush_hour"] = (
        (~df["is_weekend"].astype(bool))
        & (df["heure"].between(7, 9) | df["heure"].between(17, 20))
    ).astype(int)
    df["cluster_depart"]  = kmeans.predict(df[["pickup_latitude",  "pickup_longitude"]].values)
    df["cluster_arrivee"] = kmeans.predict(df[["dropoff_latitude", "dropoff_longitude"]].values)
    return df


def main() -> None:
    print(f"Chargement du modèle : {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        artefact = pickle.load(f)
    modele          = artefact["modele"]
    kmeans          = artefact["kmeans"]
    features        = artefact["features"]
    paire_stats     = artefact["paire_stats"]
    mediane_globale = artefact["mediane_globale"]

    con = sqlite3.connect(DB_PATH)
    sample = pd.read_sql(
        "SELECT * FROM test ORDER BY RANDOM() LIMIT 10",
        con, parse_dates=["pickup_datetime"],
    )
    con.close()

    sample = ajouter_features(sample, kmeans)
    sample = sample.merge(paire_stats, on=["cluster_depart", "cluster_arrivee"], how="left")
    sample["duree_mediane_paire"] = sample["duree_mediane_paire"].fillna(mediane_globale)
    X = sample[features].values
    preds_log = modele.predict(X)
    preds_sec = np.expm1(preds_log)

    resultats = sample[["id", "pickup_datetime",
                         "pickup_longitude", "pickup_latitude",
                         "dropoff_longitude", "dropoff_latitude"]].copy()
    resultats["trip_duration_pred_sec"] = preds_sec.round(0).astype(int)
    resultats["trip_duration_pred_min"] = (preds_sec / 60).round(1)

    print("\nPrédictions sur 10 trajets aléatoires du jeu de test :")
    print(resultats.to_string(index=False))


if __name__ == "__main__":
    main()
