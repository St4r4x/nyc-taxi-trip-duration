"""
Entraîne un modèle LightGBM sur le jeu train_split et le sérialise.

Usage :
    python -m model.train

Prérequis :
    data/processed/nyc_taxi.db doit contenir les tables train_split et val_split
    (générées par le notebook EDA.ipynb, section Partie II).

Produit :
    models/nyc_taxi.model  — modèle sérialisé (pickle)
"""

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit
from sklearn.cluster import MiniBatchKMeans
import lightgbm as lgb

DB_PATH    = Path("data/processed/nyc_taxi.db")
MODEL_PATH = Path("models/nyc_taxi.model")

NYC_LON      = (-74.3, -73.6)
NYC_LAT      = (40.4,  41.0)
KM_PAR_DEGRE = 111.0
FEATURES     = [
    "dist_haversine_km", "bearing_sin", "bearing_cos", "dist_manhattan_km",
    "heure", "jour_semaine", "mois", "jour_annee",
    "is_rush_hour", "is_weekend", "is_nuit",
    "vendor_id", "passenger_count",
    "cluster_depart", "cluster_arrivee",
    "duree_mediane_paire",
]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def filtre_outliers(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["trip_duration"].between(60, 7200)
        & df["pickup_longitude"].between(*NYC_LON)
        & df["pickup_latitude"].between(*NYC_LAT)
        & df["dropoff_longitude"].between(*NYC_LON)
        & df["dropoff_latitude"].between(*NYC_LAT)
    )
    return df[m].copy()


def ajouter_features(df: pd.DataFrame, kmeans: MiniBatchKMeans) -> pd.DataFrame:
    df = df.copy()
    # Distance
    dep = list(zip(df["pickup_latitude"],  df["pickup_longitude"]))
    arr = list(zip(df["dropoff_latitude"], df["dropoff_longitude"]))
    df["dist_haversine_km"] = haversine_vector(dep, arr, unit=Unit.KILOMETERS)
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing_deg = (np.degrees(np.arctan2(x, y)) + 360) % 360
    df["bearing_deg"]       = bearing_deg
    df["bearing_sin"]       = np.sin(np.radians(bearing_deg))
    df["bearing_cos"]       = np.cos(np.radians(bearing_deg))
    df["dist_manhattan_km"] = (
        np.abs(df["dropoff_latitude"]  - df["pickup_latitude"])  * KM_PAR_DEGRE
        + np.abs(df["dropoff_longitude"] - df["pickup_longitude"]) * KM_PAR_DEGRE * np.cos(lat1)
    )
    # Temporel
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
    # Clusters
    df["cluster_depart"]  = kmeans.predict(df[["pickup_latitude",  "pickup_longitude"]].values)
    df["cluster_arrivee"] = kmeans.predict(df[["dropoff_latitude", "dropoff_longitude"]].values)
    return df


def main() -> None:
    con = sqlite3.connect(DB_PATH)

    print("Chargement du train_split …")
    train = pd.read_sql("SELECT * FROM train_split", con,
                        parse_dates=["pickup_datetime"])
    print("Chargement du val_split …")
    val = pd.read_sql("SELECT * FROM val_split", con,
                      parse_dates=["pickup_datetime"])
    con.close()

    print("Nettoyage des outliers …")
    train = filtre_outliers(train)

    print("Entraînement du KMeans (20 clusters) …")
    kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=3, batch_size=10_000)
    kmeans.fit(train[["pickup_latitude", "pickup_longitude"]].values)

    print("Calcul des features …")
    train = ajouter_features(train, kmeans)
    val   = ajouter_features(val,   kmeans)

    # Target encoding : durée médiane par paire (cluster_depart, cluster_arrivee)
    # Calculé sur train uniquement, puis appliqué au val
    print("Target encoding paires de clusters …")
    paire_stats = (
        train.groupby(["cluster_depart", "cluster_arrivee"])["trip_duration"]
        .median()
        .rename("duree_mediane_paire")
        .reset_index()
    )
    mediane_globale = train["trip_duration"].median()
    train = train.merge(paire_stats, on=["cluster_depart", "cluster_arrivee"], how="left")
    val   = val.merge(paire_stats,   on=["cluster_depart", "cluster_arrivee"], how="left")
    train["duree_mediane_paire"] = train["duree_mediane_paire"].fillna(mediane_globale)
    val["duree_mediane_paire"]   = val["duree_mediane_paire"].fillna(mediane_globale)

    X_train = train[FEATURES].values
    y_train = np.log1p(train["trip_duration"].values)
    X_val   = val[FEATURES].values
    y_val   = val["trip_duration"].values

    print("Entraînement LightGBM …")
    modele = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    modele.fit(
        X_train, y_train,
        eval_set=[(X_val, np.log1p(y_val))],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=100)],
    )

    score = rmsle(y_val, np.expm1(modele.predict(X_val)))
    print(f"RMSLE validation : {score:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artefact = {"modele": modele, "kmeans": kmeans, "features": FEATURES,
                "paire_stats": paire_stats, "mediane_globale": mediane_globale}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefact, f)
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
