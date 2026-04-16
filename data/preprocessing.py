"""
Pipeline de preprocessing partagé entre entraînement et inférence.

INFÉRENCE (train + val + test + api) :
    preparer_inference(req, kmeans, paire_stats, mediane_globale)
    → pd.DataFrame d'une ligne prêt pour model.predict()

ENTRAÎNEMENT uniquement (nécessite trip_duration) :
    filtre_outliers(df)
    construire_kmeans(df)
    calculer_paire_stats(df, kmeans)

UTILITAIRE :
    rmsle(y_true, y_pred)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit
from sklearn.cluster import MiniBatchKMeans

from config import CFG

# ── Constantes ────────────────────────────────────────────────────────────────

NYC_LON      = (CFG.geo.lon_min, CFG.geo.lon_max)
NYC_LAT      = (CFG.geo.lat_min, CFG.geo.lat_max)
KM_PAR_DEGRE = 111.0
N_CLUSTERS   = CFG.clustering.n_clusters

FEATURES = [
    "dist_haversine_km", "bearing_sin", "bearing_cos", "dist_manhattan_km",
    "heure", "jour_semaine", "mois", "jour_annee",
    "is_rush_hour", "is_weekend", "is_nuit",
    "cluster_depart", "cluster_arrivee",
    "duree_mediane_paire",
]

# Jours 2016 avec trafic anormalement bas (Thanksgiving + Noël)
JOURS_ANORMAUX = {
    "2016-11-24", "2016-11-25",  # Thanksgiving
    "2016-12-25", "2016-12-26",  # Noël
}


# ── Métrique ──────────────────────────────────────────────────────────────────

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# ── INFÉRENCE — fonction principale ──────────────────────────────────────────

def preparer_inference(
    req,
    kmeans: MiniBatchKMeans,
    paire_stats: pd.DataFrame,
    mediane_globale: float,
) -> pd.DataFrame:
    """
    Construit un DataFrame d'une ligne avec toutes les features, à partir d'un
    objet PredictInput (ou tout objet avec pickup_lat, pickup_lon, dropoff_lat,
    dropoff_lon, pickup_datetime). Utilisable à l'inférence (API, test unitaire).

    kmeans, paire_stats et mediane_globale proviennent de l'artefact modèle.
    Retourne un DataFrame avec les colonnes FEATURES dans le bon ordre.
    """
    df = pd.DataFrame([{
        "pickup_latitude":   req.pickup_lat,
        "pickup_longitude":  req.pickup_lon,
        "dropoff_latitude":  req.dropoff_lat,
        "dropoff_longitude": req.dropoff_lon,
        "pickup_datetime":   req.pickup_datetime,
    }])
    return _ajouter_features(df, kmeans, paire_stats, mediane_globale)


def preparer_dataframe(
    df: pd.DataFrame,
    kmeans: MiniBatchKMeans,
    paire_stats: pd.DataFrame,
    mediane_globale: float,
) -> pd.DataFrame:
    """
    Applique le pipeline de features à un DataFrame entier (val, test, batch).
    Même logique que preparer_inference, sur plusieurs lignes.
    """
    return _ajouter_features(df.copy(), kmeans, paire_stats, mediane_globale)


# ── Feature engineering (logique interne) ────────────────────────────────────

def _ajouter_features(
    df: pd.DataFrame,
    kmeans: MiniBatchKMeans,
    paire_stats: pd.DataFrame,
    mediane_globale: float,
) -> pd.DataFrame:
    df = df.copy()

    # Distances
    dep = list(zip(df["pickup_latitude"],  df["pickup_longitude"]))
    arr = list(zip(df["dropoff_latitude"], df["dropoff_longitude"]))
    df["dist_haversine_km"] = haversine_vector(dep, arr, unit=Unit.KILOMETERS)

    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    df["bearing_sin"]       = np.sin(np.radians(bearing))
    df["bearing_cos"]       = np.cos(np.radians(bearing))
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
    df["is_nuit"]      = (
        dt.dt.hour.between(CFG.temporal.nuit_start, 23)
        | dt.dt.hour.between(0, CFG.temporal.nuit_end)
    ).astype(int)
    df["is_rush_hour"] = (
        (~df["is_weekend"].astype(bool))
        & df["heure"].between(CFG.temporal.rush_hour_start, CFG.temporal.rush_hour_end)
    ).astype(int)

    # Clusters & target encoding
    df["cluster_depart"]  = kmeans.predict(df[["pickup_latitude",  "pickup_longitude"]].values)
    df["cluster_arrivee"] = kmeans.predict(df[["dropoff_latitude", "dropoff_longitude"]].values)
    df = df.merge(paire_stats, on=["cluster_depart", "cluster_arrivee"], how="left")
    df["duree_mediane_paire"] = df["duree_mediane_paire"].fillna(mediane_globale)

    return df[FEATURES]


# ── ENTRAÎNEMENT uniquement ───────────────────────────────────────────────────

def filtre_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les trajets aberrants. Requiert la colonne trip_duration."""
    m = (
        df["trip_duration"].between(CFG.outliers.duration_min_sec, CFG.outliers.duration_max_sec)
        & df["pickup_longitude"].between(*NYC_LON)
        & df["pickup_latitude"].between(*NYC_LAT)
        & df["dropoff_longitude"].between(*NYC_LON)
        & df["dropoff_latitude"].between(*NYC_LAT)
    )
    return df[m].copy()


def construire_kmeans(df: pd.DataFrame) -> MiniBatchKMeans:
    """Entraîne le KMeans géographique sur les pickups. Train uniquement."""
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=42, n_init=3, batch_size=10_000
    )
    kmeans.fit(df[["pickup_latitude", "pickup_longitude"]].values)
    return kmeans


def calculer_paire_stats(
    df: pd.DataFrame,
    kmeans: MiniBatchKMeans,
) -> tuple[pd.DataFrame, float]:
    """
    Calcule la durée médiane par paire de clusters. Train uniquement.
    Requiert trip_duration. Retourne (paire_stats, mediane_globale).
    """
    tmp = df.copy()
    tmp["cluster_depart"]  = kmeans.predict(tmp[["pickup_latitude",  "pickup_longitude"]].values)
    tmp["cluster_arrivee"] = kmeans.predict(tmp[["dropoff_latitude", "dropoff_longitude"]].values)
    paire_stats = (
        tmp.groupby(["cluster_depart", "cluster_arrivee"])["trip_duration"]
        .median()
        .rename("duree_mediane_paire")
        .reset_index()
    )
    return paire_stats, df["trip_duration"].median()
