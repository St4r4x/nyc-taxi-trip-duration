"""
Tuning des hyperparamètres LightGBM avec Optuna.

Usage :
    python -m model.tune [--trials N]   (défaut : 50)

Produit :
    models/nyc_taxi_tuned.model  — meilleur modèle trouvé
"""

import argparse
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from haversine import haversine_vector, Unit
from sklearn.cluster import MiniBatchKMeans

DB_PATH    = Path("data/processed/nyc_taxi.db")
MODEL_PATH = Path("models/nyc_taxi_tuned.model")
KM_PAR_DEGRE = 111.0
NYC_LON = (-74.3, -73.6)
NYC_LAT = (40.4,  41.0)

FEATURES = [
    "dist_haversine_km", "bearing_sin", "bearing_cos", "dist_manhattan_km",
    "heure", "jour_semaine", "mois", "jour_annee",
    "is_rush_hour", "is_weekend", "is_nuit",
    "vendor_id", "passenger_count",
    "cluster_depart", "cluster_arrivee",
    "duree_mediane_paire",
]


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def filtre_outliers(df):
    m = (
        df["trip_duration"].between(60, 7200)
        & df["pickup_longitude"].between(*NYC_LON)
        & df["pickup_latitude"].between(*NYC_LAT)
        & df["dropoff_longitude"].between(*NYC_LON)
        & df["dropoff_latitude"].between(*NYC_LAT)
    )
    return df[m].copy()


def ajouter_features(df, kmeans, paire_stats, mediane_globale):
    df = df.copy()
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
    df = df.merge(paire_stats, on=["cluster_depart", "cluster_arrivee"], how="left")
    df["duree_mediane_paire"] = df["duree_mediane_paire"].fillna(mediane_globale)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    print(f"=== Tuning LightGBM — {args.trials} trials ===\n")

    # ── Chargement & préparation ─────────────────────────────────────────────
    print("Chargement des données...")
    con = sqlite3.connect(DB_PATH)
    train = pd.read_sql("SELECT * FROM train_split", con, parse_dates=["pickup_datetime"])
    val   = pd.read_sql("SELECT * FROM val_split",   con, parse_dates=["pickup_datetime"])
    con.close()
    print(f"  train brut : {len(train):,} lignes | val : {len(val):,} lignes")

    train = filtre_outliers(train)
    print(f"  train filtré : {len(train):,} lignes")

    print("Entraînement KMeans (20 clusters)...")
    kmeans = MiniBatchKMeans(n_clusters=20, random_state=42, n_init=3, batch_size=10_000)
    kmeans.fit(train[["pickup_latitude", "pickup_longitude"]].values)

    # Clusters nécessaires avant le target encoding
    train["cluster_depart"]  = kmeans.predict(train[["pickup_latitude",  "pickup_longitude"]].values)
    train["cluster_arrivee"] = kmeans.predict(train[["dropoff_latitude", "dropoff_longitude"]].values)

    paire_stats = (
        train.groupby(["cluster_depart", "cluster_arrivee"])["trip_duration"]
        .median().rename("duree_mediane_paire").reset_index()
    )
    mediane_globale = train["trip_duration"].median()

    print("Feature engineering...")
    train = ajouter_features(train, kmeans, paire_stats, mediane_globale)
    val   = ajouter_features(val,   kmeans, paire_stats, mediane_globale)
    print(f"  {len(FEATURES)} features : {FEATURES}\n")

    X_train = train[FEATURES].values
    y_train = np.log1p(train["trip_duration"].values)
    X_val   = val[FEATURES].values
    y_val   = val["trip_duration"].values

    # ── Objectif Optuna ──────────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":      2000,
            "learning_rate":     trial.suggest_float("learning_rate",     0.01,  0.1,  log=True),
            "num_leaves":        trial.suggest_int("num_leaves",          31,    255),
            "min_child_samples": trial.suggest_int("min_child_samples",   10,    100),
            "subsample":         trial.suggest_float("subsample",         0.5,   1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.5,   1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-4,  10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-4,  10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, np.log1p(y_val))],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return rmsle(y_val, np.expm1(model.predict(X_val)))

    def trial_callback(study, trial):
        is_best = trial.value == study.best_value
        marker = " *" if is_best else ""
        print(
            f"  Trial {trial.number + 1:>3}/{args.trials}"
            f"  RMSLE={trial.value:.4f}"
            f"  meilleur={study.best_value:.4f}"
            f"{marker}"
        )

    # ── Lancement de l'étude ─────────────────────────────────────────────────
    print(f"Démarrage de l'optimisation ({args.trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, callbacks=[trial_callback])

    print(f"\nMeilleur RMSLE : {study.best_value:.4f}")
    print(f"Meilleurs params : {study.best_params}")

    # ── Réentraînement avec les meilleurs params ─────────────────────────────
    best = study.best_params
    best["n_estimators"] = 2000
    best["random_state"] = 42
    best["verbose"]      = -1

    modele_final = lgb.LGBMRegressor(**best)
    modele_final.fit(
        X_train, y_train,
        eval_set=[(X_val, np.log1p(y_val))],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=200)],
    )
    score_final = rmsle(y_val, np.expm1(modele_final.predict(X_val)))
    print(f"RMSLE final : {score_final:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artefact = {
        "modele": modele_final, "kmeans": kmeans, "features": FEATURES,
        "paire_stats": paire_stats, "mediane_globale": mediane_globale,
        "best_params": study.best_params,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefact, f)
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
