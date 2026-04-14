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

from data.preprocessing import (
    FEATURES,
    calculer_paire_stats,
    construire_kmeans,
    filtre_outliers,
    preparer_dataframe,
    rmsle,
)

DB_PATH    = Path("data/processed/nyc_taxi.db")
MODEL_PATH = Path("models/nyc_taxi_tuned.model")


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
    kmeans = construire_kmeans(train)

    print("Target encoding paires de clusters...")
    paire_stats, mediane_globale = calculer_paire_stats(train, kmeans)

    y_train = np.log1p(train["trip_duration"].values)
    y_val   = val["trip_duration"].values

    print("Feature engineering...")
    print(f"  {len(FEATURES)} features : {FEATURES}\n")
    X_train = preparer_dataframe(train, kmeans, paire_stats, mediane_globale).values
    X_val   = preparer_dataframe(val,   kmeans, paire_stats, mediane_globale).values

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
