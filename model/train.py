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
MODEL_PATH = Path("models/nyc_taxi.model")


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
    kmeans = construire_kmeans(train)

    print("Target encoding paires de clusters …")
    paire_stats, mediane_globale = calculer_paire_stats(train, kmeans)

    y_train = np.log1p(train["trip_duration"].values)
    y_val   = val["trip_duration"].values

    print("Calcul des features …")
    X_train = preparer_dataframe(train, kmeans, paire_stats, mediane_globale).values
    X_val   = preparer_dataframe(val,   kmeans, paire_stats, mediane_globale).values

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
