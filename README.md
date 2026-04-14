# NYC Taxi Trip Duration

Compétition Kaggle : prédiction de la durée d'un trajet en taxi à New York (en secondes) à partir des coordonnées GPS, de l'heure et de quelques métadonnées.

**Modèle** : LightGBM — métrique : **RMSLE** (log1p des secondes)

---

## Prérequis

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda
- Données Kaggle dans `data/raw/` : `train.csv`, `test.csv`, `sample_submission.csv`

---

## Installation

```bash
conda env create -f environment.yml
conda activate nyc-taxi
```

---

## Utilisation étape par étape

### 1. Charger les données brutes dans SQLite

```bash
python -m data.download_data
```

Lit les CSV dans `data/raw/` et crée `data/processed/nyc_taxi.db` avec les tables `train` et `test`.

---

### 2. Exploration et feature engineering (notebook)

```bash
jupyter notebook
```

Ouvrir `notebooks/nyc_taxi_analysis.ipynb` avec le kernel **Python (nyc-taxi)**.

Le notebook fait :
- EDA sur les données brutes
- Split 80/20 stratifié sur le mois → tables `train_split` et `val_split` dans SQLite
- Feature engineering et baseline

---

### 3. Entraîner le modèle

```bash
python -m model.train
```

Charge `train_split`, applique le pipeline de features, entraîne LightGBM avec early stopping sur `val_split`.  
Produit : `models/nyc_taxi.model`

---

### 4. (Optionnel) Tuner les hyperparamètres

```bash
python -m model.tune --trials 50
```

Recherche Optuna sur 50 trials. Produit : `models/nyc_taxi_tuned.model`

---

### 5. Tester l'inférence

```bash
python -m model.test_model
```

Charge le modèle, applique les features au jeu `test`, affiche des prédictions sur un échantillon aléatoire.

---

### 6. Lancer l'application web

```bash
streamlit run api/app.py
```

Interface Streamlit : saisir les coordonnées de départ/arrivée, la date et l'heure → prédiction instantanée + carte du trajet.

---

## Structure du projet

```
data/
  raw/                  # CSVs Kaggle originaux (ne pas modifier)
  processed/            # nyc_taxi.db (SQLite)
  download_data.py      # Étape 1 : chargement des CSV
model/
  train.py              # Étape 3 : entraînement
  test_model.py         # Étape 5 : test d'inférence
  tune.py               # Étape 4 : tuning Optuna
models/
  nyc_taxi.model        # Artefact sérialisé
notebooks/
  nyc_taxi_analysis.ipynb
api/
  app.py                # Application Streamlit
environment.yml
```

---

## Features utilisées

| Feature | Description |
|---|---|
| `dist_haversine_km` | Distance à vol d'oiseau (km) |
| `dist_manhattan_km` | Distance Manhattan (km) |
| `bearing_sin/cos` | Direction du trajet (encodage circulaire) |
| `heure`, `jour_semaine`, `mois`, `jour_annee` | Temporel |
| `is_rush_hour`, `is_weekend`, `is_nuit` | Indicateurs temporels |
| `vendor_id`, `passenger_count` | Métadonnées |
| `cluster_depart`, `cluster_arrivee` | Clusters géographiques KMeans (20) |
| `duree_mediane_paire` | Target encoding : médiane par paire de clusters |
