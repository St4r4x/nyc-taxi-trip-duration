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

### 6. Lancer les services

**Les deux services en parallèle (recommandé) :**

```bash
python run.py
```

Lance FastAPI sur `http://localhost:8000` et Streamlit sur `http://localhost:8501`. Ctrl+C pour tout arrêter.

**FastAPI seul :**

```bash
python -m api.main
```

**Streamlit seul :**

```bash
streamlit run api/app.py
```

---

## API REST

Documentation interactive : `http://localhost:8000/docs`

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | GET | Statut du service et modèles disponibles |
| `/models` | GET | Metadata de tous les modèles disponibles |
| `/predict` | POST | Prédiction pour un trajet unique |
| `/predict/batch` | POST | Prédiction pour plusieurs trajets (max 500) |
| `/docs` | GET | Documentation interactive (Swagger UI) |

Tous les endpoints `/predict` acceptent un paramètre `?model=` pour choisir la version du modèle (ex : `nyc_taxi`, `nyc_taxi_tuned`). Par défaut : `nyc_taxi`.

### Exemple — prédiction simple

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_lat": 40.758,
    "pickup_lon": -73.9855,
    "dropoff_lat": 40.6413,
    "dropoff_lon": -73.7781,
    "pickup_datetime": "2016-06-15T17:30:00"
  }'
```

Réponse :

```json
{
  "trip_duration_sec": 5275.5,
  "trip_duration_min": 87.93,
  "distance_km": 21.773,
  "model_version": "nyc_taxi",
  "predicted_at": "2026-04-15T10:00:00+00:00"
}
```

### Exemple — prédiction batch

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"pickup_lat": 40.758, "pickup_lon": -73.9855, "dropoff_lat": 40.6413, "dropoff_lon": -73.7781, "pickup_datetime": "2016-06-15T17:30:00"},
      {"pickup_lat": 40.7506, "pickup_lon": -73.9971, "dropoff_lat": 40.7527, "dropoff_lon": -73.9772, "pickup_datetime": "2016-06-15T08:15:00"}
    ]
  }'
```

---

## Structure du projet

```
api/
  __init__.py
  main.py            # Point d'entrée : python -m api.main
  server.py          # Application FastAPI (endpoints)
  logger.py          # Logging des prédictions → table predictions (SQLite)
  registry.py        # Registre des modèles avec cache
  app.py             # Interface Streamlit
data/
  __init__.py
  download_data.py   # Étape 1 : chargement des CSV
  schema.py          # Schémas Pydantic partagés (TripRaw, PredictInput, …)
  preprocessing.py   # Pipeline partagé entraînement + inférence
  postprocessing.py  # Conversion log1p → secondes + distance haversine
  raw/               # CSVs Kaggle originaux — ne pas modifier
  processed/         # nyc_taxi.db (SQLite) avec tables : train, test, train_split, val_split, scores, predictions
model/
  __init__.py
  train.py           # Étape 3 : entraînement
  test_model.py      # Étape 5 : test d'inférence
  tune.py            # Étape 4 : tuning Optuna
models/
  nyc_taxi.model     # Artefact sérialisé (dict : modele, kmeans, features, paire_stats, mediane_globale)
notebooks/
  nyc_taxi_analysis.ipynb
run.py               # Lance FastAPI + Streamlit en parallèle
environment.yml
```

---

## Schéma des données

### Données brutes (`data/raw/train.csv`)

| Colonne | Type | Description |
|---|---|---|
| `id` | string | Identifiant unique du trajet |
| `vendor_id` | int | Prestataire (1 ou 2) |
| `pickup_datetime` | datetime | Date et heure de prise en charge |
| `dropoff_datetime` | datetime | Date et heure de dépôt *(absent du jeu test)* |
| `passenger_count` | int | Nombre de passagers (1–6) |
| `pickup_longitude` | float | Longitude du point de départ |
| `pickup_latitude` | float | Latitude du point de départ |
| `dropoff_longitude` | float | Longitude du point d'arrivée |
| `dropoff_latitude` | float | Latitude du point d'arrivée |
| `store_and_fwd_flag` | string | Flag de transmission différée (Y/N) |
| `trip_duration` | int | **Cible** — durée en secondes *(absent du jeu test)* |

Zone géographique valide : longitude ∈ [−74.3, −73.6], latitude ∈ [40.4, 41.0].  
Outliers filtrés à l'entraînement : durée ∉ [60, 7200] sec.

### Requête API (`POST /predict`)

| Champ | Type | Contrainte |
|---|---|---|
| `pickup_lat` | float | [40.4, 41.0] |
| `pickup_lon` | float | [−74.3, −73.6] |
| `dropoff_lat` | float | [40.4, 41.0] |
| `dropoff_lon` | float | [−74.3, −73.6] |
| `pickup_datetime` | datetime | ISO 8601 — ex : `2016-06-15T17:30:00` |

Validation supplémentaire : distance pickup→dropoff ≥ 50 m.

### Réponse API

| Champ | Type | Description |
|---|---|---|
| `trip_duration_sec` | float | Durée estimée en secondes |
| `trip_duration_min` | float | Durée estimée en minutes |
| `distance_km` | float | Distance à vol d'oiseau (km) |
| `model_version` | string | Version du modèle utilisé |
| `predicted_at` | string | Timestamp UTC de la prédiction (ISO 8601) |

---

## Features utilisées

| Feature | Description |
|---|---|
| `dist_haversine_km` | Distance à vol d'oiseau (km) |
| `dist_manhattan_km` | Distance Manhattan (km) |
| `bearing_sin/cos` | Direction du trajet (encodage circulaire) |
| `heure`, `jour_semaine`, `mois`, `jour_annee` | Temporel |
| `is_rush_hour`, `is_weekend`, `is_nuit` | Indicateurs temporels |
| `vendor_id`, `passenger_count` | Métadonnées du trajet |
| `cluster_depart`, `cluster_arrivee` | Clusters géographiques KMeans (20) |
| `duree_mediane_paire` | Target encoding : médiane par paire de clusters |

---

## Scores

| Modèle | RMSLE val |
|---|---|
| Baseline (moyenne) | ~0.65 |
| LightGBM v1 (500 arbres, bearing brut) | 0.4225 |
| LightGBM v2 (2000 arbres, bearing sin/cos, target encoding) | 0.4178 |
| Objectif sans données externes | ~0.38 |
| Top Kaggle (avec OSRM) | ~0.30 |

---

## Logging des prédictions

Chaque appel à `/predict` ou `/predict/batch` est automatiquement enregistré dans la table `predictions` de `data/processed/nyc_taxi.db`. Cela permet de surveiller la distribution des prédictions en production et de détecter un éventuel data drift.
