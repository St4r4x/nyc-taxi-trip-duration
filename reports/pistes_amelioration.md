# Pistes d'amélioration — NYC Taxi Trip Duration

Synthèse des approches identifiées dans les top solutions de la compétition (top 4%, top 6%), le notebook EDA de référence, et le notebook de cours (LAB 5 — Regression Linear Models).

---

## 1. Données externes

### OSRM — Open Source Routing Machine ⭐ fort impact
Kaggle a fourni des fichiers `fastest_routes_train/test.csv` avec les vrais itinéraires routiers.

| Feature | Description |
|---|---|
| `total_distance` | Distance réelle par les routes (vs. haversine à vol d'oiseau) |
| `total_travel_time` | Estimation OSRM de la durée |
| `number_of_steps` | Nombre d'intersections / virages |
| `turns_left` / `turns_right` | Comptage des tournants gauche/droite |

> Télécharger sur la page data de la compétition Kaggle. Gain documenté : ~0.01 RMSLE.

### Météo
Enrichir avec des données météo NYC 2016 (pluie, neige, température, visibilité).  
Forte corrélation avec les ralentissements de trafic.

---

## 2. Feature engineering géographique

| Idée | Détail | Difficulté |
|---|---|---|
| **PCA sur les coordonnées** | NYC est orienté à ~29° du nord. Tourner les axes lat/lon par PCA capture mieux la grille de Manhattan. Feature standard dans les top solutions. | Faible |
| **Zones aéroports** | Détecter JFK, LaGuardia, Newark comme zones spéciales (durées systématiquement différentes). | Faible |
| **Plus de clusters KMeans** | Passer de 20 à 40-80 clusters pour affiner le target encoding `duree_mediane_paire`. | Faible |
| **DBSCAN** | Remplacer KMeans par DBSCAN pour détecter les zones denses naturellement, sans fixer le nombre de clusters. | Moyenne |
| **Densité trafic par zone/heure** | Compter le nombre de départs/arrivées dans chaque cluster à chaque heure → proxy du trafic en temps réel. Calculable depuis `train_split`. | Moyenne |
| **Distance au centre** | Distance haversine depuis le pickup/dropoff jusqu'au centre de Manhattan. | Faible |

---

## 3. Feature engineering temporel

| Idée | Détail | Difficulté |
|---|---|---|
| **Jours fériés US 2016** | Flag `is_holiday` (Memorial Day, Independence Day, Labor Day, Thanksgiving, Noël…) | Faible |
| **Vitesse moyenne par heure × jour** | Calculer `vitesse = distance / durée` dans `train_split`, agréger par `(heure, jour_semaine)` → feature de contexte trafic. | Faible |
| **Semaine de l'année** | `dt.isocalendar().week` — capture les variations hebdomadaires fines. | Faible |

---

## 4. Feature engineering sur les distances

| Idée | Détail |
|---|---|
| **Log de la distance** | `log1p(dist_haversine_km)` — la relation distance/durée est log-linéaire |
| **Ratio haversine / manhattan** | Proxy de la "courbure" du trajet (trajets diagonaux vs. trajets en L) |

---

## 5. Apports du notebook de cours (LAB 5)

Ces idées sont absentes de notre pipeline actuel et validées expérimentalement dans le notebook.

### Features à ajouter directement

| Feature | Détail | Justification |
|---|---|---|
| **`log_distance_haversine`** | `log1p(dist_haversine_km)` | Relation théorique : `log(t) = log(s) − log(v)`. Corrélation avec la cible passe de 0.57 → 0.75. C'est la feature la plus importante du notebook. |
| **`abnormal_period`** | Flag binaire pour les jours avec < ~6300 trajets (Thanksgiving et Noël 2016) | Ces jours ont paradoxalement des durées plus longues (moins de taxis en circulation). Non capturé par `is_holiday` générique. |
| **`is_high_speed_trip`** | Nuit profonde : 2h–5h semaine, 4h–7h week-end | Complémentaire à notre `is_nuit` (22h–5h). La distinction nuit/aube compte. |
| **`is_rare_pickup_point`** | 1 si pickup en dehors du percentile 1–99.5 des coordonnées | Capture les trajets très excentrés (aéroports lointains, New Jersey…). |
| **`is_rare_dropoff_point`** | Idem pour le dropoff | Même logique. |

### Observations méthodologiques utiles

- **`vendor_id`, `store_and_fwd_flag`, `passenger_count`** : testés et confirmés sans impact significatif. On peut les retirer du pipeline pour simplifier.
- **`speed` (distance/durée)** ne doit jamais être une feature : c'est une fuite de données directe (la durée est dans le calcul).
- La **heatmap vitesse × (heure, jour_semaine)** sur `train_split` est un outil d'analyse utile pour valider nos features temporelles.

---

## 6. Modélisation

| Idée | Détail | Gain estimé |
|---|---|---|
| **Ensemble multi-seeds** | Entraîner 5-10 LightGBM avec des `random_state` différents et moyenner les prédictions → réduit la variance sans coût de tuning supplémentaire. | ~0.002 RMSLE |
| **Stacking XGBoost + LightGBM** | Les deux modèles font des erreurs différentes. Un meta-modèle simple (Ridge) sur leurs prédictions OOF gagne typiquement ~0.005 RMSLE. | ~0.005 RMSLE |
| **Validation croisée k-fold** | Actuellement split unique 80/20. Une CV 5-fold stratifiée sur le mois est plus robuste pour le tuning Optuna et réduit l'overfitting sur la val. | Robustesse |

---

## Priorité suggérée

| Priorité | Amélioration | Effort | Impact |
|---|---|---|---|
| 1 | `log_distance_haversine` + `abnormal_period` + `is_rare_*_point` (LAB 5) | Faible | Fort |
| 2 | OSRM (données Kaggle) | Moyen | Très fort |
| 3 | PCA sur coordonnées | Faible | Fort |
| 4 | Densité trafic par zone/heure | Moyen | Moyen |
| 5 | Ensemble multi-seeds | Faible | Moyen |
| 6 | Jours fériés + vitesse moyenne par (heure, jour) | Faible | Faible-Moyen |
| 7 | Météo | Élevé | Moyen |
| 8 | Stacking XGB + LGBM | Élevé | Moyen |

---

## Sources

- `notebooks/LAB_5_Regression_Linear_Models_SOLUTION.ipynb` — notebook de cours (Prof)
- [pklauke — Top 4% (LightGBM)](https://github.com/pklauke/Kaggle-NYCTaxi)
- [yennanliu — Top 6% (XGBoost + LightGBM)](https://github.com/yennanliu/NYC_Taxi_Trip_Duration)
- [headsortails — NYC Taxi EDA (The fast & the curious)](https://www.kaggle.com/code/headsortails/nyc-taxi-eda-update-the-fast-the-curious)
- [NYC Taxi OSRM/time series notebook](https://www.kaggle.com/code/atmarouane/nyc-taxi-trips-osrm-time-series)