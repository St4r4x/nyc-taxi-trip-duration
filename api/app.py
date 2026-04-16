"""
Application Streamlit — Prédiction de la durée d'un trajet en taxi NYC.

Lancement (depuis la racine du projet) :
    conda activate nyc-taxi
    streamlit run api/app.py
"""

import sys
from pathlib import Path

# Garantit que la racine du projet est dans sys.path (nécessaire avec streamlit run)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from api.registry import ModelRegistry
from config import CFG
from data.postprocessing import postprocesser
from data.preprocessing import preparer_inference
from data.schema import PredictInput

# ── Configuration ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NYC Taxi Duration",
    page_icon="🚕",
    layout="wide",
)

# ── Lieux célèbres ────────────────────────────────────────────────────────────

LIEUX = {
    "Times Square":         (40.7580, -73.9855),
    "JFK Airport":          (40.6413, -73.7781),
    "LaGuardia Airport":    (40.7769, -73.8740),
    "Penn Station":         (40.7506, -73.9971),
    "Grand Central":        (40.7527, -73.9772),
    "Empire State Building":(40.7484, -73.9857),
    "Brooklyn Bridge":      (40.7061, -73.9969),
    "Central Park":         (40.7851, -73.9683),
    "Wall Street":          (40.7074, -74.0113),
    "Statue of Liberty":    (40.6892, -74.0445),
}

LIEUX_NOMS = list(LIEUX.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚕 NYC Taxi Duration")
    st.caption("Modèle LightGBM · Kaggle 2016")

    st.divider()

    # Sélection du modèle
    versions_dispo = ModelRegistry.versions()
    if versions_dispo:
        version_choisie = st.selectbox(
            "Modèle",
            options=versions_dispo,
            index=0,
        )
    else:
        st.error("Aucun modèle trouvé. Entraînez-le d'abord :")
        st.code("python -m model.train", language="bash")
        st.stop()

    st.divider()

    # Raccourcis lieux
    st.subheader("Raccourcis lieux")
    lieu_depart  = st.selectbox("Départ",  LIEUX_NOMS, index=0, key="lieu_dep")
    lieu_arrivee = st.selectbox("Arrivée", LIEUX_NOMS, index=1, key="lieu_arr")

    if st.button("Appliquer", width="stretch"):
        lat_d, lon_d = LIEUX[lieu_depart]
        lat_a, lon_a = LIEUX[lieu_arrivee]
        st.session_state["plat"] = lat_d
        st.session_state["plon"] = lon_d
        st.session_state["dlat"] = lat_a
        st.session_state["dlon"] = lon_a

    st.divider()

    # Info modèle
    try:
        _, info = ModelRegistry.get(version_choisie)
        with st.expander("Détails du modèle"):
            st.write(f"**Version :** `{info.version}`")
            st.write(f"**Features :** {info.n_features}")
            st.write(f"**Créé le :** {info.created_at[:10]}")
    except KeyError:
        pass

# ── Chargement modèle ─────────────────────────────────────────────────────────

@st.cache_resource
def charger_modele(version: str):
    return ModelRegistry.get(version)

try:
    artefact, info = charger_modele(version_choisie)
except KeyError as e:
    st.error(str(e))
    st.stop()

kmeans          = artefact["kmeans"]
paire_stats     = artefact["paire_stats"]
mediane_globale = artefact["mediane_globale"]
modele          = artefact["modele"]

# ── Initialisation session state ──────────────────────────────────────────────

DEFAULT_PICKUP  = LIEUX["Times Square"]
DEFAULT_DROPOFF = LIEUX["JFK Airport"]

for key, val in [
    ("plat", DEFAULT_PICKUP[0]),
    ("plon", DEFAULT_PICKUP[1]),
    ("dlat", DEFAULT_DROPOFF[0]),
    ("dlon", DEFAULT_DROPOFF[1]),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Titre principal ───────────────────────────────────────────────────────────

st.title("Prédiction de durée de trajet")

# ── Formulaire ────────────────────────────────────────────────────────────────

col_gauche, col_droite = st.columns([1, 1], gap="large")

with col_gauche:
    st.subheader("Point de départ")
    pickup_lat = st.number_input(
        "Latitude", min_value=40.4, max_value=41.0,
        step=0.001, format="%.4f", key="plat",
    )
    pickup_lon = st.number_input(
        "Longitude", min_value=-74.3, max_value=-73.6,
        step=0.001, format="%.4f", key="plon",
    )

    st.subheader("Point d'arrivée")
    dropoff_lat = st.number_input(
        "Latitude", min_value=40.4, max_value=41.0,
        step=0.001, format="%.4f", key="dlat",
    )
    dropoff_lon = st.number_input(
        "Longitude", min_value=-74.3, max_value=-73.6,
        step=0.001, format="%.4f", key="dlon",
    )

with col_droite:
    st.subheader("Informations du trajet")
    pickup_date = st.date_input(
        "Date de prise en charge",
        value=datetime(2016, 6, 15),
        min_value=datetime(2016, 1, 1),
        max_value=datetime(2016, 12, 31),
    )
    pickup_time = st.time_input(
        "Heure de prise en charge",
        value=datetime(2016, 6, 15, 17, 30).time(),
    )
    pickup_dt = datetime.combine(pickup_date, pickup_time)

    st.subheader("Contexte")
    heure = pickup_dt.hour
    jour  = pickup_dt.weekday()
    is_rush    = (jour < 5) and (CFG.temporal.rush_hour_start <= heure <= CFG.temporal.rush_hour_end)
    is_weekend = jour >= 5
    is_nuit    = heure >= CFG.temporal.nuit_start or heure <= CFG.temporal.nuit_end

    badges = []
    if is_rush:    badges.append("🔴 Heure de pointe")
    if is_weekend: badges.append("🟢 Week-end")
    if is_nuit:    badges.append("🌙 Nuit")
    if not badges: badges.append("⚪ Circulation normale")

    for b in badges:
        st.info(b)

# ── Prédiction ────────────────────────────────────────────────────────────────

st.divider()

if st.button("Calculer la durée estimée", type="primary", width="stretch"):
    try:
        req = PredictInput(
            pickup_lat=pickup_lat, pickup_lon=pickup_lon,
            dropoff_lat=dropoff_lat, dropoff_lon=dropoff_lon,
            pickup_datetime=pickup_dt,
        )
        X = preparer_inference(req, kmeans, paire_stats, mediane_globale)
        y_log  = modele.predict(X.values)[0]
        output = postprocesser(y_log, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

        duree_sec = output.trip_duration_sec
        duree_min = output.trip_duration_min
        dist_km   = output.distance_km
        vitesse   = dist_km / (duree_sec / 3600) if duree_sec > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Durée estimée",           f"{int(duree_min)} min {int(duree_sec % 60)} s")
        c2.metric("Distance (vol d'oiseau)", f"{dist_km:.2f} km")
        c3.metric("Vitesse moyenne",         f"{vitesse:.0f} km/h")
        c4.metric("Modèle utilisé",          info.version)

        if "historique" not in st.session_state:
            st.session_state["historique"] = []

        st.session_state["historique"].append({
            "Départ (lat, lon)":   f"{pickup_lat:.4f}, {pickup_lon:.4f}",
            "Arrivée (lat, lon)":  f"{dropoff_lat:.4f}, {dropoff_lon:.4f}",
            "Date/heure":          pickup_dt.strftime("%Y-%m-%d %H:%M"),
            "Distance (km)":       round(dist_km, 2),
            "Durée estimée":       f"{int(duree_min)} min {int(duree_sec % 60)} s",
            "Vitesse (km/h)":      round(vitesse, 0),
        })

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# ── Carte ─────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Visualisation du trajet")

points_df = pd.DataFrame([
    {"lat": pickup_lat,  "lon": pickup_lon,  "type": "Départ",  "couleur": [0, 128, 255, 220]},
    {"lat": dropoff_lat, "lon": dropoff_lon, "type": "Arrivée", "couleur": [255, 64, 64, 220]},
])
ligne_df = pd.DataFrame([{
    "depart":  [pickup_lon,  pickup_lat],
    "arrivee": [dropoff_lon, dropoff_lat],
}])

layer_points = pdk.Layer(
    "ScatterplotLayer",
    data=points_df,
    get_position="[lon, lat]",
    get_color="couleur",
    get_radius=250,
    pickable=True,
)
layer_ligne = pdk.Layer(
    "LineLayer",
    data=ligne_df,
    get_source_position="depart",
    get_target_position="arrivee",
    get_color=[80, 80, 80, 160],
    get_width=4,
)

vue = pdk.ViewState(
    latitude=(pickup_lat + dropoff_lat) / 2,
    longitude=(pickup_lon + dropoff_lon) / 2,
    zoom=10,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer_ligne, layer_points],
    initial_view_state=vue,
    tooltip={"text": "{type}"},
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
))

# ── Historique ────────────────────────────────────────────────────────────────

if st.session_state.get("historique"):
    st.subheader("Historique des prédictions")
    df_hist = pd.DataFrame(st.session_state["historique"][::-1])
    st.dataframe(df_hist, hide_index=True)

    if st.button("Effacer l'historique"):
        st.session_state["historique"] = []
        st.rerun()
