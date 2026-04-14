"""
Application Streamlit — Prédiction de la durée d'un trajet en taxi NYC.

Lancement (depuis la racine du projet) :
    conda activate nyc-taxi
    streamlit run api/app.py
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from haversine import haversine, Unit

from data.preprocessing import preparer_inference

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "nyc_taxi.model"

# Coordonnées par défaut : Times Square → JFK Airport
DEFAULT_PICKUP  = (40.7580, -73.9855)   # Times Square
DEFAULT_DROPOFF = (40.6413, -73.7781)   # JFK

st.set_page_config(
    page_title="NYC Taxi Duration",
    page_icon="🚕",
    layout="wide",
)

# ── Chargement du modèle ─────────────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artefact = charger_modele()



# ── Interface ────────────────────────────────────────────────────────────────
st.title("🚕 NYC Taxi — Prédiction de durée de trajet")
st.caption("Modèle LightGBM entraîné sur les données Kaggle NYC Taxi Trip Duration (2016)")

if artefact is None:
    st.error(
        "Modèle introuvable. Entraînez-le d'abord :\n\n"
        "```bash\nconda activate nyc-taxi\npython -m model.train\n```"
    )
    st.stop()

modele          = artefact["modele"]
kmeans          = artefact["kmeans"]
paire_stats     = artefact["paire_stats"]
mediane_globale = artefact["mediane_globale"]

# ── Colonnes principales ─────────────────────────────────────────────────────
col_gauche, col_droite = st.columns([1, 1], gap="large")

with col_gauche:
    st.subheader("Point de départ")
    pickup_lat = st.number_input(
        "Latitude", value=DEFAULT_PICKUP[0], min_value=40.4, max_value=41.0,
        step=0.001, format="%.4f", key="plat"
    )
    pickup_lon = st.number_input(
        "Longitude", value=DEFAULT_PICKUP[1], min_value=-74.3, max_value=-73.6,
        step=0.001, format="%.4f", key="plon"
    )

    st.subheader("Point d'arrivée")
    dropoff_lat = st.number_input(
        "Latitude", value=DEFAULT_DROPOFF[0], min_value=40.4, max_value=41.0,
        step=0.001, format="%.4f", key="dlat"
    )
    dropoff_lon = st.number_input(
        "Longitude", value=DEFAULT_DROPOFF[1], min_value=-74.3, max_value=-73.6,
        step=0.001, format="%.4f", key="dlon"
    )

with col_droite:
    st.subheader("Informations du trajet")
    pickup_date = st.date_input("Date de prise en charge", value=datetime(2016, 6, 15))
    pickup_time = st.time_input("Heure de prise en charge", value=datetime(2016, 6, 15, 17, 30))
    pickup_dt   = datetime.combine(pickup_date, pickup_time)


# ── Carte ────────────────────────────────────────────────────────────────────
st.subheader("Visualisation du trajet")

points_df = pd.DataFrame([
    {"lat": pickup_lat,  "lon": pickup_lon,  "type": "Départ",  "couleur": [0, 128, 255, 200]},
    {"lat": dropoff_lat, "lon": dropoff_lon, "type": "Arrivée", "couleur": [255, 64, 64, 200]},
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
    get_radius=200,
    pickable=True,
)
layer_ligne = pdk.Layer(
    "LineLayer",
    data=ligne_df,
    get_source_position="depart",
    get_target_position="arrivee",
    get_color=[100, 100, 100, 180],
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
    map_style="mapbox://styles/mapbox/light-v10",
))

# ── Prédiction ───────────────────────────────────────────────────────────────
st.divider()
if st.button("Calculer la durée estimée", type="primary", use_container_width=True):
    X = preparer_inference(
        pickup_lat, pickup_lon,
        dropoff_lat, dropoff_lon,
        pickup_dt,
        kmeans, paire_stats, mediane_globale,
    )
    duree_sec = float(np.expm1(modele.predict(X.values)[0]))
    duree_min = duree_sec / 60
    dist_km   = haversine(
        (pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon), unit=Unit.KILOMETERS
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Durée estimée", f"{duree_min:.0f} min {duree_sec % 60:.0f} sec")
    c2.metric("Distance à vol d'oiseau", f"{dist_km:.2f} km")
    c3.metric("Vitesse moyenne estimée", f"{dist_km / (duree_sec / 3600):.0f} km/h")

    heure = pickup_dt.hour
    if 7 <= heure <= 9 or 17 <= heure <= 20:
        st.info("Heure de pointe détectée — durée potentiellement plus longue.")
    elif heure >= 22 or heure <= 5:
        st.success("Trajet de nuit — trafic faible, durée potentiellement plus courte.")
