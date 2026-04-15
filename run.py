"""
Lance l'API FastAPI et l'app Streamlit en parallèle.

Usage :
    conda activate nyc-taxi
    python run.py

    FastAPI  → http://localhost:8000   (Swagger : http://localhost:8000/docs)
    Streamlit → http://localhost:8501
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

processes = [
    {
        "name": "FastAPI",
        "cmd": [sys.executable, "-m", "api.main"],
        "url": "http://localhost:8000/docs",
    },
    {
        "name": "Streamlit",
        "cmd": [sys.executable, "-m", "streamlit", "run", "api/app.py"],
        "url": "http://localhost:8501",
    },
]

procs = []
for p in processes:
    print(f"[{p['name']}] Démarrage → {p['url']}")
    procs.append(subprocess.Popen(p["cmd"], cwd=ROOT))

print("\nCtrl+C pour tout arrêter.\n")

try:
    for proc in procs:
        proc.wait()
except KeyboardInterrupt:
    print("\nArrêt...")
    for proc in procs:
        proc.terminate()
    for proc in procs:
        proc.wait()
    print("Arrêté.")
