"""
Registre des modèles disponibles.

Charge et met en cache les artefacts depuis models/*.model.
Calcule la metadata de chaque modèle à partir des métadonnées du fichier.

Usage :
    from api.registry import ModelRegistry

    artefact, info = ModelRegistry.get("nyc_taxi")
    versions = ModelRegistry.versions()
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

from data.schema import ModelInfo

MODELS_DIR   = Path("models")
DEFAULT_MODEL = "nyc_taxi"


class ModelRegistry:
    _cache: dict[str, tuple[dict, ModelInfo]] = {}

    @classmethod
    def versions(cls) -> list[str]:
        """Retourne la liste des versions disponibles (stems des fichiers .model)."""
        return sorted(p.stem for p in MODELS_DIR.glob("*.model"))

    @classmethod
    def get(cls, version: str) -> tuple[dict, ModelInfo]:
        """
        Charge et met en cache le modèle demandé.

        Args:
            version : nom du fichier sans extension (ex : "nyc_taxi", "nyc_taxi_tuned")

        Returns:
            (artefact dict, ModelInfo)

        Raises:
            KeyError : si la version n'existe pas
        """
        if version not in cls._cache:
            path = MODELS_DIR / f"{version}.model"
            if not path.exists():
                available = cls.versions()
                raise KeyError(
                    f"Modèle '{version}' introuvable. "
                    f"Versions disponibles : {available}"
                )
            with open(path, "rb") as f:
                artefact = pickle.load(f)

            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            info  = ModelInfo(
                version    = version,
                path       = str(path),
                created_at = mtime.isoformat(),
                features   = artefact["features"],
                n_features = len(artefact["features"]),
            )
            cls._cache[version] = (artefact, info)

        return cls._cache[version]
