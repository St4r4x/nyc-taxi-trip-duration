"""
Chargement de la configuration du projet depuis config.yaml.

Usage :
    from config import CFG

    CFG.geo.lon_min
    CFG.temporal.rush_hour_start
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _load() -> SimpleNamespace:
    with open(_CONFIG_PATH) as f:
        return _to_namespace(yaml.safe_load(f))


CFG = _load()
