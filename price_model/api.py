"""Public demand -> price API: electricity_price / hydrogen_price."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .config import COMMODITIES
from .multivariate import demand_only_row, predict as _predict

_OUTPUTS = Path(__file__).resolve().parent.parent / "data_exchange" / "02_train_output__benders_input"


@lru_cache(maxsize=None)
def _bundle(commodity: str) -> dict:
    path = _OUTPUTS / COMMODITIES[commodity]["model"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found — run `train_model.py` first ({path})."
        )
    return joblib.load(path)


def available_zones(commodity: str) -> list[str]:
    """Zones with a trained model for this commodity."""
    return sorted(_bundle(commodity)["zones"])


def _price(commodity: str, zone: str, demand, context: dict):
    bundle = _bundle(commodity)
    if zone not in bundle["zones"]:
        raise KeyError(
            f"No {commodity} model for zone {zone!r}. "
            f"Available: {', '.join(available_zones(commodity))}"
        )
    entry = bundle["zones"][zone]
    features = entry.get("features", bundle["features"])
    demand_col = bundle["demand"]
    medians = entry["medians"]

    unknown = set(context) - set(features)
    if unknown:
        raise TypeError(f"Unknown feature(s) {sorted(unknown)}; valid: {features}")

    demand = np.atleast_1d(np.asarray(demand, dtype=float))
    n = len(demand)
    X = demand_only_row(features, demand_col, medians, demand, context)
    out = _predict(bundle, zone, X)
    return float(out[0]) if n == 1 else out


def electricity_price(zone: str, demand, **context):
    """Electricity price [EUR/MWh] for ``zone`` at the given electricity ``demand`` [MW]."""
    return _price("electricity", zone, demand, context)


def hydrogen_price(zone: str, h2_demand, **context):
    """Hydrogen price [EUR/MWhH2] for ``zone`` at the given hydrogen demand [MWH2]."""
    return _price("hydrogen", zone, h2_demand, context)
