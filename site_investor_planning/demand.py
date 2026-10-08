"""Hourly internal demand profiles of one industrial site, read from ``inputs/site_demand.csv``.

The CSV holds one row per candidate country and model-year hour (8736 per country) with columns
``country``, ``hour`` and one column per service in MW: ``electricity``, ``space_heat``,
``lt_process_heat``, ``mt_process_heat``, ``ht_heat``, ``steam``, ``cooling`` and ``hydrogen``
(MW_th for heat and cooling, MW_LHV for hydrogen). A site built in a country gets that country's
profiles. Without a ``country`` column, the same profiles apply in every country.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SERVICES

ROOT = Path(__file__).resolve().parent.parent
SITE_DEMAND_CSV = ROOT / "inputs" / "site_demand.csv"

HOURS_PER_DAY = 24
YEAR_DAYS = 364
YEAR_HOURS = YEAR_DAYS * HOURS_PER_DAY


@dataclass(frozen=True)
class SiteDemandAssumptions:
    """Demand flexibility of one site.

    ``flex_fraction`` is how far each hour's demand may move up or down as a share of itself,
    with the shifts netting to zero over each day.
    """

    flex_fraction: dict[str, float] = field(default_factory=lambda: {
        "electricity": 0.10, "space_heat": 0.10, "lt_process_heat": 0.10, "mt_process_heat": 0.10,
        "ht_heat": 0.10, "steam": 0.10, "cooling": 0.10, "hydrogen": 0.10,
    })


SITE_DEMAND = SiteDemandAssumptions()


@lru_cache(maxsize=1)
def _load_csv() -> pd.DataFrame:
    """Read and validate ``SITE_DEMAND_CSV``."""
    if not SITE_DEMAND_CSV.exists():
        raise FileNotFoundError(f"site demand profiles not found: {SITE_DEMAND_CSV}")
    df = pd.read_csv(SITE_DEMAND_CSV)
    missing = [c for c in ["hour", *SERVICES] if c not in df.columns]
    if missing:
        raise ValueError(f"{SITE_DEMAND_CSV} is missing column(s) {missing}")
    groups = df.groupby("country") if "country" in df.columns else [(None, df)]
    for country, g in groups:
        if len(g) != YEAR_HOURS or set(g["hour"]) != set(range(YEAR_HOURS)):
            raise ValueError(f"{SITE_DEMAND_CSV}: {country or 'profile'} needs hours 0-{YEAR_HOURS - 1} "
                             f"exactly once, got {len(g)} rows")
    return df


@lru_cache(maxsize=32)
def year_profiles(country: str) -> dict[str, np.ndarray]:
    """Full-year (8736h) MW profile of every service for a site built in ``country``."""
    df = _load_csv()
    if "country" in df.columns:
        df = df[df["country"] == country]
        if df.empty:
            raise ValueError(f"{SITE_DEMAND_CSV} has no demand profiles for country {country!r}")
    df = df.sort_values("hour")
    return {s: df[s].to_numpy(dtype=float) for s in SERVICES}


def annual_demand_mwh(country: str) -> dict[str, float]:
    """Annual demand of every service, MWh/yr, for a site built in ``country``."""
    return {s: float(v.sum()) for s, v in year_profiles(country).items()}


def site_demand(country: str, hours: np.ndarray) -> dict[str, np.ndarray]:
    """MW demand of every service at the given model-year hour positions."""
    full = year_profiles(country)
    idx = np.asarray(hours, dtype=int)
    return {s: full[s][idx] for s in SERVICES}


def peak_demand_mw(country: str, services: list[str]) -> float:
    """Full-year peak MW of the combined demand of ``services``, used to cap thermal asset sizes."""
    full = year_profiles(country)
    return float(sum(full[s] for s in services).max())
