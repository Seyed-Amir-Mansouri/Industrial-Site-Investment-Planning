"""Hourly internal demand profiles of one industrial site: per-unit shapes times peak MW.

``inputs/site_demand.csv`` holds the per-unit shapes: one row per candidate country and model-year
hour (8736 per country) with columns ``country``, ``hour`` and one column per service:
``electricity``, ``space_heat``, ``process_heat`` (low/medium-temperature), ``steam``
(high-temperature heat / steam), ``cooling`` and ``hydrogen``. Each value is that hour's demand as
a share of the column's annual peak (1.0 = the peak hour). Without a ``country`` column, the same
shapes apply in every country.

``inputs/site_demand_peaks.csv`` holds the peaks: one row per country with the same service
columns, in MW (MW_th for heat and cooling, MW_LHV for hydrogen). A run can override any of them
through a JSON file named by the ``PLANNER_DEMAND_PEAK_OVERRIDES`` environment variable, shaped
``{"peaks": {country: {service: MW}}}``. A site's hourly MW demand is its shape times its peak.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SERVICES

ROOT = Path(__file__).resolve().parent.parent
SITE_DEMAND_CSV = ROOT / "inputs" / "site_demand.csv"
SITE_DEMAND_PEAKS_CSV = ROOT / "inputs" / "site_demand_peaks.csv"
DEMAND_PEAK_OVERRIDES_ENV = "PLANNER_DEMAND_PEAK_OVERRIDES"

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
        "electricity": 0.10, "space_heat": 0.10, "process_heat": 0.10, "steam": 0.10,
        "cooling": 0.10, "hydrogen": 0.10,
    })


SITE_DEMAND = SiteDemandAssumptions()


@lru_cache(maxsize=1)
def default_peaks_mw() -> dict[str, dict[str, float]]:
    """Peak MW of every service per country, from ``SITE_DEMAND_PEAKS_CSV``."""
    df = pd.read_csv(SITE_DEMAND_PEAKS_CSV)
    missing = [c for c in ["country", *SERVICES] if c not in df.columns]
    if missing:
        raise ValueError(f"{SITE_DEMAND_PEAKS_CSV} is missing column(s) {missing}")
    return {row["country"]: {s: float(row[s]) for s in SERVICES} for _, row in df.iterrows()}


@lru_cache(maxsize=1)
def peaks_mw() -> dict[str, dict[str, float]]:
    """Peak MW per country and service: the defaults, with this run's overrides applied if any."""
    peaks = {c: dict(v) for c, v in default_peaks_mw().items()}
    override_path = os.environ.get(DEMAND_PEAK_OVERRIDES_ENV)
    if override_path:
        for country, values in json.loads(Path(override_path).read_text(encoding="utf-8"))["peaks"].items():
            for service, mw in values.items():
                if service not in SERVICES:
                    raise ValueError(f"demand peak override has unknown service {service!r} -- choices: {SERVICES}")
                peaks.setdefault(country, {})[service] = float(mw)
    return peaks


@lru_cache(maxsize=1)
def _load_csv() -> pd.DataFrame:
    """Read and validate the per-unit shapes in ``SITE_DEMAND_CSV``."""
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
    if (df[SERVICES] < 0).any().any():
        raise ValueError(f"{SITE_DEMAND_CSV} has negative per-unit demand values")
    return df


@lru_cache(maxsize=32)
def year_profiles(country: str) -> dict[str, np.ndarray]:
    """Full-year (8736h) MW profile of every service for a site built in ``country``: its per-unit
    shape times its peak MW."""
    df = _load_csv()
    if "country" in df.columns:
        df = df[df["country"] == country]
        if df.empty:
            raise ValueError(f"{SITE_DEMAND_CSV} has no demand profiles for country {country!r}")
    peaks = peaks_mw().get(country)
    if peaks is None:
        raise ValueError(f"{SITE_DEMAND_PEAKS_CSV} has no demand peaks for country {country!r}")
    df = df.sort_values("hour")
    return {s: df[s].to_numpy(dtype=float) * peaks[s] for s in SERVICES}


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
