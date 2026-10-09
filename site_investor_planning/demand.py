"""Hourly internal demand of an industrial site: one per-unit yearly curve per demand times the site's peaks.

``inputs/site_demand.csv`` holds the per-unit curves: 8736 rows (``hour`` 0-8735, the model year)
and one column per service: ``electricity``, ``space_heat``, ``process_heat`` (low/medium-temperature),
``steam`` (high-temperature heat / steam), ``cooling`` and ``hydrogen``. Each value is that hour's
demand as a share of the site's daily peak (1.0 = a normal day's peak hour; a day can go above or
below it). The curves are shared by every site. The shipped file repeats one daily curve for the
whole year.

Each site's daily peaks, green hydrogen share and flexibility come from its ``SiteSpec``. A run's
sites can be read from a JSON file shaped ``{"sites": [{"name": .., "peaks_mw": {service: MW},
"green_share": .., "flex_fraction": ..}, ...]}``; missing values fall back to the defaults.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_SITE_PEAKS_MW, SERVICES, SiteSpec

ROOT = Path(__file__).resolve().parent.parent
SITE_DEMAND_CSV = ROOT / "inputs" / "site_demand.csv"

HOURS_PER_DAY = 24
YEAR_DAYS = 364
YEAR_HOURS = YEAR_DAYS * HOURS_PER_DAY


@lru_cache(maxsize=1)
def yearly_curves() -> dict[str, np.ndarray]:
    """The 8736-hour per-unit curve of every service, read and checked from ``SITE_DEMAND_CSV``."""
    if not SITE_DEMAND_CSV.exists():
        raise FileNotFoundError(f"site demand curves not found: {SITE_DEMAND_CSV}")
    df = pd.read_csv(SITE_DEMAND_CSV)
    missing = [c for c in ["hour", *SERVICES] if c not in df.columns]
    if missing:
        raise ValueError(f"{SITE_DEMAND_CSV} is missing column(s) {missing}")
    if len(df) != YEAR_HOURS or set(df["hour"]) != set(range(YEAR_HOURS)):
        raise ValueError(f"{SITE_DEMAND_CSV} needs hours 0-{YEAR_HOURS - 1} exactly once, got {len(df)} rows")
    if (df[SERVICES] < 0).any().any():
        raise ValueError(f"{SITE_DEMAND_CSV} has negative per-unit demand values")
    df = df.sort_values("hour")
    return {s: df[s].to_numpy(dtype=float) for s in SERVICES}


def year_profiles(peaks_mw: dict[str, float]) -> dict[str, np.ndarray]:
    """Full-year (8736h) MW profile of every service for a site with these daily peaks."""
    return {s: curve * float(peaks_mw[s]) for s, curve in yearly_curves().items()}


def annual_demand_mwh(peaks_mw: dict[str, float]) -> dict[str, float]:
    """Annual demand of every service, MWh/yr, for a site with these daily peaks."""
    return {s: float(v.sum()) for s, v in year_profiles(peaks_mw).items()}


def site_demand(peaks_mw: dict[str, float], hours: np.ndarray) -> dict[str, np.ndarray]:
    """MW demand of every service at the given model-year hour positions."""
    idx = np.asarray(hours, dtype=int)
    return {s: v[idx] for s, v in year_profiles(peaks_mw).items()}


def peak_demand_mw(peaks_mw: dict[str, float], services: list[str]) -> float:
    """Full-year peak MW of the combined demand of ``services``, used to cap thermal asset sizes."""
    full = year_profiles(peaks_mw)
    return float(sum(full[s] for s in services).max())


def default_sites(n_sites: int, green_share: float | None = None,
                  flex_fraction: float | None = None) -> list[SiteSpec]:
    """``n_sites`` sites named Site 1..N with the default peaks, optionally overriding the green
    share and flexibility of all of them."""
    sites = []
    for i in range(1, n_sites + 1):
        spec = SiteSpec(name=f"Site {i}", peaks_mw=dict(DEFAULT_SITE_PEAKS_MW))
        if green_share is not None:
            spec.green_share = green_share
        if flex_fraction is not None:
            spec.flex_fraction = flex_fraction
        sites.append(spec)
    return sites


def load_sites(path: Path) -> list[SiteSpec]:
    """Sites from a ``{"sites": [...]}`` JSON file (see module docstring); names must be unique."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["sites"]
    if not entries:
        raise ValueError(f"{path} defines no sites")
    sites = []
    for i, e in enumerate(entries, start=1):
        unknown = [k for k in e.get("peaks_mw", {}) if k not in SERVICES]
        if unknown:
            raise ValueError(f"site {i} in {path} has unknown service(s) {unknown} -- choices: {SERVICES}")
        peaks = {**DEFAULT_SITE_PEAKS_MW, **e.get("peaks_mw", {})}
        sites.append(SiteSpec(name=str(e.get("name") or f"Site {i}"), peaks_mw=peaks,
                              green_share=float(e.get("green_share", 0.42)),
                              flex_fraction=float(e.get("flex_fraction", 0.10))))
    names = [s.name for s in sites]
    if len(set(names)) != len(names):
        raise ValueError(f"site names in {path} must be unique, got {names}")
    return sites
