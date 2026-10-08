"""Hourly internal demand profiles of one industrial site (electricity, space heat, low/medium-temperature
process heat, high-temperature heat, steam, cooling, hydrogen).

The defaults are synthetic: each service's annual total is spread over the model year with a
simple shape (shift pattern for process loads, seasonal + occupancy shape for space heating and
comfort cooling). Space heating scales with the country's heating degree days and comfort cooling
with its cooling degree days, so the same site has a different demand depending on where it's
built. Heating and cooling degree days are approximate long-run national values (Eurostat
nrg_chdd_a order of magnitude, rounded); only their ratio to the reference values matters. Drop a
CSV at ``inputs/site_demand.csv`` (columns ``hour`` + one per service, MW, 8736 rows) to use real
site data instead -- it then applies unchanged in every country.
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

HEATING_DEGREE_DAYS = {"AT": 3300, "BE": 2700, "CZ": 3300, "DE": 3000, "FR": 2300, "HR": 2300,
                       "HU": 2700, "LU": 2900, "NL": 2700, "PL": 3300, "RO": 2800, "SI": 2700,
                       "SK": 3100}
COOLING_DEGREE_DAYS = {"AT": 40, "BE": 10, "CZ": 25, "DE": 20, "FR": 70, "HR": 250, "HU": 150,
                       "LU": 15, "NL": 10, "PL": 30, "RO": 200, "SI": 90, "SK": 70}
REFERENCE_HDD = 3000.0
REFERENCE_CDD = 50.0


@dataclass(frozen=True)
class SiteDemandAssumptions:
    """Annual internal demand of one site, MWh/yr (MWh_th for heat/cooling, MWh_LHV for hydrogen).

    ``space_heat`` is at ``REFERENCE_HDD``; ``cooling`` is the process part, with
    ``comfort_cooling_mwh`` (at ``REFERENCE_CDD``) added on top. ``flex_fraction`` is how far each
    hour's demand may move up or down as a share of itself, with the shifts netting to zero over
    each day. The process loads follow a shift pattern (fraction of peak): ``shift_day`` on weekdays
    from ``shift_start_hour`` to ``shift_end_hour``, ``shift_night`` on weekday nights and
    ``shift_weekend`` all weekend.
    """

    annual_mwh: dict[str, float] = field(default_factory=lambda: {
        "electricity": 50_000.0,
        "space_heat": 8_000.0,
        "lt_process_heat": 20_000.0,
        "mt_process_heat": 20_000.0,
        "ht_heat": 12_000.0,
        "steam": 18_000.0,
        "cooling": 12_000.0,
        "hydrogen": 10_000.0,
    })
    comfort_cooling_mwh: float = 3_000.0
    flex_fraction: dict[str, float] = field(default_factory=lambda: {
        "electricity": 0.10, "space_heat": 0.10, "lt_process_heat": 0.10, "mt_process_heat": 0.10,
        "ht_heat": 0.10, "steam": 0.10, "cooling": 0.10, "hydrogen": 0.10,
    })
    shift_day: float = 1.0
    shift_night: float = 0.6
    shift_weekend: float = 0.5
    shift_start_hour: int = 6
    shift_end_hour: int = 22


SITE_DEMAND = SiteDemandAssumptions()


def _day_hour_grid() -> tuple[np.ndarray, np.ndarray]:
    """(day-of-year 1..364, hour-of-day 0..23) for every model-year hour; day 1 is taken as a Monday."""
    h = np.arange(YEAR_HOURS)
    return h // HOURS_PER_DAY + 1, h % HOURS_PER_DAY


def _normalized(shape: np.ndarray, annual_mwh: float) -> np.ndarray:
    total = shape.sum()
    return shape * (annual_mwh / total) if total > 0 else np.zeros_like(shape)


def _shift_shape(a: SiteDemandAssumptions) -> np.ndarray:
    """Two-shift weekday pattern of the process loads, with lower weekday-night and weekend levels."""
    day, hod = _day_hour_grid()
    weekend = ((day - 1) % 7) >= 5
    day_shift = (hod >= a.shift_start_hour) & (hod < a.shift_end_hour)
    return np.where(weekend, a.shift_weekend, np.where(day_shift, a.shift_day, a.shift_night))


def _space_heat_shape() -> np.ndarray:
    """Seasonal space heating shape peaking in mid-January, higher during 6:00-20:00 occupancy."""
    day, hod = _day_hour_grid()
    seasonal = np.clip(np.cos(2 * np.pi * (day - 15) / YEAR_DAYS) + 0.2, 0.0, None)
    occupancy = np.where((hod >= 6) & (hod < 20), 1.0, 0.7)
    return seasonal * occupancy


def _comfort_cooling_shape() -> np.ndarray:
    """Seasonal comfort cooling shape peaking in mid-July, with a 9:00-19:00 afternoon bump."""
    day, hod = _day_hour_grid()
    seasonal = np.clip(np.cos(2 * np.pi * (day - 200) / YEAR_DAYS) - 0.3, 0.0, None)
    afternoon = 1.0 + 0.6 * np.clip(np.sin(np.pi * (hod - 9) / 10), 0.0, None)
    return seasonal * afternoon


@lru_cache(maxsize=1)
def _csv_profiles() -> dict[str, np.ndarray] | None:
    if not SITE_DEMAND_CSV.exists():
        return None
    df = pd.read_csv(SITE_DEMAND_CSV).sort_values("hour")
    missing = [s for s in SERVICES if s not in df.columns]
    if missing or len(df) != YEAR_HOURS:
        raise ValueError(f"{SITE_DEMAND_CSV} needs {YEAR_HOURS} rows and columns hour,{','.join(SERVICES)} "
                         f"(missing: {missing}, rows: {len(df)})")
    return {s: df[s].to_numpy(dtype=float) for s in SERVICES}


@lru_cache(maxsize=32)
def year_profiles(country: str, assumptions: SiteDemandAssumptions = SITE_DEMAND) -> dict[str, np.ndarray]:
    """Full-year (8736h) MW profile of every service for a site built in ``country``."""
    from_csv = _csv_profiles()
    if from_csv is not None:
        return from_csv
    a = assumptions
    shift = _shift_shape(a)
    hdd = HEATING_DEGREE_DAYS.get(country, REFERENCE_HDD) / REFERENCE_HDD
    cdd = COOLING_DEGREE_DAYS.get(country, REFERENCE_CDD) / REFERENCE_CDD
    return {
        "electricity": _normalized(shift, a.annual_mwh["electricity"]),
        "space_heat": _normalized(_space_heat_shape(), a.annual_mwh["space_heat"] * hdd),
        "lt_process_heat": _normalized(shift, a.annual_mwh["lt_process_heat"]),
        "mt_process_heat": _normalized(shift, a.annual_mwh["mt_process_heat"]),
        "ht_heat": _normalized(shift, a.annual_mwh["ht_heat"]),
        "steam": _normalized(shift, a.annual_mwh["steam"]),
        "cooling": (_normalized(shift, a.annual_mwh["cooling"])
                    + _normalized(_comfort_cooling_shape(), a.comfort_cooling_mwh * cdd)),
        "hydrogen": _normalized(shift, a.annual_mwh["hydrogen"]),
    }


def site_demand(country: str, hours: np.ndarray) -> dict[str, np.ndarray]:
    """MW demand of every service at the given model-year hour positions."""
    full = year_profiles(country)
    idx = np.asarray(hours, dtype=int)
    return {s: full[s][idx] for s in SERVICES}


def peak_demand_mw(country: str, services: list[str]) -> float:
    """Full-year peak MW of the combined demand of ``services``, used to cap thermal asset sizes."""
    full = year_profiles(country)
    return float(sum(full[s] for s in services).max())
