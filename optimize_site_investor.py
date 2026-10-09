"""Industrial Site Investor operating LP: one or more industrial sites meeting their own electricity,
heat, steam, cooling and hydrogen demand, trading surplus/deficit at the trained price proxy's prices."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import linopy
from economic_dispatch.config import RunConfig, discover_zones
from economic_dispatch import model as ed_model
from economic_dispatch import data_loader as ed_dl

from price_model.multivariate import predict as model_predict
from price_model.neighbors import add_neighbor_features, add_candidate_neighbor_prices, load_adjacency
from price_model import api as price_api
from site_investor_planning.config import (ASSETS, GREEN_H2, HEAT_SERVICES, SERVICES, SITE_TECH,
                                          THERMAL_ASSET_SERVICES, SiteSpec)
from site_investor_planning.demand import site_demand

HOURS_PER_DAY = 24
TOTAL_YEAR_DAYS = 364
N_MONTHS = 12
TOTAL_YEAR_HOURS = TOTAL_YEAR_DAYS * HOURS_PER_DAY

UNCERTAINTY_SCENARIOS_PATH = ROOT / "inputs" / "uncertainty_scenarios.json"
RESCALE_TARGET_MAX_PCT = 50.0


SCENARIO_OVERRIDES_ENV = "PLANNER_SCENARIO_OVERRIDES"


@lru_cache(maxsize=1)
def load_uncertainty_scenarios() -> dict:
    scenarios = json.loads(UNCERTAINTY_SCENARIOS_PATH.read_text())["scenarios"]
    override_path = os.environ.get(SCENARIO_OVERRIDES_ENV)
    if override_path:
        for name, edit in json.loads(Path(override_path).read_text())["scenarios"].items():
            if name in scenarios:
                scenarios[name].update(edit)
    return scenarios


@lru_cache(maxsize=1)
def _global_max_error_pct() -> float:
    """Largest wind/solar capacity_scale error (%) across every unc scenario/country, used
    as the rescale reference point so RESCALE_TARGET_MAX_PCT corresponds to that worst case."""
    scenarios = load_uncertainty_scenarios()
    max_err = 0.0
    for name, sc in scenarios.items():
        if name == "p100":
            continue
        for resource in ("wind", "solar"):
            for scale in sc[resource].values():
                err = (1.0 - scale) * 100.0
                if err > max_err:
                    max_err = err
    return max_err


def _rescaled_capacity_scale(scenario: str | None, country: str) -> tuple[float, float]:
    """(wind_scale, solar_scale) for this scenario/country, rescaled so the worst case across
    every unc scenario/country corresponds to RESCALE_TARGET_MAX_PCT error (not the real,
    larger error baked into the training data) -- (1.0, 1.0) if scenario is None/unknown or
    has no entry for this country (e.g. 'p100'). Only used for the candidate's OWN available
    capacity (and hence its own contribution to the price-model input); the rest of the
    system's price features keep reflecting the real, un-rescaled scenario severity, since
    those come straight from edf/hdf's real per-scenario dispatch data."""
    if not scenario:
        return 1.0, 1.0
    sc = load_uncertainty_scenarios().get(scenario)
    if sc is None:
        return 1.0, 1.0
    factor = RESCALE_TARGET_MAX_PCT / _global_max_error_pct()

    def rescale(raw_scale: float) -> float:
        err = (1.0 - raw_scale) * 100.0
        return 1.0 - (err * factor) / 100.0

    wind_scale = rescale(sc["wind"].get(country, 1.0))
    solar_scale = rescale(sc["solar"].get(country, 1.0))
    return wind_scale, solar_scale


def _month_boundaries(total_days: int = TOTAL_YEAR_DAYS, n_months: int = N_MONTHS) -> list[int]:
    """``n_months + 1`` cut points partitioning ``[0, total_days)`` into ``n_months`` chunks."""
    return [round(i * total_days / n_months) for i in range(n_months + 1)]


def representative_days(n: int, total_days: int = TOTAL_YEAR_DAYS, n_months: int = N_MONTHS,
                        ) -> tuple[list[int], list[float]]:
    """``n`` evenly-spaced representative days per month-chunk, plus each day's weight; returns ``(days, weights)``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bounds = _month_boundaries(total_days, n_months)
    days: list[int] = []
    weights: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        length = hi - lo
        if n >= length:
            days.extend(range(lo + 1, hi + 1))
            weights.extend([1.0] * length)
            continue
        picks = np.unique(np.round(np.linspace(lo, hi - 1, n)).astype(int))
        w = length / len(picks)
        days.extend(int(p) + 1 for p in picks)
        weights.extend([w] * len(picks))
    return days, weights


def first_n_days(n: int, total_days: int = TOTAL_YEAR_DAYS, n_months: int = N_MONTHS,
                 ) -> tuple[list[int], list[float]]:
    """Same weighting scheme as ``representative_days``, but picks the first ``n`` calendar days of each month-chunk."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bounds = _month_boundaries(total_days, n_months)
    days: list[int] = []
    weights: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        length = hi - lo
        if n >= length:
            days.extend(range(lo + 1, hi + 1))
            weights.extend([1.0] * length)
            continue
        w = length / n
        days.extend(range(lo + 1, lo + 1 + n))
        weights.extend([w] * n)
    return days, weights

_ELEC_ZONE_OVERRIDES = {"BE": "BE00", "NL": "NL00"}

TANK_EFFICIENCY = 0.99

DEFAULT_SITE_CAPACITIES = {"wind_mw": 10.0, "pv_mw": 10.0, "battery_mw": 5.0,
                           "heat_pump_mw": 2.0, "industrial_heat_pump_mw": 5.0,
                           "electric_boiler_mw": 5.0, "electric_chiller_mw": 2.0,
                           "electrolyser_mw": 5.0, "tank_mw": 1.0}


def _run_config(zone: str, start_day: int, end_day: int) -> RunConfig:
    return RunConfig(
        zones=[zone], start_day=start_day, end_day=end_day,
        zones_db=ROOT / "inputs" / "zones_2030.parquet",
        networks_db=ROOT / "inputs" / "networks_2030.parquet",
        g_investor_tank_efficiency=TANK_EFFICIENCY,
    )


def _donor_candidates(country: str, host_zone: str, resource_idx: int,
                      profile_info: dict, all_zones: list[str]) -> list[str]:
    """Zone(s) that could supply this resource's (wind=0/solar=1) weather profile for ``host_zone``."""
    def has_data(z):
        return profile_info.get(z, (False, 0.0, False, 0.0))[resource_idx * 2]
    if has_data(host_zone):
        return [host_zone]
    sibs = [z for z in all_zones if z[:2] == country and z != host_zone and has_data(z)]
    return sibs or [host_zone]


def _donor_zone(candidates: list[str], cap_key: str, zdata: dict) -> str:
    """Best-capacity zone among ``candidates`` (see ``_donor_candidates``)."""
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda z: zdata[z].capacities.get(cap_key, 0.0))


@lru_cache(maxsize=64)
def _zone_raw_profile(zone: str, hours_key: tuple[int, ...]):
    """Capacity- and scenario-independent part of ``sizing_and_profiles``: host-zone validation,
    wind/PV donor resolution, and the raw (un-rescaled) normalized capacity-factor profiles.
    Cached because it's invariant across every scenario/trial-point/core-point call within a
    Benders run for a given zone and representative-hour selection -- only the scenario rescale
    and the candidate's own MW sizing (applied by the caller) actually vary call to call."""
    hours = np.array(hours_key, dtype=int)
    start_day = int(hours.min()) // HOURS_PER_DAY + 1
    end_day = int(hours.max()) // HOURS_PER_DAY + 1
    cfg = _run_config(zone, start_day, end_day)
    country = zone[:2]
    host_zone = ed_model._h2_main_zones(cfg)[country]
    if host_zone != zone:
        raise ValueError(f"{zone} is not {country}'s main H2 zone (that's {host_zone}); "
                         f"pass the main H2 zone instead.")

    profile_info = ed_model._g_investor_renewable_profile_info(str(cfg.zones_db))
    all_zones = discover_zones(cfg.zones_db)

    wind_candidates = _donor_candidates(country, host_zone, 0, profile_info, all_zones)
    pv_candidates = _donor_candidates(country, host_zone, 1, profile_info, all_zones)
    needed_zones = sorted({host_zone, *wind_candidates, *pv_candidates})

    zdata = ed_dl.load_zones_from_db(needed_zones, cfg.zones_db, 0, TOTAL_YEAR_HOURS)
    wind_donor = _donor_zone(wind_candidates, "Wind (onshore) (MW)", zdata)
    pv_donor = _donor_zone(pv_candidates, "Solar (MW)", zdata)

    wind_max = max(profile_info.get(wind_donor, (False, 0.0, False, 0.0))[1], 1e-9)
    pv_max = max(profile_info.get(pv_donor, (False, 0.0, False, 0.0))[3], 1e-9)
    wind_cf_full = zdata[wind_donor].profiles["Wind_Onshore Profile"].to_numpy(dtype=float)
    pv_cf_full = zdata[pv_donor].profiles["Solar Profile"].to_numpy(dtype=float)
    wind_cf = np.clip(wind_cf_full[hours], 0.0, None)
    pv_cf = np.clip(pv_cf_full[hours], 0.0, None)
    wind_cf_norm = np.clip(wind_cf / wind_max, 0.0, 1.0)
    pv_cf_norm = np.clip(pv_cf / pv_max, 0.0, 1.0)
    return host_zone, wind_donor, pv_donor, wind_cf_norm, pv_cf_norm


def sizing_and_profiles(zone: str, hours: np.ndarray, capacities: dict, scenario: str | None = None):
    """Resolve this site's sizing (``capacities`` plus derived storage MWh) and wind/PV availability
    upper bounds for the given year-hour positions. If ``scenario`` is given, the site's OWN wind/PV
    capacity factor (and hence its own available-capacity bound and its own contribution to the price
    feature row) is derated by that scenario's country-level capacity_scale, rescaled so the worst
    case across all scenarios/countries corresponds to RESCALE_TARGET_MAX_PCT error -- NOT the real,
    larger error baked into the training data (see ``_rescaled_capacity_scale``). The rest of the
    system's price features are untouched, still reflecting the real severity."""
    start_day = int(hours.min()) // HOURS_PER_DAY + 1
    end_day = int(hours.max()) // HOURS_PER_DAY + 1
    cfg = _run_config(zone, start_day, end_day)
    country = zone[:2]
    sizing = {a: float(capacities.get(a, 0.0)) for a in ASSETS}
    sizing["battery_mwh"] = sizing["battery_mw"] * cfg.g_investor_battery_duration_hours
    sizing["tank_mwh"] = sizing["tank_mw"] * cfg.g_investor_tank_duration_hours

    hours_key = tuple(int(h) for h in hours)
    host_zone, wind_donor, pv_donor, wind_cf_norm, pv_cf_norm = _zone_raw_profile(zone, hours_key)

    wind_scale, pv_scale = _rescaled_capacity_scale(scenario, country)
    wind_cf_norm = wind_cf_norm * wind_scale
    pv_cf_norm = pv_cf_norm * pv_scale

    wind_upper = wind_cf_norm * sizing["wind_mw"]
    pv_upper = pv_cf_norm * sizing["pv_mw"]
    return cfg, sizing, host_zone, wind_donor, pv_donor, wind_upper, pv_upper, wind_cf_norm, pv_cf_norm


_BACKTEST_SCENARIO = "p100"


def _backfill_required_price_cols(df: pd.DataFrame, raw_df: pd.DataFrame, target_col: str,
                                  commodity: str) -> pd.DataFrame:
    """Ensure every ``price_<zone>`` column any trained zone model actually needs is present.

    ``add_candidate_neighbor_prices`` recomputes each zone's top-5 correlated neighbours from
    whatever (possibly one-scenario, possibly unusual) data it's given -- a scenario extreme
    enough to shift that correlation ranking can silently drop a column the trained model still
    expects (KeyError at predict time). The model's own required feature list is fixed at
    training time, so we just backfill any of ITS needed ``price_<zone>`` columns straight from
    ``raw_df`` (same scenario, so ``hour`` keys are unique) regardless of this scenario's own
    correlation structure.
    """
    bundle = price_api._bundle(commodity)
    needed_zones = {n[len("price_"):] for z in bundle["zones"] for n in bundle["zones"][z]["features"]
                    if n.startswith("price_")}
    missing = [z for z in needed_zones if f"price_{z}" not in df.columns and z in raw_df["zone"].values]
    if not missing:
        return df
    wide = raw_df[raw_df["zone"].isin(missing)].pivot_table(index="hour", columns="zone", values=target_col,
                                                            aggfunc="first")
    for z in missing:
        df[f"price_{z}"] = df["hour"].map(wide[z])
    return df


def enriched_elec_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full elec_samples.parquet enriched with neighbour/candidate-price columns, filtered to one capacity scenario."""
    ROOT_IN = ROOT / "inputs"
    ROOT_EXCHANGE = ROOT / "data_exchange" / "01_dispatch_output__train_input"
    edf = pd.read_parquet(ROOT_EXCHANGE / "elec_samples.parquet")
    if "scenario" in edf.columns:
        edf = edf[edf["scenario"] == scenario].drop(columns="scenario")
    raw_edf = edf
    eadj = load_adjacency(ROOT_IN / "elec_adjacency.json")
    edf, _ = add_neighbor_features(edf, "demand", eadj, "residual_load")
    edf, _ = add_candidate_neighbor_prices(edf, "price_eur_mwh", eadj)
    edf = _backfill_required_price_cols(edf, raw_edf, "price_eur_mwh", "electricity")
    return edf


def enriched_h2_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full h2_samples.parquet enriched with neighbour/candidate-price columns, filtered to one capacity scenario."""
    ROOT_IN = ROOT / "inputs"
    ROOT_EXCHANGE = ROOT / "data_exchange" / "01_dispatch_output__train_input"
    hdf = pd.read_parquet(ROOT_EXCHANGE / "h2_samples.parquet")
    if "scenario" in hdf.columns:
        hdf = hdf[hdf["scenario"] == scenario].drop(columns="scenario")
    raw_hdf = hdf
    hadj = load_adjacency(ROOT_IN / "h2_adjacency.json")
    hdf, _ = add_neighbor_features(hdf, "h2_demand", hadj, None)
    hdf, _ = add_candidate_neighbor_prices(hdf, "h2_price", hadj)
    hdf = _backfill_required_price_cols(hdf, raw_hdf, "h2_price", "hydrogen")
    return hdf


def _capacity_scale(values: np.ndarray, capacity_mw: np.ndarray, add_mw: float) -> np.ndarray:
    """Scale a real activity series by (scenario capacity + candidate add) / scenario capacity; no-op where capacity is 0."""
    cap = np.asarray(capacity_mw, dtype=float)
    factor = np.divide(cap + add_mw, cap, out=np.ones_like(cap), where=cap > 0)
    return values * factor


def proxy_price_series(elec_zone: str, hours: np.ndarray, h2_zone: str | None = None,
                       edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                       wind_gen_add: np.ndarray | None = None, pv_gen_add: np.ndarray | None = None,
                       battery_add: float = 0.0, electrolyser_add: float = 0.0,
                       tank_add: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Predict (elec_price, h2_price) for the given year-hour positions from the trained proxy models, optionally reflecting the candidate's own added wind/PV/battery/electrolyser/tank capacity in the feature row first."""
    if h2_zone is None:
        h2_zone = elec_zone
    hours = list(hours)
    H = len(hours)

    elec_bundle = price_api._bundle("electricity")
    h2_bundle = price_api._bundle("hydrogen")

    if edf is None:
        edf = enriched_elec_df()
    erow = edf[(edf["zone"] == elec_zone) & (edf["hour"].isin(hours))].sort_values("hour")
    efeats = elec_bundle["zones"][elec_zone]["features"]
    has_add = (wind_gen_add is not None or pv_gen_add is not None
              or battery_add or electrolyser_add or tank_add)
    if has_add:
        assert (wind_gen_add is None or len(wind_gen_add) == H) and \
              (pv_gen_add is None or len(pv_gen_add) == H), \
              f"wind_gen_add/pv_gen_add must have length {H}, matching hours"
        w_add = np.zeros(H) if wind_gen_add is None else np.asarray(wind_gen_add, dtype=float)
        s_add = np.zeros(H) if pv_gen_add is None else np.asarray(pv_gen_add, dtype=float)
        erow = erow.copy()
        if "wind" in erow.columns:
            erow["wind"] = erow["wind"].to_numpy() + w_add
        if "solar" in erow.columns:
            erow["solar"] = erow["solar"].to_numpy() + s_add
        if "residual_load" in erow.columns:
            erow["residual_load"] = erow["residual_load"].to_numpy() - w_add - s_add
        if "wind_capacity_mw" in erow.columns:
            erow["wind_capacity_mw"] = erow["wind_capacity_mw"].to_numpy() + w_add.max(initial=0.0)
        if "pv_capacity_mw" in erow.columns:
            erow["pv_capacity_mw"] = erow["pv_capacity_mw"].to_numpy() + s_add.max(initial=0.0)
        if battery_add and "battery" in erow.columns and "battery_capacity_mw" in erow.columns:
            erow["battery"] = _capacity_scale(erow["battery"].to_numpy(),
                                              erow["battery_capacity_mw"].to_numpy(), battery_add)
    p_elec = model_predict(elec_bundle, elec_zone, erow[efeats])

    if hdf is None:
        hdf = enriched_h2_df()
    hrow = hdf[(hdf["zone"] == h2_zone) & (hdf["hour"].isin(hours))].sort_values("hour")
    hfeats = h2_bundle["zones"][h2_zone]["features"]
    if has_add:
        hrow = hrow.copy()
        if "wind_capacity_mw" in hrow.columns:
            hrow["wind_capacity_mw"] = hrow["wind_capacity_mw"].to_numpy() + w_add.max(initial=0.0)
        if "pv_capacity_mw" in hrow.columns:
            hrow["pv_capacity_mw"] = hrow["pv_capacity_mw"].to_numpy() + s_add.max(initial=0.0)
        if electrolyser_add and "electrolyser_gen" in hrow.columns and "electrolyser_capacity_mw" in hrow.columns:
            hrow["electrolyser_gen"] = _capacity_scale(hrow["electrolyser_gen"].to_numpy(),
                                                        hrow["electrolyser_capacity_mw"].to_numpy(), electrolyser_add)
        if tank_add and "storage" in hrow.columns and "tank_capacity_mw" in hrow.columns:
            hrow["storage"] = _capacity_scale(hrow["storage"].to_numpy(),
                                              hrow["tank_capacity_mw"].to_numpy(), tank_add)
    p_h2 = model_predict(h2_bundle, h2_zone, hrow[hfeats])

    assert len(p_elec) == H and len(p_h2) == H, f"expected {H} hours, got {len(p_elec)}/{len(p_h2)}"
    return np.asarray(p_elec, dtype=float), np.asarray(p_h2, dtype=float)



def _capacity_bounded(m: linopy.Model, name: str, coords: list[pd.Index], upper_val,
                      return_duals: bool, cap_name: str):
    """Build a [0, ``upper_val``]-bounded variable, using a named constraint instead of a plain bound when duals are needed."""
    if not return_duals:
        return m.add_variables(lower=0.0, upper=upper_val, coords=coords, name=name)
    v = m.add_variables(lower=0.0, coords=coords, name=name)
    m.add_constraints(v <= upper_val, name=cap_name)
    return v


_THERMAL_VARS = {"heat_pump_mw": "hp", "industrial_heat_pump_mw": "ihp",
                 "electric_boiler_mw": "eb", "electric_chiller_mw": "ch"}
_THERMAL_LABELS = {"heat_pump_mw": "heat pump", "industrial_heat_pump_mw": "industrial heat pump",
                   "electric_boiler_mw": "electric boiler", "electric_chiller_mw": "electric chiller"}
_BACKUP_VARS = {**{svc: f"gas_{svc}" for svc in HEAT_SERVICES}, "cooling": "legacy_cold"}
_BALANCES = {svc: f"{svc}_balance" for svc in SERVICES}


def _sample_days(rep_days_per_month: int, day_selection: str) -> tuple[list[int], list[float]]:
    if day_selection == "first":
        return first_n_days(rep_days_per_month)
    return representative_days(rep_days_per_month)


def _solve_sites(units: dict[str, tuple[str, SiteSpec]], capacities: dict[str, dict], sites: dict[str, float],
                 days: list[int], day_weights: list[float], block_len: int,
                 return_duals: bool, edf, hdf, quiet: bool, scenario: str | None) -> dict:
    """Build and solve every unit's site LP together over ``days`` (``day_weights`` per day), cut
    into blocks of ``block_len`` hours (24 = representative days, each its own storage cycle; the
    whole horizon = one contiguous run). A unit is one site placed in one zone:
    ``units[u] = (zone, spec)``, the zone giving prices, wind/PV profiles and the scenario derating,
    the ``SiteSpec`` giving demand peaks, green hydrogen share and flexibility. ``sites[u]`` scales
    unit ``u``'s internal demand: 1 = the site is built there, 0 = not (default 1). The site's
    ``flex_fraction`` lets each hour's demand move by up to that fraction of itself, netting to zero
    over each block.

    Each demand has its own hourly balance, served by its own asset(s) and its backup (gas boiler
    for heat, legacy chiller for cooling, market imports for electricity and hydrogen). The site's
    own wind/PV exported to the grid earns Guarantees of Origin it can sell, unless that output is
    claimed for green hydrogen. The electrolyser's green load in each hour must be covered by that
    hour's own (new, hence additional) wind/PV or GOs bought from additional plants, and green
    production plus certified green H2 purchases must reach the site's ``green_share`` of its annual
    hydrogen demand.

    Cut coefficients: each asset's is the sum of the duals on its capacity constraints; the site's
    is the sensitivity to switching the site on, which scales every balance's demand, every
    flexibility band and the green hydrogen requirement. A solve HiGHS reports as failed is retried
    once with presolve off."""
    t0 = time.time()
    green = GREEN_H2
    tech = SITE_TECH
    unit_ids = list(units)
    hours = np.array([(d - 1) * HOURS_PER_DAY + h for d in days for h in range(HOURS_PER_DAY)], dtype=int)
    H = len(hours)
    n_blocks = H // block_len

    unit_idx = pd.Index(unit_ids, name="unit")
    day_idx = pd.RangeIndex(n_blocks, name="day")
    hid_idx = pd.RangeIndex(block_len, name="hid")
    coords = [unit_idx, day_idx, hid_idx]
    weight_flat = np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY)
    w = xr.DataArray(weight_flat.reshape(n_blocks, block_len), coords=[day_idx, hid_idx])

    def r2d(a):
        return np.asarray(a, dtype=float).reshape(n_blocks, block_len)

    cfgs, sizings, host_zones = {}, {}, {}
    stacks = {k: [] for k in ("wind_upper", "pv_upper", "wind_cf", "pv_cf", "p_elec", "p_h2",
                              *(f"demand_{s}" for s in _BALANCES))}
    for u in unit_ids:
        z, spec = units[u]
        (cfg, s, host_zone, wind_donor, pv_donor, wind_upper, pv_upper,
         wind_cf_norm, pv_cf_norm) = sizing_and_profiles(z, hours, capacities[u], scenario)
        cfgs[u], sizings[u], host_zones[u] = cfg, s, host_zone
        elec_zone = _ELEC_ZONE_OVERRIDES.get(z[:2], z)
        p_elec, p_h2 = proxy_price_series(elec_zone, hours, h2_zone=z, edf=edf, hdf=hdf,
                                          wind_gen_add=wind_cf_norm * s["wind_mw"],
                                          pv_gen_add=pv_cf_norm * s["pv_mw"],
                                          battery_add=s["battery_mw"], electrolyser_add=s["electrolyser_mw"],
                                          tank_add=s["tank_mw"])
        demand = site_demand(spec.peaks_mw, hours)
        for key, arr in (("wind_upper", wind_upper), ("pv_upper", pv_upper), ("wind_cf", wind_cf_norm),
                         ("pv_cf", pv_cf_norm), ("p_elec", p_elec), ("p_h2", p_h2)):
            stacks[key].append(r2d(arr))
        for svc in _BALANCES:
            stacks[f"demand_{svc}"].append(r2d(demand[svc]))
    da = {k: xr.DataArray(np.stack(v), coords=coords) for k, v in stacks.items()}
    site_da = xr.DataArray([float(sites.get(u, 1.0)) for u in unit_ids], coords=[unit_idx])
    flex_da = xr.DataArray([float(units[u][1].flex_fraction) for u in unit_ids], coords=[unit_idx])
    green_da = xr.DataArray([float(units[u][1].green_share) for u in unit_ids], coords=[unit_idx])
    any_flex = bool((flex_da > 0).any())
    any_green = bool((green_da > 0).any())

    def per_zone(key):
        return xr.DataArray([sizings[u][key] for u in unit_ids], coords=[unit_idx])

    cfg0 = cfgs[unit_ids[0]]
    ely_eff = cfg0.g_investor_electrolyser_efficiency
    batt_eff, tank_eff = cfg0.g_investor_battery_efficiency, cfg0.g_investor_tank_efficiency
    sto_cost = cfg0.storage_op_cost_eur_per_mwh
    batt_soc0 = cfg0.initial_soc_fraction * per_zone("battery_mwh")
    tank_soc0 = cfg0.initial_soc_fraction * per_zone("tank_mwh")

    m = linopy.Model()
    wind_p = _capacity_bounded(m, "wind_p", coords, da["wind_upper"], return_duals, "wind_cap")
    pv_p = _capacity_bounded(m, "pv_p", coords, da["pv_upper"], return_duals, "pv_cap")
    batt_dis = _capacity_bounded(m, "batt_dis", coords, per_zone("battery_mw"), return_duals, "batt_dis_cap")
    batt_ch = _capacity_bounded(m, "batt_ch", coords, per_zone("battery_mw"), return_duals, "batt_ch_cap")
    batt_soc = _capacity_bounded(m, "batt_soc", coords, per_zone("battery_mwh"), return_duals, "batt_soc_cap")
    tank_dis = _capacity_bounded(m, "tank_dis", coords, per_zone("tank_mw"), return_duals, "tank_dis_cap")
    tank_ch = _capacity_bounded(m, "tank_ch", coords, per_zone("tank_mw"), return_duals, "tank_ch_cap")
    tank_soc = _capacity_bounded(m, "tank_soc", coords, per_zone("tank_mwh"), return_duals, "tank_soc_cap")
    ely_p = _capacity_bounded(m, "ely_p", coords, per_zone("electrolyser_mw"), return_duals, "ely_cap")
    thermal = {(a, svc): m.add_variables(lower=0.0, coords=coords, name=f"{v}_{svc}")
               for a, v in _THERMAL_VARS.items() for svc in THERMAL_ASSET_SERVICES[a]}
    for a, v in _THERMAL_VARS.items():
        m.add_constraints(sum(thermal[a, svc] for svc in THERMAL_ASSET_SERVICES[a]) <= per_zone(a),
                          name=f"{v}_cap")
    backup = {svc: m.add_variables(lower=0.0, coords=coords, name=v) for svc, v in _BACKUP_VARS.items()}
    grid_buy = m.add_variables(lower=0.0, coords=coords, name="grid_buy")
    grid_sell = m.add_variables(lower=0.0, coords=coords, name="grid_sell")
    h2_buy = m.add_variables(lower=0.0, coords=coords, name="h2_buy")
    h2_sell = m.add_variables(lower=0.0, coords=coords, name="h2_sell")

    for name, soc, ch, dis, soc0, eff in (("batt", batt_soc, batt_ch, batt_dis, batt_soc0, batt_eff),
                                          ("tank", tank_soc, tank_ch, tank_dis, tank_soc0, tank_eff)):
        m.add_constraints(soc.isel(hid=0) - soc0 - eff * ch.isel(hid=0) + dis.isel(hid=0) == 0,
                          name=f"{name}_balance_0")
        m.add_constraints(soc.isel(hid=slice(1, None)) - soc.isel(hid=slice(None, -1))
                          - eff * ch.isel(hid=slice(1, None)) + dis.isel(hid=slice(1, None)) == 0,
                          name=f"{name}_balance")
        if cfg0.cyclic_storage:
            m.add_constraints(soc.isel(hid=-1) >= soc0, name=f"{name}_cyclic")

    shift = {}
    for svc in _BALANCES:
        if not any_flex:
            continue
        sh = m.add_variables(coords=coords, name=f"shift_{svc}")
        band = flex_da * da[f"demand_{svc}"] * site_da
        m.add_constraints(sh <= band, name=f"shift_up_{svc}")
        m.add_constraints(-1 * sh <= band, name=f"shift_down_{svc}")
        m.add_constraints(sh.sum("hid") == 0, name=f"shift_net_{svc}")
        shift[svc] = sh

    def served(svc, expr):
        """Supply side of ``svc``'s balance, net of its demand shift (balance RHS stays demand * site)."""
        return expr - shift[svc] if svc in shift else expr

    thermal_elec = sum((1.0 / tech.cop(a, svc)) * q for (a, svc), q in thermal.items())
    m.add_constraints(served("electricity", wind_p + pv_p + batt_dis - batt_ch - ely_p - thermal_elec
                                            - (1.0 / tech.legacy_chiller_cop) * backup["cooling"]
                                            + grid_buy - grid_sell)
                      == da["demand_electricity"] * site_da, name=_BALANCES["electricity"])
    m.add_constraints(served("hydrogen", ely_eff * ely_p + tank_dis - tank_ch + h2_buy - h2_sell)
                      == da["demand_hydrogen"] * site_da, name=_BALANCES["hydrogen"])
    for (a, svc), q in thermal.items():
        m.add_constraints(served(svc, q + backup[svc]) == da[f"demand_{svc}"] * site_da,
                          name=_BALANCES[svc])

    go_sell = m.add_variables(lower=0.0, coords=coords, name="go_sell")
    m.add_constraints(go_sell - grid_sell <= 0, name="go_sell_export")
    green_cost = -green.go_sell_price_eur_per_mwh * (w * go_sell).sum()
    if any_green:
        ely_green = m.add_variables(lower=0.0, coords=coords, name="ely_green")
        res_to_ely = m.add_variables(lower=0.0, coords=coords, name="res_to_ely")
        go_buy = m.add_variables(lower=0.0, coords=coords, name="go_buy")
        h2_buy_green = m.add_variables(lower=0.0, coords=coords, name="h2_buy_green")
        m.add_constraints(ely_green - ely_p <= 0, name="ely_green_cap")
        m.add_constraints(ely_green - res_to_ely - go_buy == 0, name="ely_green_matching")
        m.add_constraints(res_to_ely + go_sell - wind_p - pv_p <= 0, name="res_attribution")
        m.add_constraints(h2_buy_green - h2_buy <= 0, name="h2_buy_green_cap")
        annual_h2 = (w * da["demand_hydrogen"]).sum(["day", "hid"])
        m.add_constraints((w * (ely_eff * ely_green + h2_buy_green)).sum(["day", "hid"])
                          >= green_da * annual_h2 * site_da, name="green_h2_share")
        green_cost = (green_cost + green.go_buy_price_eur_per_mwh * (w * go_buy).sum()
                      + green.green_h2_premium_eur_per_mwh * (w * h2_buy_green).sum())

    gas_cost = tech.gas_heat_cost_eur_per_mwh_th
    gas_heat = sum(backup[svc] for svc in HEAT_SERVICES)
    cost = ((w * (da["p_elec"] + tech.grid_import_fee_eur_per_mwh) * grid_buy).sum()
            - (w * da["p_elec"] * grid_sell).sum()
            + (w * (da["p_h2"] + tech.h2_import_fee_eur_per_mwh) * h2_buy).sum()
            - (w * da["p_h2"] * h2_sell).sum()
            + gas_cost * (w * gas_heat).sum()
            + sto_cost * ((w * batt_ch).sum() + (w * batt_dis).sum() + (w * tank_ch).sum() + (w * tank_dis).sum())
            + green_cost)
    m.add_objective(cost)

    build_s = time.time() - t0
    t1 = time.time()
    solver_kwargs = {"output_flag": False} if quiet else {}
    status, condition = m.solve(solver_name="highs", **solver_kwargs)
    if status != "ok":
        status, condition = m.solve(solver_name="highs", presolve="off", **solver_kwargs)
    solve_s = time.time() - t1
    if status != "ok":
        raise RuntimeError(f"site solve failed: {status}/{condition}")

    cut_coeffs, site_coeffs = None, None
    if return_duals:
        hd = ["day", "hid"]
        dual = m.dual
        soc0_terms = {}
        for name in ("batt", "tank"):
            t = dual[f"{name}_balance_0"].sum("day")
            if cfg0.cyclic_storage:
                t = t + dual[f"{name}_cyclic"].sum("day")
            soc0_terms[name] = t
        mu = {
            "wind_mw": (dual["wind_cap"] * da["wind_cf"]).sum(hd),
            "pv_mw": (dual["pv_cap"] * da["pv_cf"]).sum(hd),
            "battery_mw": (dual["batt_dis_cap"].sum(hd) + dual["batt_ch_cap"].sum(hd)
                           + dual["batt_soc_cap"].sum(hd) * cfg0.g_investor_battery_duration_hours
                           + soc0_terms["batt"] * cfg0.initial_soc_fraction * cfg0.g_investor_battery_duration_hours),
            "tank_mw": (dual["tank_dis_cap"].sum(hd) + dual["tank_ch_cap"].sum(hd)
                        + dual["tank_soc_cap"].sum(hd) * cfg0.g_investor_tank_duration_hours
                        + soc0_terms["tank"] * cfg0.initial_soc_fraction * cfg0.g_investor_tank_duration_hours),
            "electrolyser_mw": dual["ely_cap"].sum(hd),
            **{a: dual[f"{v}_cap"].sum(hd) for a, v in _THERMAL_VARS.items()},
        }
        lam = sum((dual[con] * da[f"demand_{svc}"]).sum(hd) for svc, con in _BALANCES.items())
        for svc in shift:
            band_per_site = flex_da * da[f"demand_{svc}"]
            lam = lam + ((dual[f"shift_up_{svc}"] + dual[f"shift_down_{svc}"]) * band_per_site).sum(hd)
        if any_green:
            lam = lam + dual["green_h2_share"] * green_da * annual_h2
        cut_coeffs = {u: {a: float(mu[a].sel(unit=u)) for a in ASSETS} for u in unit_ids}
        site_coeffs = {u: float(lam.sel(unit=u)) for u in unit_ids}

    sol = m.solution
    schedules, objective_by_zone = {}, {}
    for u in unit_ids:
        def v(name):
            return np.asarray(sol[name].sel(unit=u).values).reshape(-1)
        p_elec_z = np.asarray(da["p_elec"].sel(unit=u).values).reshape(-1)
        p_h2_z = np.asarray(da["p_h2"].sel(unit=u).values).reshape(-1)
        gb, gs, hb, hs = v("grid_buy"), v("grid_sell"), v("h2_buy"), v("h2_sell")
        gas = sum(v(_BACKUP_VARS[svc]) for svc in HEAT_SERVICES)
        bc, bd, tc, td = v("batt_ch"), v("batt_dis"), v("tank_ch"), v("tank_dis")
        zero = np.zeros(H)
        gos = v("go_sell")
        gob, hbg, eg, re = ((v("go_buy"), v("h2_buy_green"), v("ely_green"), v("res_to_ely"))
                            if any_green else (zero, zero, zero, zero))
        z_cost = float((weight_flat * ((p_elec_z + tech.grid_import_fee_eur_per_mwh) * gb - p_elec_z * gs
                                       + (p_h2_z + tech.h2_import_fee_eur_per_mwh) * hb - p_h2_z * hs
                                       + gas_cost * gas + sto_cost * (bc + bd + tc + td)
                                       + green.go_buy_price_eur_per_mwh * gob
                                       - green.go_sell_price_eur_per_mwh * gos
                                       + green.green_h2_premium_eur_per_mwh * hbg)).sum())
        objective_by_zone[u] = z_cost

        site_z = float(site_da.sel(unit=u))
        out = pd.DataFrame({"hour": np.arange(H)})
        for svc in _BALANCES:
            out[f"Site {svc} demand (MW)"] = site_z * np.asarray(da[f"demand_{svc}"].sel(unit=u).values).reshape(-1)
            out[f"Site {svc} demand shift (MW)"] = v(f"shift_{svc}") if svc in shift else 0.0
        out["Site wind (MW)"] = v("wind_p")
        out["Site pv (MW)"] = v("pv_p")
        out["Site battery discharge (MW)"] = bd
        out["Site battery charge (-) (MW)"] = -bc
        out["Site grid import (MW)"] = gb
        out["Site grid export (-) (MW)"] = -gs
        out["Site electrolyser load (-) (MW)"] = -v("ely_p")
        out["Site electrolyser production (MW)"] = ely_eff * v("ely_p")
        out["Site tank discharge (MW)"] = td
        out["Site tank charge (-) (MW)"] = -tc
        out["Site H2 import (MW)"] = hb
        out["Site H2 export (-) (MW)"] = -hs
        out["Site electrolyser green load (MW)"] = eg
        out["Site green H2 produced (MW)"] = ely_eff * eg
        out["Site own wind/PV claimed for green H2 (MW)"] = re
        out["Site GOs bought (MWh/h)"] = gob
        out["Site GOs sold (MWh/h)"] = gos
        out["Site green H2 bought (MW)"] = hbg
        for (a, svc) in thermal:
            out[f"Site {_THERMAL_LABELS[a]} -> {svc} (MW)"] = v(f"{_THERMAL_VARS[a]}_{svc}")
        for svc in HEAT_SERVICES:
            out[f"Site gas boiler -> {svc} (MW)"] = v(_BACKUP_VARS[svc])
        out["Site legacy chiller -> cooling (MW)"] = v("legacy_cold")
        out["day_of_year"] = np.repeat(days, HOURS_PER_DAY)
        out["day_weight"] = weight_flat
        out.attrs.update({"objective": z_cost, "p_elec": p_elec_z, "p_h2": p_h2_z, "host_zone": host_zones[u],
                          "site_name": units[u][1].name,
                          "sizing": sizings[u], "capacities": {a: sizings[u][a] for a in ASSETS},
                          "site": site_z, "cut_coeffs": cut_coeffs[u] if cut_coeffs else None,
                          "green_h2_share": (float((weight_flat * (ely_eff * eg + hbg)).sum()
                                                   / (weight_flat * out["Site hydrogen demand (MW)"]).sum())
                                             if site_z > 0 else None),
                          "site_coeff": site_coeffs[u] if site_coeffs else None})
        schedules[u] = out

    return {
        "objective": float(m.objective.value),
        "objective_by_zone": objective_by_zone,
        "schedules": schedules,
        "cut_coeffs": cut_coeffs,
        "site_coeffs": site_coeffs,
        "build_seconds": build_s,
        "solve_seconds": solve_s,
    }


def solve(zone: str, start_day: int | None = None, end_day: int | None = None,
          rep_days_per_month: int | None = None,
          capacities: dict | None = None, return_duals: bool = False,
          edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
          day_selection: str = "first", quiet: bool = False,
          scenario: str | None = None, site: float = 1.0,
          spec: SiteSpec | None = None) -> pd.DataFrame:
    """Solve one site's LP over a contiguous day range or representative-day sample, priced by the proxy models.
    ``spec`` gives the site's demand peaks, green hydrogen share and flexibility (default: a default site).
    ``scenario``, if given, derates the site's OWN wind/PV capacity (rescaled, see
    ``sizing_and_profiles``) -- pass the same scenario name used to build ``edf``/``hdf``."""
    representative = rep_days_per_month is not None
    if representative:
        if start_day is not None or end_day is not None:
            raise ValueError("rep_days_per_month is mutually exclusive with "
                             "start_day/end_day -- representative sampling always "
                             "spans the full 364-day year.")
        days, day_weights = _sample_days(rep_days_per_month, day_selection)
        block_len = HOURS_PER_DAY
    else:
        if start_day is None:
            raise ValueError("either start_day (contiguous mode) or "
                             "rep_days_per_month (representative-sampling mode) is required.")
        if end_day is None:
            end_day = start_day
        days = list(range(start_day, end_day + 1))
        day_weights = [1.0] * len(days)
        block_len = len(days) * HOURS_PER_DAY

    capacities = capacities if capacities is not None else DEFAULT_SITE_CAPACITIES
    spec = spec if spec is not None else SiteSpec(name="Site 1")
    result = _solve_sites({zone: (zone, spec)}, {zone: capacities}, {zone: site}, days, day_weights, block_len,
                          return_duals, edf, hdf, quiet, scenario)
    out = result["schedules"][zone]
    H = len(out)
    if not representative:
        out = out.drop(columns=["day_of_year", "day_weight"])
        out.attrs = result["schedules"][zone].attrs
    out.attrs.update({"build_seconds": result["build_seconds"], "solve_seconds": result["solve_seconds"],
                      "n_hours": H, "annualized_hours": TOTAL_YEAR_HOURS if representative else H,
                      "rep_days_per_month": rep_days_per_month,
                      "sampled_days": days if representative else None})
    return out


def solve_joint(units: dict[str, tuple[str, SiteSpec]], capacities: dict[str, dict], rep_days_per_month: int,
                return_duals: bool = True,
                edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                day_selection: str = "first", quiet: bool = False,
                scenario: str | None = None, sites: dict[str, float] | None = None) -> dict:
    """Solve every unit's site LP jointly (representative-day horizon only) for a given trial capacity
    vector and site choice. ``units[u] = (zone, spec)`` places site ``spec`` in ``zone``; ``sites[u]``
    = 1 if that placement is built (default all 1). ``scenario``, if given, derates every unit's own
    wind/PV capacity (rescaled, see ``sizing_and_profiles``) -- pass the same scenario name used to
    build ``edf``/``hdf``."""
    days, day_weights = _sample_days(rep_days_per_month, day_selection)
    result = _solve_sites(units, capacities, sites or {}, days, day_weights, HOURS_PER_DAY,
                          return_duals, edf, hdf, quiet, scenario)
    for out in result["schedules"].values():
        out.attrs.update({"annualized_hours": TOTAL_YEAR_HOURS, "rep_days_per_month": rep_days_per_month,
                          "sampled_days": days})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DE00")
    ap.add_argument("--day", type=int, default=None, help="single day (shorthand for --start-day/--end-day)")
    ap.add_argument("--start-day", type=int, default=None,
                    help="contiguous-range start day, 1-364 (default 5 if no other horizon flag is given)")
    ap.add_argument("--end-day", type=int, default=None)
    ap.add_argument("--rep-days-per-month", type=int, default=None,
                    help="representative-sampling mode: N evenly-spaced days per month (1-29), "
                         "mutually exclusive with --day/--start-day/--end-day")
    args = ap.parse_args()
    if args.rep_days_per_month is not None:
        if args.day is not None or args.start_day is not None or args.end_day is not None:
            ap.error("--rep-days-per-month is mutually exclusive with --day/--start-day/--end-day")
        df = solve(args.zone, rep_days_per_month=args.rep_days_per_month)
    else:
        if args.day is not None:
            start_day, end_day = args.day, args.day
        else:
            start_day, end_day = (args.start_day or 5), (args.end_day or args.start_day or 5)
        df = solve(args.zone, start_day, end_day)
    if df.attrs["n_hours"] <= 48:
        print(df.to_string(index=False))
    else:
        print(df.describe().to_string())
    print(f"\nhours solved: {df.attrs['n_hours']} (annualized as {df.attrs['annualized_hours']}) | "
          f"build: {df.attrs['build_seconds']:.1f}s | solve: {df.attrs['solve_seconds']:.1f}s")
    print("operating cost (EUR, energy purchases + backup fuel - sales):", round(df.attrs["objective"], 2))
