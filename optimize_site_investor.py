"""Industrial Site Investor operating LP: one or more industrial sites meeting their own electricity,
heat, steam, space and process cooling and hydrogen demand, trading surplus/deficit at the trained price proxy's prices."""
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

import highspy
import linopy
from linopy.io import to_highspy
from economic_dispatch.config import RunConfig, discover_zones
from economic_dispatch import model as ed_model
from economic_dispatch import data_loader as ed_dl

from price_model.multivariate import predict as model_predict
from price_model.neighbors import add_neighbor_features, add_candidate_neighbor_prices, load_adjacency
from price_model import api as price_api
from site_investor_planning.config import (ASSETS, CAPACITY_KEYS, DEFAULT_STORAGE_HOURS, GREEN_H2, SERVICES,
                                          SITE_TECH, STORAGE_ENERGY, THERMAL_SERVICES, THERMAL_ASSET_SERVICES,
                                          SiteSpec)
from site_investor_planning.demand import site_demand

HOURS_PER_DAY = 24
TOTAL_YEAR_DAYS = 364
N_MONTHS = 12
TOTAL_YEAR_HOURS = TOTAL_YEAR_DAYS * HOURS_PER_DAY

UNCERTAINTY_SCENARIOS_PATH = ROOT / "inputs" / "uncertainty_scenarios.json"


SCENARIO_OVERRIDES_ENV = "PLANNER_SCENARIO_OVERRIDES"
BASELINE_SCENARIO = "p100"


@lru_cache(maxsize=1)
def load_uncertainty_scenarios() -> dict:
    """Scenarios from ``inputs/uncertainty_scenarios.json``, with a run's overrides applied.

    The override file (``PLANNER_SCENARIO_OVERRIDES``) may edit existing scenarios and add custom
    ones marked ``"custom": true``: a custom scenario starts from the price data of
    ``price_scenario`` (default the trained dispatch scenario nearest to its capacities, see
    ``nearest_trained_scenario``) with each country's wind and solar capacity then scaled to the
    factors it gives (see ``scale_country_vre``)."""
    scenarios = json.loads(UNCERTAINTY_SCENARIOS_PATH.read_text())["scenarios"]
    override_path = os.environ.get(SCENARIO_OVERRIDES_ENV)
    if override_path:
        for name, edit in json.loads(Path(override_path).read_text())["scenarios"].items():
            if name in scenarios:
                scenarios[name].update(edit)
            elif edit.get("custom"):
                scenarios[name] = {"price_scenario": edit.get("price_scenario") or nearest_trained_scenario(edit),
                                   **edit}
    return scenarios


@lru_cache(maxsize=1)
def trained_capacity_ratios() -> pd.DataFrame:
    """Each trained dispatch scenario's installed capacity per country relative to the Baseline's: one
    row per scenario, columns ``(kind, country)`` with ``kind`` one of ``"wind"``, ``"solar"``,
    ``"battery"``, ``"electrolyser"`` and ``"tank"``."""
    cols = {"wind_capacity_mw": "wind", "pv_capacity_mw": "solar", "battery_capacity_mw": "battery",
            "electrolyser_capacity_mw": "electrolyser", "tank_capacity_mw": "tank"}
    path = ROOT / "data_exchange" / "01_dispatch_output__train_input" / "elec_samples.parquet"
    df = pd.read_parquet(path, columns=["scenario", "zone", *cols])
    per_zone = df.groupby(["scenario", "zone"])[list(cols)].max().reset_index()
    per_zone["country"] = per_zone["zone"].str[:2]
    per_country = per_zone.groupby(["scenario", "country"])[list(cols)].sum()
    ratio = (per_country / per_country.loc[BASELINE_SCENARIO].replace(0.0, np.nan)).fillna(1.0)
    return ratio.rename(columns=cols).unstack("country")


def nearest_trained_scenario(scenario: dict) -> str:
    """The trained dispatch scenario whose country capacities are closest (least squares over every
    country) to ``scenario``: its ``wind``/``solar`` factors, a country it doesn't list counting as 1,
    and every other asset at the Baseline's capacity. Ties go to the Baseline."""
    ratios = trained_capacity_ratios()
    ratios = ratios.loc[[BASELINE_SCENARIO, *(s for s in ratios.index if s != BASELINE_SCENARIO)]]
    target = np.array([scenario.get(kind, {}).get(country, 1.0) for kind, country in ratios.columns])
    return str(ratios.index[int(np.argmin(((ratios.to_numpy() - target) ** 2).sum(axis=1)))])


def price_scenario(scenario: str) -> str:
    """The dispatch scenario whose market prices ``scenario`` uses: itself, or for a custom
    scenario its ``price_scenario``."""
    sc = load_uncertainty_scenarios().get(scenario, {})
    return sc.get("price_scenario", scenario)


def scale_country_vre(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """``df`` (price-model samples of a custom ``scenario``'s ``price_scenario``, one row per zone and
    hour) with each country's wind and solar capacity rescaled from that scenario's level to the
    custom scenario's ``wind``/``solar`` factors (1.2 = 20% more than the Baseline), together with
    the wind and solar generation they produce and the residual load and VRE totals that follow from
    it. Unchanged for a scenario that isn't custom; a country the scenario doesn't list is taken at
    the Baseline's capacity."""
    sc = load_uncertainty_scenarios().get(scenario, {})
    if not sc.get("custom"):
        return df
    source = trained_capacity_ratios().loc[sc["price_scenario"]]
    df = df.copy()
    country = df["zone"].str[:2]
    for kind, gen_col, cap_col in (("wind", "wind", "wind_capacity_mw"), ("solar", "solar", "pv_capacity_mw")):
        target = country.map(sc.get(kind, {})).fillna(1.0).astype(float)
        factor = (target / country.map(source[kind]).fillna(1.0)).to_numpy()
        if cap_col in df.columns:
            df[cap_col] = df[cap_col].to_numpy() * factor
        if gen_col in df.columns:
            extra = df[gen_col].to_numpy() * (factor - 1.0)
            df[gen_col] = df[gen_col].to_numpy() + extra
            if "vre" in df.columns:
                df["vre"] = df["vre"].to_numpy() + extra
            if "residual_load" in df.columns:
                df["residual_load"] = df["residual_load"].to_numpy() - extra
    return df


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
                           "heat_pump_mw": 2.0, "ac_chiller_mw": 2.0, "industrial_heat_pump_mw": 5.0,
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
    wind/PV donor resolution, and the raw normalized capacity-factor profiles.
    Cached because it's invariant across every scenario/trial-point/core-point call within a
    Benders run for a given zone and representative-hour selection -- only the candidate's own MW
    sizing (applied by the caller) varies call to call."""
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


def with_storage_energy(capacities: dict) -> dict:
    """``capacities`` as MW per asset plus MWh per storage asset (``CAPACITY_KEYS``); a storage
    asset whose MWh isn't given holds its MW times the duration of its default catalog product."""
    out = {a: float(capacities.get(a, 0.0)) for a in ASSETS}
    for a, e in STORAGE_ENERGY.items():
        out[e] = float(capacities[e]) if e in capacities else out[a] * DEFAULT_STORAGE_HOURS[a]
    return out


def sizing_and_profiles(zone: str, hours: np.ndarray, capacities: dict):
    """Resolve this site's sizing (``capacities`` plus derived storage MWh) and wind/PV availability
    upper bounds for the given year-hour positions. The site's own capacity factors are the same in
    every scenario; a scenario acts through its price data."""
    start_day = int(hours.min()) // HOURS_PER_DAY + 1
    end_day = int(hours.max()) // HOURS_PER_DAY + 1
    cfg = _run_config(zone, start_day, end_day)
    sizing = with_storage_energy(capacities)

    hours_key = tuple(int(h) for h in hours)
    host_zone, wind_donor, pv_donor, wind_cf_norm, pv_cf_norm = _zone_raw_profile(zone, hours_key)

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
    """Full elec_samples.parquet enriched with neighbour/candidate-price columns, filtered to one capacity
    scenario; a custom scenario uses its ``price_scenario``'s samples with its country wind and solar
    scaling applied (``scale_country_vre``)."""
    ROOT_IN = ROOT / "inputs"
    ROOT_EXCHANGE = ROOT / "data_exchange" / "01_dispatch_output__train_input"
    edf = pd.read_parquet(ROOT_EXCHANGE / "elec_samples.parquet")
    if "scenario" in edf.columns:
        edf = edf[edf["scenario"] == price_scenario(scenario)].drop(columns="scenario")
    edf = scale_country_vre(edf, scenario)
    raw_edf = edf
    eadj = load_adjacency(ROOT_IN / "elec_adjacency.json")
    edf, _ = add_neighbor_features(edf, "demand", eadj, "residual_load")
    edf, _ = add_candidate_neighbor_prices(edf, "price_eur_mwh", eadj)
    edf = _backfill_required_price_cols(edf, raw_edf, "price_eur_mwh", "electricity")
    return edf


def enriched_h2_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full h2_samples.parquet enriched with neighbour/candidate-price columns, filtered to one capacity
    scenario; a custom scenario uses its ``price_scenario``'s samples with its country wind and solar
    capacity scaled (``scale_country_vre``)."""
    ROOT_IN = ROOT / "inputs"
    ROOT_EXCHANGE = ROOT / "data_exchange" / "01_dispatch_output__train_input"
    hdf = pd.read_parquet(ROOT_EXCHANGE / "h2_samples.parquet")
    if "scenario" in hdf.columns:
        hdf = hdf[hdf["scenario"] == price_scenario(scenario)].drop(columns="scenario")
    hdf = scale_country_vre(hdf, scenario)
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



_THERMAL_VARS = {"heat_pump_mw": "hp", "ac_chiller_mw": "ac", "industrial_heat_pump_mw": "ihp",
                 "electric_boiler_mw": "eb", "electric_chiller_mw": "ch"}
_THERMAL_LABELS = {"heat_pump_mw": "heat pump", "ac_chiller_mw": "AC chiller",
                   "industrial_heat_pump_mw": "industrial heat pump",
                   "electric_boiler_mw": "electric boiler", "electric_chiller_mw": "process chiller"}
_UNMET_VARS = {svc: f"unmet_{svc}" for svc in THERMAL_SERVICES}
_HEAT_SERVICES = ["space_heat", "process_heat"]
_COOL_SERVICES = ["space_cool", "process_cool"]
_MULTI_OUTPUTS = {"chp": _HEAT_SERVICES, "sth": _HEAT_SERVICES, "hrc_heat": _HEAT_SERVICES, "abs": _COOL_SERVICES}
_MULTI_LABELS = {"chp": "CHP heat", "sth": "solar thermal", "hrc_heat": "heat-recovery chiller heat",
                 "abs": "absorption chiller"}
_BALANCES = {svc: f"{svc}_balance" for svc in SERVICES}


def _sample_days(rep_days_per_month: int, day_selection: str) -> tuple[list[int], list[float]]:
    if day_selection == "first":
        return first_n_days(rep_days_per_month)
    return representative_days(rep_days_per_month)


class SiteLP:
    """Every unit's site LP over ``days`` (``day_weights`` per day), cut into blocks of ``block_len``
    hours (24 = representative days, each its own storage cycle; the whole horizon = one contiguous
    run), built once as a HiGHS model and re-solved for any capacities and site choice. A unit is one
    site placed in one zone: ``units[u] = (zone, spec)``, the zone giving prices and wind/PV profiles,
    the ``SiteSpec`` giving demand peaks, green hydrogen share and flexibility.
    ``sites[u]`` scales unit ``u``'s internal demand: 1 = the site is built there, 0 = not (default 1).
    The site's ``flex_fraction`` of each service lets each hour's demand of that service move by up
    to that fraction of itself, netting to zero over each block.

    Each demand has its own hourly balance, served by its own asset(s). Electricity and hydrogen can
    also be bought on the market; there is no existing plant, so heat or cooling the new assets can't
    supply is left unmet and charged the unmet-demand penalty. Some assets link balances: the biogas
    CHP makes electricity plus space or process heat, solar thermal serves space or process heat,
    the heat-recovery chiller's cooling goes to process cooling and its rejected heat to space or
    process heat, the absorption chiller turns process heat into space or process cooling, and the
    thermal battery charges with electricity and discharges steam. Hot-water and chilled-water / ice
    storage shift space heating and space cooling within the day. Heat that CHP, solar thermal and
    the heat-recovery chiller can't use is released. Market hydrogen is not green: the site meets
    its ``green_share`` with Guarantees of Origin (GOs, one per MWh of renewable electricity), either
    bought or kept from its own wind/PV output. Over the year, the GOs it uses times the
    electrolyser efficiency (the hydrogen that electricity would make) must reach ``green_share`` of
    the hydrogen it keeps: bought plus made by the electrolyser minus sold. Every MWh of own wind/PV
    output earns a GO, used on site or exported; the site sells the GOs it doesn't keep, and a GO
    it keeps can't also be sold.

    Each asset's MW, each storage asset's MWh and each unit's on/off placement are columns fixed at
    the values being solved,
    so a new point only changes those bounds and the electricity and hydrogen price costs (the price
    proxy sees the site's own added capacity), and HiGHS restarts from the previous basis. The cut
    coefficients are those fixed columns' reduced costs: each asset's is the sensitivity to its MW
    (or MWh) across every constraint it bounds, the site's is the sensitivity to switching the site on, which
    scales every balance's demand, every flexibility band and the green hydrogen requirement. A solve
    HiGHS reports as failed is retried once from scratch with presolve off."""

    def __init__(self, units: dict[str, tuple[str, SiteSpec]], days: list[int], day_weights: list[float],
                 block_len: int, edf, hdf, quiet: bool, scenario: str | None):
        t0 = time.time()
        self.units, self.unit_ids, self.scenario = units, list(units), scenario
        self.days = list(days)
        self.hours = np.array([(d - 1) * HOURS_PER_DAY + h for d in days for h in range(HOURS_PER_DAY)], dtype=int)
        self.H = H = len(self.hours)
        n_blocks = H // block_len
        unit_ids = self.unit_ids
        unit_idx = pd.Index(unit_ids, name="unit")
        day_idx = pd.RangeIndex(n_blocks, name="day")
        hid_idx = pd.RangeIndex(block_len, name="hid")
        coords = [unit_idx, day_idx, hid_idx]
        self.weight_flat = np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY)
        w = xr.DataArray(self.weight_flat.reshape(n_blocks, block_len), coords=[day_idx, hid_idx])
        self.w = w
        edf = edf if edf is not None else enriched_elec_df()
        hdf = hdf if hdf is not None else enriched_h2_df()

        def r2d(a):
            return np.asarray(a, dtype=float).reshape(n_blocks, block_len)

        self.cfgs, self.host_zones, self.cf, self.frames = {}, {}, {}, {}
        stacks = {k: [] for k in ("wind_cf", "pv_cf", *(f"demand_{s}" for s in _BALANCES))}
        for u in unit_ids:
            z, spec = units[u]
            cfg, _, host_zone, _, _, _, _, wind_cf_norm, pv_cf_norm = sizing_and_profiles(z, self.hours, {})
            self.cfgs[u], self.host_zones[u] = cfg, host_zone
            self.cf[u] = (wind_cf_norm, pv_cf_norm)
            elec_zone = _ELEC_ZONE_OVERRIDES.get(z[:2], z)
            self.frames[u] = (elec_zone, z,
                              edf[(edf["zone"] == elec_zone) & (edf["hour"].isin(self.hours))],
                              hdf[(hdf["zone"] == z) & (hdf["hour"].isin(self.hours))])
            stacks["wind_cf"].append(r2d(wind_cf_norm))
            stacks["pv_cf"].append(r2d(pv_cf_norm))
            demand = site_demand(spec.peaks_mw, self.hours)
            for svc in _BALANCES:
                stacks[f"demand_{svc}"].append(r2d(demand[svc]))
        self.da = da = {k: xr.DataArray(np.stack(v), coords=coords) for k, v in stacks.items()}
        flex_da = {svc: xr.DataArray([units[u][1].flex_fraction[svc] for u in unit_ids], coords=[unit_idx])
                   for svc in _BALANCES}
        green_da = xr.DataArray([float(units[u][1].green_share) for u in unit_ids], coords=[unit_idx])
        self.any_green = any_green = bool((green_da > 0).any())

        self.cfg0 = cfg0 = self.cfgs[unit_ids[0]]
        self.ely_eff = ely_eff = cfg0.g_investor_electrolyser_efficiency
        batt_eff, tank_eff = cfg0.g_investor_battery_efficiency, cfg0.g_investor_tank_efficiency
        self.sto_cost = sto_cost = cfg0.storage_op_cost_eur_per_mwh
        soc0_frac = cfg0.initial_soc_fraction

        m = linopy.Model()
        cap = {a: m.add_variables(lower=0.0, upper=0.0, coords=[unit_idx], name=f"cap_{a}") for a in CAPACITY_KEYS}
        site_on = m.add_variables(lower=0.0, upper=0.0, coords=[unit_idx], name="site_on")

        def bounded(name, coef, a, cap_name):
            v = m.add_variables(lower=0.0, coords=coords, name=name)
            m.add_constraints(v - coef * cap[a] <= 0, name=cap_name)
            return v

        wind_p = bounded("wind_p", da["wind_cf"], "wind_mw", "wind_cap")
        pv_p = bounded("pv_p", da["pv_cf"], "pv_mw", "pv_cap")
        batt_dis = bounded("batt_dis", 1.0, "battery_mw", "batt_dis_cap")
        batt_ch = bounded("batt_ch", 1.0, "battery_mw", "batt_ch_cap")
        batt_soc = bounded("batt_soc", 1.0, "battery_mwh", "batt_soc_cap")
        tank_dis = bounded("tank_dis", 1.0, "tank_mw", "tank_dis_cap")
        tank_ch = bounded("tank_ch", 1.0, "tank_mw", "tank_ch_cap")
        tank_soc = bounded("tank_soc", 1.0, "tank_mwh", "tank_soc_cap")
        ely_p = bounded("ely_p", 1.0, "electrolyser_mw", "ely_cap")
        tech = SITE_TECH
        chp_el = bounded("chp_el", 1.0, "chp_mw", "chp_cap")
        hw_dis = bounded("hw_dis", 1.0, "hot_water_storage_mw", "hw_dis_cap")
        hw_ch = bounded("hw_ch", 1.0, "hot_water_storage_mw", "hw_ch_cap")
        hw_soc = bounded("hw_soc", 1.0, "hot_water_storage_mwh", "hw_soc_cap")
        cs_dis = bounded("cs_dis", 1.0, "cold_storage_mw", "cs_dis_cap")
        cs_ch = bounded("cs_ch", 1.0, "cold_storage_mw", "cs_ch_cap")
        cs_soc = bounded("cs_soc", 1.0, "cold_storage_mwh", "cs_soc_cap")
        tb_dis = bounded("tb_dis", 1.0, "thermal_battery_mw", "tb_dis_cap")
        tb_ch = bounded("tb_ch", 1.0, "thermal_battery_mw", "tb_ch_cap")
        tb_soc = bounded("tb_soc", 1.0, "thermal_battery_mwh", "tb_soc_cap")
        hrc_cool = bounded("hrc_cool", 1.0, "heat_recovery_chiller_mw", "hrc_cap")
        thermal = {(a, svc): m.add_variables(lower=0.0, coords=coords, name=f"{v}_{svc}")
                   for a, v in _THERMAL_VARS.items() for svc in THERMAL_ASSET_SERVICES[a]}
        for a, v in _THERMAL_VARS.items():
            m.add_constraints(sum(thermal[a, svc] for svc in THERMAL_ASSET_SERVICES[a]) - cap[a] <= 0,
                              name=f"{v}_cap")
        self.thermal_keys = list(thermal)
        unmet = {svc: m.add_variables(lower=0.0, coords=coords, name=v) for svc, v in _UNMET_VARS.items()}
        multi = {name: {svc: m.add_variables(lower=0.0, coords=coords, name=f"{name}_{svc}") for svc in svcs}
                 for name, svcs in _MULTI_OUTPUTS.items()}
        m.add_constraints(sum(multi["chp"].values())
                          - (tech.chp_heat_efficiency / tech.chp_electric_efficiency) * chp_el <= 0,
                          name="chp_heat_cap")
        m.add_constraints(sum(multi["sth"].values()) - da["pv_cf"] * cap["solar_thermal_mw"] <= 0, name="sth_cap")
        m.add_constraints(sum(multi["hrc_heat"].values()) - (1.0 + 1.0 / tech.heat_recovery_chiller_cop) * hrc_cool
                          <= 0, name="hrc_heat_cap")
        m.add_constraints(sum(multi["abs"].values()) - cap["absorption_chiller_mw"] <= 0, name="abs_cap")

        grid_buy = m.add_variables(lower=0.0, coords=coords, name="grid_buy")
        grid_sell = m.add_variables(lower=0.0, coords=coords, name="grid_sell")
        h2_buy = m.add_variables(lower=0.0, coords=coords, name="h2_buy")
        h2_sell = m.add_variables(lower=0.0, coords=coords, name="h2_sell")

        for name, soc, ch, dis, eff, e in (
                ("batt", batt_soc, batt_ch, batt_dis, batt_eff, "battery_mwh"),
                ("tank", tank_soc, tank_ch, tank_dis, tank_eff, "tank_mwh"),
                ("hw", hw_soc, hw_ch, hw_dis, tech.hot_water_storage_efficiency, "hot_water_storage_mwh"),
                ("cs", cs_soc, cs_ch, cs_dis, tech.cold_storage_efficiency, "cold_storage_mwh"),
                ("tb", tb_soc, tb_ch, tb_dis, tech.thermal_battery_efficiency, "thermal_battery_mwh")):
            m.add_constraints(soc.isel(hid=0) - soc0_frac * cap[e] - eff * ch.isel(hid=0) + dis.isel(hid=0) == 0,
                              name=f"{name}_balance_0")
            m.add_constraints(soc.isel(hid=slice(1, None)) - soc.isel(hid=slice(None, -1))
                              - eff * ch.isel(hid=slice(1, None)) + dis.isel(hid=slice(1, None)) == 0,
                              name=f"{name}_balance")
            if cfg0.cyclic_storage:
                m.add_constraints(soc.isel(hid=-1) - soc0_frac * cap[e] >= 0, name=f"{name}_cyclic")

        shift = {}
        for svc in _BALANCES:
            if not bool((flex_da[svc] > 0).any()):
                continue
            sh = m.add_variables(coords=coords, name=f"shift_{svc}")
            band = flex_da[svc] * da[f"demand_{svc}"]
            m.add_constraints(sh - band * site_on <= 0, name=f"shift_up_{svc}")
            m.add_constraints(-1 * sh - band * site_on <= 0, name=f"shift_down_{svc}")
            m.add_constraints(sh.sum("hid") == 0, name=f"shift_net_{svc}")
            shift[svc] = sh
        self.shift_svcs = list(shift)

        def served(svc, expr):
            """Supply side of ``svc``'s balance, net of its demand shift."""
            return expr - shift[svc] if svc in shift else expr

        thermal_elec = sum((1.0 / SITE_TECH.cop(a, svc)) * q for (a, svc), q in thermal.items())
        m.add_constraints(served("electricity", wind_p + pv_p + batt_dis - batt_ch - ely_p - thermal_elec
                                                + chp_el - tb_ch - (1.0 / tech.heat_recovery_chiller_cop) * hrc_cool
                                                + grid_buy - grid_sell)
                          - da["demand_electricity"] * site_on == 0, name=_BALANCES["electricity"])
        m.add_constraints(served("hydrogen", ely_eff * ely_p + tank_dis - tank_ch + h2_buy - h2_sell)
                          - da["demand_hydrogen"] * site_on == 0, name=_BALANCES["hydrogen"])
        supply = {svc: 1.0 * unmet[svc] for svc in THERMAL_SERVICES}
        for (a, svc), q in thermal.items():
            supply[svc] = supply[svc] + q
        for name in ("chp", "sth", "hrc_heat", "abs"):
            for svc, q in multi[name].items():
                supply[svc] = supply[svc] + q
        supply["space_heat"] = supply["space_heat"] + hw_dis - hw_ch
        supply["space_cool"] = supply["space_cool"] + cs_dis - cs_ch
        supply["steam"] = supply["steam"] + tb_dis
        supply["process_cool"] = supply["process_cool"] + hrc_cool
        supply["process_heat"] = supply["process_heat"] - (1.0 / tech.absorption_chiller_cop) * sum(multi["abs"].values())
        for svc in THERMAL_SERVICES:
            m.add_constraints(served(svc, supply[svc]) - da[f"demand_{svc}"] * site_on == 0, name=_BALANCES[svc])

        green = GREEN_H2
        go_sell = m.add_variables(lower=0.0, coords=coords, name="go_sell")
        green_cost = -green.go_sell_price_eur_per_mwh * (w * go_sell).sum()
        if any_green:
            go_buy = m.add_variables(lower=0.0, coords=coords, name="go_buy")
            go_keep = m.add_variables(lower=0.0, coords=coords, name="go_keep")
            m.add_constraints(go_keep + go_sell - wind_p - pv_p <= 0, name="res_attribution")
            h2_kept = (w * (h2_buy + ely_eff * ely_p - h2_sell)).sum(["day", "hid"])
            m.add_constraints(ely_eff * (w * (go_buy + go_keep)).sum(["day", "hid"])
                              - green_da * h2_kept >= 0, name="green_h2_share")
            green_cost = green_cost + green.go_buy_price_eur_per_mwh * (w * go_buy).sum()
        else:
            m.add_constraints(go_sell - wind_p - pv_p <= 0, name="res_attribution")

        unmet_total = sum(unmet[svc] for svc in THERMAL_SERVICES)
        zero_price = xr.zeros_like(da["wind_cf"])
        m.add_objective((w * zero_price * grid_buy).sum() + (w * zero_price * grid_sell).sum()
                        + (w * zero_price * h2_buy).sum() + (w * zero_price * h2_sell).sum()
                        + tech.unmet_demand_penalty_eur_per_mwh * (w * unmet_total).sum()
                        + sto_cost * sum((w * x).sum() for x in (batt_ch, batt_dis, tank_ch, tank_dis, hw_ch, hw_dis,
                                                                  cs_ch, cs_dis, tb_ch, tb_dis))
                        + (tech.biogas_price_eur_per_mwh / tech.chp_electric_efficiency) * (w * chp_el).sum()
                        + green_cost)

        M = m.matrices
        vpos = np.full(m._xCounter, -1, dtype=np.int64)
        vpos[M.vlabels] = np.arange(len(M.vlabels))
        self.var_pos = {}
        for name in m.variables:
            pos = vpos[m.variables[name].labels.values]
            self.var_pos[name] = pos
        self.cap_pos = {a: self.var_pos[f"cap_{a}"] for a in CAPACITY_KEYS}
        self.site_pos = self.var_pos["site_on"]
        self.price_pos = {k: self.var_pos[k].ravel() for k in ("grid_buy", "grid_sell", "h2_buy", "h2_sell")}
        for name in (*(f"cap_{a}" for a in CAPACITY_KEYS), "site_on", *self.price_pos):
            assert (self.var_pos[name] >= 0).all(), f"{name} is missing from the HiGHS model"
        self.h = to_highspy(m, set_names=False)
        self.h.setOptionValue("output_flag", not quiet)
        del m, M
        self.setup_seconds = time.time() - t0

    def solve(self, capacities: dict[str, dict], sites: dict[str, float], return_duals: bool = True) -> dict:
        """Solve at ``capacities[u]`` (MW per asset and MWh per storage asset, see ``with_storage_energy``)
        and ``sites[u]`` (on/off, default 1) for every unit; returns the same result dict as
        ``_solve_sites``."""
        t0 = time.time()
        tech, green = SITE_TECH, GREEN_H2
        h, unit_ids, H = self.h, self.unit_ids, self.H
        sizings, prices = {}, {}
        for u in unit_ids:
            z, _ = self.units[u]
            sizing = with_storage_energy(capacities[u])
            sizings[u] = sizing
            wind_cf_norm, pv_cf_norm = self.cf[u]
            elec_zone, h2_zone, erows, hrows = self.frames[u]
            prices[u] = proxy_price_series(elec_zone, self.hours, h2_zone=h2_zone, edf=erows, hdf=hrows,
                                           wind_gen_add=wind_cf_norm * sizing["wind_mw"],
                                           pv_gen_add=pv_cf_norm * sizing["pv_mw"],
                                           battery_add=sizing["battery_mw"],
                                           electrolyser_add=sizing["electrolyser_mw"], tank_add=sizing["tank_mw"])
        site_vals = {u: float(sites.get(u, 1.0)) for u in unit_ids}

        fixed_idx = np.concatenate([self.cap_pos[a] for a in CAPACITY_KEYS] + [self.site_pos]).astype(np.int32)
        fixed_val = np.array([sizings[u][a] for a in CAPACITY_KEYS for u in unit_ids]
                             + [site_vals[u] for u in unit_ids], dtype=float)
        h.changeColsBounds(len(fixed_idx), fixed_idx, fixed_val, fixed_val)
        wf = self.weight_flat
        cost = {"grid_buy": [wf * (prices[u][0] + tech.grid_import_fee_eur_per_mwh) for u in unit_ids],
                "grid_sell": [-wf * prices[u][0] for u in unit_ids],
                "h2_buy": [wf * (prices[u][1] + tech.h2_import_fee_eur_per_mwh) for u in unit_ids],
                "h2_sell": [-wf * prices[u][1] for u in unit_ids]}
        cost_idx = np.concatenate([self.price_pos[k] for k in cost]).astype(np.int32)
        cost_val = np.concatenate([np.concatenate(cost[k]) for k in cost])
        h.changeColsCost(len(cost_idx), cost_idx, cost_val)
        build_s = time.time() - t0

        t1 = time.time()
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            h.clearSolver()
            h.setOptionValue("presolve", "off")
            h.run()
            h.setOptionValue("presolve", "choose")
        solve_s = time.time() - t1
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"site solve failed: {h.modelStatusToString(h.getModelStatus())}")

        sol = h.getSolution()
        col = np.asarray(sol.col_value)
        cut_coeffs = site_coeffs = None
        if return_duals:
            rc = np.asarray(sol.col_dual)
            cut_coeffs = {u: {a: float(rc[self.cap_pos[a][i]]) for a in CAPACITY_KEYS} for i, u in enumerate(unit_ids)}
            site_coeffs = {u: float(rc[self.site_pos[i]]) for i, u in enumerate(unit_ids)}

        penalty = tech.unmet_demand_penalty_eur_per_mwh
        weight_flat, ely_eff, sto_cost = self.weight_flat, self.ely_eff, self.sto_cost
        schedules, objective_by_zone = {}, {}
        for i, u in enumerate(unit_ids):
            def v(name):
                return col[self.var_pos[name][i]].reshape(-1)
            p_elec_z, p_h2_z = prices[u]
            gb, gs, hb, hs = v("grid_buy"), v("grid_sell"), v("h2_buy"), v("h2_sell")
            unmet_mw = sum(v(_UNMET_VARS[svc]) for svc in THERMAL_SERVICES)
            bc, bd, tc, td = v("batt_ch"), v("batt_dis"), v("tank_ch"), v("tank_dis")
            sto_mwh = bc + bd + tc + td + sum(v(f"{k}_{d}") for k in ("hw", "cs", "tb") for d in ("ch", "dis"))
            chp = v("chp_el")
            fuel = chp / tech.chp_electric_efficiency
            zero = np.zeros(H)
            gos = v("go_sell")
            gob, gok = (v("go_buy"), v("go_keep")) if self.any_green else (zero, zero)
            z_cost = float((weight_flat * ((p_elec_z + tech.grid_import_fee_eur_per_mwh) * gb - p_elec_z * gs
                                           + (p_h2_z + tech.h2_import_fee_eur_per_mwh) * hb - p_h2_z * hs
                                           + penalty * unmet_mw + sto_cost * sto_mwh
                                           + tech.biogas_price_eur_per_mwh * fuel
                                           + green.go_buy_price_eur_per_mwh * gob
                                           - green.go_sell_price_eur_per_mwh * gos)).sum())
            objective_by_zone[u] = z_cost

            site_z = site_vals[u]
            out = pd.DataFrame({"hour": np.arange(H)})
            for svc in _BALANCES:
                out[f"Site {svc} demand (MW)"] = site_z * np.asarray(self.da[f"demand_{svc}"].sel(unit=u).values).reshape(-1)
                out[f"Site {svc} demand shift (MW)"] = v(f"shift_{svc}") if svc in self.shift_svcs else 0.0
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
            out["Site GOs bought (MWh/h)"] = gob
            out["Site own wind/PV GOs kept (MWh/h)"] = gok
            out["Site GOs sold (MWh/h)"] = gos
            out["Site green H2 covered by GOs (MW)"] = ely_eff * (gob + gok)
            for (a, svc) in self.thermal_keys:
                q = v(f"{_THERMAL_VARS[a]}_{svc}")
                out[f"Site {_THERMAL_LABELS[a]} -> {svc} (MW)"] = q
                out[f"Site {_THERMAL_LABELS[a]} electricity (-) (MW)"] = -q / tech.cop(a, svc)
            out["Site CHP electricity (MW)"] = chp
            out["Site CHP biogas (MW)"] = fuel
            for name, svcs in _MULTI_OUTPUTS.items():
                for svc in svcs:
                    out[f"Site {_MULTI_LABELS[name]} -> {svc} (MW)"] = v(f"{name}_{svc}")
            hrc = v("hrc_cool")
            out["Site heat-recovery chiller -> process_cool (MW)"] = hrc
            out["Site heat-recovery chiller electricity (-) (MW)"] = -hrc / tech.heat_recovery_chiller_cop
            out["Site absorption chiller heat (-) (MW)"] = -sum(v(f"abs_{svc}") for svc in _COOL_SERVICES) \
                / tech.absorption_chiller_cop
            for k, label in (("hw", "hot-water storage"), ("cs", "cold storage"), ("tb", "thermal battery")):
                out[f"Site {label} discharge (MW)"] = v(f"{k}_dis")
                out[f"Site {label} charge (-) (MW)"] = -v(f"{k}_ch")
            for svc in THERMAL_SERVICES:
                out[f"Site unmet {svc} demand (MW)"] = v(_UNMET_VARS[svc])
            out["day_of_year"] = np.repeat(self.days, HOURS_PER_DAY)
            out["day_weight"] = weight_flat
            out.attrs.update({"objective": z_cost, "p_elec": p_elec_z, "p_h2": p_h2_z,
                              "host_zone": self.host_zones[u], "site_name": self.units[u][1].name,
                              "sizing": sizings[u], "capacities": {a: sizings[u][a] for a in CAPACITY_KEYS},
                              "site": site_z, "cut_coeffs": cut_coeffs[u] if cut_coeffs else None,
                              "green_h2_share": (float((weight_flat * ely_eff * (gob + gok)).sum()
                                                       / (weight_flat * (hb + ely_eff * v("ely_p") - hs)).sum())
                                                 if site_z > 0 and (weight_flat * (hb + ely_eff * v("ely_p") - hs)).sum() > 0
                                                 else None),
                              "site_coeff": site_coeffs[u] if site_coeffs else None})
            schedules[u] = out

        return {
            "objective": float(h.getInfo().objective_function_value),
            "objective_by_zone": objective_by_zone,
            "schedules": schedules,
            "cut_coeffs": cut_coeffs,
            "site_coeffs": site_coeffs,
            "build_seconds": build_s,
            "solve_seconds": solve_s,
        }


def _solve_sites(units: dict[str, tuple[str, SiteSpec]], capacities: dict[str, dict], sites: dict[str, float],
                 days: list[int], day_weights: list[float], block_len: int,
                 return_duals: bool, edf, hdf, quiet: bool, scenario: str | None) -> dict:
    """Build every unit's site LP (see ``SiteLP``) and solve it once at ``capacities`` and ``sites``."""
    t0 = time.time()
    lp = SiteLP(units, days, day_weights, block_len, edf, hdf, quiet, scenario)
    result = lp.solve(capacities, sites, return_duals)
    result["build_seconds"] = time.time() - t0 - result["solve_seconds"]
    return result


_PRICE_FRAMES: dict = {}
_SITE_LPS: dict = {}


def set_price_frames(frames: dict) -> None:
    """Keep each scenario's ``(edf, hdf)`` price frames in this process for ``solve_unit``; used as a
    worker initializer so the frames are sent once per worker instead of with every job."""
    _PRICE_FRAMES.clear()
    _PRICE_FRAMES.update(frames)


def solve_unit(unit: str, entry: tuple[str, SiteSpec], capacities: dict, site: float, rep_days_per_month: int,
               scenario: str, quiet: bool = False, day_selection: str = "first") -> dict:
    """Solve one unit's site LP for ``scenario`` at ``capacities`` and on/off ``site``, reusing the
    ``SiteLP`` this process already built for that unit and scenario (built on first use with the
    frames from ``set_price_frames``). Same result dict as ``solve_joint``."""
    key = (unit, scenario, rep_days_per_month, day_selection)
    lp = _SITE_LPS.get(key)
    setup_s = 0.0
    if lp is None:
        days, day_weights = _sample_days(rep_days_per_month, day_selection)
        edf, hdf = _PRICE_FRAMES[scenario]
        lp = _SITE_LPS[key] = SiteLP({unit: entry}, days, day_weights, HOURS_PER_DAY, edf, hdf, quiet, scenario)
        setup_s = lp.setup_seconds
    result = lp.solve({unit: capacities}, {unit: site})
    result["build_seconds"] += setup_s
    for out in result["schedules"].values():
        out.attrs.update({"annualized_hours": TOTAL_YEAR_HOURS, "rep_days_per_month": rep_days_per_month,
                          "sampled_days": lp.days})
    return result


def solve(zone: str, start_day: int | None = None, end_day: int | None = None,
          rep_days_per_month: int | None = None,
          capacities: dict | None = None, return_duals: bool = False,
          edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
          day_selection: str = "first", quiet: bool = False,
          scenario: str | None = None, site: float = 1.0,
          spec: SiteSpec | None = None) -> pd.DataFrame:
    """Solve one site's LP over a contiguous day range or representative-day sample, priced by the proxy models.
    ``spec`` gives the site's demand peaks, green hydrogen share and flexibility (default: a default site).
    ``scenario`` names the scenario ``edf``/``hdf`` were built for."""
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
    = 1 if that placement is built (default all 1). ``scenario`` names the scenario ``edf``/``hdf``
    were built for."""
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
    print("operating cost (EUR, energy purchases + unmet demand penalty - sales):", round(df.attrs["objective"], 2))
