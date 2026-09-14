"""Standalone Hydrogen Producer optimization for one zone/day-range, priced by our
trained proxy.

Reuses Project 3's own Hydrogen Producer physics -- sizing and internal
constraints, ``economic_dispatch/model.py::_build_h2_producer`` (vendored locally,
copied from Project 3) -- but does
NOT run Project 3's full joint network LP. Instead of coupling
``prod_grid_net``/``prod_h2_net`` into a real multi-zone balance, this treats the
host zone as an external price-taker market: the cost of every MWh imported
(or revenue of every MWh exported) is priced using OUR trained price_model
proxy (``electricity_price``/``hydrogen_price``), evaluated at that zone's own
REAL historical hourly conditions (demand, weather, and -- since this is a
backtest against already-realized history, not a forward what-if -- real
historical neighbour prices too). See Formulation.md SS2 for the full math.

Solves any (start_day, end_day) range for one zone -- a single day, or the full
364-day/8736-hour year (``--start-day 1 --end-day 364``), which scopes storage
cyclic-closure, downstream-demand conservation, and the RED III quota to the
same horizon Project 3's own full-year joint solve uses, removing the
single-day horizon-scope artifact documented in Formulation.md SS2.5. Training
always stays on the full year / all CORE zones regardless (Formulation.md SS1)
-- only solving is ever scoped down.

Usage:
    python optimize_h2_producer.py --zone DE00 --day 5                    # one day
    python optimize_h2_producer.py --zone DE00 --start-day 1 --end-day 364  # full year
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import linopy
from economic_dispatch.config import RunConfig, discover_zones
from economic_dispatch import model as p3model
from economic_dispatch import data_loader as p3dl

import day_sampling
from price_model.multivariate import predict as model_predict
from price_model.neighbors import add_neighbor_features, add_candidate_neighbor_prices, load_adjacency
from price_model.extract import _read_balance_csv, DEFAULT_ELEC_CSV, DEFAULT_H2_CSV
from price_model import api as price_api

HOURS_PER_DAY = 24
TOTAL_YEAR_DAYS = day_sampling.TOTAL_DAYS
TOTAL_YEAR_HOURS = TOTAL_YEAR_DAYS * HOURS_PER_DAY

_ELEC_PRODUCER_COLS = ["H2 Producer wind (MW)", "H2 Producer pv (MW)",
                      "H2 Producer battery discharge (MW)", "H2 Producer battery charge (-) (MW)",
                      "H2 Producer electrolyser load (-) (MW)", "H2 Producer grid exchange (MW)"]
_H2_PRODUCER_COLS = ["H2 Producer electrolyser production (MW)", "H2 Producer tank discharge (MW)",
                    "H2 Producer tank charge (-) (MW)", "H2 Producer downstream demand (-) (MW)",
                    "H2 Producer pipeline exchange (MW)"]


def load_actual_schedule(zone: str, start_day: int, end_day: int) -> pd.DataFrame:
    """Actual H2 Producer schedule for this zone/day-range straight from Project 3's
    real full-year joint solve (``inputs/hourly_balance_{elec,h2}.csv``'s
    "H2 Producer *" columns)."""
    hours = list(range((start_day - 1) * HOURS_PER_DAY, end_day * HOURS_PER_DAY))
    ezones, ecats, evals = _read_balance_csv(DEFAULT_ELEC_CSV)
    hzones, hcats, hvals = _read_balance_csv(DEFAULT_H2_CSV)

    def col(zones, cats, vals, cat):
        idx = [i for i, (z, c) in enumerate(zip(zones, cats)) if z == zone and c == cat]
        if not idx:
            return np.full(len(hours), np.nan)
        return vals[hours, idx[0]]

    data = {"hour": list(range(len(hours)))}
    for c in _ELEC_PRODUCER_COLS:
        data[c] = col(ezones, ecats, evals, c)
    for c in _H2_PRODUCER_COLS:
        data[c] = col(hzones, hcats, hvals, c)
    return pd.DataFrame(data)


_ELEC_ZONE_OVERRIDES = {"BE": "BE00", "NL": "NL00"}

_CAPACITY_OVERRIDE_FIELDS = {
    "electrolyser_mw": "h2_producer_electrolyser_mw_overrides",
    "wind_mw": "h2_producer_wind_mw_overrides",
    "pv_mw": "h2_producer_pv_mw_overrides",
    "battery_mw": "h2_producer_battery_mw_overrides",
    "tank_mw": "h2_producer_tank_mw_overrides",
}

TANK_EFFICIENCY = 0.99

REN_SELF_USE_PRIORITY_EUR_PER_MWH = 0.001


def _run_config(zone: str, start_day: int, end_day: int, capacities: dict | None = None) -> RunConfig:
    overrides = {}
    if capacities:
        country = zone[:2]
        for key, field in _CAPACITY_OVERRIDE_FIELDS.items():
            if key in capacities:
                overrides[field] = {country: float(capacities[key])}
    return RunConfig(
        zones=[zone], start_day=start_day, end_day=end_day,
        zones_db=ROOT / "inputs" / "zones_2030.parquet",
        networks_db=ROOT / "inputs" / "networks_2030.parquet",
        enable_h2_producer=True,
        h2_producer_tank_efficiency=TANK_EFFICIENCY,
        **overrides,
    )


def _donor_candidates(country: str, host_zone: str, resource_idx: int,
                      profile_info: dict, all_zones: list[str]) -> list[str]:
    """Zone(s) that could supply this resource's (wind=0/solar=1) weather profile for
    ``host_zone``: just ``[host_zone]`` if its own profile is real, otherwise every
    same-country sibling with real profile data (e.g. BEOF/LUB1/NLLL H2-hub nodes have
    none of their own -- see ``_h2_producer_renewable_profile_info``), or ``[host_zone]``
    again if no sibling has any either. Doesn't need capacity data (``zdata``) -- that's
    only needed to RANK multiple candidates, done separately by ``_donor_zone`` once
    every candidate this function returns has actually been loaded (see
    ``sizing_and_profiles``, which is why this is split out: the old single-function
    version tried to rank before loading, and crashed with a bare ``KeyError`` for any
    country with more than one real candidate for a resource its host zone lacks)."""
    def has_data(z):
        return profile_info.get(z, (False, 0.0, False, 0.0))[resource_idx * 2]
    if has_data(host_zone):
        return [host_zone]
    sibs = [z for z in all_zones if z[:2] == country and z != host_zone and has_data(z)]
    return sibs or [host_zone]


def _donor_zone(candidates: list[str], cap_key: str, zdata: dict) -> str:
    """Best-capacity zone among ``candidates`` (see ``_donor_candidates``) -- mirrors
    model.py::_build_h2_producer's nested ``_donor_zone`` (not importable directly
    since it's a closure). ``zdata`` must already cover every zone in ``candidates``."""
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda z: zdata[z].capacities.get(cap_key, 0.0))


def sizing_and_profiles(zone: str, hours: np.ndarray, capacities: dict | None = None):
    """Resolve this country's Producer sizing + wind/PV availability upper bounds for
    the given 0-indexed YEAR-hour positions ``hours`` (0..8735 -- need not be
    contiguous; see ``day_sampling.representative_days``). Sizing/donor-zone selection
    is entirely independent of which hours are requested (``_h2_producer_sizing``/
    ``_h2_main_zones`` always rank against the FULL stored year regardless of any day
    selection, see their own docstrings) -- only the returned ``wind_upper``/
    ``pv_upper`` arrays actually depend on ``hours``. ``capacities``, if given,
    overrides the rank-derived sizing for this one country (see ``_run_config``) --
    ``sizing`` (and therefore ``wind_upper``/``pv_upper``, which scale off it) reflects
    the override transparently."""
    start_day = int(hours.min()) // HOURS_PER_DAY + 1
    end_day = int(hours.max()) // HOURS_PER_DAY + 1
    cfg = _run_config(zone, start_day, end_day, capacities)
    country = zone[:2]
    sizing = p3model._h2_producer_sizing(cfg)[country]
    host_zone = p3model._h2_main_zones(cfg)[country]
    if host_zone != zone:
        raise ValueError(f"{zone} is not {country}'s main H2 zone (that's {host_zone}); "
                         f"pass the main H2 zone instead.")

    profile_info = p3model._h2_producer_renewable_profile_info(str(cfg.zones_db))
    all_zones = discover_zones(cfg.zones_db)

    wind_candidates = _donor_candidates(country, host_zone, 0, profile_info, all_zones)
    pv_candidates = _donor_candidates(country, host_zone, 1, profile_info, all_zones)
    needed_zones = sorted({host_zone, *wind_candidates, *pv_candidates})

    zdata = p3dl.load_zones_from_db(needed_zones, cfg.zones_db, 0, TOTAL_YEAR_HOURS)
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

    wind_upper = wind_cf_norm * sizing["wind_mw"]
    pv_upper = pv_cf_norm * sizing["pv_mw"]
    return cfg, sizing, host_zone, wind_donor, pv_donor, wind_upper, pv_upper, wind_cf_norm, pv_cf_norm


_BACKTEST_SCENARIO = "p100"  # the "as-planned NT2030" capacity scenario -- see
# run_capacity_scenarios.py. elec_samples.parquet/h2_samples.parquet are now pooled
# across all capacity-uncertainty scenarios (one row per zone/hour/scenario, added
# 2026-09-14) for training, so a real single-history backtest here must filter down to
# one scenario's rows first -- otherwise every (zone, hour) lookup below would silently
# return every scenario's row stacked together instead of the single real value a
# backtest needs. p100 is the right choice: it's the un-shortfalled baseline, the
# closest of the 4 to "history as it actually happened".


def enriched_elec_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full elec_samples.parquet enriched with neighbour/candidate-price columns --
    the expensive step in proxy_price_series, cached here so a multi-zone caller
    (e.g. exporting every country) only pays it once instead of once per zone.
    ``scenario`` selects which capacity-uncertainty scenario's rows to use (default
    the "as-planned" baseline); ignored if the samples parquet predates capacity
    scenarios (no ``scenario`` column at all)."""
    ROOT_IN = ROOT / "inputs"
    edf = pd.read_parquet(ROOT_IN / "elec_samples.parquet")
    if "scenario" in edf.columns:
        edf = edf[edf["scenario"] == scenario].drop(columns="scenario")
    eadj = load_adjacency(ROOT_IN / "elec_adjacency.json")
    edf, _ = add_neighbor_features(edf, "demand", eadj, "residual_load")
    edf, _ = add_candidate_neighbor_prices(edf, "price_eur_mwh", eadj)
    return edf


def enriched_h2_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full h2_samples.parquet enriched with neighbour/candidate-price columns --
    see enriched_elec_df. ``scenario`` as there."""
    ROOT_IN = ROOT / "inputs"
    hdf = pd.read_parquet(ROOT_IN / "h2_samples.parquet")
    if "scenario" in hdf.columns:
        hdf = hdf[hdf["scenario"] == scenario].drop(columns="scenario")
    hadj = load_adjacency(ROOT_IN / "h2_adjacency.json")
    hdf, _ = add_neighbor_features(hdf, "h2_demand", hadj, None)
    hdf, _ = add_candidate_neighbor_prices(hdf, "h2_price", hadj)
    return hdf


def proxy_price_series(elec_zone: str, hours: np.ndarray, h2_zone: str | None = None,
                       edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """The requested 0-indexed YEAR-hour positions' ``hours`` (elec_price, h2_price) as
    predicted by OUR trained proxy, fed real historical feature values (including real
    neighbour prices -- legitimate here since this is a backtest over already-realized
    history, not a forward what-if scenario -- see the price-taker/circularity
    discussion in Formulation.md SS2). ``h2_zone`` defaults to ``elec_zone`` (true for
    11 of 13 countries); pass it explicitly for the 2 whose H2-hub zone has no
    electricity demand of its own and so no trained electricity model (Belgium: BE00
    elec / BEOF H2; Netherlands: NL00 elec / NLLL H2). Pass ``edf``/``hdf`` (from
    enriched_elec_df/enriched_h2_df) to reuse already-enriched frames across many zones
    instead of re-enriching per call.

    ``hours`` need not be contiguous -- filtered via ``.isin`` either way, and
    ``sort_values("hour")`` afterward keeps the result in ascending-hour order, which
    for the representative-day caller (day_sampling.representative_days) is already
    identical to (day, hour-in-day) order since each day's 24 hours occupy their own
    non-overlapping block and days are already ascending."""
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
    p_elec = model_predict(elec_bundle, elec_zone, erow[efeats])

    if hdf is None:
        hdf = enriched_h2_df()
    hrow = hdf[(hdf["zone"] == h2_zone) & (hdf["hour"].isin(hours))].sort_values("hour")
    hfeats = h2_bundle["zones"][h2_zone]["features"]
    p_h2 = model_predict(h2_bundle, h2_zone, hrow[hfeats])

    assert len(p_elec) == H and len(p_h2) == H, f"expected {H} hours, got {len(p_elec)}/{len(p_h2)}"
    return np.asarray(p_elec, dtype=float), np.asarray(p_h2, dtype=float)


def _capacity_bounded(m: linopy.Model, name: str, coords: list[pd.Index], upper_val,
                      return_duals: bool, cap_name: str):
    """Build a [0, ``upper_val``]-bounded variable over ``coords`` (a single ``[hour]``
    index for the contiguous path, or ``[day, hid]`` for the representative-day path --
    linopy/xarray coords is already a list, so either shape works unchanged). When
    ``return_duals`` is False (the default, standalone-CLI path) the upper bound is a
    plain variable bound, byte-identical to this module's original behavior. When True
    (the Benders planning subproblem path, ``h2_planning``), the bound instead becomes
    an explicit NAMED constraint (``cap_name``) so its dual -- the marginal value of
    relaxing that capacity by 1 MW -- can be read back afterwards via
    ``m.dual[cap_name]``. See Formulation.md SS4.5 for how these duals become Benders
    optimality-cut coefficients."""
    if not return_duals:
        return m.add_variables(lower=0.0, upper=upper_val, coords=coords, name=name)
    v = m.add_variables(lower=0.0, coords=coords, name=name)
    m.add_constraints(v <= upper_val, name=cap_name)
    return v


def solve(zone: str, start_day: int | None = None, end_day: int | None = None,
         rep_days_per_month: int | None = None, fix_storage: bool = False,
         capacities: dict | None = None, return_duals: bool = False,
         downstream_load_mw: float | None = None, downstream_load_flex_pct: float | None = None,
         grid_cap_mw: float | None = None, h2_cap_mw: float | None = None,
         edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
         quiet: bool = False) -> pd.DataFrame:
    """Two mutually exclusive ways to pick the solved horizon:

    * ``start_day``/``end_day`` (the original mode) -- a single CONTIGUOUS day range,
      solved as one continuous storage-arbitrage chain (one cyclic closure at the very
      end, if ``cfg.cyclic_storage``). A single day or the full 364-day year
      (``start_day=1, end_day=364``).
    * ``rep_days_per_month`` (1..29) -- ``n`` evenly-spaced REPRESENTATIVE days per
      month sampled across the full year (``day_sampling.representative_days``, see
      Formulation.md SS2.6), each carrying a WEIGHT (how many real days it stands in
      for). Every hour-indexed SUM that's meant to represent an annual total --
      ``demand_conservation``, the RED III quota, and the objective's operating cost --
      is weighted by that hour's day-weight so it approximates the same full-year
      quantity a contiguous ``start_day=1, end_day=364`` solve would produce, just from
      far fewer hours (speed/accuracy trade-off: 1 rep-day/month = 12 solved days, ~30x
      fewer hours than the full year). Storage is treated as an INDEPENDENT 24h block
      per representative day (its own cyclic closure, if any, back to the same starting
      SoC) rather than one continuous chain -- there's no principled way to say what a
      battery's state of charge should be jumping from, say, day 12 to day 43 when the
      days in between were never solved, so no arbitrage value is assumed to carry
      between non-adjacent sampled days (the standard assumption in representative-day
      energy models). Mutually exclusive with ``fix_storage`` (that diagnostic needs a
      real contiguous historical trajectory to replay). Used both by the standalone CLI
      and, optionally, by the Benders capacity-planning subproblems
      (``plan_h2_capacity.py --rep-days-per-month``) as a faster approximate subproblem.

    ``fix_storage=True`` fixes battery and H2-tank charge/discharge to their ACTUAL
    historical values (Project 3's real full-year joint solve) instead of leaving them
    as free decision variables -- everything else (electrolyser, grid/pipeline exchange,
    downstream demand) still re-optimizes against the proxy price signal. Diagnostic:
    isolates whether storage-arbitrage error (the weakest-fitting variables, see
    Formulation.md SS3.4) is bleeding into grid/pipeline exchange and electrolyser
    dispatch through the balance equations, versus those being wrong on their own.

    ``capacities`` (optional, see ``_run_config``) overrides this country's rank-derived
    asset sizing -- used by the Benders capacity-planning loop (``h2_planning``) to
    price one trial capacity vector. ``return_duals=True`` additionally builds every
    capacity-linked bound as a named constraint (``_capacity_bounded``) and, after
    solving, stashes per-asset Benders optimality-cut coefficients in
    ``out.attrs["cut_coeffs"]`` (Formulation.md SS4.5); incompatible with
    ``fix_storage=True`` (storage bounds aren't capacity-linked under that diagnostic).
    ``edf``/``hdf`` are forwarded to ``proxy_price_series`` to reuse an
    already-enriched frame across many calls (see ``enriched_elec_df``/
    ``enriched_h2_df``) instead of re-enriching the full sample parquets every time --
    essential for a Benders loop that calls ``solve()`` many times. ``quiet=True``
    silences HiGHS's per-solve console output (``output_flag=False``), useful for the
    same many-iterations reason.

    ``downstream_load_mw`` (optional) decouples the H2 Producer's downstream demand
    baseline from electrolyser capacity: instead of the original
    ``h2_producer_downstream_demand_pct_of_electrolyser_capacity * (ely_mw * ely_eff)``
    (a fixed PERCENTAGE of whatever electrolyser capacity happens to be selected), the
    baseline becomes this exact FIXED MW figure regardless of ``ely_mw``/``capacities``
    -- e.g. "every country's downstream load is 10 MW" independent of how big its
    electrolyser is. ``downstream_load_flex_pct`` sets demand flexibility as a percentage
    of THAT fixed baseline (default: reuses ``cfg.h2_producer_demand_flex_pct``) instead
    of a percentage of electrolyser capacity. Since baseline/flex no longer depend on
    ``ely_mw`` at all under this mode, the Benders electrolyser cut coefficient's
    cross-coupling terms (``beta``/``phi`` below) correctly drop to zero -- see the
    ``return_duals`` block's comment. Typically paired with a MASTER-level "must install
    at least the downstream load" constraint (``h2_planning/master.py::build_master``'s
    ``downstream_load_mw`` parameter) so the electrolyser can't be sized below what it's
    being asked to serve.

    With every new argument left at its default (``start_day`` given, everything else
    default), this reproduces the module's original behavior exactly (same bound-mode
    variables, same fresh-enrichment ``proxy_price_series`` call) -- Formulation.md
    SS3's committed validation numbers stay reproducible byte-for-byte."""
    if fix_storage and return_duals:
        raise ValueError("return_duals=True is incompatible with fix_storage=True: "
                         "storage bounds are fixed-to-actual under fix_storage, not "
                         "capacity-linked, so no battery/tank cut coefficients exist.")
    representative = rep_days_per_month is not None
    if representative:
        if start_day is not None or end_day is not None:
            raise ValueError("rep_days_per_month is mutually exclusive with "
                             "start_day/end_day -- representative sampling always "
                             "spans the full 364-day year.")
        if fix_storage:
            raise ValueError("fix_storage=True needs a real contiguous historical "
                             "trajectory to replay; incompatible with "
                             "rep_days_per_month (independent-per-day storage).")
        days, day_weights = day_sampling.representative_days(rep_days_per_month)
    else:
        if start_day is None:
            raise ValueError("either start_day (contiguous mode) or "
                             "rep_days_per_month (representative-sampling mode) is required.")
        if end_day is None:
            end_day = start_day
        days = list(range(start_day, end_day + 1))
        day_weights = [1.0] * len(days)

    n_days = len(days)
    hours = np.array([(d - 1) * HOURS_PER_DAY + h for d in days for h in range(HOURS_PER_DAY)], dtype=int)
    H = len(hours)

    t0 = time.time()
    (cfg, s, host_zone, wind_donor, pv_donor, wind_upper, pv_upper,
    wind_cf_norm, pv_cf_norm) = sizing_and_profiles(zone, hours, capacities)
    elec_zone = _ELEC_ZONE_OVERRIDES.get(zone[:2], zone)
    p_elec, p_h2 = proxy_price_series(elec_zone, hours, h2_zone=zone, edf=edf, hdf=hdf)

    ely_mw, batt_mw, batt_mwh = s["electrolyser_mw"], s["battery_mw"], s["battery_mwh"]
    tank_mw, tank_mwh = s["tank_mw"], s["tank_mwh"]
    ely_eff = cfg.h2_producer_electrolyser_efficiency
    capacity_h2 = ely_mw * ely_eff
    if downstream_load_mw is not None:
        flex_pct = (cfg.h2_producer_demand_flex_pct if downstream_load_flex_pct is None
                   else downstream_load_flex_pct)
        baseline = downstream_load_mw
        flex = flex_pct * downstream_load_mw
    else:
        baseline = cfg.h2_producer_downstream_demand_pct_of_electrolyser_capacity * capacity_h2
        flex = cfg.h2_producer_demand_flex_pct * capacity_h2
    grid_cap = grid_cap_mw if grid_cap_mw is not None else cfg.h2_producer_grid_connection_mw
    h2_cap = h2_cap_mw if h2_cap_mw is not None else cfg.h2_producer_h2_connection_mw
    batt_eff, tank_eff = cfg.h2_producer_battery_efficiency, cfg.h2_producer_tank_efficiency
    quota = max(cfg.h2_producer_renewable_h2_quota, 0.0)
    gc_price = cfg.h2_producer_gc_price_eur_per_mwh
    sto_cost = cfg.storage_op_cost_eur_per_mwh
    batt_soc0 = cfg.initial_soc_fraction * batt_mwh
    tank_soc0 = cfg.initial_soc_fraction * tank_mwh
    annual_hours = TOTAL_YEAR_HOURS if representative else H

    if fix_storage:
        act = load_actual_schedule(zone, start_day, end_day)
        fixed_batt_dis = np.nan_to_num(act["H2 Producer battery discharge (MW)"].to_numpy(), nan=0.0)
        fixed_batt_ch = np.nan_to_num(-act["H2 Producer battery charge (-) (MW)"].to_numpy(), nan=0.0)
        fixed_tank_dis = np.nan_to_num(act["H2 Producer tank discharge (MW)"].to_numpy(), nan=0.0)
        fixed_tank_ch = np.nan_to_num(-act["H2 Producer tank charge (-) (MW)"].to_numpy(), nan=0.0)

    m = linopy.Model()

    if not representative:
        hour_idx = pd.RangeIndex(H, name="hour")
        coords = [hour_idx]
        wind_p = _capacity_bounded(m, "wind_p", coords, wind_upper, return_duals, "wind_cap")
        pv_p = _capacity_bounded(m, "pv_p", coords, pv_upper, return_duals, "pv_cap")
        if fix_storage:
            batt_dis = m.add_variables(lower=fixed_batt_dis, upper=fixed_batt_dis, coords=coords, name="batt_dis")
            batt_ch = m.add_variables(lower=fixed_batt_ch, upper=fixed_batt_ch, coords=coords, name="batt_ch")
            tank_dis = m.add_variables(lower=fixed_tank_dis, upper=fixed_tank_dis, coords=coords, name="tank_dis")
            tank_ch = m.add_variables(lower=fixed_tank_ch, upper=fixed_tank_ch, coords=coords, name="tank_ch")
        else:
            batt_dis = _capacity_bounded(m, "batt_dis", coords, batt_mw, return_duals, "batt_dis_cap")
            batt_ch = _capacity_bounded(m, "batt_ch", coords, batt_mw, return_duals, "batt_ch_cap")
            tank_dis = _capacity_bounded(m, "tank_dis", coords, tank_mw, return_duals, "tank_dis_cap")
            tank_ch = _capacity_bounded(m, "tank_ch", coords, tank_mw, return_duals, "tank_ch_cap")
        if fix_storage:
            soc_slack = max(1.0, 0.02 * batt_mwh)
            tank_slack = max(1.0, 0.02 * tank_mwh)
            batt_soc = m.add_variables(lower=-soc_slack, upper=batt_mwh + soc_slack, coords=coords, name="batt_soc")
            tank_soc = m.add_variables(lower=-tank_slack, upper=tank_mwh + tank_slack, coords=coords, name="tank_soc")
        else:
            batt_soc = _capacity_bounded(m, "batt_soc", coords, batt_mwh, return_duals, "batt_soc_cap")
            tank_soc = _capacity_bounded(m, "tank_soc", coords, tank_mwh, return_duals, "tank_soc_cap")
        ely_p = _capacity_bounded(m, "ely_p", coords, ely_mw, return_duals, "ely_cap")
        x_grid = m.add_variables(lower=-grid_cap, upper=grid_cap, coords=coords, name="x_grid")
        x_h2 = m.add_variables(lower=-h2_cap, upper=h2_cap, coords=coords, name="x_h2")
        if return_duals:
            demand = m.add_variables(coords=coords, name="demand")
            m.add_constraints(demand >= baseline - flex, name="demand_lb")
            m.add_constraints(demand <= baseline + flex, name="demand_ub")
        else:
            demand = m.add_variables(lower=baseline - flex, upper=baseline + flex, coords=coords, name="demand")
        ely_ren = m.add_variables(lower=0.0, coords=coords, name="ely_ren")
        gc_buy = m.add_variables(lower=0.0, name="gc_buy")
        gc_sell = m.add_variables(lower=0.0, name="gc_sell")

        m.add_constraints(wind_p + pv_p + batt_dis - batt_ch - ely_p - x_grid == 0, name="elec_balance")
        m.add_constraints(ely_eff * ely_p + tank_dis - tank_ch - demand - x_h2 == 0, name="h2_balance")
        m.add_constraints(ely_ren <= wind_p + pv_p, name="ely_ren_cap_avail")
        m.add_constraints(ely_ren <= ely_p, name="ely_ren_cap_ely")
        m.add_constraints(gc_buy <= ely_p.sum() - ely_ren.sum(), name="gc_buy_cap")
        m.add_constraints(gc_sell <= (wind_p + pv_p).sum() - ely_ren.sum(), name="gc_sell_cap")

        m.add_constraints(batt_soc.isel(hour=0) - batt_soc0 - batt_eff * batt_ch.isel(hour=0) + batt_dis.isel(hour=0) == 0,
                          name="batt_balance_0")
        m.add_constraints(batt_soc.isel(hour=slice(1, None)) - batt_soc.isel(hour=slice(None, -1))
                          - batt_eff * batt_ch.isel(hour=slice(1, None)) + batt_dis.isel(hour=slice(1, None)) == 0,
                          name="batt_balance")
        m.add_constraints(tank_soc.isel(hour=0) - tank_soc0 - tank_eff * tank_ch.isel(hour=0) + tank_dis.isel(hour=0) == 0,
                          name="tank_balance_0")
        m.add_constraints(tank_soc.isel(hour=slice(1, None)) - tank_soc.isel(hour=slice(None, -1))
                          - tank_eff * tank_ch.isel(hour=slice(1, None)) + tank_dis.isel(hour=slice(1, None)) == 0,
                          name="tank_balance")
        if cfg.cyclic_storage and not fix_storage:
            m.add_constraints(batt_soc.isel(hour=-1) >= batt_soc0, name="batt_cyclic")
            m.add_constraints(tank_soc.isel(hour=-1) >= tank_soc0, name="tank_cyclic")

        m.add_constraints(demand.sum() == baseline * annual_hours, name="demand_conservation")
        if quota > 0.0:
            m.add_constraints(ely_eff * (ely_ren.sum() + gc_buy) >= quota * demand.sum(), name="red3_quota")

        cost = (-(p_elec * x_grid).sum() - (p_h2 * x_h2).sum()
               + sto_cost * (batt_ch.sum() + batt_dis.sum() + tank_ch.sum() + tank_dis.sum())
               + gc_price * gc_buy - gc_price * gc_sell
               - REN_SELF_USE_PRIORITY_EUR_PER_MWH * ely_ren.sum())
        m.add_objective(cost)
    else:
        day_idx = pd.RangeIndex(n_days, name="day")
        hid_idx = pd.RangeIndex(HOURS_PER_DAY, name="hid")
        coords = [day_idx, hid_idx]

        def r2d(a):
            return np.asarray(a).reshape(n_days, HOURS_PER_DAY)

        wind_upper2d, pv_upper2d = r2d(wind_upper), r2d(pv_upper)
        p_elec2d, p_h2_2d = r2d(p_elec), r2d(p_h2)
        w = xr.DataArray(np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY).reshape(n_days, HOURS_PER_DAY),
                         coords=coords)

        wind_p = _capacity_bounded(m, "wind_p", coords, wind_upper2d, return_duals, "wind_cap")
        pv_p = _capacity_bounded(m, "pv_p", coords, pv_upper2d, return_duals, "pv_cap")
        batt_dis = _capacity_bounded(m, "batt_dis", coords, batt_mw, return_duals, "batt_dis_cap")
        batt_ch = _capacity_bounded(m, "batt_ch", coords, batt_mw, return_duals, "batt_ch_cap")
        tank_dis = _capacity_bounded(m, "tank_dis", coords, tank_mw, return_duals, "tank_dis_cap")
        tank_ch = _capacity_bounded(m, "tank_ch", coords, tank_mw, return_duals, "tank_ch_cap")
        batt_soc = _capacity_bounded(m, "batt_soc", coords, batt_mwh, return_duals, "batt_soc_cap")
        tank_soc = _capacity_bounded(m, "tank_soc", coords, tank_mwh, return_duals, "tank_soc_cap")
        ely_p = _capacity_bounded(m, "ely_p", coords, ely_mw, return_duals, "ely_cap")
        x_grid = m.add_variables(lower=-grid_cap, upper=grid_cap, coords=coords, name="x_grid")
        x_h2 = m.add_variables(lower=-h2_cap, upper=h2_cap, coords=coords, name="x_h2")
        if return_duals:
            demand = m.add_variables(coords=coords, name="demand")
            m.add_constraints(demand >= baseline - flex, name="demand_lb")
            m.add_constraints(demand <= baseline + flex, name="demand_ub")
        else:
            demand = m.add_variables(lower=baseline - flex, upper=baseline + flex, coords=coords, name="demand")
        ely_ren = m.add_variables(lower=0.0, coords=coords, name="ely_ren")
        gc_buy = m.add_variables(lower=0.0, name="gc_buy")
        gc_sell = m.add_variables(lower=0.0, name="gc_sell")

        m.add_constraints(wind_p + pv_p + batt_dis - batt_ch - ely_p - x_grid == 0, name="elec_balance")
        m.add_constraints(ely_eff * ely_p + tank_dis - tank_ch - demand - x_h2 == 0, name="h2_balance")
        m.add_constraints(ely_ren <= wind_p + pv_p, name="ely_ren_cap_avail")
        m.add_constraints(ely_ren <= ely_p, name="ely_ren_cap_ely")
        m.add_constraints(gc_buy <= (w * ely_p).sum() - (w * ely_ren).sum(), name="gc_buy_cap")
        m.add_constraints(gc_sell <= (w * (wind_p + pv_p)).sum() - (w * ely_ren).sum(), name="gc_sell_cap")

        m.add_constraints(batt_soc.isel(hid=0) - batt_soc0 - batt_eff * batt_ch.isel(hid=0) + batt_dis.isel(hid=0) == 0,
                          name="batt_balance_0")
        m.add_constraints(batt_soc.isel(hid=slice(1, None)) - batt_soc.isel(hid=slice(None, -1))
                          - batt_eff * batt_ch.isel(hid=slice(1, None)) + batt_dis.isel(hid=slice(1, None)) == 0,
                          name="batt_balance")
        m.add_constraints(tank_soc.isel(hid=0) - tank_soc0 - tank_eff * tank_ch.isel(hid=0) + tank_dis.isel(hid=0) == 0,
                          name="tank_balance_0")
        m.add_constraints(tank_soc.isel(hid=slice(1, None)) - tank_soc.isel(hid=slice(None, -1))
                          - tank_eff * tank_ch.isel(hid=slice(1, None)) + tank_dis.isel(hid=slice(1, None)) == 0,
                          name="tank_balance")
        if cfg.cyclic_storage:
            m.add_constraints(batt_soc.isel(hid=-1) >= batt_soc0, name="batt_cyclic")
            m.add_constraints(tank_soc.isel(hid=-1) >= tank_soc0, name="tank_cyclic")

        m.add_constraints((w * demand).sum() == baseline * annual_hours, name="demand_conservation")
        if quota > 0.0:
            m.add_constraints(ely_eff * ((w * ely_ren).sum() + gc_buy) >= quota * (w * demand).sum(), name="red3_quota")

        cost = (-(w * p_elec2d * x_grid).sum() - (w * p_h2_2d * x_h2).sum()
               + sto_cost * ((w * batt_ch).sum() + (w * batt_dis).sum()
                            + (w * tank_ch).sum() + (w * tank_dis).sum())
               + gc_price * gc_buy - gc_price * gc_sell
               - REN_SELF_USE_PRIORITY_EUR_PER_MWH * (w * ely_ren).sum())
        m.add_objective(cost)

    build_s = time.time() - t0
    t1 = time.time()
    solver_kwargs = {"output_flag": False} if quiet else {}
    status, condition = m.solve(solver_name="highs", **solver_kwargs)
    solve_s = time.time() - t1
    if status != "ok":
        raise RuntimeError(f"solve failed: {status}/{condition}")

    cut_coeffs = None
    if return_duals:
        beta = cfg.h2_producer_downstream_demand_pct_of_electrolyser_capacity * ely_eff
        phi = cfg.h2_producer_demand_flex_pct * ely_eff
        mu_electrolyser = (float(m.dual["ely_cap"].sum())
                          + float((m.dual["demand_lb"] * (beta - phi)).sum())
                          + float((m.dual["demand_ub"] * (beta + phi)).sum())
                          + float(m.dual["demand_conservation"]) * beta * annual_hours)
        wind_cf_shaped = wind_cf_norm.reshape(n_days, HOURS_PER_DAY) if representative else wind_cf_norm
        pv_cf_shaped = pv_cf_norm.reshape(n_days, HOURS_PER_DAY) if representative else pv_cf_norm
        mu_wind = float((m.dual["wind_cap"] * wind_cf_shaped).sum())
        mu_pv = float((m.dual["pv_cap"] * pv_cf_shaped).sum())
        # Battery/tank MW also sets the INITIAL state of charge (batt_soc0/tank_soc0 =
        # initial_soc_fraction * duration_hours * mw) -- the RHS of batt_balance_0/
        # tank_balance_0 (and, when cyclic_storage is on, batt_cyclic/tank_cyclic too)
        # depends on capacity too, a real second channel for d(objective)/d(mw) missing
        # from this formula until 2026-09-11 -- see solve_joint's identical fix and
        # h2_planning/master.py::add_optimality_cut's docstring for the verified repro
        # (return_duals=True + fix_storage=True is rejected above, so cyclic_storage
        # alone correctly matches whether batt_cyclic/tank_cyclic were even added).
        batt_soc0_terms = float(m.dual["batt_balance_0"].sum())
        tank_soc0_terms = float(m.dual["tank_balance_0"].sum())
        if cfg.cyclic_storage:
            batt_soc0_terms += float(m.dual["batt_cyclic"].sum())
            tank_soc0_terms += float(m.dual["tank_cyclic"].sum())
        mu_battery = (float(m.dual["batt_dis_cap"].sum()) + float(m.dual["batt_ch_cap"].sum())
                     + float(m.dual["batt_soc_cap"].sum()) * cfg.h2_producer_battery_duration_hours
                     + batt_soc0_terms * cfg.initial_soc_fraction * cfg.h2_producer_battery_duration_hours)
        mu_tank = (float(m.dual["tank_dis_cap"].sum()) + float(m.dual["tank_ch_cap"].sum())
                  + float(m.dual["tank_soc_cap"].sum()) * cfg.h2_producer_tank_duration_hours
                  + tank_soc0_terms * cfg.initial_soc_fraction * cfg.h2_producer_tank_duration_hours)
        cut_coeffs = {"electrolyser_mw": mu_electrolyser, "wind_mw": mu_wind, "pv_mw": mu_pv,
                     "battery_mw": mu_battery, "tank_mw": mu_tank}

    sol = m.solution
    out = pd.DataFrame({
        "hour": np.arange(H),
        "H2 Producer wind (MW)": np.asarray(sol["wind_p"].values).reshape(-1),
        "H2 Producer pv (MW)": np.asarray(sol["pv_p"].values).reshape(-1),
        "H2 Producer battery discharge (MW)": np.asarray(sol["batt_dis"].values).reshape(-1),
        "H2 Producer battery charge (-) (MW)": -np.asarray(sol["batt_ch"].values).reshape(-1),
        "H2 Producer electrolyser load (-) (MW)": -np.asarray(sol["ely_p"].values).reshape(-1),
        "H2 Producer grid exchange (MW)": np.asarray(sol["x_grid"].values).reshape(-1),
        "H2 Producer electrolyser production (MW)": ely_eff * np.asarray(sol["ely_p"].values).reshape(-1),
        "H2 Producer tank discharge (MW)": np.asarray(sol["tank_dis"].values).reshape(-1),
        "H2 Producer tank charge (-) (MW)": -np.asarray(sol["tank_ch"].values).reshape(-1),
        "H2 Producer downstream demand (-) (MW)": -np.asarray(sol["demand"].values).reshape(-1),
        "H2 Producer pipeline exchange (MW)": np.asarray(sol["x_h2"].values).reshape(-1),
        "H2 Producer renewable-covered electrolyser load (MW)": np.asarray(sol["ely_ren"].values).reshape(-1),
    })
    if representative:
        out["day_of_year"] = np.repeat(days, HOURS_PER_DAY)
        out["day_weight"] = np.repeat(day_weights, HOURS_PER_DAY)
    out.attrs["objective"] = float(m.objective.value)
    out.attrs["p_elec"] = p_elec
    out.attrs["p_h2"] = p_h2
    out.attrs["host_zone"] = host_zone
    out.attrs["sizing"] = s
    out.attrs["build_seconds"] = build_s
    out.attrs["solve_seconds"] = solve_s
    out.attrs["n_hours"] = H
    out.attrs["annualized_hours"] = annual_hours
    out.attrs["rep_days_per_month"] = rep_days_per_month
    out.attrs["sampled_days"] = days if representative else None
    out.attrs["fix_storage"] = fix_storage
    out.attrs["capacities"] = {"electrolyser_mw": ely_mw, "wind_mw": s["wind_mw"], "pv_mw": s["pv_mw"],
                              "battery_mw": batt_mw, "tank_mw": tank_mw}
    out.attrs["cut_coeffs"] = cut_coeffs
    out.attrs["gc_buy_mwh"] = float(sol["gc_buy"].values)
    out.attrs["gc_sell_mwh"] = float(sol["gc_sell"].values)
    return out


def solve_joint(zones: list[str], capacities: dict[str, dict], rep_days_per_month: int,
                pool_mw: float, return_duals: bool = True,
                grid_cap_mw: float | None = None, h2_cap_mw: float | None = None,
                demand_flex_pct: float = 0.20, ens_penalty_eur_per_mwh: float | None = None,
                edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                quiet: bool = False) -> dict:
    """Joint multi-country H2 Producer LP: every zone in ``zones`` solved TOGETHER in
    ONE linopy model, sharing a single downstream-demand POOL (``sum_zone
    demand_base_{zone} == pool_mw``, ONE constraint -- NOT per-hour, since
    ``demand_base`` has no hour index, see below) instead of each zone having its own
    independent demand target. Everything else -- storage, RED III quota, grid/pipeline
    exchange, GC buy/sell -- stays PER-ZONE/separable, exactly as ``solve()``'s
    representative-day path; only the demand baseline is shared. Representative-day
    horizon only (no contiguous mode -- this project's planning workflow never needs one
    for a multi-zone joint solve).

    ``capacities`` (REQUIRED, unlike ``solve()`` which falls back to rank-derived
    sizing) -- ``{zone: {"electrolyser_mw"/"wind_mw"/"pv_mw"/"battery_mw"/"tank_mw":
    MW}}`` -- the trial capacity vector the Benders master proposed for every zone
    this iteration.

    Demand is two decision layers, not one:

    * ``demand_base_{zone}`` -- ONE value per zone (no hour index), $\\ge 0$. This is
      what the pool constraint sums to ``pool_mw`` -- a zone with electrolyser_mw=0
      (skipped by the master) still CAN take a positive share here in principle, but
      the capacity ceiling below pins its realized demand to 0 every hour regardless,
      so in practice the pool redistributes only across zones that actually installed
      an electrolyser, same as before.
    * ``shift_{zone,day,h}`` -- free (+/-) per zone AND representative hour, bounded
      ``-demand_flex_pct * demand_base_zone <= shift_{zone,day,h} <= +demand_flex_pct *
      demand_base_zone`` (default ``demand_flex_pct=0.20``, i.e. +/-20% of that zone's
      OWN baseline, every representative hour independently) -- how much that zone's
      REALIZED hour can deviate from its flat baseline. Net-zero PER ZONE PER
      REPRESENTATIVE DAY (a 24h cycle, matching the battery/tank storage cyclic
      constraints' own horizon, NOT an annual aggregate): ``sum_h shift_{zone,day,h}
      == 0`` for every ``(zone, day)`` independently -- every MWh shifted up within one
      representative day must be offset by shifting an equal MWh down within THAT SAME
      day, for THAT SAME zone (no borrowing across days, no borrowing from other
      zones' pool share, no net demand creation/destruction). Realized demand
      ``demand_{zone,day,h} = demand_base_zone + shift_{zone,day,h}`` is what actually
      appears in the H2 balance/RED III quota below, and is what's exported in the
      schedule's "H2 Producer downstream demand (-) (MW)" column -- ``demand_base``/
      ``shift`` themselves aren't separately exported today.

    Realized demand is still bounded ``0 <= demand_{zone,day,h} <= electrolyser_mw_zone
    * eta_ely`` every hour (redundant on the lower side given the two bullets above,
    kept implicit rather than as an extra constraint) -- "the electrolyser must be big
    enough for whatever it's asked to produce" still holds BY CONSTRUCTION every hour.
    Because the ±20% band is per-zone and the net-zero constraint is per-zone-per-day
    too (not system-wide), the REALIZED cross-zone total at any single hour is no
    longer pinned to exactly ``pool_mw`` the way it was before this shift mechanism
    existed -- only each zone's own baseline, and each zone's own per-day net shift,
    are pinned. This REPLACES the old fixed/coupled-to-capacity ``downstream_load_mw``
    demand mechanism entirely for this joint mode -- there is no ``downstream_load_mw``/
    ``downstream_load_flex_pct`` parameter here (unrelated to ``demand_flex_pct``
    above, which is this joint mode's own, differently-shaped flexibility knob).

    Returns ``{"objective": float, "objective_by_zone": {zone: float}, "schedules":
    {zone: DataFrame}, "cut_coeffs": {zone: {asset: mu}} or None, "gc_buy_mwh"/
    "gc_sell_mwh": {zone: float}, "build_seconds", "solve_seconds"}``. Each zone's
    schedule DataFrame has the same columns as ``solve()``'s (plus day_of_year/
    day_weight), so downstream code (e.g. the artifact's schedule export) doesn't need
    to know it came from a joint solve.

    NOTE: because every zone's own physical constraints (balance, storage, capacity
    bounds) are still fully separable per zone -- only the ``demand_pool`` equality
    actually links zones together, and it doesn't appear in the objective itself --
    the total objective decomposes exactly into a sum of independently-computed
    per-zone contributions (``objective_by_zone``), which is what feeds each zone's
    own Benders cut. The Benders cut for one zone is still only a LOCAL supporting
    hyperplane in that zone's own capacity (standard Benders assumption) even though
    the pool constraint means one zone's true cost also depends on every other zone's
    capacity -- this is weaker than a fully separable subproblem, so this joint mode
    may need more iterations to converge than the independent-per-zone mode.

    ``ens_penalty_eur_per_mwh`` (optional, default ``None`` -- every existing caller's
    behavior is byte-for-byte unchanged): when set, adds two non-negative slack
    variables -- ``ens_elec`` (Energy Not Served, folded into ``elec_balance``) and
    ``ens_h2`` (Hydrogen Not Served, folded into ``demand_ub``: ``demand - ens_h2 <=
    demand_upper`` instead of ``demand <= demand_upper``) -- each penalized at this
    EUR/MWh rate in the objective. This exists so a trial capacity vector too small to
    physically cover ``pool_mw`` (e.g. the master's all-skip iteration-1 guess once the
    ``min_total_electrolyser_mw`` floor is ALSO removed, see
    ``h2_planning.master.build_master``'s docstring) makes the LP expensive rather than
    INFEASIBLE -- turning what would otherwise be a hard ``RuntimeError`` (the "no
    feasibility cuts implemented" limitation) into an ordinary, Benders-cuttable
    objective value. ``ens_h2`` is added to the capacity-ceiling constraint rather than
    to ``h2_balance`` itself, because ``h2_balance`` is already unconstrained-feasible
    whenever ``h2_cap_mw`` is unlimited (``x_h2`` can absorb any supply/demand mismatch
    on its own) -- the actual infeasibility this is designed to avoid comes from
    ``demand_ub`` capping realized demand at installed electrolyser capacity regardless
    of pipeline import ability. ``ens_elec`` is added symmetrically to ``elec_balance``
    for the same reason on the electricity side, even though it won't bind while
    ``grid_cap_mw`` stays unlimited too. When ``quota > 0``, ``red3_quota``'s RHS also
    switches from ``demand`` to ``demand - ens_h2`` (the actually-served portion) --
    otherwise a zero-electrolyser trial still has to satisfy 42% renewable coverage of
    its full, un-servable demand target with zero real production, which is infeasible
    on its own regardless of ``ens_h2`` fixing ``demand_ub`` (found by direct testing:
    ``demand_ub`` alone was NOT sufficient to restore feasibility)."""
    t0 = time.time()
    days, day_weights = day_sampling.representative_days(rep_days_per_month)
    n_days = len(days)
    hours = np.array([(d - 1) * HOURS_PER_DAY + h for d in days for h in range(HOURS_PER_DAY)], dtype=int)
    H = len(hours)
    annual_hours = TOTAL_YEAR_HOURS

    zone_idx = pd.Index(zones, name="zone")
    day_idx = pd.RangeIndex(n_days, name="day")
    hid_idx = pd.RangeIndex(HOURS_PER_DAY, name="hid")
    coords = [zone_idx, day_idx, hid_idx]

    w = xr.DataArray(np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY).reshape(n_days, HOURS_PER_DAY),
                     coords=[day_idx, hid_idx])

    cfgs, ely_effs, ely_mws, batt_mws, batt_mwhs, tank_mws, tank_mwhs = {}, {}, {}, {}, {}, {}, {}
    wind_uppers, pv_uppers, wind_cf_norms, pv_cf_norms, p_elecs, p_h2s = [], [], [], [], [], []
    host_zones = {}
    for z in zones:
        (cfg, s, host_zone, wind_donor, pv_donor, wind_upper, pv_upper,
        wind_cf_norm, pv_cf_norm) = sizing_and_profiles(z, hours, capacities[z])
        cfgs[z] = cfg
        host_zones[z] = host_zone
        ely_effs[z] = cfg.h2_producer_electrolyser_efficiency
        ely_mws[z] = s["electrolyser_mw"]
        batt_mws[z], batt_mwhs[z] = s["battery_mw"], s["battery_mwh"]
        tank_mws[z], tank_mwhs[z] = s["tank_mw"], s["tank_mwh"]
        wind_uppers.append(wind_upper.reshape(n_days, HOURS_PER_DAY))
        pv_uppers.append(pv_upper.reshape(n_days, HOURS_PER_DAY))
        wind_cf_norms.append(wind_cf_norm.reshape(n_days, HOURS_PER_DAY))
        pv_cf_norms.append(pv_cf_norm.reshape(n_days, HOURS_PER_DAY))
        elec_zone = _ELEC_ZONE_OVERRIDES.get(z[:2], z)
        p_elec, p_h2 = proxy_price_series(elec_zone, hours, h2_zone=z, edf=edf, hdf=hdf)
        p_elecs.append(p_elec.reshape(n_days, HOURS_PER_DAY))
        p_h2s.append(p_h2.reshape(n_days, HOURS_PER_DAY))

    wind_upper_da = xr.DataArray(np.stack(wind_uppers), coords=coords)
    pv_upper_da = xr.DataArray(np.stack(pv_uppers), coords=coords)
    wind_cf_norm_da = xr.DataArray(np.stack(wind_cf_norms), coords=coords)
    pv_cf_norm_da = xr.DataArray(np.stack(pv_cf_norms), coords=coords)
    p_elec_da = xr.DataArray(np.stack(p_elecs), coords=coords)
    p_h2_da = xr.DataArray(np.stack(p_h2s), coords=coords)

    ely_mw_da = xr.DataArray([ely_mws[z] for z in zones], coords=[zone_idx])
    ely_eff_da = xr.DataArray([ely_effs[z] for z in zones], coords=[zone_idx])
    batt_mw_da = xr.DataArray([batt_mws[z] for z in zones], coords=[zone_idx])
    batt_mwh_da = xr.DataArray([batt_mwhs[z] for z in zones], coords=[zone_idx])
    tank_mw_da = xr.DataArray([tank_mws[z] for z in zones], coords=[zone_idx])
    tank_mwh_da = xr.DataArray([tank_mwhs[z] for z in zones], coords=[zone_idx])

    cfg0 = cfgs[zones[0]]
    grid_cap = grid_cap_mw if grid_cap_mw is not None else cfg0.h2_producer_grid_connection_mw
    h2_cap = h2_cap_mw if h2_cap_mw is not None else cfg0.h2_producer_h2_connection_mw
    batt_eff, tank_eff = cfg0.h2_producer_battery_efficiency, cfg0.h2_producer_tank_efficiency
    quota = max(cfg0.h2_producer_renewable_h2_quota, 0.0)
    gc_price = cfg0.h2_producer_gc_price_eur_per_mwh
    sto_cost = cfg0.storage_op_cost_eur_per_mwh
    batt_soc0_da = cfg0.initial_soc_fraction * batt_mwh_da
    tank_soc0_da = cfg0.initial_soc_fraction * tank_mwh_da

    m = linopy.Model()
    wind_p = _capacity_bounded(m, "wind_p", coords, wind_upper_da, return_duals, "wind_cap")
    pv_p = _capacity_bounded(m, "pv_p", coords, pv_upper_da, return_duals, "pv_cap")
    batt_dis = _capacity_bounded(m, "batt_dis", coords, batt_mw_da, return_duals, "batt_dis_cap")
    batt_ch = _capacity_bounded(m, "batt_ch", coords, batt_mw_da, return_duals, "batt_ch_cap")
    tank_dis = _capacity_bounded(m, "tank_dis", coords, tank_mw_da, return_duals, "tank_dis_cap")
    tank_ch = _capacity_bounded(m, "tank_ch", coords, tank_mw_da, return_duals, "tank_ch_cap")
    batt_soc = _capacity_bounded(m, "batt_soc", coords, batt_mwh_da, return_duals, "batt_soc_cap")
    tank_soc = _capacity_bounded(m, "tank_soc", coords, tank_mwh_da, return_duals, "tank_soc_cap")
    ely_p = _capacity_bounded(m, "ely_p", coords, ely_mw_da, return_duals, "ely_cap")
    x_grid = m.add_variables(lower=-grid_cap, upper=grid_cap, coords=coords, name="x_grid")
    x_h2 = m.add_variables(lower=-h2_cap, upper=h2_cap, coords=coords, name="x_h2")

    ens_on = ens_penalty_eur_per_mwh is not None
    if ens_on:
        ens_elec = m.add_variables(lower=0.0, coords=coords, name="ens_elec")
        ens_h2 = m.add_variables(lower=0.0, coords=coords, name="ens_h2")

    demand_upper = ely_mw_da * ely_eff_da
    demand_base = m.add_variables(lower=0.0, coords=[zone_idx], name="demand_base")
    shift = m.add_variables(coords=coords, name="shift")
    demand = m.add_variables(coords=coords, name="demand")
    m.add_constraints(demand == demand_base + shift, name="demand_def")
    demand_ub_lhs = demand if not ens_on else demand - ens_h2
    m.add_constraints(demand_ub_lhs <= demand_upper, name="demand_ub")
    m.add_constraints(shift - demand_flex_pct * demand_base <= 0.0, name="shift_ub")
    m.add_constraints(shift + demand_flex_pct * demand_base >= 0.0, name="shift_lb")
    m.add_constraints(shift.sum("hid") == 0.0, name="shift_net_zero")
    ely_ren = m.add_variables(lower=0.0, coords=coords, name="ely_ren")
    gc_buy = m.add_variables(lower=0.0, coords=[zone_idx], name="gc_buy")
    gc_sell = m.add_variables(lower=0.0, coords=[zone_idx], name="gc_sell")

    elec_balance_lhs = wind_p + pv_p + batt_dis - batt_ch - ely_p - x_grid
    if ens_on:
        elec_balance_lhs = elec_balance_lhs + ens_elec
    m.add_constraints(elec_balance_lhs == 0, name="elec_balance")
    m.add_constraints(ely_eff_da * ely_p + tank_dis - tank_ch - demand - x_h2 == 0, name="h2_balance")
    m.add_constraints(ely_ren <= wind_p + pv_p, name="ely_ren_cap_avail")
    m.add_constraints(ely_ren <= ely_p, name="ely_ren_cap_ely")
    m.add_constraints(gc_buy <= (w * ely_p).sum(["day", "hid"]) - (w * ely_ren).sum(["day", "hid"]), name="gc_buy_cap")
    m.add_constraints(gc_sell <= (w * (wind_p + pv_p)).sum(["day", "hid"]) - (w * ely_ren).sum(["day", "hid"]),
                      name="gc_sell_cap")

    m.add_constraints(batt_soc.isel(hid=0) - batt_soc0_da - batt_eff * batt_ch.isel(hid=0) + batt_dis.isel(hid=0) == 0,
                      name="batt_balance_0")
    m.add_constraints(batt_soc.isel(hid=slice(1, None)) - batt_soc.isel(hid=slice(None, -1))
                      - batt_eff * batt_ch.isel(hid=slice(1, None)) + batt_dis.isel(hid=slice(1, None)) == 0,
                      name="batt_balance")
    m.add_constraints(tank_soc.isel(hid=0) - tank_soc0_da - tank_eff * tank_ch.isel(hid=0) + tank_dis.isel(hid=0) == 0,
                      name="tank_balance_0")
    m.add_constraints(tank_soc.isel(hid=slice(1, None)) - tank_soc.isel(hid=slice(None, -1))
                      - tank_eff * tank_ch.isel(hid=slice(1, None)) + tank_dis.isel(hid=slice(1, None)) == 0,
                      name="tank_balance")
    if cfg0.cyclic_storage:
        m.add_constraints(batt_soc.isel(hid=-1) >= batt_soc0_da, name="batt_cyclic")
        m.add_constraints(tank_soc.isel(hid=-1) >= tank_soc0_da, name="tank_cyclic")

    if quota > 0.0:
        quota_demand = demand if not ens_on else demand - ens_h2
        m.add_constraints(ely_eff_da * ((w * ely_ren).sum(["day", "hid"]) + gc_buy)
                          >= quota * (w * quota_demand).sum(["day", "hid"]), name="red3_quota")

    m.add_constraints(demand_base.sum("zone") == pool_mw, name="demand_pool")

    cost = (-(w * p_elec_da * x_grid).sum() - (w * p_h2_da * x_h2).sum()
           + sto_cost * ((w * batt_ch).sum() + (w * batt_dis).sum()
                        + (w * tank_ch).sum() + (w * tank_dis).sum())
           + gc_price * gc_buy.sum() - gc_price * gc_sell.sum()
           - REN_SELF_USE_PRIORITY_EUR_PER_MWH * (w * ely_ren).sum())
    if ens_on:
        cost = cost + ens_penalty_eur_per_mwh * ((w * ens_elec).sum() + (w * ens_h2).sum())
    m.add_objective(cost)

    build_s = time.time() - t0
    t1 = time.time()
    solver_kwargs = {"output_flag": False} if quiet else {}
    status, condition = m.solve(solver_name="highs", **solver_kwargs)
    solve_s = time.time() - t1
    if status != "ok":
        raise RuntimeError(f"joint solve failed: {status}/{condition}")

    sol = m.solution
    cut_coeffs = None
    if return_duals:
        mu_electrolyser_da = (m.dual["ely_cap"].sum(["day", "hid"])
                             + (m.dual["demand_ub"] * ely_eff_da).sum(["day", "hid"]))
        mu_wind_da = (m.dual["wind_cap"] * wind_cf_norm_da).sum(["day", "hid"])
        mu_pv_da = (m.dual["pv_cap"] * pv_cf_norm_da).sum(["day", "hid"])
        # Battery/tank MW also sets each representative day's INITIAL state of charge
        # (batt_soc0_da/tank_soc0_da = initial_soc_fraction * duration_hours * mw), which
        # is the RHS of batt_balance_0/tank_balance_0 (and, when cyclic_storage is on,
        # batt_cyclic/tank_cyclic too) -- a real, second channel for d(objective)/d(mw)
        # beyond the plain capacity bounds below. Omitting it (as this formula did until
        # 2026-09-11) makes mu an incomplete/wrong subgradient, which can make the
        # resulting Benders cut genuinely INVALID (violated by a later, true resolve) --
        # reproduced directly with a 2-country run whose battery/tank swung from the
        # catalog max to near-zero between iterations; see
        # h2_planning/master.py::add_optimality_cut's docstring for the full repro.
        batt_soc0_terms = m.dual["batt_balance_0"].sum("day")
        tank_soc0_terms = m.dual["tank_balance_0"].sum("day")
        if cfg0.cyclic_storage:
            batt_soc0_terms = batt_soc0_terms + m.dual["batt_cyclic"].sum("day")
            tank_soc0_terms = tank_soc0_terms + m.dual["tank_cyclic"].sum("day")
        mu_battery_da = (m.dual["batt_dis_cap"].sum(["day", "hid"]) + m.dual["batt_ch_cap"].sum(["day", "hid"])
                        + m.dual["batt_soc_cap"].sum(["day", "hid"]) * cfg0.h2_producer_battery_duration_hours
                        + batt_soc0_terms * cfg0.initial_soc_fraction * cfg0.h2_producer_battery_duration_hours)
        mu_tank_da = (m.dual["tank_dis_cap"].sum(["day", "hid"]) + m.dual["tank_ch_cap"].sum(["day", "hid"])
                    + m.dual["tank_soc_cap"].sum(["day", "hid"]) * cfg0.h2_producer_tank_duration_hours
                    + tank_soc0_terms * cfg0.initial_soc_fraction * cfg0.h2_producer_tank_duration_hours)
        cut_coeffs = {z: {"electrolyser_mw": float(mu_electrolyser_da.sel(zone=z)),
                          "wind_mw": float(mu_wind_da.sel(zone=z)),
                          "pv_mw": float(mu_pv_da.sel(zone=z)),
                          "battery_mw": float(mu_battery_da.sel(zone=z)),
                          "tank_mw": float(mu_tank_da.sel(zone=z))}
                     for z in zones}

    schedules, gc_buy_mwh, gc_sell_mwh, objective_by_zone = {}, {}, {}, {}
    ens_elec_mwh, ens_h2_mwh = {}, {}
    day_weight_flat = np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY)
    for zi, z in enumerate(zones):
        wp = np.asarray(sol["wind_p"].sel(zone=z).values).reshape(-1)
        pp = np.asarray(sol["pv_p"].sel(zone=z).values).reshape(-1)
        bd = np.asarray(sol["batt_dis"].sel(zone=z).values).reshape(-1)
        bc = np.asarray(sol["batt_ch"].sel(zone=z).values).reshape(-1)
        ep = np.asarray(sol["ely_p"].sel(zone=z).values).reshape(-1)
        xg = np.asarray(sol["x_grid"].sel(zone=z).values).reshape(-1)
        td = np.asarray(sol["tank_dis"].sel(zone=z).values).reshape(-1)
        tc = np.asarray(sol["tank_ch"].sel(zone=z).values).reshape(-1)
        dm = np.asarray(sol["demand"].sel(zone=z).values).reshape(-1)
        xh = np.asarray(sol["x_h2"].sel(zone=z).values).reshape(-1)
        er = np.asarray(sol["ely_ren"].sel(zone=z).values).reshape(-1)
        out = pd.DataFrame({
            "hour": np.arange(H),
            "H2 Producer wind (MW)": wp,
            "H2 Producer pv (MW)": pp,
            "H2 Producer battery discharge (MW)": bd,
            "H2 Producer battery charge (-) (MW)": -bc,
            "H2 Producer electrolyser load (-) (MW)": -ep,
            "H2 Producer grid exchange (MW)": xg,
            "H2 Producer electrolyser production (MW)": ely_effs[z] * ep,
            "H2 Producer tank discharge (MW)": td,
            "H2 Producer tank charge (-) (MW)": -tc,
            "H2 Producer downstream demand (-) (MW)": -dm,
            "H2 Producer pipeline exchange (MW)": xh,
            "H2 Producer renewable-covered electrolyser load (MW)": er,
        })
        out["day_of_year"] = np.repeat(days, HOURS_PER_DAY)
        out["day_weight"] = day_weight_flat
        if ens_on:
            out["H2 Producer energy not served (MW)"] = np.asarray(sol["ens_elec"].sel(zone=z).values).reshape(-1)
            out["H2 Producer hydrogen not served (MW)"] = np.asarray(sol["ens_h2"].sel(zone=z).values).reshape(-1)

        p_elec_z = np.asarray(p_elecs[zi]).reshape(-1)
        p_h2_z = np.asarray(p_h2s[zi]).reshape(-1)
        gcb, gcs = float(sol["gc_buy"].sel(zone=z).values), float(sol["gc_sell"].sel(zone=z).values)
        z_cost = (-(p_elec_z * day_weight_flat * xg).sum() - (p_h2_z * day_weight_flat * xh).sum()
                 + sto_cost * (day_weight_flat * (bc + bd + tc + td)).sum()
                 + gc_price * gcb - gc_price * gcs
                 - REN_SELF_USE_PRIORITY_EUR_PER_MWH * (day_weight_flat * er).sum())
        if ens_on:
            ee = np.asarray(sol["ens_elec"].sel(zone=z).values).reshape(-1)
            eh = np.asarray(sol["ens_h2"].sel(zone=z).values).reshape(-1)
            z_cost = z_cost + ens_penalty_eur_per_mwh * (day_weight_flat * (ee + eh)).sum()
            ens_elec_mwh[z] = float((day_weight_flat * ee).sum())
            ens_h2_mwh[z] = float((day_weight_flat * eh).sum())
        objective_by_zone[z] = float(z_cost)

        out.attrs["objective"] = float(z_cost)
        out.attrs["p_elec"] = p_elec_z
        out.attrs["p_h2"] = p_h2_z
        out.attrs["host_zone"] = host_zones[z]
        out.attrs["capacities"] = capacities[z]
        out.attrs["annualized_hours"] = annual_hours
        out.attrs["rep_days_per_month"] = rep_days_per_month
        out.attrs["sampled_days"] = days
        out.attrs["gc_buy_mwh"] = gcb
        out.attrs["gc_sell_mwh"] = gcs
        out.attrs["cut_coeffs"] = cut_coeffs[z] if cut_coeffs else None
        schedules[z] = out
        gc_buy_mwh[z], gc_sell_mwh[z] = gcb, gcs

    return {
        "objective": float(m.objective.value),
        "objective_by_zone": objective_by_zone,
        "schedules": schedules,
        "cut_coeffs": cut_coeffs,
        "gc_buy_mwh": gc_buy_mwh,
        "gc_sell_mwh": gc_sell_mwh,
        "ens_elec_mwh": ens_elec_mwh if ens_on else None,
        "ens_h2_mwh": ens_h2_mwh if ens_on else None,
        "build_seconds": build_s,
        "solve_seconds": solve_s,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DE00")
    ap.add_argument("--day", type=int, default=None, help="single day (shorthand for --start-day/--end-day)")
    ap.add_argument("--start-day", type=int, default=None,
                    help="contiguous-range start day, 1-364 (default 5 if no other horizon flag is given)")
    ap.add_argument("--end-day", type=int, default=None)
    ap.add_argument("--rep-days-per-month", type=int, default=None,
                    help="representative-sampling mode instead of a contiguous range: N evenly-spaced "
                         "days per month (1-29), weighted to approximate the full year -- mutually "
                         "exclusive with --day/--start-day/--end-day. See day_sampling.py / "
                         "Formulation.md SS2.6")
    ap.add_argument("--fix-storage", action="store_true",
                    help="fix battery/tank charge-discharge to actual historical values "
                         "(contiguous mode only)")
    args = ap.parse_args()
    if args.rep_days_per_month is not None:
        if args.day is not None or args.start_day is not None or args.end_day is not None:
            ap.error("--rep-days-per-month is mutually exclusive with --day/--start-day/--end-day")
        df = solve(args.zone, rep_days_per_month=args.rep_days_per_month, fix_storage=args.fix_storage)
    else:
        if args.day is not None:
            start_day, end_day = args.day, args.day
        else:
            start_day, end_day = (args.start_day or 5), (args.end_day or args.start_day or 5)
        df = solve(args.zone, start_day, end_day, fix_storage=args.fix_storage)
    if df.attrs["n_hours"] <= 48:
        print(df.to_string(index=False))
    else:
        print(df.describe().to_string())
    print(f"\nhours solved: {df.attrs['n_hours']} (annualized as {df.attrs['annualized_hours']}) | "
         f"build: {df.attrs['build_seconds']:.1f}s | solve: {df.attrs['solve_seconds']:.1f}s")
    print("objective (EUR, negative = net revenue):", round(df.attrs["objective"], 2))
