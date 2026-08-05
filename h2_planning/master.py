"""Benders master MILP for H2 Producer capacity planning: binary candidate selection
per (country, asset), a system-wide raw-CAPEX budget, and per-country recourse
(``theta``) variables tightened iteratively by optimality cuts built from
``optimize_h2_producer.solve(..., return_duals=True)``'s subproblem duals. See
``Formulation.md`` SS4.4-4.6.
"""
from __future__ import annotations

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .config import ASSETS


def build_master(countries: list[str], cand_mw: dict[str, dict[str, np.ndarray]],
                 unit_cost: dict[str, float], budget: float | None, crf: dict[str, float],
                 theta_lower: float, downstream_load_mw: float | None = None,
                 require_electrolyser_for_others: bool = False,
                 min_total_electrolyser_mw: float | None = None,
                 disabled_assets: list[str] | None = None) -> linopy.Model:
    """Fresh master problem, no cuts yet (iteration 1 will pick candidates purely by
    annualized CAPEX, with every ``theta`` pinned at its lower bound -- expected, see
    Formulation.md SS4.6). ``crf`` is per-asset (``CapexAssumptions.
    capital_recovery_factors()``) since each asset can carry its own ``lifetime_years``
    -- it only ever scales the objective's CAPEX term, never the raw ``budget``
    constraint (Formulation.md SS4.3/4.4).

    ``budget`` may be ``None`` to DEACTIVATE the budget constraint entirely -- every
    country then sizes purely off subproblem economics (via ``theta``) and annualized
    CAPEX, with no system-wide spend cap at all. Useful to see what capacities the
    proxy economics alone would pick, independent of any budget assumption.

    ``downstream_load_mw`` (optional), if given, adds
    ``cap_{c,electrolyser}(y) >= downstream_load_mw`` for every country -- "each H2
    Producer must install AT LEAST its downstream load" worth of electrolyser capacity,
    rather than relying on pipeline imports to cover demand it can't itself produce.
    Only meaningful paired with ``optimize_h2_producer.solve(...,
    downstream_load_mw=...)`` in the subproblem (same fixed MW figure), since otherwise
    the subproblem's own demand baseline still scales with whatever electrolyser MW
    happens to be chosen. This constraint overrides ``one_hot``'s ``<=1`` skip option
    for the electrolyser specifically -- with ``downstream_load_mw > 0`` the
    electrolyser candidate 0 (skip) can never satisfy it, so every country is forced to
    build at least enough electrolyser capacity to serve its own load.

    ``require_electrolyser_for_others`` (optional), if True, adds
    ``sum_k y_{c,a,k} <= sum_k y_{c,electrolyser,k}`` for every non-electrolyser asset
    -- wind/PV/battery/tank can only be selected (any positive candidate) for a country
    that also selected a positive electrolyser candidate; skipping the electrolyser
    forces every other asset to be skipped too. Physically: this facility is a
    Hydrogen Producer, so its renewables/storage exist to serve the electrolyser, not
    as a standalone power plant. NOTE: automatically satisfied (vacuously) and changes
    nothing whenever ``downstream_load_mw`` is also set to a positive value, since that
    already forces the electrolyser on in every country -- it only bites in a run
    where the electrolyser itself is genuinely optional (``downstream_load_mw`` unset).

    ``min_total_electrolyser_mw`` (optional), if given, adds
    ``sum_c cap_{c,electrolyser}(y) >= min_total_electrolyser_mw`` -- an AGGREGATE
    floor across every country combined (unlike ``downstream_load_mw``, which is
    per-country). Needed whenever the subproblem enforces a shared demand POOL across
    countries (``optimize_h2_producer.solve_joint``'s ``pool_mw``): with no cuts yet,
    iteration 1's master would otherwise pick the globally cheapest combination --
    skip every electrolyser everywhere -- which makes the joint subproblem
    INFEASIBLE (a 0-capacity fleet can't supply any pool demand at all, let alone
    ``pool_mw``). This constraint only guarantees the fleet COULD satisfy the pool in
    aggregate (typically ``pool_mw / eta_ely``); the master is still free to decide
    HOW that capacity is distributed across countries.

    ``disabled_assets`` (optional), a list of ``ASSETS`` entries (e.g.
    ``["wind_mw", "pv_mw"]``) to force to 0 MW for EVERY country -- adds
    ``sum_k y_{c,a,k} == 0`` for each named asset, i.e. fixes that asset's binary
    selection off entirely rather than leaving it to the optimizer. Diagnostic: e.g.
    disabling wind/PV isolates whether they're what's driving other assets'
    (especially electrolyser's) sizing via ``require_electrolyser_for_others``."""
    m = linopy.Model()
    country_idx = pd.Index(countries, name="country")
    asset_idx = pd.Index(ASSETS, name="asset")
    n_k = len(cand_mw[countries[0]][ASSETS[0]])
    k_idx = pd.Index(range(n_k), name="k")

    y = m.add_variables(binary=True, coords=[country_idx, asset_idx, k_idx], name="y")
    theta = m.add_variables(lower=theta_lower, coords=[country_idx], name="theta")

    cand_da = xr.DataArray(
        np.array([[cand_mw[c][a] for a in ASSETS] for c in countries]),
        coords=[country_idx, asset_idx, k_idx],
    )
    value_cost = xr.DataArray(
        np.array([[cand_mw[c][a] * unit_cost[a] for a in ASSETS] for c in countries]),
        coords=[country_idx, asset_idx, k_idx],
    )
    annualized_value_cost = xr.DataArray(
        np.array([[cand_mw[c][a] * unit_cost[a] * crf[a] for a in ASSETS] for c in countries]),
        coords=[country_idx, asset_idx, k_idx],
    )

    m.add_constraints(y.sum("k") <= 1, name="one_hot")
    annualized_capex_expr = (annualized_value_cost * y).sum()
    if budget is not None:
        capex_expr = (value_cost * y).sum()
        m.add_constraints(capex_expr <= budget, name="budget")
    if downstream_load_mw is not None:
        ely_cap = (cand_da * y).sum("k").sel(asset="electrolyser_mw")
        m.add_constraints(ely_cap >= downstream_load_mw, name="min_electrolyser_vs_load")
    if min_total_electrolyser_mw is not None:
        ely_cap_total = (cand_da * y).sum(["k", "country"]).sel(asset="electrolyser_mw")
        m.add_constraints(ely_cap_total >= min_total_electrolyser_mw, name="min_total_electrolyser")
    if require_electrolyser_for_others:
        ely_installed = y.sum("k").sel(asset="electrolyser_mw")
        for a in ASSETS:
            if a == "electrolyser_mw":
                continue
            m.add_constraints(y.sum("k").sel(asset=a) <= ely_installed, name=f"require_ely_for_{a}")
    if disabled_assets:
        for a in disabled_assets:
            m.add_constraints(y.sum("k").sel(asset=a) == 0, name=f"disable_{a}")
    m.add_objective(annualized_capex_expr + theta.sum())
    return m


def add_optimality_cut(m: linopy.Model, country: str, iteration: int, Q: float,
                       mu: dict[str, float], cap_star: dict[str, float],
                       cand_mw: dict[str, dict[str, np.ndarray]]) -> None:
    """Add country ``country``'s Benders optimality cut for this iteration:

        theta_c >= Q(cap*) + sum_a mu_a * (cap_a(y) - cap*_a)

    ``mu``/``cap_star`` come straight from ``optimize_h2_producer.solve(...,
    return_duals=True)``'s ``out.attrs["cut_coeffs"]``/``out.attrs["capacities"]`` at
    the trial capacities the master just proposed. Cut names include the iteration
    index -- must stay unique for the whole run (the caller is responsible for passing
    a fresh ``iteration`` each call)."""
    y = m.variables["y"]
    theta = m.variables["theta"]
    k_idx = y.coords["k"]
    cap_expr = sum(
        mu[a] * (xr.DataArray(cand_mw[country][a], coords=[k_idx]) * y.sel(country=country, asset=a)).sum("k")
        for a in ASSETS
    )
    rhs = Q - sum(mu[a] * cap_star[a] for a in ASSETS)
    m.add_constraints(theta.sel(country=country) - cap_expr >= rhs, name=f"cut_{country}_{iteration}")


def extract_capacities(m: linopy.Model, countries: list[str],
                       cand_mw: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Read the master's current ``y`` solution and translate it back into MW per
    (country, asset) -- HiGHS MIP solutions can return ~0.9999997 rather than exactly
    1.0 for a chosen binary, so this thresholds at > 0.5, never ``== 1``. Since
    ``one_hot`` is ``<=1`` (not ``==1``), no candidate being chosen is a legitimate
    outcome -- "skip this asset" -- reported as 0.0 MW, not an arbitrary fallback
    candidate."""
    y_sol = m.variables["y"].solution
    out: dict[str, dict[str, float]] = {}
    for c in countries:
        out[c] = {}
        for a in ASSETS:
            chosen = np.flatnonzero(y_sol.sel(country=c, asset=a).to_numpy() > 0.5)
            out[c][a] = float(cand_mw[c][a][int(chosen[0])]) if len(chosen) else 0.0
    return out
