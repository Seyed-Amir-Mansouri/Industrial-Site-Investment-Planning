"""Benders master MILP for Industrial Site Investor planning: site locations plus discrete asset sizes."""
from __future__ import annotations

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .config import ASSETS


def build_master(units: list[str], cand_mw: dict[str, dict[str, np.ndarray]],
                 cand_capex: dict[str, dict[str, np.ndarray]], budget: float | None,
                 crf: dict[str, float], theta_lower: float,
                 scenario_probs: dict[str, float] | None = None,
                 cvar_alpha: float | None = None,
                 site_max_mw: dict[str, dict[str, float]] | None = None,
                 max_units_per_candidate: int | None = None,
                 site_units: dict[str, list[str]] | None = None,
                 symmetric_sites: list[list[str]] | None = None) -> linopy.Model:
    """Build a fresh Benders master MILP (site location, candidate selection, budget, and optional
    CVaR/scenario objective) with no cuts yet. A unit is one site placed in one candidate country;
    ``site_units`` maps each site to its units, and every site is built in exactly one of them
    (several sites may share a country). Only the chosen unit may host capacity, up to
    ``site_max_mw`` of each asset. The recourse cost has one ``theta`` per unit (and scenario),
    bounded below by ``theta_lower`` and by that unit's own cuts. Each group in ``symmetric_sites``
    lists sites with identical settings, which are interchangeable; each must be placed in the same
    or a later country (in ``site_units`` order) than the one before it, so the master doesn't search
    mirror-image plans."""
    if site_units is None:
        site_units = {"site": list(units)}
    m = linopy.Model()
    unit_idx = pd.Index(units, name="unit")
    asset_idx = pd.Index(ASSETS, name="asset")
    n_k = len(cand_mw[units[0]][ASSETS[0]])
    k_idx = pd.Index(range(n_k), name="k")

    units_upper = np.inf if max_units_per_candidate is None else max_units_per_candidate
    y = m.add_variables(lower=0, upper=units_upper, integer=True, coords=[unit_idx, asset_idx, k_idx], name="y")
    site = m.add_variables(binary=True, coords=[unit_idx], name="site")
    for i, members in enumerate(site_units.values()):
        m.add_constraints(site.sel(unit=members).sum() == 1, name=f"place_site_{i}")
    for g, group in enumerate(symmetric_sites or []):
        for j, (a, b) in enumerate(zip(group, group[1:])):
            order_a = sum(k * site.sel(unit=u) for k, u in enumerate(site_units[a]) if k)
            order_b = sum(k * site.sel(unit=u) for k, u in enumerate(site_units[b]) if k)
            m.add_constraints(order_a - order_b <= 0, name=f"site_order_{g}_{j}")
    if cvar_alpha is not None and scenario_probs is None:
        raise ValueError("cvar_alpha requires scenario_probs")
    if scenario_probs is not None and len(scenario_probs) > 1 and cvar_alpha is None:
        raise ValueError("more than one scenario needs a CVaR confidence level (cvar_alpha)")
    if scenario_probs is None:
        theta = m.add_variables(lower=theta_lower, coords=[unit_idx], name="theta")
        theta_term = theta.sum()
    else:
        scenario_idx = pd.Index(list(scenario_probs), name="scenario")
        theta = m.add_variables(lower=theta_lower, coords=[scenario_idx, unit_idx], name="theta")
        theta_s = theta.sum("unit")
        prob_da = xr.DataArray(list(scenario_probs.values()), coords=[scenario_idx])
        if cvar_alpha is None or len(scenario_probs) == 1:
            theta_term = (theta_s * prob_da).sum()
        else:
            zeta = m.add_variables(name="zeta")
            u = m.add_variables(lower=0, coords=[scenario_idx], name="cvar_excess")
            m.add_constraints(u - theta_s + zeta >= 0, name="cvar_excess_def")
            theta_term = zeta + (u * prob_da).sum() / (1 - cvar_alpha)

    value_cost = xr.DataArray(
        np.array([[cand_capex[c][a] for a in ASSETS] for c in units]),
        coords=[unit_idx, asset_idx, k_idx],
    )
    annualized_value_cost = xr.DataArray(
        np.array([[cand_capex[c][a] * crf[a] for a in ASSETS] for c in units]),
        coords=[unit_idx, asset_idx, k_idx],
    )

    annualized_capex_expr = (annualized_value_cost * y).sum()
    if budget is not None:
        capex_expr = (value_cost * y).sum()
        m.add_constraints(capex_expr <= budget, name="budget")

    if site_max_mw is not None:
        cand_mw_da = xr.DataArray(
            np.array([[cand_mw[c][a] for a in ASSETS] for c in units]),
            coords=[unit_idx, asset_idx, k_idx],
        )
        cap_da = xr.DataArray(np.array([[site_max_mw[c][a] for a in ASSETS] for c in units]),
                              coords=[unit_idx, asset_idx])
        m.add_constraints((cand_mw_da * y).sum("k") - cap_da * site <= 0, name="site_max_mw")

    m.add_objective(annualized_capex_expr + theta_term)
    return m


def add_optimality_cut(m: linopy.Model, units: list[str], iteration: int,
                       Q: dict[str, float], mu: dict[str, dict[str, float]],
                       cap_star: dict[str, dict[str, float]],
                       cand_mw: dict[str, dict[str, np.ndarray]],
                       scenario: str | None = None,
                       lam: dict[str, float] | None = None,
                       site_star: dict[str, float] | None = None, tag: str = "core") -> None:
    """Add one Benders optimality cut per unit in ``units`` (for one scenario, or the only one) to
    the master, each bounding that unit's own ``theta``. ``mu`` are the recourse cost's
    sensitivities to each asset's MW, ``lam`` its sensitivity to the site's on/off (its internal
    demand switching on), both taken at ``cap_star``/``site_star``."""
    y = m.variables["y"]
    site = m.variables["site"]
    theta = m.variables["theta"]
    theta_var = theta.sel(scenario=scenario) if scenario is not None else theta
    k_idx = y.coords["k"]
    base = f"cut_{tag}_{scenario}_{iteration}" if scenario is not None else f"cut_{tag}_{iteration}"
    for c in units:
        cap_expr = sum(mu[c][a] * (xr.DataArray(cand_mw[c][a], coords=[k_idx]) * y.sel(unit=c, asset=a)).sum("k")
                       for a in ASSETS)
        rhs = Q[c] - sum(mu[c][a] * cap_star[c][a] for a in ASSETS)
        if lam is not None:
            cap_expr = cap_expr + lam[c] * site.sel(unit=c)
            rhs -= lam[c] * site_star[c]
        m.add_constraints(theta_var.sel(unit=c) - cap_expr >= rhs, name=f"{base}_{c}")


def cvar_value(values: dict[str, float], probs: dict[str, float], alpha: float) -> float:
    """Exact CVaR_alpha of a discrete distribution given by per-scenario values and probabilities."""
    tail_mass = 1.0 - alpha
    remaining = tail_mass
    total = 0.0
    for s in sorted(values, key=lambda s: values[s], reverse=True):
        take = min(probs[s], remaining)
        total += take * values[s]
        remaining -= take
        if remaining <= 1e-12:
            break
    return total / tail_mass


def extract_capacities(m: linopy.Model, units: list[str],
                       cand_mw: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Read the master's current ``y`` solution and translate it into total MW per (unit, asset)."""
    y_sol = m.variables["y"].solution
    out: dict[str, dict[str, float]] = {}
    for c in units:
        out[c] = {}
        for a in ASSETS:
            counts = np.rint(y_sol.sel(unit=c, asset=a).to_numpy())
            out[c][a] = float(np.sum(counts * cand_mw[c][a]))
    return out


def extract_sites(m: linopy.Model, units: list[str]) -> dict[str, float]:
    """1.0 for every unit (site, country) the master's current solution builds, else 0.0."""
    site_sol = m.variables["site"].solution
    return {c: float(np.rint(site_sol.sel(unit=c))) for c in units}


def extract_units(m: linopy.Model, units: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Integer unit count per (unit, asset) for each candidate product, as read from the master's ``y`` solution."""
    y_sol = m.variables["y"].solution
    return {c: {a: np.rint(y_sol.sel(unit=c, asset=a).to_numpy()).astype(int) for a in ASSETS}
            for c in units}


def extract_capex(m: linopy.Model, units: list[str],
                  cand_capex: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Same ``y``-solution reading as ``extract_capacities``, but returns each chosen candidate's absolute CAPEX (EUR) instead of MW."""
    y_sol = m.variables["y"].solution
    out: dict[str, dict[str, float]] = {}
    for c in units:
        out[c] = {}
        for a in ASSETS:
            counts = np.rint(y_sol.sel(unit=c, asset=a).to_numpy())
            out[c][a] = float(np.sum(counts * cand_capex[c][a]))
    return out
