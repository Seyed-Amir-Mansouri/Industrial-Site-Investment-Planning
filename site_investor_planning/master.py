"""Benders master MILP for Industrial Site Investor planning: site locations plus discrete asset sizes."""
from __future__ import annotations

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .config import ASSETS


def build_master(countries: list[str], cand_mw: dict[str, dict[str, np.ndarray]],
                 cand_capex: dict[str, dict[str, np.ndarray]], budget: float | None,
                 crf: dict[str, float], theta_lower: float,
                 scenario_probs: dict[str, float] | None = None,
                 cvar_alpha: float | None = None,
                 site_max_mw: dict[str, dict[str, float]] | None = None,
                 max_units_per_candidate: int | None = None,
                 n_sites: int = 1) -> linopy.Model:
    """Build a fresh Benders master MILP (site selection, candidate selection, budget, and optional
    CVaR/scenario objective) with no cuts yet. Exactly ``n_sites`` of ``countries`` get a site, and
    only a chosen site may host capacity, up to ``site_max_mw`` of each asset."""
    if not 1 <= n_sites <= len(countries):
        raise ValueError(f"n_sites must be between 1 and the {len(countries)} candidate countries, got {n_sites}")
    m = linopy.Model()
    country_idx = pd.Index(countries, name="country")
    asset_idx = pd.Index(ASSETS, name="asset")
    n_k = len(cand_mw[countries[0]][ASSETS[0]])
    k_idx = pd.Index(range(n_k), name="k")

    units_upper = np.inf if max_units_per_candidate is None else max_units_per_candidate
    y = m.add_variables(lower=0, upper=units_upper, integer=True, coords=[country_idx, asset_idx, k_idx], name="y")
    site = m.add_variables(binary=True, coords=[country_idx], name="site")
    m.add_constraints(site.sum() == n_sites, name="n_sites")
    if cvar_alpha is not None and scenario_probs is None:
        raise ValueError("cvar_alpha requires scenario_probs")
    if scenario_probs is None:
        theta = m.add_variables(lower=theta_lower * len(countries), name="theta")
        theta_term = theta
    else:
        scenario_idx = pd.Index(list(scenario_probs), name="scenario")
        theta = m.add_variables(lower=theta_lower * len(countries), coords=[scenario_idx], name="theta")
        prob_da = xr.DataArray(list(scenario_probs.values()), coords=[scenario_idx])
        if cvar_alpha is None:
            theta_term = (theta * prob_da).sum()
        else:
            zeta = m.add_variables(name="zeta")
            u = m.add_variables(lower=0, coords=[scenario_idx], name="cvar_excess")
            m.add_constraints(u >= theta - zeta, name="cvar_excess_def")
            theta_term = zeta + (u * prob_da).sum() / (1 - cvar_alpha)

    value_cost = xr.DataArray(
        np.array([[cand_capex[c][a] for a in ASSETS] for c in countries]),
        coords=[country_idx, asset_idx, k_idx],
    )
    annualized_value_cost = xr.DataArray(
        np.array([[cand_capex[c][a] * crf[a] for a in ASSETS] for c in countries]),
        coords=[country_idx, asset_idx, k_idx],
    )

    annualized_capex_expr = (annualized_value_cost * y).sum()
    if budget is not None:
        capex_expr = (value_cost * y).sum()
        m.add_constraints(capex_expr <= budget, name="budget")

    if site_max_mw is not None:
        cand_mw_da = xr.DataArray(
            np.array([[cand_mw[c][a] for a in ASSETS] for c in countries]),
            coords=[country_idx, asset_idx, k_idx],
        )
        cap_da = xr.DataArray(np.array([[site_max_mw[c][a] for a in ASSETS] for c in countries]),
                              coords=[country_idx, asset_idx])
        m.add_constraints((cand_mw_da * y).sum("k") - cap_da * site <= 0, name="site_max_mw")

    m.add_objective(annualized_capex_expr + theta_term)
    return m


def add_optimality_cut(m: linopy.Model, countries: list[str], iteration: int,
                       Q: dict[str, float], mu: dict[str, dict[str, float]],
                       cap_star: dict[str, dict[str, float]],
                       cand_mw: dict[str, dict[str, np.ndarray]],
                       scenario: str | None = None,
                       lam: dict[str, float] | None = None,
                       site_star: dict[str, float] | None = None, tag: str = "core") -> None:
    """Add one combined Benders optimality cut (across all countries, or for one scenario) to the master.
    ``mu`` are the recourse cost's sensitivities to each asset's MW, ``lam`` its sensitivity to each
    site's on/off (its internal demand switching on), both taken at ``cap_star``/``site_star``."""
    y = m.variables["y"]
    site = m.variables["site"]
    theta = m.variables["theta"]
    theta_var = theta.sel(scenario=scenario) if scenario is not None else theta
    k_idx = y.coords["k"]
    cap_expr = sum(
        mu[c][a] * (xr.DataArray(cand_mw[c][a], coords=[k_idx]) * y.sel(country=c, asset=a)).sum("k")
        for c in countries for a in ASSETS
    )
    rhs = sum(Q[c] - sum(mu[c][a] * cap_star[c][a] for a in ASSETS) for c in countries)
    if lam is not None:
        cap_expr = cap_expr + sum(lam[c] * site.sel(country=c) for c in countries)
        rhs -= sum(lam[c] * site_star[c] for c in countries)
    cut_name = f"cut_{tag}_{scenario}_{iteration}" if scenario is not None else f"cut_{tag}_{iteration}"
    m.add_constraints(theta_var - cap_expr >= rhs, name=cut_name)


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


def extract_capacities(m: linopy.Model, countries: list[str],
                       cand_mw: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Read the master's current ``y`` solution and translate it into total MW per (country, asset)."""
    y_sol = m.variables["y"].solution
    out: dict[str, dict[str, float]] = {}
    for c in countries:
        out[c] = {}
        for a in ASSETS:
            counts = np.rint(y_sol.sel(country=c, asset=a).to_numpy())
            out[c][a] = float(np.sum(counts * cand_mw[c][a]))
    return out


def extract_sites(m: linopy.Model, countries: list[str]) -> dict[str, float]:
    """1.0 for every country the master's current solution builds a site in, else 0.0."""
    site_sol = m.variables["site"].solution
    return {c: float(np.rint(site_sol.sel(country=c))) for c in countries}


def extract_units(m: linopy.Model, countries: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Integer unit count per (country, asset) for each candidate product, as read from the master's ``y`` solution."""
    y_sol = m.variables["y"].solution
    return {c: {a: np.rint(y_sol.sel(country=c, asset=a).to_numpy()).astype(int) for a in ASSETS}
            for c in countries}


def extract_capex(m: linopy.Model, countries: list[str],
                  cand_capex: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Same ``y``-solution reading as ``extract_capacities``, but returns each chosen candidate's absolute CAPEX (EUR) instead of MW."""
    y_sol = m.variables["y"].solution
    out: dict[str, dict[str, float]] = {}
    for c in countries:
        out[c] = {}
        for a in ASSETS:
            counts = np.rint(y_sol.sel(country=c, asset=a).to_numpy())
            out[c][a] = float(np.sum(counts * cand_capex[c][a]))
    return out
