"""Candidate site locations and discrete capacity grids for Industrial Site Investor planning."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from economic_dispatch import model as ed_model
from economic_dispatch.config import RunConfig

import math

from .config import ASSETS, STORAGE_ENERGY, CapexAssumptions, SiteSpec

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZONES_DB = ROOT / "inputs" / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = ROOT / "inputs" / "networks_2030.parquet"


def candidate_site_zones(zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB) -> dict[str, str]:
    """{country: host zone} for every country a site can be built in -- the countries with both an
    electricity and a hydrogen market (and so a hydrogen price model), hosted in their main H2 zone."""
    cfg = RunConfig(zones_db=zones_db, networks_db=networks_db)
    with_h2 = ed_model._g_investor_sizing(cfg)
    main_zones = ed_model._h2_main_zones(cfg)
    return {c: main_zones[c] for c in sorted(with_h2)}


def build_candidates(countries: list[str], capex_cfg: CapexAssumptions | None = None,
                     zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB):
    """Per-country candidate MW/CAPEX grids and host zones for the given countries. The MW grid also
    holds each storage asset's energy (MWh) per product under its ``STORAGE_ENERGY`` key, on the
    same product index as its MW.

    Assets may have different numbers of products in the catalog; shorter lists are padded with
    zero-MW, zero-CAPEX products so every asset has the same grid length in the master, and a
    padded product can never add capacity or cost."""
    capex_cfg = capex_cfg or CapexAssumptions()
    site_zones = candidate_site_zones(zones_db, networks_db)
    missing = [c for c in countries if c not in site_zones]
    if missing:
        raise ValueError(f"no candidate site for {missing} -- eligible countries: {sorted(site_zones)}")

    n_k = max(len(capex_cfg.catalog[a]) for a in ASSETS)
    for a in STORAGE_ENERGY:
        if any(not cand.mwh or cand.mwh <= 0 for cand in capex_cfg.catalog[a]):
            raise ValueError(f"every {a} product needs an energy capacity (mwh > 0)")

    def padded(values: list[float]) -> np.ndarray:
        """Candidate values padded with zeros to the longest asset's product count."""
        return np.asarray(values + [0.0] * (n_k - len(values)), dtype=float)

    cand_mw_shared = {a: padded([cand.mw for cand in capex_cfg.catalog[a]]) for a in ASSETS}
    cand_mw_shared.update({e: padded([cand.mwh for cand in capex_cfg.catalog[a]]) for a, e in STORAGE_ENERGY.items()})
    cand_capex_shared = {a: padded([cand.capex_eur for cand in capex_cfg.catalog[a]]) for a in ASSETS}
    cand_mw = {c: dict(cand_mw_shared) for c in countries}
    cand_capex = {c: dict(cand_capex_shared) for c in countries}

    host_zone = {c: site_zones[c] for c in countries}
    return cand_mw, cand_capex, host_zone


def rule_mw(rule: tuple[float, list[dict[str, float]]], peaks_mw: dict[str, float], product_mw: float) -> float:
    """One cap rule at these annual peaks: its factor times the largest of its weighted peak sums,
    rounded up to a whole number of ``product_mw`` products and then to a whole MW."""
    factor, groups = rule
    need = round(factor * max(sum(w * peaks_mw.get(s, 0.0) for s, w in g.items()) for g in groups), 6)
    if need <= 0:
        return 0.0
    if product_mw > 0:
        need = round(math.ceil(round(need / product_mw, 6)) * product_mw, 6)
    return float(math.ceil(need))


def default_max_mw(peaks_mw: dict[str, float], capex_cfg: CapexAssumptions | None = None) -> dict[str, float]:
    """Default largest MW of each asset for a site with these annual peaks: each ``cap_rules`` rule
    (see ``rule_mw``) with the asset's smallest catalog product."""
    capex_cfg = capex_cfg or CapexAssumptions()
    return {a: rule_mw(rule, peaks_mw, min((c.mw for c in capex_cfg.catalog[a] if c.mw > 0), default=0.0))
            for a, rule in capex_cfg.cap_rules.items()}


def site_max_mw(spec: SiteSpec, capex_cfg: CapexAssumptions | None = None) -> dict[str, float]:
    """Largest MW of each asset one site may host: ``default_max_mw`` of the site's peaks, replaced by
    the site's own ``max_mw`` where it gives one; an asset outside its ``active_assets`` gets 0."""
    caps = default_max_mw(spec.peaks_mw, capex_cfg)
    caps.update(spec.max_mw)
    if spec.active_assets is not None:
        caps = {a: (caps[a] if a in spec.active_assets else 0.0) for a in caps}
    return caps
