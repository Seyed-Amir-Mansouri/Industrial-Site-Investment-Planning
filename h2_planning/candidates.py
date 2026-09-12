"""Discrete candidate capacity grids for H2 Producer capacity planning -- one fixed
real-world product catalog per asset (``config.CANDIDATE_CATALOG``, same MW/CAPEX for
every country), plus today's Project-3-rank-derived sizing
(``economic_dispatch/model.py::_h2_producer_sizing``) kept only as a reference value
for display (no longer used to place the candidate grid). See ``Formulation.md`` SS4.2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from economic_dispatch import model as p3model
from economic_dispatch.config import RunConfig

from .config import ASSETS, CapexAssumptions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZONES_DB = ROOT / "inputs" / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = ROOT / "inputs" / "networks_2030.parquet"


def default_sizing_and_zones(zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB):
    """(sizing, main_zones) for EVERY country with a Hydrogen Producer -- both
    ``_h2_producer_sizing`` and ``_h2_main_zones`` rank/resolve against the FULL stored
    zone database regardless of ``cfg.zones`` (see their own docstrings: this stability
    is deliberate, so a subset-of-countries planning run doesn't shift any other
    country's sizing), so one plain ``RunConfig`` covering all 20 CORE zones is enough
    to get every eligible country's row at once, independent of which countries this
    planning run actually targets."""
    cfg = RunConfig(zones_db=zones_db, networks_db=networks_db, enable_h2_producer=True)
    sizing = p3model._h2_producer_sizing(cfg)
    main_zones = p3model._h2_main_zones(cfg)
    return sizing, main_zones


def build_candidates(countries: list[str], capex_cfg: CapexAssumptions | None = None,
                     zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB):
    """For the given 2-letter ``countries``, return ``(default_mw, cand_mw, cand_capex,
    host_zone)``:

    * ``default_mw[c][a]`` -- today's rank-derived MW for asset ``a`` (see
      ``config.ASSETS``), country ``c`` -- reference value only (shown as a tick in the
      artifact UI), no longer used to place the candidate grid.
    * ``cand_mw[c][a]`` -- that asset's candidate MW ``np.ndarray``, straight off
      ``capex_cfg.catalog[a]`` (SAME for every country -- this catalog has no
      per-country cost/size variation).
    * ``cand_capex[c][a]`` -- that asset's candidate absolute CAPEX (EUR) ``np.ndarray``,
      same shape/order as ``cand_mw[c][a]`` -- ``h2_planning.master.build_master`` prices
      candidate ``k`` at ``cand_capex[c][a][k]`` directly (no EUR/MW multiplication;
      the catalog's economies-of-scale are real and non-linear in MW, see
      ``Formulation.md`` SS4.3).
    * ``host_zone[c]`` -- ``c``'s main H2 zone (what ``optimize_h2_producer.solve()``
      needs as its ``zone`` argument).

    Raises ``ValueError`` naming the eligible-country list if any requested country has
    no Hydrogen Producer sizing (no H2 demand, or simply not a CORE-region country)."""
    capex_cfg = capex_cfg or CapexAssumptions()
    sizing, main_zones = default_sizing_and_zones(zones_db, networks_db)
    missing = [c for c in countries if c not in sizing]
    if missing:
        raise ValueError(f"no Hydrogen Producer sizing for {missing} -- eligible "
                         f"countries: {sorted(sizing)}")

    default_mw = {c: {a: float(sizing[c][a]) for a in ASSETS} for c in countries}
    cand_mw_shared = {a: np.asarray([cand.mw for cand in capex_cfg.catalog[a]], dtype=float)
                      for a in ASSETS}
    cand_capex_shared = {a: np.asarray([cand.capex_eur for cand in capex_cfg.catalog[a]], dtype=float)
                         for a in ASSETS}
    cand_mw = {c: dict(cand_mw_shared) for c in countries}
    cand_capex = {c: dict(cand_capex_shared) for c in countries}
    host_zone = {c: main_zones[c] for c in countries}
    return default_mw, cand_mw, cand_capex, host_zone
