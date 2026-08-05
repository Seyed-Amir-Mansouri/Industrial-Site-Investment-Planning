"""Discrete candidate capacity grids for H2 Producer capacity planning, centered on
today's Project-3-rank-derived sizing (``economic_dispatch/model.py::_h2_producer_sizing``).
See ``Formulation.md`` SS4.2.
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


def build_candidate_grid(default_mw: float, asset: str, cfg: CapexAssumptions) -> np.ndarray:
    """Candidate MW values for ``asset``, in priority order:

    1. ``cfg.candidate_grid_mw`` set -> that ABSOLUTE MW list, as-is -- same grid for
       every asset and every country, ignoring ``default_mw`` entirely (the country's
       default is still recorded separately and shown as a reference tick, just no
       longer used to place the grid).
    2. ``cfg.candidate_span_above_default_mw``/``candidate_step_mw`` both set -> a
       ONE-SIDED grid starting AT ``default_mw`` and stepping up by ``candidate_step_mw``
       until the span is covered, e.g. default=5, span=2, step=0.5 ->
       ``[5, 5.5, 6, 6.5, 7]``. Never goes below ``default_mw`` (unlike offsets below,
       which are symmetric) -- for a "must build at least the default" style run this
       guarantees every candidate already satisfies that floor, not just the default
       itself.
    3. Otherwise (original default): ``default_mw`` + ``cfg.candidate_offsets_mw[asset]``
       (a fixed, per-asset array of MW offsets, symmetric around default), floored at
       ``min_candidate_mw`` so no candidate is <= 0.

    Note: for a small-default asset (e.g. some country's ``pv_mw`` near the low end),
    the offset-based floor (mode 3) can make two of the lowest candidates collapse to
    the same clipped value -- harmless, since one-hot selection still picks exactly one
    candidate, just with a wasted duplicate in the grid."""
    if cfg.candidate_grid_mw is not None:
        return np.asarray(cfg.candidate_grid_mw, dtype=float)
    if cfg.candidate_span_above_default_mw is not None and cfg.candidate_step_mw is not None:
        n = int(round(cfg.candidate_span_above_default_mw / cfg.candidate_step_mw)) + 1
        return default_mw + np.arange(n) * cfg.candidate_step_mw
    offsets = np.asarray(cfg.candidate_offsets_mw[asset], dtype=float)
    return np.maximum(cfg.min_candidate_mw, default_mw + offsets)


def build_candidates(countries: list[str], capex_cfg: CapexAssumptions | None = None,
                     zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB):
    """For the given 2-letter ``countries``, return ``(default_mw, cand_mw, host_zone)``:

    * ``default_mw[c][a]`` -- today's rank-derived MW for asset ``a`` (see
      ``config.ASSETS``), country ``c`` -- or, if ``capex_cfg.default_mw_override`` is
      set, that FIXED value for every country alike (still per-asset, no longer
      per-country).
    * ``cand_mw[c][a]`` -- that asset's candidate ``np.ndarray`` (see
      ``build_candidate_grid``).
    * ``host_zone[c]`` -- ``c``'s main H2 zone (what ``optimize_h2_producer.solve()``
      needs as its ``zone`` argument).

    Raises ``ValueError`` naming the eligible-country list if any requested country has
    no Hydrogen Producer sizing (no H2 demand, or simply not a CORE-region country) --
    still checked even under ``default_mw_override``, since ``host_zone`` always comes
    from the rank engine's ``main_zones``."""
    capex_cfg = capex_cfg or CapexAssumptions()
    sizing, main_zones = default_sizing_and_zones(zones_db, networks_db)
    missing = [c for c in countries if c not in sizing]
    if missing:
        raise ValueError(f"no Hydrogen Producer sizing for {missing} -- eligible "
                         f"countries: {sorted(sizing)}")

    if capex_cfg.default_mw_override is not None:
        default_mw = {c: dict(capex_cfg.default_mw_override) for c in countries}
    else:
        default_mw = {c: {a: float(sizing[c][a]) for a in ASSETS} for c in countries}
    cand_mw = {c: {a: build_candidate_grid(default_mw[c][a], a, capex_cfg) for a in ASSETS}
              for c in countries}
    host_zone = {c: main_zones[c] for c in countries}
    return default_mw, cand_mw, host_zone
