"""H2 Producer capacity PLANNING via Benders decomposition -- discrete/binary capacity
choice for the 5 Hydrogen Producer assets (electrolyser, wind, PV, battery, H2 tank)
under a system-wide CAPEX budget, built on top of ``optimize_h2_producer.py``'s fixed-
capacity operational LP. See ``Formulation.md`` SS4 and ``plan_h2_capacity.py``.
"""
from .config import ASSETS, AssetCandidate, CANDIDATE_CATALOG, CapexAssumptions
from .candidates import build_candidates, default_sizing_and_zones
from .master import add_optimality_cut, build_master, extract_capacities, extract_capex

__all__ = ["ASSETS", "AssetCandidate", "CANDIDATE_CATALOG", "CapexAssumptions",
          "build_candidates", "default_sizing_and_zones", "add_optimality_cut",
          "build_master", "extract_capacities", "extract_capex"]
