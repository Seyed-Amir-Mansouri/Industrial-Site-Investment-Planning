"""Candidate-catalog CAPEX / lifetime / annualization assumptions for H2 Producer
capacity PLANNING (as opposed to price_model's demand -> price fitting, or
optimize_h2_producer's fixed-capacity operational LP).

``CANDIDATE_CATALOG`` below is sourced from ``Help/Candidates (Edited).docx``'s
candidate-product table (2030 CAPEX/lifetime columns -- the docx also has a
"current"-year column, not used here per user instruction) -- four discrete,
named real-world product sizes per asset, each with its own absolute CAPEX (already
bundling power + energy cost for battery/tank, no separate EUR/MW vs. EUR/MWh split
needed) and design lifetime. This replaced an earlier placeholder/synthetic
CAPEX model (single blended EUR/MW figure per asset, candidate grid centered on
today's rank-derived default +/- an offset array) -- see git history around
2026-08-06 to restore that approach. See ``Formulation.md`` SS4.2/SS4.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

ASSETS = ["electrolyser_mw", "wind_mw", "pv_mw", "battery_mw", "tank_mw"]


class AssetCandidate(NamedTuple):
    """One discrete candidate product for an asset -- MW (and, for battery/tank,
    MWh), absolute CAPEX (EUR), and design lifetime (years). Straight from one row
    of ``Help/Candidates (Edited).docx``'s table."""
    mw: float
    capex_eur: float
    lifetime_years: float
    mwh: float | None = None


# 2030 CAPEX/lifetime columns from Help/Candidates (Edited).docx. Every asset has
# exactly 4 candidates, sorted by MW ascending -- h2_planning/master.py::build_master
# sizes the one-hot selection variable's candidate dimension k off
# len(cand_mw[countries[0]][ASSETS[0]]), a single shared k axis across every asset,
# so all five lists must stay equal length. Battery/tank MWh here (each asset's own
# power:energy ratio varies candidate-to-candidate -- e.g. battery is 2h at the two
# smaller sizes but 4h at the two larger ones) is NOT separately enforced by the
# subproblem LP, which still sizes MWh off optimize_h2_producer.py's fixed
# h2_producer_{battery,tank}_duration_hours (2h/24h) regardless of which candidate
# was nominally selected -- a known simplification, see Formulation.md SS4.3.1.
CANDIDATE_CATALOG: dict[str, list[AssetCandidate]] = {
    "electrolyser_mw": [
        AssetCandidate(mw=5.0, capex_eur=5_210_000.0, lifetime_years=25.0),
        AssetCandidate(mw=20.0, capex_eur=18_500_000.0, lifetime_years=25.0),
        AssetCandidate(mw=50.0, capex_eur=38_600_000.0, lifetime_years=25.0),
        AssetCandidate(mw=100.0, capex_eur=65_500_000.0, lifetime_years=25.0),
    ],
    "wind_mw": [
        AssetCandidate(mw=5.0, capex_eur=6_640_000.0, lifetime_years=30.0),
        AssetCandidate(mw=10.0, capex_eur=13_300_000.0, lifetime_years=30.0),
        AssetCandidate(mw=50.0, capex_eur=66_400_000.0, lifetime_years=30.0),
        AssetCandidate(mw=100.0, capex_eur=133_000_000.0, lifetime_years=30.0),
    ],
    "pv_mw": [
        AssetCandidate(mw=5.0, capex_eur=2_500_000.0, lifetime_years=40.0),
        AssetCandidate(mw=25.0, capex_eur=12_500_000.0, lifetime_years=40.0),
        AssetCandidate(mw=50.0, capex_eur=25_000_000.0, lifetime_years=40.0),
        AssetCandidate(mw=100.0, capex_eur=50_000_000.0, lifetime_years=40.0),
    ],
    "battery_mw": [
        AssetCandidate(mw=2.0, mwh=4.0, capex_eur=1_130_000.0, lifetime_years=20.0),
        AssetCandidate(mw=10.0, mwh=20.0, capex_eur=5_640_000.0, lifetime_years=20.0),
        AssetCandidate(mw=20.0, mwh=80.0, capex_eur=20_600_000.0, lifetime_years=20.0),
        AssetCandidate(mw=50.0, mwh=200.0, capex_eur=51_400_000.0, lifetime_years=20.0),
    ],
    "tank_mw": [
        AssetCandidate(mw=1.0, mwh=16.7, capex_eur=950_000.0, lifetime_years=30.0),
        AssetCandidate(mw=5.0, mwh=166.7, capex_eur=3_750_000.0, lifetime_years=30.0),
        AssetCandidate(mw=20.0, mwh=666.7, capex_eur=14_000_000.0, lifetime_years=30.0),
        AssetCandidate(mw=50.0, mwh=3_333.0, capex_eur=65_000_000.0, lifetime_years=30.0),
    ],
}


@dataclass
class CapexAssumptions:
    """Candidate catalog (CAPEX + lifetime, per asset) plus the annualization/budget
    parameters used by the Benders capacity-planning master problem
    (``h2_planning/master.py``). ``catalog`` defaults to ``CANDIDATE_CATALOG`` above
    (Help/Candidates (Edited).docx's 2030 column) -- override it to substitute a
    different candidate set (e.g. the docx's "current"-year column, or real vendor
    quotes) without touching any other code."""

    catalog: dict[str, list[AssetCandidate]] = field(default_factory=lambda: CANDIDATE_CATALOG)

    discount_rate: float = 0.05
    default_budget_eur: float = 500_000_000.0
    theta_lower_bound_eur: float = -1e8

    lifetime_years: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.lifetime_years = self._lifetime_years_from_catalog()

    def _lifetime_years_from_catalog(self) -> dict[str, float]:
        """Each asset's lifetime -- constant across its candidates in ``catalog``
        (true for the 2030 docx data). Raises if a substituted catalog varies
        lifetime WITHIN an asset, since the master's per-asset CRF (below) can't
        represent that."""
        out = {}
        for a in ASSETS:
            years = {c.lifetime_years for c in self.catalog[a]}
            if len(years) != 1:
                raise ValueError(f"{a}: candidates have mixed lifetimes {sorted(years)} -- "
                                 f"the master's per-asset CRF can't represent that; set "
                                 f"capex_cfg.lifetime_years[{a!r}] explicitly instead")
            out[a] = years.pop()
        return out

    @staticmethod
    def _crf(r: float, n: float) -> float:
        if r == 0:
            return 1.0 / n
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def capital_recovery_factors(self) -> dict[str, float]:
        """Per-asset CRF, using the shared ``discount_rate`` and each asset's own
        ``lifetime_years`` entry (defaults from the catalog; override ``lifetime_years``
        directly, e.g. via ``plan_h2_capacity.py --lifetime-years``, for a different
        service-life assumption without touching the CAPEX catalog itself)."""
        return {a: self._crf(self.discount_rate, n) for a, n in self.lifetime_years.items()}
