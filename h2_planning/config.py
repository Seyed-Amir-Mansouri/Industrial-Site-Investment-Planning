"""Candidate-catalog CAPEX / lifetime / annualization assumptions for H2 Producer capacity planning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

ASSETS = ["electrolyser_mw", "wind_mw", "pv_mw", "battery_mw", "tank_mw"]


class AssetCandidate(NamedTuple):
    """One discrete candidate product for an asset: MW, absolute CAPEX (EUR), and design lifetime (years)."""
    mw: float
    capex_eur: float
    lifetime_years: float
    mwh: float | None = None


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
    """Candidate catalog plus annualization/budget parameters for the Benders master problem."""

    catalog: dict[str, list[AssetCandidate]] = field(default_factory=lambda: CANDIDATE_CATALOG)

    discount_rate: float = 0.05
    default_budget_eur: float = 500_000_000.0
    theta_lower_bound_eur: float = -1e8

    lifetime_years: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.lifetime_years = self._lifetime_years_from_catalog()

    def _lifetime_years_from_catalog(self) -> dict[str, float]:
        """Each asset's lifetime, constant across its candidates in ``catalog``."""
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
        """Per-asset capital recovery factor from ``discount_rate`` and ``lifetime_years``."""
        return {a: self._crf(self.discount_rate, n) for a, n in self.lifetime_years.items()}
