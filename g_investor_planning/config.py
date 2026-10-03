"""Candidate-catalog CAPEX / lifetime / annualization assumptions for General Investor capacity planning."""
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
    # PEM electrolyser modules, 25yr design life. Larger modules get cheaper per MW
    # (balance-of-plant/compression economies of scale); ~5.0MW module is the original
    # Candidates-doc reference point, others are realistic adjacent module sizes.
    "electrolyser_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_300_000.0, lifetime_years=25.0),
        AssetCandidate(mw=2.5, capex_eur=2_750_000.0, lifetime_years=25.0),
        AssetCandidate(mw=5.0, capex_eur=5_210_000.0, lifetime_years=25.0),
        AssetCandidate(mw=10.0, capex_eur=9_500_000.0, lifetime_years=25.0),
        AssetCandidate(mw=20.0, capex_eur=17_000_000.0, lifetime_years=25.0),
    ],
    # Onshore wind turbines, 30yr design life -- real commercial turbine classes
    # (nameplate MW with turbine+BOP+grid-connection installed cost).
    "wind_mw": [
        AssetCandidate(mw=2.0, capex_eur=2_900_000.0, lifetime_years=30.0),    # Vestas V100-2.0 class
        AssetCandidate(mw=2.75, capex_eur=3_795_000.0, lifetime_years=30.0),   # GE 2.75-120 class
        AssetCandidate(mw=4.2, capex_eur=5_376_000.0, lifetime_years=30.0),    # Vestas V150-4.2 class
        AssetCandidate(mw=5.0, capex_eur=6_640_000.0, lifetime_years=30.0),    # original reference point
        AssetCandidate(mw=5.8, capex_eur=7_076_000.0, lifetime_years=30.0),    # Siemens Gamesa SG 5.8-170 class
    ],
    # Utility-scale fixed-tilt PV, 40yr design life -- larger plants get cheaper per MWp.
    "pv_mw": [
        AssetCandidate(mw=1.0, capex_eur=650_000.0, lifetime_years=40.0),
        AssetCandidate(mw=2.5, capex_eur=1_400_000.0, lifetime_years=40.0),
        AssetCandidate(mw=5.0, capex_eur=2_500_000.0, lifetime_years=40.0),
        AssetCandidate(mw=10.0, capex_eur=4_600_000.0, lifetime_years=40.0),
        AssetCandidate(mw=20.0, capex_eur=8_400_000.0, lifetime_years=40.0),
    ],
    # Li-ion battery energy storage, fixed 2h duration, 20yr design life.
    "battery_mw": [
        AssetCandidate(mw=1.0, mwh=2.0, capex_eur=640_000.0, lifetime_years=20.0),
        AssetCandidate(mw=2.0, mwh=4.0, capex_eur=1_130_000.0, lifetime_years=20.0),
        AssetCandidate(mw=5.0, mwh=10.0, capex_eur=2_700_000.0, lifetime_years=20.0),
        AssetCandidate(mw=10.0, mwh=20.0, capex_eur=5_100_000.0, lifetime_years=20.0),
        AssetCandidate(mw=20.0, mwh=40.0, capex_eur=9_600_000.0, lifetime_years=20.0),
    ],
    # Compressed-H2 tank storage, fixed ~16.7h duration (matches the original reference
    # candidate's MW:MWh ratio), 30yr design life.
    "tank_mw": [
        AssetCandidate(mw=0.5, mwh=8.3, capex_eur=540_000.0, lifetime_years=30.0),
        AssetCandidate(mw=1.0, mwh=16.7, capex_eur=950_000.0, lifetime_years=30.0),
        AssetCandidate(mw=2.0, mwh=33.3, capex_eur=1_733_000.0, lifetime_years=30.0),
        AssetCandidate(mw=5.0, mwh=83.3, capex_eur=4_000_000.0, lifetime_years=30.0),
        AssetCandidate(mw=10.0, mwh=166.7, capex_eur=7_333_000.0, lifetime_years=30.0),
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
