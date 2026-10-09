"""Candidate catalog, technology parameters and CAPEX/annualization assumptions for Industrial Site Investor planning.

Assets and the internal demand each one serves:

- electricity: wind, PV, battery
- space heating: heat pump
- low/medium-temperature process heat (up to ~150 C): industrial heat pump
- high-temperature process heat / steam: electric (electrode / resistance) boiler
- cooling: electric chiller
- hydrogen: electrolyser, H2 tank

Thermal assets are sized in MW of useful output (MW_th / MW_cold), the electrolyser in MW of
electrical input. ``HEAT_SERVICES`` are the demands the existing gas boiler backs up.

``DEMAND_PEAKS_MW`` is the default peak demand of a site in each country, in MW (MW_th for heat and
cooling, MW_LHV for hydrogen); a site's hourly demand is its per-unit profile in
``inputs/site_demand.csv`` times these peaks.

Each asset has one default product, which the master may build any number of times (up to
the site cap). Catalog sources: wind (Vestas V100-2.0 turbine class), PV (fixed-tilt, 40yr), battery (Li-ion, 2h, 20yr), electrolyser (PEM,
25yr) and H2 tank (compressed, ~16.7h, 30yr) follow the 2030 candidate-product table in
``Help/Candidates (Edited).docx``. Heat pump (20yr), industrial heat pump (25yr), electric boiler
(25yr) and electric chiller (20yr) are indicative 2030 installed costs in the range of public technology catalogues (e.g. the Danish
Energy Agency's); replace them with vendor quotes for a real site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

ASSETS = ["wind_mw", "pv_mw", "battery_mw",
          "heat_pump_mw", "industrial_heat_pump_mw", "electric_boiler_mw", "electric_chiller_mw",
          "electrolyser_mw", "tank_mw"]

SERVICES = ["electricity", "space_heat", "process_heat", "steam", "cooling", "hydrogen"]
HEAT_SERVICES = ["space_heat", "process_heat", "steam"]

DEMAND_PEAKS_MW: dict[str, dict[str, float]] = {
    "AT": {"electricity": 7.512, "space_heat": 3.2528, "process_heat": 6.0096, "steam": 4.5072, "cooling": 3.2571, "hydrogen": 1.5024},
    "BE": {"electricity": 7.512, "space_heat": 2.6614, "process_heat": 6.0096, "steam": 4.5072, "cooling": 2.1664, "hydrogen": 1.5024},
    "CZ": {"electricity": 7.512, "space_heat": 3.2528, "process_heat": 6.0096, "steam": 4.5072, "cooling": 2.7117, "hydrogen": 1.5024},
    "DE": {"electricity": 7.512, "space_heat": 2.9571, "process_heat": 6.0096, "steam": 4.5072, "cooling": 2.53, "hydrogen": 1.5024},
    "FR": {"electricity": 7.512, "space_heat": 2.2671, "process_heat": 6.0096, "steam": 4.5072, "cooling": 4.3477, "hydrogen": 1.5024},
    "HR": {"electricity": 7.512, "space_heat": 2.2671, "process_heat": 6.0096, "steam": 4.5072, "cooling": 10.8915, "hydrogen": 1.5024},
    "HU": {"electricity": 7.512, "space_heat": 2.6614, "process_heat": 6.0096, "steam": 4.5072, "cooling": 7.256, "hydrogen": 1.5024},
    "LU": {"electricity": 7.512, "space_heat": 2.8585, "process_heat": 6.0096, "steam": 4.5072, "cooling": 2.3482, "hydrogen": 1.5024},
    "NL": {"electricity": 7.512, "space_heat": 2.6614, "process_heat": 6.0096, "steam": 4.5072, "cooling": 2.1664, "hydrogen": 1.5024},
    "PL": {"electricity": 7.512, "space_heat": 3.2528, "process_heat": 6.0096, "steam": 4.5072, "cooling": 2.8935, "hydrogen": 1.5024},
    "RO": {"electricity": 7.512, "space_heat": 2.76, "process_heat": 6.0096, "steam": 4.5072, "cooling": 9.0738, "hydrogen": 1.5024},
    "SI": {"electricity": 7.512, "space_heat": 2.6614, "process_heat": 6.0096, "steam": 4.5072, "cooling": 5.0748, "hydrogen": 1.5024},
    "SK": {"electricity": 7.512, "space_heat": 3.0557, "process_heat": 6.0096, "steam": 4.5072, "cooling": 4.3477, "hydrogen": 1.5024},
}

THERMAL_ASSET_SERVICES = {
    "heat_pump_mw": ["space_heat"],
    "industrial_heat_pump_mw": ["process_heat"],
    "electric_boiler_mw": ["steam"],
    "electric_chiller_mw": ["cooling"],
}


class AssetCandidate(NamedTuple):
    """One discrete candidate product for an asset: MW, absolute CAPEX (EUR), and design lifetime (years)."""
    mw: float
    capex_eur: float
    lifetime_years: float
    mwh: float | None = None


CANDIDATE_CATALOG: dict[str, list[AssetCandidate]] = {
    "wind_mw": [
        AssetCandidate(mw=2.0, capex_eur=2_900_000.0, lifetime_years=30.0),
    ],
    "pv_mw": [
        AssetCandidate(mw=1.0, capex_eur=650_000.0, lifetime_years=40.0),
    ],
    "battery_mw": [
        AssetCandidate(mw=1.0, mwh=2.0, capex_eur=640_000.0, lifetime_years=20.0),
    ],
    "heat_pump_mw": [
        AssetCandidate(mw=0.5, capex_eur=450_000.0, lifetime_years=20.0),
    ],
    "industrial_heat_pump_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_000_000.0, lifetime_years=25.0),
    ],
    "electric_boiler_mw": [
        AssetCandidate(mw=1.0, capex_eur=150_000.0, lifetime_years=25.0),
    ],
    "electric_chiller_mw": [
        AssetCandidate(mw=0.5, capex_eur=200_000.0, lifetime_years=20.0),
    ],
    "electrolyser_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_300_000.0, lifetime_years=25.0),
    ],
    "tank_mw": [
        AssetCandidate(mw=0.5, mwh=8.3, capex_eur=540_000.0, lifetime_years=30.0),
    ],
}


@dataclass(frozen=True)
class SiteTechParams:
    """Operating parameters of the site's conversion assets, its existing backup plant and its market access.

    COPs/efficiencies are useful output per MWh of electricity in. The existing backup plant (sunk,
    no CAPEX, unlimited capacity) is a gas boiler for every heat demand and a legacy chiller running
    on site electricity for cooling, which keeps every subproblem feasible whatever the master
    proposes. The site buys and sells electricity and hydrogen at the modeled market price and pays
    the import fees (network charges, levies) on every MWh it buys.
    """

    heat_pump_cop: float = 3.0
    industrial_heat_pump_cop: float = 2.5
    electric_boiler_efficiency: float = 0.99
    electric_chiller_cop: float = 4.5

    gas_price_eur_per_mwh: float = 35.0
    co2_price_eur_per_t: float = 90.0
    gas_emission_t_per_mwh: float = 0.202
    gas_boiler_efficiency: float = 0.90
    legacy_chiller_cop: float = 3.0

    grid_import_fee_eur_per_mwh: float = 15.0
    h2_import_fee_eur_per_mwh: float = 0.0

    @property
    def gas_heat_cost_eur_per_mwh_th(self) -> float:
        """Cost of one MWh of heat from the existing gas boiler (fuel + CO2)."""
        return (self.gas_price_eur_per_mwh + self.co2_price_eur_per_t * self.gas_emission_t_per_mwh) \
            / self.gas_boiler_efficiency

    def cop(self, asset: str, service: str) -> float:
        """Useful thermal output per MWh of electricity when ``asset`` serves ``service``."""
        return {("heat_pump_mw", "space_heat"): self.heat_pump_cop,
                ("industrial_heat_pump_mw", "process_heat"): self.industrial_heat_pump_cop,
                ("electric_boiler_mw", "steam"): self.electric_boiler_efficiency,
                ("electric_chiller_mw", "cooling"): self.electric_chiller_cop}[(asset, service)]


SITE_TECH = SiteTechParams()


@dataclass(frozen=True)
class GreenH2Params:
    """Green (RFNBO) hydrogen rules and the green certificate market.

    At least ``green_share`` of the site's annual hydrogen demand must be green. Green hydrogen is
    either made by the site's electrolyser from additional renewable electricity, matched hour by
    hour (EU RFNBO temporal correlation from 2030), or bought as certified green hydrogen at
    ``green_h2_premium`` above the hydrogen market price. Additional renewable electricity is the
    site's own new wind/PV, or Guarantees of Origin (GOs) bought from additional plants. The site's
    own wind/PV exported to the grid earns GOs it can sell, unless that output is claimed for green
    hydrogen.

    Defaults: 42% green share (RED III 2030 RFNBO target for industrial hydrogen), GOs bought at
    8 and sold at 6 EUR/MWh, and a certified green H2 premium of 120 EUR/MWh (~EUR 4/kg).
    """

    green_share: float = 0.42
    go_buy_price_eur_per_mwh: float = 8.0
    go_sell_price_eur_per_mwh: float = 6.0
    green_h2_premium_eur_per_mwh: float = 120.0


GREEN_H2 = GreenH2Params()


@dataclass
class CapexAssumptions:
    """Candidate catalog plus annualization/budget parameters for the Benders master problem.

    ``site_max_mw`` is the largest MW of each market-facing asset one site may host (land, grid
    connection). Thermal assets are instead capped at ``thermal_oversize_factor`` times the site's
    own peak demand for their service, since their output can't be sold.
    """

    catalog: dict[str, list[AssetCandidate]] = field(default_factory=lambda: dict(CANDIDATE_CATALOG))

    discount_rate: float = 0.05
    default_budget_eur: float = 1_500_000_000.0
    theta_lower_bound_eur: float = -1e8

    site_max_mw: dict[str, float] = field(default_factory=lambda: {
        "wind_mw": 60.0, "pv_mw": 60.0, "battery_mw": 40.0, "electrolyser_mw": 40.0, "tank_mw": 20.0,
    })
    thermal_oversize_factor: float = 1.25

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
