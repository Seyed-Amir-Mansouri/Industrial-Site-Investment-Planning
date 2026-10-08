"""Candidate catalog, technology parameters and CAPEX/annualization assumptions for Industrial Site Investor planning.

Assets and the internal demand each one serves:

- electricity: wind, PV, battery
- space heating: heat pump
- low-temperature process heat (< ~100 C): low-temperature industrial heat pump
- medium-temperature process heat (~100-150 C): medium-temperature industrial heat pump
- high-temperature heat (> ~150 C, non-steam): electric (resistance / induction) heater
- high-temperature steam: electric (electrode) steam boiler
- cooling: electric chiller
- hydrogen: electrolyser, H2 tank

Thermal assets are sized in MW of useful output (MW_th / MW_cold), the electrolyser in MW of
electrical input. ``HEAT_SERVICES`` are the demands the existing gas boiler backs up.

Catalog sources: wind (real turbine classes: Vestas V100-2.0, GE 2.75-120, Vestas V150-4.2,
Siemens Gamesa SG 5.8-170), PV (fixed-tilt, 40yr), battery (Li-ion, 2h, 20yr), electrolyser (PEM,
25yr) and H2 tank (compressed, ~16.7h, 30yr) follow the 2030 candidate-product table in
``Help/Candidates (Edited).docx``. Heat pump (20yr), low/medium-temperature industrial heat pumps
(25yr), electric heater (20yr), electric steam boiler (25yr) and electric chiller (20yr) are
indicative 2030 installed costs in the range of public technology catalogues (e.g. the Danish
Energy Agency's); replace them with vendor quotes for a real site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

ASSETS = ["wind_mw", "pv_mw", "battery_mw",
          "heat_pump_mw", "lt_heat_pump_mw", "mt_heat_pump_mw", "electric_heater_mw", "electric_boiler_mw",
          "electric_chiller_mw", "electrolyser_mw", "tank_mw"]

SERVICES = ["electricity", "space_heat", "lt_process_heat", "mt_process_heat", "ht_heat", "steam",
            "cooling", "hydrogen"]
HEAT_SERVICES = ["space_heat", "lt_process_heat", "mt_process_heat", "ht_heat", "steam"]

THERMAL_ASSET_SERVICES = {
    "heat_pump_mw": ["space_heat"],
    "lt_heat_pump_mw": ["lt_process_heat"],
    "mt_heat_pump_mw": ["mt_process_heat"],
    "electric_heater_mw": ["ht_heat"],
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
        AssetCandidate(mw=2.75, capex_eur=3_795_000.0, lifetime_years=30.0),
        AssetCandidate(mw=4.2, capex_eur=5_376_000.0, lifetime_years=30.0),
        AssetCandidate(mw=5.0, capex_eur=6_640_000.0, lifetime_years=30.0),
        AssetCandidate(mw=5.8, capex_eur=7_076_000.0, lifetime_years=30.0),
    ],
    "pv_mw": [
        AssetCandidate(mw=1.0, capex_eur=650_000.0, lifetime_years=40.0),
        AssetCandidate(mw=2.5, capex_eur=1_400_000.0, lifetime_years=40.0),
        AssetCandidate(mw=5.0, capex_eur=2_500_000.0, lifetime_years=40.0),
        AssetCandidate(mw=10.0, capex_eur=4_600_000.0, lifetime_years=40.0),
        AssetCandidate(mw=20.0, capex_eur=8_400_000.0, lifetime_years=40.0),
    ],
    "battery_mw": [
        AssetCandidate(mw=1.0, mwh=2.0, capex_eur=640_000.0, lifetime_years=20.0),
        AssetCandidate(mw=2.0, mwh=4.0, capex_eur=1_130_000.0, lifetime_years=20.0),
        AssetCandidate(mw=5.0, mwh=10.0, capex_eur=2_700_000.0, lifetime_years=20.0),
        AssetCandidate(mw=10.0, mwh=20.0, capex_eur=5_100_000.0, lifetime_years=20.0),
        AssetCandidate(mw=20.0, mwh=40.0, capex_eur=9_600_000.0, lifetime_years=20.0),
    ],
    "heat_pump_mw": [
        AssetCandidate(mw=0.5, capex_eur=450_000.0, lifetime_years=20.0),
        AssetCandidate(mw=1.0, capex_eur=800_000.0, lifetime_years=20.0),
        AssetCandidate(mw=2.0, capex_eur=1_500_000.0, lifetime_years=20.0),
        AssetCandidate(mw=5.0, capex_eur=3_500_000.0, lifetime_years=20.0),
        AssetCandidate(mw=10.0, capex_eur=6_500_000.0, lifetime_years=20.0),
    ],
    "lt_heat_pump_mw": [
        AssetCandidate(mw=1.0, capex_eur=900_000.0, lifetime_years=25.0),
        AssetCandidate(mw=2.5, capex_eur=2_000_000.0, lifetime_years=25.0),
        AssetCandidate(mw=5.0, capex_eur=3_600_000.0, lifetime_years=25.0),
        AssetCandidate(mw=10.0, capex_eur=6_800_000.0, lifetime_years=25.0),
        AssetCandidate(mw=20.0, capex_eur=12_500_000.0, lifetime_years=25.0),
    ],
    "mt_heat_pump_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_200_000.0, lifetime_years=25.0),
        AssetCandidate(mw=2.5, capex_eur=2_750_000.0, lifetime_years=25.0),
        AssetCandidate(mw=5.0, capex_eur=5_000_000.0, lifetime_years=25.0),
        AssetCandidate(mw=10.0, capex_eur=9_500_000.0, lifetime_years=25.0),
        AssetCandidate(mw=20.0, capex_eur=18_000_000.0, lifetime_years=25.0),
    ],
    "electric_heater_mw": [
        AssetCandidate(mw=1.0, capex_eur=250_000.0, lifetime_years=20.0),
        AssetCandidate(mw=5.0, capex_eur=1_000_000.0, lifetime_years=20.0),
        AssetCandidate(mw=10.0, capex_eur=1_800_000.0, lifetime_years=20.0),
        AssetCandidate(mw=20.0, capex_eur=3_200_000.0, lifetime_years=20.0),
        AssetCandidate(mw=40.0, capex_eur=6_000_000.0, lifetime_years=20.0),
    ],
    "electric_boiler_mw": [
        AssetCandidate(mw=1.0, capex_eur=150_000.0, lifetime_years=25.0),
        AssetCandidate(mw=5.0, capex_eur=500_000.0, lifetime_years=25.0),
        AssetCandidate(mw=10.0, capex_eur=850_000.0, lifetime_years=25.0),
        AssetCandidate(mw=20.0, capex_eur=1_500_000.0, lifetime_years=25.0),
        AssetCandidate(mw=40.0, capex_eur=2_800_000.0, lifetime_years=25.0),
    ],
    "electric_chiller_mw": [
        AssetCandidate(mw=0.5, capex_eur=200_000.0, lifetime_years=20.0),
        AssetCandidate(mw=1.0, capex_eur=350_000.0, lifetime_years=20.0),
        AssetCandidate(mw=2.0, capex_eur=650_000.0, lifetime_years=20.0),
        AssetCandidate(mw=5.0, capex_eur=1_500_000.0, lifetime_years=20.0),
        AssetCandidate(mw=10.0, capex_eur=2_800_000.0, lifetime_years=20.0),
    ],
    "electrolyser_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_300_000.0, lifetime_years=25.0),
        AssetCandidate(mw=2.5, capex_eur=2_750_000.0, lifetime_years=25.0),
        AssetCandidate(mw=5.0, capex_eur=5_210_000.0, lifetime_years=25.0),
        AssetCandidate(mw=10.0, capex_eur=9_500_000.0, lifetime_years=25.0),
        AssetCandidate(mw=20.0, capex_eur=17_000_000.0, lifetime_years=25.0),
    ],
    "tank_mw": [
        AssetCandidate(mw=0.5, mwh=8.3, capex_eur=540_000.0, lifetime_years=30.0),
        AssetCandidate(mw=1.0, mwh=16.7, capex_eur=950_000.0, lifetime_years=30.0),
        AssetCandidate(mw=2.0, mwh=33.3, capex_eur=1_733_000.0, lifetime_years=30.0),
        AssetCandidate(mw=5.0, mwh=83.3, capex_eur=4_000_000.0, lifetime_years=30.0),
        AssetCandidate(mw=10.0, mwh=166.7, capex_eur=7_333_000.0, lifetime_years=30.0),
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
    lt_heat_pump_cop: float = 3.0
    mt_heat_pump_cop: float = 2.0
    electric_heater_efficiency: float = 0.98
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
                ("lt_heat_pump_mw", "lt_process_heat"): self.lt_heat_pump_cop,
                ("mt_heat_pump_mw", "mt_process_heat"): self.mt_heat_pump_cop,
                ("electric_heater_mw", "ht_heat"): self.electric_heater_efficiency,
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

    catalog: dict[str, list[AssetCandidate]] = field(default_factory=lambda: CANDIDATE_CATALOG)

    discount_rate: float = 0.05
    default_budget_eur: float = 500_000_000.0
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
