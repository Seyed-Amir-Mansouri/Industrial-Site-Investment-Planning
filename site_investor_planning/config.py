"""Candidate catalog, technology parameters and CAPEX/annualization assumptions for Industrial Site Investor planning.

Assets and the internal demand each one serves:

- electricity: wind, PV, battery
- space heating: heat pump
- space cooling (air conditioning of buildings): AC chiller
- process heat (up to ~150 C): industrial heat pump
- steam (above ~150 C): electric (electrode / resistance) boiler
- process cooling (machines, products, cold stores): process chiller
- hydrogen: electrolyser, H2 tank

Thermal assets are sized in MW of useful output (MW_th / MW_cold), the electrolyser in MW of
electrical input. ``THERMAL_SERVICES`` are the heat and cooling demands. There is no existing
plant on site: heat or cooling the new assets can't cover is left unmet at a penalty.

Every site to build is a ``SiteSpec``: its own annual peak demand per service (MW_th for heat and
cooling, MW_LHV for hydrogen), green hydrogen share and demand flexibility. A site's hourly demand
is the per-unit yearly curve in ``inputs/site_demand.csv`` times its peaks, wherever it is built.
``DEFAULT_SITE_PEAKS_MW`` are the peaks a new site starts from.

Each asset has one default product, which the master may build any number of times (up to
the site cap). Catalog sources: wind (Vestas V100-2.0 turbine class), PV (fixed-tilt, 40yr), battery (Li-ion, 2h, 20yr), electrolyser (PEM,
25yr) and H2 tank (compressed, ~16.7h, 30yr) follow the 2030 candidate-product table in
``Help/Candidates (Edited).docx``. Heat pump (20yr), industrial heat pump (25yr), electric boiler
(25yr), process chiller (20yr) and AC chiller (20yr) are indicative 2030 installed costs in the range of public technology catalogues (e.g. the Danish
Energy Agency's); replace them with vendor quotes for a real site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

ASSETS = ["wind_mw", "pv_mw", "battery_mw",
          "heat_pump_mw", "ac_chiller_mw", "industrial_heat_pump_mw", "electric_boiler_mw",
          "electric_chiller_mw", "electrolyser_mw", "tank_mw"]

SERVICES = ["electricity", "space_heat", "space_cool", "process_heat", "steam", "process_cool", "hydrogen"]
THERMAL_SERVICES = ["space_heat", "space_cool", "process_heat", "steam", "process_cool"]

SERVICE_LABELS = {
    "electricity": "Electricity",
    "space_heat": "Space heating",
    "space_cool": "Space cooling",
    "process_heat": "Process heat (up to 150 °C)",
    "steam": "Steam (above 150 °C)",
    "process_cool": "Process cooling",
    "hydrogen": "Hydrogen",
}

DEFAULT_SITE_PEAKS_MW: dict[str, float] = {
    "electricity": 8.0, "space_heat": 4.0, "space_cool": 3.0, "process_heat": 7.0,
    "steam": 5.0, "process_cool": 2.0, "hydrogen": 1.5,
}

THERMAL_ASSET_SERVICES = {
    "heat_pump_mw": ["space_heat"],
    "ac_chiller_mw": ["space_cool"],
    "industrial_heat_pump_mw": ["process_heat"],
    "electric_boiler_mw": ["steam"],
    "electric_chiller_mw": ["process_cool"],
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
    "ac_chiller_mw": [
        AssetCandidate(mw=0.5, capex_eur=175_000.0, lifetime_years=20.0),
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
    """Operating parameters of the site's conversion assets, its unmet-demand penalty and its market access.

    COPs/efficiencies are useful output per MWh of electricity in. The site has no existing plant:
    any heat or cooling the new assets can't supply is left unmet and charged
    ``unmet_demand_penalty_eur_per_mwh``, which keeps every subproblem feasible whatever the master
    proposes and in practice makes the planner build enough capacity. The site buys and sells electricity and hydrogen at the modeled market price and pays
    the import fees (network charges, levies) on every MWh it buys.
    """

    heat_pump_cop: float = 3.0
    industrial_heat_pump_cop: float = 2.5
    electric_boiler_efficiency: float = 0.99
    electric_chiller_cop: float = 4.5
    ac_chiller_cop: float = 3.5

    unmet_demand_penalty_eur_per_mwh: float = 5000.0

    grid_import_fee_eur_per_mwh: float = 15.0
    h2_import_fee_eur_per_mwh: float = 0.0

    def cop(self, asset: str, service: str) -> float:
        """Useful thermal output per MWh of electricity when ``asset`` serves ``service``."""
        return {("heat_pump_mw", "space_heat"): self.heat_pump_cop,
                ("industrial_heat_pump_mw", "process_heat"): self.industrial_heat_pump_cop,
                ("electric_boiler_mw", "steam"): self.electric_boiler_efficiency,
                ("electric_chiller_mw", "process_cool"): self.electric_chiller_cop,
                ("ac_chiller_mw", "space_cool"): self.ac_chiller_cop}[(asset, service)]


SITE_TECH = SiteTechParams()


@dataclass(frozen=True)
class GreenH2Params:
    """Green (RFNBO) hydrogen rules and the green certificate market.

    Each site sets its own minimum green share of annual hydrogen demand (``SiteSpec.green_share``). Green hydrogen is
    either made by the site's electrolyser from additional renewable electricity, matched hour by
    hour (EU RFNBO temporal correlation from 2030), or bought as certified green hydrogen at
    ``green_h2_premium`` above the hydrogen market price. Additional renewable electricity is the
    site's own new wind/PV, or Guarantees of Origin (GOs) bought from additional plants. The site's
    own wind/PV exported to the grid earns GOs it can sell, unless that output is claimed for green
    hydrogen.

    Defaults: GOs bought at 8 and sold at 6 EUR/MWh, and a certified green H2 premium of
    120 EUR/MWh (~EUR 4/kg).
    """

    go_buy_price_eur_per_mwh: float = 8.0
    go_sell_price_eur_per_mwh: float = 6.0
    green_h2_premium_eur_per_mwh: float = 120.0


GREEN_H2 = GreenH2Params()


@dataclass
class SiteSpec:
    """One industrial site to build: its name, annual peak demand per service (MW), minimum green
    share of its annual hydrogen demand (default 42%, the RED III 2030 RFNBO target for industrial
    hydrogen), and demand flexibility (how far each hour's demand may move up or down as a share of
    itself, the shifts netting to zero over each day; default 10%)."""

    name: str
    peaks_mw: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SITE_PEAKS_MW))
    green_share: float = 0.42
    flex_fraction: float = 0.10

    def __post_init__(self) -> None:
        missing = [svc for svc in SERVICES if svc not in self.peaks_mw]
        if missing:
            raise ValueError(f"site {self.name!r} is missing peak(s) for {missing}")
        self.peaks_mw = {svc: float(self.peaks_mw[svc]) for svc in SERVICES}


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
