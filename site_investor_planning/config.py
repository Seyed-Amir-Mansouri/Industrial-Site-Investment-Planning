"""Candidate catalog, technology parameters and CAPEX/annualization assumptions for Industrial Site Investor planning.

Assets and the internal demand each one serves:

- electricity: wind, PV, battery, biogas CHP (which also gives space or process heat)
- space heating: heat pump, hot-water storage tank
- space cooling (air conditioning of buildings): AC chiller, chilled-water / ice storage
- process heat (up to ~150 C): industrial heat pump, solar thermal collectors (also space heating)
- steam (above ~150 C): electric (electrode / resistance) boiler, thermal battery (charged with electricity)
- process cooling (machines, products, cold stores): process chiller, heat-recovery chiller (whose
  rejected heat serves space or process heat), absorption chiller (driven by process heat, also
  serves space cooling)
- hydrogen: electrolyser, H2 tank

Thermal assets are sized in MW of useful output (MW_th / MW_cold), the storage assets in MW of
charge and discharge, the CHP in MW of electricity and the electrolyser in MW of electrical input. ``THERMAL_SERVICES`` are the heat and cooling demands. There is no existing
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
Energy Agency's), as are biogas CHP (gas engine, 20yr), hot-water storage (6h, 30yr), chilled-water /
ice storage (6h, 25yr), solar thermal collectors (MW_th peak, 25yr), thermal battery (8h, 25yr),
heat-recovery chiller (20yr) and absorption chiller (single effect, 25yr); replace them with vendor
quotes for a real site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

ASSETS = ["wind_mw", "pv_mw", "battery_mw", "chp_mw",
          "heat_pump_mw", "hot_water_storage_mw", "ac_chiller_mw", "cold_storage_mw",
          "industrial_heat_pump_mw", "solar_thermal_mw", "electric_boiler_mw", "thermal_battery_mw",
          "electric_chiller_mw", "heat_recovery_chiller_mw", "absorption_chiller_mw",
          "electrolyser_mw", "tank_mw"]

STORAGE_ENERGY = {"battery_mw": "battery_mwh", "hot_water_storage_mw": "hot_water_storage_mwh",
                  "cold_storage_mw": "cold_storage_mwh", "thermal_battery_mw": "thermal_battery_mwh",
                  "tank_mw": "tank_mwh"}
CAPACITY_KEYS = ASSETS + list(STORAGE_ENERGY.values())
CAPACITY_ASSET = {**{a: a for a in ASSETS}, **{e: a for a, e in STORAGE_ENERGY.items()}}

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

DEFAULT_FLEX_FRACTION = 0.10

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
    "chp_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_200_000.0, lifetime_years=20.0),
    ],
    "hot_water_storage_mw": [
        AssetCandidate(mw=1.0, mwh=6.0, capex_eur=150_000.0, lifetime_years=30.0),
    ],
    "cold_storage_mw": [
        AssetCandidate(mw=0.5, mwh=3.0, capex_eur=120_000.0, lifetime_years=25.0),
    ],
    "solar_thermal_mw": [
        AssetCandidate(mw=1.0, capex_eur=450_000.0, lifetime_years=25.0),
    ],
    "thermal_battery_mw": [
        AssetCandidate(mw=1.0, mwh=8.0, capex_eur=700_000.0, lifetime_years=25.0),
    ],
    "heat_recovery_chiller_mw": [
        AssetCandidate(mw=0.5, capex_eur=300_000.0, lifetime_years=20.0),
    ],
    "absorption_chiller_mw": [
        AssetCandidate(mw=0.5, capex_eur=200_000.0, lifetime_years=25.0),
    ],
    "electrolyser_mw": [
        AssetCandidate(mw=1.0, capex_eur=1_300_000.0, lifetime_years=25.0),
    ],
    "tank_mw": [
        AssetCandidate(mw=0.5, mwh=8.3, capex_eur=540_000.0, lifetime_years=30.0),
    ],
}

DEFAULT_STORAGE_HOURS = {a: CANDIDATE_CATALOG[a][0].mwh / CANDIDATE_CATALOG[a][0].mw for a in STORAGE_ENERGY}


@dataclass(frozen=True)
class SiteTechParams:
    """Operating parameters of the site's conversion assets, its unmet-demand penalty and its market access.

    COPs/efficiencies are useful output per MWh of electricity in, except the absorption chiller's
    (cooling per MWh of process heat in). The biogas CHP burns biogas bought at ``biogas_price``
    and turns ``chp_electric_efficiency`` of it into electricity and up to ``chp_heat_efficiency``
    into space or process heat. The heat-recovery chiller's rejected heat (its cooling plus its
    electricity) can serve space or process heat. Hot-water, chilled-water / ice storage and the
    thermal battery hold the MWh of the products built and lose ``1 - *_efficiency`` of what they charge;
    the thermal battery charges with electricity and discharges steam. The site has no existing plant:
    any heat or cooling the new assets can't supply is left unmet and charged
    ``unmet_demand_penalty_eur_per_mwh``, which keeps every subproblem feasible whatever the master
    proposes. It is far above the yearly cost of any asset per MW, so the planner never leaves
    demand unmet to save on building when the site caps allow enough capacity. The site buys and sells electricity and hydrogen at the modeled market price and pays
    the import fees (network charges, levies) on every MWh it buys.
    """

    heat_pump_cop: float = 3.0
    industrial_heat_pump_cop: float = 2.5
    electric_boiler_efficiency: float = 0.99
    electric_chiller_cop: float = 4.5
    ac_chiller_cop: float = 3.5
    heat_recovery_chiller_cop: float = 3.0
    absorption_chiller_cop: float = 0.7

    chp_electric_efficiency: float = 0.40
    chp_heat_efficiency: float = 0.45
    biogas_price_eur_per_mwh: float = 60.0

    hot_water_storage_efficiency: float = 0.95
    cold_storage_efficiency: float = 0.95
    thermal_battery_efficiency: float = 0.95

    unmet_demand_penalty_eur_per_mwh: float = 1_000_000.0

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


_CHP_MW_PER_HEAT = SITE_TECH.chp_electric_efficiency / SITE_TECH.chp_heat_efficiency
_ABS_DRIVE = 1.0 / SITE_TECH.absorption_chiller_cop
_COOLING_DRIVE = {"space_cool": _ABS_DRIVE, "process_cool": _ABS_DRIVE}

CAP_RULES: dict[str, tuple[float, list[dict[str, float]]]] = {
    "wind_mw": (7.5, [{"electricity": 1.0}]),
    "pv_mw": (7.5, [{"electricity": 1.0}]),
    "battery_mw": (5.0, [{"electricity": 1.0}]),
    "chp_mw": (1.25, [{"electricity": 1.0},
                      {svc: _CHP_MW_PER_HEAT * w for svc, w in
                       {"space_heat": 1.0, "process_heat": 1.0, **_COOLING_DRIVE}.items()}]),
    "heat_pump_mw": (1.25, [{"space_heat": 1.0}]),
    "hot_water_storage_mw": (1.25, [{"space_heat": 1.0}]),
    "ac_chiller_mw": (1.25, [{"space_cool": 1.0}]),
    "cold_storage_mw": (1.25, [{"space_cool": 1.0}]),
    "industrial_heat_pump_mw": (1.25, [{"process_heat": 1.0, **_COOLING_DRIVE}]),
    "solar_thermal_mw": (1.25, [{"space_heat": 1.0, "process_heat": 1.0}]),
    "electric_boiler_mw": (1.25, [{"steam": 1.0}]),
    "thermal_battery_mw": (1.25, [{"steam": 1.0 / SITE_TECH.thermal_battery_efficiency}]),
    "electric_chiller_mw": (1.25, [{"process_cool": 1.0}]),
    "heat_recovery_chiller_mw": (1.25, [{"process_cool": 1.0}]),
    "absorption_chiller_mw": (1.25, [{"space_cool": 1.0, "process_cool": 1.0}]),
    "electrolyser_mw": (1.25 / 0.68, [{"hydrogen": 1.0}]),
    "tank_mw": (1.25, [{"hydrogen": 1.0}]),
}


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
    hydrogen), demand flexibility per service (how far each hour's demand of that service may move
    up or down as a share of itself, the shifts netting to zero over each day; default 10% for
    every service, and a single number applies to all of them), the assets it may build
    (``active_assets``, default all) and optional maximum MW per asset (``max_mw``) that replace
    the default site caps."""

    name: str
    peaks_mw: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SITE_PEAKS_MW))
    green_share: float = 0.42
    flex_fraction: dict[str, float] | float = DEFAULT_FLEX_FRACTION
    active_assets: list[str] | None = None
    max_mw: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [svc for svc in SERVICES if svc not in self.peaks_mw]
        if missing:
            raise ValueError(f"site {self.name!r} is missing peak(s) for {missing}")
        self.peaks_mw = {svc: float(self.peaks_mw[svc]) for svc in SERVICES}
        if isinstance(self.flex_fraction, dict):
            unknown = [svc for svc in self.flex_fraction if svc not in SERVICES]
            if unknown:
                raise ValueError(f"site {self.name!r} has flexibility for unknown service(s) {unknown}")
            self.flex_fraction = {svc: float(self.flex_fraction.get(svc, DEFAULT_FLEX_FRACTION)) for svc in SERVICES}
        else:
            self.flex_fraction = {svc: float(self.flex_fraction) for svc in SERVICES}
        if self.active_assets is not None:
            unknown = [a for a in self.active_assets if a not in ASSETS]
            if unknown:
                raise ValueError(f"site {self.name!r} has unknown active asset(s) {unknown} -- choices: {ASSETS}")
            self.active_assets = [a for a in ASSETS if a in self.active_assets]
        unknown = [a for a in self.max_mw if a not in ASSETS]
        if unknown:
            raise ValueError(f"site {self.name!r} has a maximum MW for unknown asset(s) {unknown}")
        self.max_mw = {a: float(v) for a, v in self.max_mw.items()}

    def settings_key(self) -> tuple:
        """Everything that shapes this site's subproblem and caps; sites with equal keys are interchangeable."""
        return (tuple(sorted(self.peaks_mw.items())), self.green_share, tuple(sorted(self.flex_fraction.items())),
                tuple(self.active_assets or ASSETS), tuple(sorted(self.max_mw.items())))


@dataclass
class CapexAssumptions:
    """Candidate catalog plus annualization/budget parameters for the Benders master problem.

    ``cap_rules`` set the default largest MW of each asset one site may host, large enough for the
    asset to cover on its own every peak it can serve: the factor times the largest of its weighted
    peak sums, rounded up to a whole number of its smallest product and then to a whole MW. Heat and
    cooling assets and their stores get 1.25 times the peak they serve (25% headroom); the biogas
    CHP 1.25 times the larger of the electricity peak and the heat for space heating, process heat
    and the absorption chiller's drive (both cooling peaks divided by its COP) in MW of electricity;
    the industrial heat pump 1.25 times process heat plus that drive heat; the thermal battery 1.25
    times the steam peak divided by its efficiency; solar thermal 1.25 times space plus process heat;
    the absorption chiller 1.25 times both cooling peaks; wind and PV 7.5 times and the battery 5
    times the electricity peak (they also sell to the market); the electrolyser 1.25 times the
    hydrogen peak divided by its 68% efficiency (it is sized in MW of electricity), and the H2 tank
    1.25 times the hydrogen peak. A site's own ``max_mw`` replaces these.
    """

    catalog: dict[str, list[AssetCandidate]] = field(default_factory=lambda: dict(CANDIDATE_CATALOG))

    discount_rate: float = 0.05
    default_budget_eur: float = 500_000_000.0
    theta_lower_bound_eur: float = -1e8

    cap_rules: dict[str, tuple[float, list[dict[str, float]]]] = field(default_factory=lambda: dict(CAP_RULES))

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
