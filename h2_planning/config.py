"""Placeholder CAPEX / discretization / annualization assumptions for H2 Producer
capacity PLANNING (as opposed to price_model's demand -> price fitting, or
optimize_h2_producer's fixed-capacity operational LP).

No investment-cost data exists anywhere in this project or in the vendored Project 3
dispatch engine (``economic_dispatch/``) -- both are dispatch-only. Every EUR figure
below is a placeholder, order-of-magnitude, circa-2030 ballpark (IRENA/IEA/BNEF-style),
NOT a real quote -- same "ASSUMPTION" convention ``economic_dispatch/config.py`` already
uses for its own assumptions. Replace with real figures before using this for anything
beyond a methodology demonstration. See ``Formulation.md`` SS4.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from economic_dispatch.config import RunConfig

_DEFAULTS = RunConfig()

ASSETS = ["electrolyser_mw", "wind_mw", "pv_mw", "battery_mw", "tank_mw"]


@dataclass
class CapexAssumptions:
    """EUR/MW (and EUR/MWh where relevant) unit costs for the 5 Hydrogen Producer
    assets, plus the discrete-candidate-grid and CAPEX-annualization parameters used by
    the Benders capacity-planning master problem (``h2_planning/master.py``). Every
    ``*_eur_*`` field is an ASSUMPTION -- a placeholder, not a real quote (see module
    docstring)."""

    electrolyser_eur_per_mw: float = 600_000.0
    wind_eur_per_mw: float = 1_300_000.0
    pv_eur_per_mw: float = 600_000.0
    battery_power_eur_per_mw: float = 150_000.0
    battery_energy_eur_per_mwh: float = 150_000.0
    tank_power_eur_per_mw: float = 50_000.0
    tank_energy_eur_per_mwh: float = 25_000.0

    candidate_offsets_mw: dict[str, list[float]] = field(default_factory=lambda: {
        "electrolyser_mw": [-8.0, -4.0, 0.0, 4.0, 8.0],
        "wind_mw": [-4.0, -2.0, 0.0, 2.0, 4.0],
        "pv_mw": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "battery_mw": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "tank_mw": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })
    min_candidate_mw: float = 0.5

    candidate_grid_mw: list[float] | None = None

    candidate_span_above_default_mw: float | None = None
    candidate_step_mw: float | None = None

    default_mw_override: dict[str, float] | None = None

    discount_rate: float = 0.05
    lifetime_years: dict[str, float] = field(default_factory=lambda: {
        "electrolyser_mw": 20,
        "wind_mw": 25,
        "pv_mw": 30,
        "battery_mw": 15,
        "tank_mw": 30,
    })

    default_budget_eur: float = 500_000_000.0
    theta_lower_bound_eur: float = -1e8

    @staticmethod
    def _crf(r: float, n: float) -> float:
        if r == 0:
            return 1.0 / n
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def capital_recovery_factors(self) -> dict[str, float]:
        """Per-asset CRF, using the shared ``discount_rate`` and each asset's own
        ``lifetime_years`` entry."""
        return {a: self._crf(self.discount_rate, n) for a, n in self.lifetime_years.items()}

    def effective_unit_cost_eur_per_mw(self) -> dict[str, float]:
        """Collapse each asset to ONE EUR/MW figure for the master's linear cost --
        battery/tank fold in their energy-cost portion via the SAME fixed MW -> MWh
        duration ratio ``optimize_h2_producer.py`` itself uses
        (``h2_producer_{battery,tank}_duration_hours``), since only MW is discretized
        here (Formulation.md SS4.2)."""
        return {
            "electrolyser_mw": self.electrolyser_eur_per_mw,
            "wind_mw": self.wind_eur_per_mw,
            "pv_mw": self.pv_eur_per_mw,
            "battery_mw": (self.battery_power_eur_per_mw
                          + _DEFAULTS.h2_producer_battery_duration_hours * self.battery_energy_eur_per_mwh),
            "tank_mw": (self.tank_power_eur_per_mw
                       + _DEFAULTS.h2_producer_tank_duration_hours * self.tank_energy_eur_per_mwh),
        }
