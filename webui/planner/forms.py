import re

from django import forms

from . import services
from .geo import country_label

ASSET_LABELS = {
    "wind_mw": "Wind",
    "pv_mw": "Solar PV",
    "battery_mw": "Battery",
    "chp_mw": "Biogas CHP",
    "heat_pump_mw": "Heat pump",
    "hot_water_storage_mw": "Hot-water storage",
    "ac_chiller_mw": "AC chiller",
    "cold_storage_mw": "Chilled-water / ice storage",
    "industrial_heat_pump_mw": "Industrial heat pump",
    "solar_thermal_mw": "Solar thermal",
    "electric_boiler_mw": "Electric boiler",
    "thermal_battery_mw": "Thermal battery",
    "electric_chiller_mw": "Process chiller",
    "heat_recovery_chiller_mw": "Heat-recovery chiller",
    "absorption_chiller_mw": "Absorption chiller",
    "electrolyser_mw": "Electrolyser",
    "tank_mw": "H2 storage",
}
FULL_ASSET_LABELS = {
    **ASSET_LABELS,
    "pv_mw": "Solar photovoltaic",
    "chp_mw": "Biogas combined heat and power",
    "ac_chiller_mw": "Air-conditioning chiller",
    "tank_mw": "Hydrogen storage",
}
ASSET_SERVES = {
    "wind_mw": ["electricity"], "pv_mw": ["electricity"], "battery_mw": ["electricity"],
    "chp_mw": ["electricity", "space_heat", "process_heat"],
    "heat_pump_mw": ["space_heat"], "hot_water_storage_mw": ["space_heat"],
    "ac_chiller_mw": ["space_cool"], "cold_storage_mw": ["space_cool"],
    "industrial_heat_pump_mw": ["process_heat"], "solar_thermal_mw": ["space_heat", "process_heat"],
    "electric_boiler_mw": ["steam"], "thermal_battery_mw": ["steam"],
    "electric_chiller_mw": ["process_cool"], "heat_recovery_chiller_mw": ["process_cool", "space_heat", "process_heat"],
    "absorption_chiller_mw": ["space_cool", "process_cool"],
    "electrolyser_mw": ["hydrogen"], "tank_mw": ["hydrogen"],
}
FIRM_SUPPLY = {
    "space_heat": {"heat_pump_mw": 1.0, "chp_mw": 0.45 / 0.40},
    "space_cool": {"ac_chiller_mw": 1.0, "absorption_chiller_mw": 1.0},
    "process_heat": {"industrial_heat_pump_mw": 1.0, "chp_mw": 0.45 / 0.40},
    "steam": {"electric_boiler_mw": 1.0, "thermal_battery_mw": 0.95},
    "process_cool": {"electric_chiller_mw": 1.0, "heat_recovery_chiller_mw": 1.0, "absorption_chiller_mw": 1.0},
}
PART_SUPPLY = {
    "space_heat": ["solar_thermal_mw", "heat_recovery_chiller_mw", "hot_water_storage_mw"],
    "space_cool": ["cold_storage_mw"],
    "process_heat": ["solar_thermal_mw", "heat_recovery_chiller_mw"],
    "steam": [], "process_cool": [],
}
SHARED_HEAT = {"chp_mw": 0.45 / 0.40}
ABSORPTION_COP = 0.7
MARKET_SUPPLY = {"electricity": ["wind_mw", "pv_mw", "chp_mw", "battery_mw"], "hydrogen": ["electrolyser_mw", "tank_mw"]}
MWH_ASSETS = {"battery_mw", "tank_mw", "hot_water_storage_mw", "cold_storage_mw", "thermal_battery_mw"}
MAX_CANDIDATES_PER_ASSET = 8
MAX_SITES = 6
CUSTOM_FIELD_RE = re.compile(r"^custom_name__(\d+)$")


def range_text(bounds: list[int] | None) -> str:
    """A trained range as shown below its box, e.g. "trained −73 to +30%"; empty when unknown."""
    if not bounds:
        return ""
    lo, hi = bounds
    sign = lambda v: f"+{v}" if v > 0 else str(v).replace("-", "−")
    return f"trained {sign(lo)} to {sign(hi)}%"


def plain(value):
    """A default shown in a number box: whole numbers without a trailing .0, others as they are."""
    if value is None or isinstance(value, bool):
        return value
    return int(value) if float(value).is_integer() else value


class PlanRunForm(forms.Form):
    name = forms.CharField(max_length=120, required=False, label="Run name",
                           widget=forms.TextInput(attrs={"placeholder": "e.g. Baseline 1.5B, PV only"}))
    budget_limited = forms.BooleanField(required=False, initial=True, label="Limited budget")
    budget = forms.FloatField(required=False, min_value=1e6, initial=1_500_000_000, label="Total CAPEX budget (EUR)",
                              help_text="Raw, unannualized budget across all sites.")
    all_countries = forms.BooleanField(required=False, initial=True, label="All eligible countries")
    countries = forms.MultipleChoiceField(required=False, label="Candidate site countries",
                                          widget=forms.CheckboxSelectMultiple)
    n_sites = forms.IntegerField(min_value=1, max_value=MAX_SITES, initial=1, widget=forms.HiddenInput)

    discount_rate_pct = forms.FloatField(min_value=0, max_value=30, initial=5, label="Discount rate (%)")
    risk_measure = forms.ChoiceField(
        required=False,
        choices=[("deterministic", "Deterministic (one scenario)"), ("cvar", "CVaR (two or more scenarios)")],
        initial="deterministic", label="Risk measure", widget=forms.RadioSelect,
        help_text="Set automatically: deterministic with one scenario, CVaR with two or more.")
    cvar_alpha = forms.FloatField(required=False, min_value=0.5, max_value=0.99, initial=0.8,
                                  label="CVaR confidence level α",
                                  help_text="Higher α focuses on the worst tail of scenarios.")

    rep_days_per_month = forms.IntegerField(min_value=1, max_value=29, initial=7,
                                            label="Days per month",
                                            help_text="Representative days solved per month, weighted to cover the "
                                                      "whole month. More days = more accurate but slower.")
    gap_tol = forms.FloatField(min_value=0.0001, max_value=0.5, initial=0.01, label="Optimality gap tolerance")
    max_iters = forms.IntegerField(min_value=1, max_value=500, initial=100, label="Max Benders iterations")
    master_time_limit = forms.FloatField(min_value=10, max_value=3600, initial=180,
                                         label="Master time limit (s)")
    workers = forms.IntegerField(min_value=1, max_value=8, initial=2, label="Parallel workers",
                                 help_text="Workers solve each iteration's scenario subproblems in parallel. "
                                           "Each one uses significant memory, so use 2 or fewer on constrained machines.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_scenarios = services.scenario_defaults()
        self.baseline = services.BASELINE_SCENARIO
        self.defaults = {self.baseline: all_scenarios[self.baseline]}
        self.scenario_countries = sorted(services.eligible_countries())
        trained = services.trained_error_ranges()
        self.error_ranges = {c: trained.get(c) for c in self.scenario_countries}
        self.fields["countries"].choices = [(c, country_label(c)) for c in services.eligible_countries()]

        self.site_defaults = services.site_defaults()
        self.demand_services = services.demand_services()
        self.demand_service_labels = [services.SERVICE_LABELS[s] for s in self.demand_services]
        self.catalog_defaults = services.capex_assumptions_defaults()
        self.assets = self.catalog_defaults["assets"]
        default_caps = self.catalog_defaults["default_max_mw"]
        for i in range(MAX_SITES):
            self.fields[f"site_name__{i}"] = forms.CharField(
                max_length=40, required=False, initial=f"Site {i + 1}", label="Name")
            self.fields[f"site_green__{i}"] = forms.FloatField(
                min_value=0, max_value=100, initial=plain(round(self.site_defaults["green_share"] * 100, 4)),
                label="Minimum green hydrogen share (%)")
            for service in self.demand_services:
                self.fields[f"site_peak__{i}__{service}"] = forms.FloatField(
                    min_value=0, initial=plain(round(self.site_defaults["peaks_mw"][service], 4)),
                    label=f"{services.SERVICE_LABELS[service]} (MW)")
                self.fields[f"site_flex__{i}__{service}"] = forms.FloatField(
                    min_value=0, max_value=50, initial=plain(round(self.site_defaults["flex_fraction"][service] * 100, 4)),
                    label=f"{services.SERVICE_LABELS[service]} flexibility (±%)")
            for asset in self.assets:
                self.fields[f"site_asset__{i}__{asset}"] = forms.BooleanField(
                    required=False, initial=False, label=ASSET_LABELS[asset])
                self.fields[f"site_maxmw__{i}__{asset}"] = forms.FloatField(
                    required=False, min_value=0, initial=plain(default_caps[asset]),
                    label=f"{ASSET_LABELS[asset]} maximum (MW)",
                    widget=forms.NumberInput(attrs={"step": "any", "data-maxmw": asset}))

        for asset in self.assets:
            rows = self.catalog_defaults["catalog"][asset]
            lifetime_default = rows[0]["lifetime_years"] if rows else 1.0
            self.fields[f"cat_lifetime__{asset}"] = forms.FloatField(
                min_value=0.1, max_value=100, initial=plain(lifetime_default), label="Lifetime (yr)")
            for i in range(MAX_CANDIDATES_PER_ASSET):
                c = rows[i] if i < len(rows) else None
                self.fields[f"cat_mw__{asset}__{i}"] = forms.FloatField(
                    required=False, min_value=0.001, initial=plain(c["mw"]) if c else None, label="Size (MW)")
                self.fields[f"cat_capex__{asset}__{i}"] = forms.FloatField(
                    required=False, min_value=0, initial=plain(c["capex_eur"]) if c else None, label="CAPEX (EUR)")
                if asset in MWH_ASSETS:
                    self.fields[f"cat_mwh__{asset}__{i}"] = forms.FloatField(
                        required=False, min_value=0, initial=plain(c["mwh"]) if c else None, label="Energy (MWh)")

        self.fields["scenario_include__" + self.baseline] = forms.BooleanField(required=False, initial=True)
        self.fields["scenario_prob__" + self.baseline] = forms.FloatField(
            min_value=0, max_value=100, initial=100, label="Probability (%)")
        self.custom_indices = sorted({int(m.group(1)) for k in (self.data or {})
                                      if (m := CUSTOM_FIELD_RE.match(k))})
        for i in self.custom_indices:
            self.fields[f"custom_name__{i}"] = forms.CharField(
                max_length=40, required=False, initial=f"Scenario {i + 1}", label="Name")
            self.fields[f"custom_prob__{i}"] = forms.FloatField(
                min_value=0, max_value=100, initial=0, label="Probability (%)")
            for c in self.scenario_countries:
                for kind in ("wind", "solar"):
                    lo, hi = self._error_bounds(c, kind)
                    self.fields[f"custom_{kind}__{i}__{c}"] = forms.FloatField(
                        min_value=lo, max_value=hi, initial=0, label=f"{c} {kind} capacity change vs baseline (%)",
                        error_messages={"min_value": f"Outside the accepted range ({lo} to {hi}%%).",
                                        "max_value": f"Outside the accepted range ({lo} to {hi}%%)."},
                        widget=forms.NumberInput(attrs={"step": "any"}))

    def _error_bounds(self, country: str, kind: str) -> list[int]:
        """Lowest and highest capacity change (%) accepted for ``country``'s ``kind`` (wind or solar): the
        trained range, or -100 to 100 when the training samples aren't available."""
        r = self.error_ranges.get(country)
        return r[kind] if r else [-100, 100]

    @property
    def scenario_country_rows(self) -> list[dict]:
        """Each scenario country's accepted wind and solar change bounds and their display text, for the
        template a new scenario card is built from."""
        return [{"code": c, "wind": self._error_bounds(c, "wind"), "solar": self._error_bounds(c, "solar"),
                 "wind_text": range_text(self.error_ranges[c] and self.error_ranges[c]["wind"]),
                 "solar_text": range_text(self.error_ranges[c] and self.error_ranges[c]["solar"])}
                for c in self.scenario_countries]

    @property
    def fill_bounds(self) -> dict:
        """Bounds of the "All countries" fill boxes: the range every country accepts."""
        return {k: [max(self._error_bounds(c, k)[0] for c in self.scenario_countries),
                    min(self._error_bounds(c, k)[1] for c in self.scenario_countries)] for k in ("wind", "solar")}

    COUNTRY_TABLE_COLUMNS = 4

    def _table_rows(self, field: str, cols: int) -> list[list]:
        """A checkbox field's options chunked into table rows of ``cols`` cells."""
        cells = list(self[field])
        return [cells[i:i + cols] for i in range(0, len(cells), cols)]

    @property
    def country_table_rows(self) -> list[list]:
        """Candidate country checkboxes laid out as table rows."""
        return self._table_rows("countries", self.COUNTRY_TABLE_COLUMNS)

    @property
    def site_rules(self) -> dict:
        """What the page's script needs to fill default maximum MW and show each demand's coverage."""
        return {"cap_rules": self.catalog_defaults["cap_rules"], "serves": ASSET_SERVES, "firm": FIRM_SUPPLY,
                "shared_heat": SHARED_HEAT, "absorption_cop": ABSORPTION_COP,
                "default_product_mw": {a: min((c["mw"] for c in self.catalog_defaults["catalog"][a] if c["mw"] > 0),
                                              default=0) for a in self.assets},
                "part": PART_SUPPLY, "market": MARKET_SUPPLY, "asset_labels": ASSET_LABELS,
                "service_labels": dict(zip(self.demand_services, self.demand_service_labels))}

    @property
    def site_rows(self) -> list[dict]:
        """One card per possible site: its name, green share, each demand's peak and flexibility, and
        each asset's active box and maximum MW."""
        return [{"index": i, "name": self[f"site_name__{i}"], "green": self[f"site_green__{i}"],
                 "peaks": [self[f"site_peak__{i}__{svc}"] for svc in self.demand_services],
                 "flexes": [self[f"site_flex__{i}__{svc}"] for svc in self.demand_services],
                 "assets": [{"key": a, "label": ASSET_LABELS[a], "active": self[f"site_asset__{i}__{a}"],
                             "maxmw": self[f"site_maxmw__{i}__{a}"],
                             "serves": ", ".join(services.SERVICE_LABELS[s] for s in ASSET_SERVES[a])}
                            for a in self.assets],
                 "services": self.demand_services}
                for i in range(MAX_SITES)]

    def _site_assets(self, data: dict, i: int) -> list[str]:
        """Assets ticked as active at site ``i``."""
        return [a for a in self.assets if data.get(f"site_asset__{i}__{a}")]

    def _active_anywhere(self, data: dict) -> list[str]:
        """Assets active at one or more of the run's sites, in catalog order."""
        used = {a for i in range(data.get("n_sites") or 0) for a in self._site_assets(data, i)}
        return [a for a in self.assets if a in used]

    @property
    def catalog_rows(self) -> list[dict]:
        rows = []
        for asset in self.assets:
            default = (self.catalog_defaults["catalog"][asset] or [{}])[0]
            rows.append({
                "asset": asset,
                "label": ASSET_LABELS[asset],
                "has_mwh": asset in MWH_ASSETS,
                "default": {k: plain(default[k]) if default.get(k) is not None else ""
                            for k in ("mw", "mwh", "capex_eur")},
                "lifetime": self[f"cat_lifetime__{asset}"],
                "candidates": [
                    {
                        "index": i,
                        "mw": self[f"cat_mw__{asset}__{i}"],
                        "capex": self[f"cat_capex__{asset}__{i}"],
                        "mwh": self[f"cat_mwh__{asset}__{i}"] if asset in MWH_ASSETS else None,
                    }
                    for i in range(MAX_CANDIDATES_PER_ASSET)
                ],
            })
        return rows

    @property
    def baseline_row(self) -> dict:
        """The Baseline scenario's include box and probability."""
        return {"name": services.scenario_label(self.baseline),
                "include": self["scenario_include__" + self.baseline],
                "prob": self["scenario_prob__" + self.baseline]}

    @property
    def custom_scenario_rows(self) -> list[dict]:
        """One card per custom scenario the user added: name, probability and each country's wind/solar error."""
        return [{"index": i, "name": self[f"custom_name__{i}"], "prob": self[f"custom_prob__{i}"],
                 "countries": [{"code": r["code"], "wind": self[f"custom_wind__{i}__{r['code']}"],
                                "solar": self[f"custom_solar__{i}__{r['code']}"],
                                "wind_text": r["wind_text"], "solar_text": r["solar_text"]}
                               for r in self.scenario_country_rows]}
                for i in self.custom_indices]

    def _included_scenarios(self, data: dict) -> list[str]:
        """Keys of every scenario in the run: the Baseline if ticked, and every added scenario as ``custom<N>``."""
        base = [self.baseline] if data.get("scenario_include__" + self.baseline) else []
        return base + [f"custom{i + 1}" for i in self.custom_indices]

    def _scenario_prob(self, data: dict, key: str) -> float | None:
        if key.startswith("custom") and key[6:].isdigit():
            return data.get(f"custom_prob__{int(key[6:]) - 1}")
        return data.get(f"scenario_prob__{key}")

    def clean(self):
        data = super().clean()
        if not data.get("all_countries") and not data.get("countries"):
            self.add_error("countries", "Select at least one country or choose all countries.")
        if data.get("budget_limited") and data.get("budget") is None:
            self.add_error("budget", "Enter a budget, or untick Limited budget to plan without one.")
        n_sites = data.get("n_sites") or 0
        names = [(data.get(f"site_name__{i}") or "").strip() or f"Site {i + 1}" for i in range(n_sites)]
        if len(set(names)) != len(names):
            self.add_error(None, f"Each site needs its own name; got {', '.join(names)}.")
        for i, name in enumerate(names):
            if not self._site_assets(data, i):
                self.add_error(None, f"{name}: activate at least one asset.")
        included = self._included_scenarios(data)
        if not included:
            self.add_error(None, "Select at least one uncertainty scenario.")
        probs = [self._scenario_prob(data, s) for s in included]
        if all(p is not None for p in probs):
            total = sum(probs)
            if abs(total - 100) > 0.01:
                self.add_error(None, f"Scenario probabilities add up to {total:.2f}%. They must add up to 100%.")
        data["risk_measure"] = "deterministic" if len(included) <= 1 else "cvar"
        if data["risk_measure"] == "cvar" and data.get("cvar_alpha") is None:
            self.add_error("cvar_alpha", "Enter a confidence level for CVaR (two or more scenarios are included).")

        for asset in self._active_anywhere(data):
            count = 0
            for i in range(MAX_CANDIDATES_PER_ASSET):
                mw = data.get(f"cat_mw__{asset}__{i}")
                capex = data.get(f"cat_capex__{asset}__{i}")
                mwh = data.get(f"cat_mwh__{asset}__{i}") if asset in MWH_ASSETS else None
                if mw is None and capex is None and mwh is None:
                    continue
                if mw is None or capex is None:
                    self.add_error(f"cat_mw__{asset}__{i}",
                                   "Enter both size (MW) and CAPEX for this candidate, or leave the row blank.")
                    continue
                if asset in MWH_ASSETS and mwh is None:
                    self.add_error(f"cat_mwh__{asset}__{i}",
                                   "Enter the energy (MWh) for this candidate, or leave the row blank.")
                    continue
                count += 1
            if count == 0:
                self.add_error(None, f"{ASSET_LABELS[asset]}: enter at least one candidate product (size + CAPEX).")
        return data

    def params(self) -> dict:
        d = self.cleaned_data
        active_anywhere = self._active_anywhere(d)
        catalog_overrides = {}
        for asset in active_anywhere:
            candidates = []
            for i in range(MAX_CANDIDATES_PER_ASSET):
                mw = d.get(f"cat_mw__{asset}__{i}")
                capex = d.get(f"cat_capex__{asset}__{i}")
                if mw is None or capex is None:
                    continue
                candidate = {"mw": mw, "capex_eur": capex, "lifetime_years": d[f"cat_lifetime__{asset}"]}
                if asset in MWH_ASSETS:
                    candidate["mwh"] = d.get(f"cat_mwh__{asset}__{i}")
                candidates.append(candidate)
            catalog_overrides[asset] = sorted(candidates, key=lambda c: c["mw"])

        sites = [{"name": (d.get(f"site_name__{i}") or "").strip() or f"Site {i + 1}",
                  "peaks_mw": {svc: d[f"site_peak__{i}__{svc}"] for svc in self.demand_services},
                  "green_share": d[f"site_green__{i}"] / 100,
                  "flex_fraction": {svc: d[f"site_flex__{i}__{svc}"] / 100 for svc in self.demand_services},
                  "active_assets": self._site_assets(d, i),
                  "max_mw": {a: d[f"site_maxmw__{i}__{a}"] for a in self._site_assets(d, i)
                             if d.get(f"site_maxmw__{i}__{a}") is not None}}
                 for i in range(d["n_sites"])]

        included = self._included_scenarios(d)
        total = sum(self._scenario_prob(d, s) for s in included)
        overrides = {}
        for s in services.all_scenario_names():
            overrides[s] = {"probability": 0.0}
        if self.baseline in included:
            overrides[self.baseline] = {"probability": d["scenario_prob__" + self.baseline] / total}
        labels = {s: services.scenario_label(s) for s in included}
        for i in self.custom_indices:
            key = f"custom{i + 1}"
            name = (d.get(f"custom_name__{i}") or "").strip() or f"Scenario {i + 1}"
            labels[key] = name
            overrides[key] = {
                "custom": True,
                "label": name,
                "probability": d[f"custom_prob__{i}"] / total,
                "wind": {c: round(1 + d[f"custom_wind__{i}__{c}"] / 100, 6) for c in self.scenario_countries},
                "solar": {c: round(1 + d[f"custom_solar__{i}__{c}"] / 100, 6) for c in self.scenario_countries},
            }
        return {
            "all_countries": d["all_countries"],
            "countries": sorted(d["countries"]),
            "n_sites": d["n_sites"],
            "sites": sites,
            "budget": d["budget"] if d["budget_limited"] else None,
            "active_assets": active_anywhere,
            "disabled_assets": [a for a in self.assets if a not in active_anywhere],
            "scenarios": included,
            "scenario_labels": labels,
            "scenario_overrides": overrides,
            "catalog_overrides": catalog_overrides,
            "discount_rate_pct": d["discount_rate_pct"],
            "risk_measure": d["risk_measure"],
            "cvar_alpha": d["cvar_alpha"],
            "rep_days_per_month": d["rep_days_per_month"],
            "gap_tol": d["gap_tol"],
            "max_iters": d["max_iters"],
            "master_time_limit": d["master_time_limit"],
            "workers": d["workers"],
        }
