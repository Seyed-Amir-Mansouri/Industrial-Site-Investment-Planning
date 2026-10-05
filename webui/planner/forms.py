from django import forms

from . import services

ASSET_LABELS = {
    "electrolyser_mw": "Electrolyser",
    "wind_mw": "Wind",
    "pv_mw": "Solar PV",
    "battery_mw": "Battery",
    "tank_mw": "H2 tank",
}


class PlanRunForm(forms.Form):
    # Problem definition
    name = forms.CharField(max_length=120, required=False, label="Run name",
                           widget=forms.TextInput(attrs={"placeholder": "e.g. Baseline 500M, PV only"}))
    all_countries = forms.BooleanField(required=False, initial=True, label="All eligible countries")
    countries = forms.MultipleChoiceField(required=False, label="Countries",
                                          widget=forms.SelectMultiple(attrs={"size": 8}))
    budget = forms.FloatField(min_value=1e6, initial=500_000_000, label="Total CAPEX budget (EUR)",
                              help_text="Raw, unannualized budget across all countries.")
    disabled_assets = forms.MultipleChoiceField(
        required=False, label="Excluded asset types",
        choices=[(a, label) for a, label in ASSET_LABELS.items()],
        widget=forms.CheckboxSelectMultiple,
    )
    max_units_per_candidate = forms.IntegerField(
        min_value=0, initial=0, label="Max units per product",
        help_text="Cap on how many units of any single product size can be built. 0 = no cap.")
    scenarios = forms.MultipleChoiceField(required=False, label="Uncertainty scenarios",
                                          widget=forms.CheckboxSelectMultiple)

    # Economics
    discount_rate_pct = forms.FloatField(min_value=0, max_value=30, initial=5, label="Discount rate (%)")
    lifetime_years = forms.FloatField(required=False, min_value=1, max_value=100,
                                      label="Lifetime override (years)",
                                      help_text="Leave blank to use each asset's own lifetime.")
    risk_measure = forms.ChoiceField(
        choices=[("cvar", "CVaR (risk-averse)"), ("expected", "Expected value (risk-neutral)")],
        initial="cvar", label="Risk measure", widget=forms.RadioSelect)
    cvar_alpha = forms.FloatField(required=False, min_value=0.5, max_value=0.99, initial=0.8,
                                  label="CVaR confidence level α",
                                  help_text="Higher α focuses on the worst tail of scenarios.")

    # Solver
    rep_days_per_month = forms.IntegerField(min_value=1, max_value=29, initial=7,
                                            label="Representative days per month",
                                            help_text="More days = more accurate but slower.")
    gap_tol = forms.FloatField(min_value=0.0001, max_value=0.5, initial=0.01, label="Optimality gap tolerance")
    max_iters = forms.IntegerField(min_value=1, max_value=100, initial=30, label="Max Benders iterations")
    master_time_limit = forms.FloatField(min_value=10, max_value=3600, initial=180,
                                         label="Master time limit (s)")
    workers = forms.IntegerField(min_value=1, max_value=8, initial=2, label="Parallel workers",
                                 help_text="Each worker uses significant memory. Use 2 or fewer on constrained machines.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["countries"].choices = [(c, c) for c in services.eligible_countries()]
        self.fields["scenarios"].choices = [(s, f"{s}  ({p:.1%})") for s, p in services.scenario_probabilities().items()]
        self.fields["scenarios"].initial = list(services.scenario_probabilities())

    def clean(self):
        data = super().clean()
        if not data.get("all_countries") and not data.get("countries"):
            self.add_error("countries", "Select at least one country or choose all countries.")
        if not data.get("scenarios"):
            self.add_error("scenarios", "Select at least one uncertainty scenario.")
        if data.get("risk_measure") == "cvar" and data.get("cvar_alpha") is None:
            self.add_error("cvar_alpha", "Enter a confidence level for CVaR.")
        return data

    def params(self) -> dict:
        d = self.cleaned_data
        return {
            "all_countries": d["all_countries"],
            "countries": sorted(d["countries"]),
            "budget": d["budget"],
            "disabled_assets": list(d["disabled_assets"]),
            "max_units_per_candidate": d["max_units_per_candidate"],
            "scenarios": list(d["scenarios"]),
            "discount_rate_pct": d["discount_rate_pct"],
            "lifetime_years": d["lifetime_years"],
            "risk_measure": d["risk_measure"],
            "cvar_alpha": d["cvar_alpha"],
            "rep_days_per_month": d["rep_days_per_month"],
            "gap_tol": d["gap_tol"],
            "max_iters": d["max_iters"],
            "master_time_limit": d["master_time_limit"],
            "workers": d["workers"],
        }
