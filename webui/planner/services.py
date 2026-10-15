"""Bridge between the Django UI and the optimizer: builds plan_capacity.py commands and reads results back."""
from __future__ import annotations

import csv
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from django.conf import settings

PROJECT_ROOT = Path(settings.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "webui"
SCENARIO_OVERRIDES_ENV = "PLANNER_SCENARIO_OVERRIDES"
CATALOG_OVERRIDES_ENV = "PLANNER_CATALOG_OVERRIDES"
SITES_FILE_NAME = "sites.json"
BASELINE_SCENARIO = "p100"
SCENARIO_LABELS = {BASELINE_SCENARIO: "Baseline"}


def scenario_label(key: str, labels: dict | None = None) -> str:
    """Display name of a scenario: a run's own label for it, else Baseline for p100, else its key."""
    return (labels or {}).get(key) or SCENARIO_LABELS.get(key, key)


@lru_cache(maxsize=1)
def planner_module():
    import plan_capacity

    return plan_capacity


@lru_cache(maxsize=1)
def demand_services() -> list[str]:
    """The site's internal demands, in balance order."""
    import site_investor_planning as hp

    return list(hp.SERVICES)


@lru_cache(maxsize=1)
def site_defaults() -> dict:
    """The settings a new site starts from on the New run page: every annual peak and flexibility at
    0 (the user enters the site's own), and the default green hydrogen share."""
    import site_investor_planning as hp

    spec = hp.SiteSpec(name="Site 1")
    return {"peaks_mw": {s: 0.0 for s in hp.SERVICES}, "green_share": spec.green_share,
            "flex_fraction": {s: 0.0 for s in hp.SERVICES}}


@lru_cache(maxsize=1)
def site_assumptions():
    import site_investor_planning as hp

    tech = hp.SITE_TECH
    spec = hp.SiteSpec(name="Site 1")
    return {
        "services": list(hp.SERVICES),
        "default_peaks": [spec.peaks_mw[s] for s in hp.SERVICES],
        "default_annual": list(hp.annual_demand_mwh(spec.peaks_mw).values()),
        "cop": [{"asset": a, "service": svc, "cop": tech.cop(a, svc)}
                for a, services in hp.THERMAL_ASSET_SERVICES.items() for svc in services],
        "electrolyser_efficiency": 0.68,
        "unmet_penalty": tech.unmet_demand_penalty_eur_per_mwh,
        "grid_import_fee": tech.grid_import_fee_eur_per_mwh,
        "h2_import_fee": tech.h2_import_fee_eur_per_mwh,
        "green_share": site_defaults()["green_share"],
        "go_buy_price": hp.GREEN_H2.go_buy_price_eur_per_mwh,
        "go_sell_price": hp.GREEN_H2.go_sell_price_eur_per_mwh,
        "green_h2_premium": hp.GREEN_H2.green_h2_premium_eur_per_mwh,
        "flex_pct": spec.flex_fraction["electricity"] * 100,
    }


@lru_cache(maxsize=1)
def capex_assumptions_defaults():
    import site_investor_planning as hp

    cfg = hp.CapexAssumptions()
    return {
        "assets": list(hp.ASSETS),
        "catalog": {a: [c._asdict() for c in cfg.catalog[a]] for a in hp.ASSETS},
        "discount_rate": cfg.discount_rate,
        "budget": cfg.default_budget_eur,
        "cap_rules": {a: [factor, [dict(g) for g in groups]] for a, (factor, groups) in cfg.cap_rules.items()},
        "default_max_mw": hp.default_max_mw(site_defaults()["peaks_mw"], cfg),
        "lifetime_years": dict(cfg.lifetime_years),
    }


@lru_cache(maxsize=1)
def trained_error_ranges() -> dict[str, dict[str, list[int]]]:
    """Per country, the change in installed wind and solar capacity (%) the price models were trained
    on: each training scenario's capacity in that country compared with the Baseline's, positive for
    more and negative for less, rounded inward to whole percent. Empty when the training samples
    aren't on disk."""
    import math

    import pandas as pd

    path = PROJECT_ROOT / "data_exchange" / "01_dispatch_output__train_input" / "elec_samples.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["scenario", "zone", "wind_capacity_mw", "pv_capacity_mw"])
    per_zone = df.groupby(["scenario", "zone"])[["wind_capacity_mw", "pv_capacity_mw"]].max().reset_index()
    per_zone["country"] = per_zone["zone"].str[:2]
    per_country = per_zone.groupby(["scenario", "country"])[["wind_capacity_mw", "pv_capacity_mw"]].sum()
    ratio = (per_country / per_country.loc["p100"]).reset_index()
    out = {}
    for country, g in ratio.groupby("country"):
        out[country] = {kind: [math.ceil(round((g[col].min() - 1) * 100, 6)), math.floor(round((g[col].max() - 1) * 100, 6))]
                        for kind, col in (("wind", "wind_capacity_mw"), ("solar", "pv_capacity_mw"))}
    return out


@lru_cache(maxsize=1)
def cap_rule_rows() -> list[dict]:
    """One row per asset for the Catalog page: how its default maximum MW per site follows the site's
    peaks, and the value at the command-line default site's peaks."""
    import site_investor_planning as hp

    cfg = hp.CapexAssumptions()
    example = hp.default_max_mw(hp.SiteSpec(name="Site 1").peaks_mw, cfg)
    rows = []
    for a, (factor, groups) in cfg.cap_rules.items():
        sums = []
        for g in groups:
            terms = [(SERVICE_LABELS[svc] if i == 0 else SERVICE_LABELS[svc][0].lower() + SERVICE_LABELS[svc][1:],
                      w) for i, (svc, w) in enumerate(g.items())]
            sums.append(" + ".join(label if abs(w - 1.0) < 1e-9 else f"{w:.2f} × {label[0].lower() + label[1:]}"
                                   for label, w in terms))
        parts = [t[0].lower() + t[1:] for t in sums]
        basis = sums[0] if len(sums) == 1 else "The larger of " + " and ".join(
            f"({t})" if len(g) > 1 else t for t, g in zip(parts, groups))
        rule = (f"1.25 × peak ÷ 68% efficiency (= {factor:.2f} × peak)" if a == "electrolyser_mw"
                else f"{factor:g} × peak")
        rows.append({"asset": a, "rule": rule, "basis": basis, "example": example[a]})
    return rows


@lru_cache(maxsize=1)
def eligible_countries() -> list[str]:
    return planner_module().eligible_countries()


ASSET_GROUPS = {
    "electricity": ["wind_mw", "pv_mw", "battery_mw", "chp_mw"],
    "space_heat": ["heat_pump_mw", "hot_water_storage_mw"],
    "space_cool": ["ac_chiller_mw", "cold_storage_mw"],
    "process_heat": ["industrial_heat_pump_mw", "solar_thermal_mw"],
    "steam": ["electric_boiler_mw", "thermal_battery_mw"],
    "process_cool": ["electric_chiller_mw", "heat_recovery_chiller_mw", "absorption_chiller_mw"],
    "hydrogen": ["electrolyser_mw", "tank_mw"],
}
GROUP_LABELS = {
    "electricity": "Electricity", "space_heat": "Space heating", "space_cool": "Space cooling",
    "process_heat": "Process heat (up to 150 °C)", "steam": "Steam (above 150 °C)",
    "process_cool": "Process cooling", "hydrogen": "Hydrogen",
}
GROUP_MAP_TOKENS = {g: f"--group-{g.replace('_', '-')}" for g in ASSET_GROUPS}
SERVICE_LABELS = GROUP_LABELS
MARKER_MIN_DIAMETER = 18
MARKER_MAX_DIAMETER = 56


def _marker_diameter(total: float, max_total: float) -> float:
    if max_total <= 0:
        return MARKER_MIN_DIAMETER
    scale = (total / max_total) ** 0.5
    return round(MARKER_MIN_DIAMETER + (MARKER_MAX_DIAMETER - MARKER_MIN_DIAMETER) * scale, 1)


def _marker_gradient(values: dict[str, float], total: float) -> str:
    if total <= 0:
        return ""
    stops = []
    acc = 0.0
    for group, members in ASSET_GROUPS.items():
        share = sum(values.get(a, 0.0) for a in members)
        if share <= 0:
            continue
        pct = share / total * 100
        stops.append(f"var({GROUP_MAP_TOKENS[group]}) {acc:.2f}% {acc + pct:.2f}%")
        acc += pct
    return "conic-gradient(" + ", ".join(stops) + ")"


def country_map_markers(params: dict, summary: dict) -> list[dict]:
    from . import geo

    assets = capex_assumptions_defaults()["assets"]
    codes = eligible_countries() if params.get("all_countries") else params.get("countries", [])
    by_country = {}
    for row in summary.get("site_rows", []):
        entry = by_country.setdefault(row["country"], {"values": {a: 0.0 for a in assets}, "sites": []})
        entry["sites"].append(row["site"])
        for a in assets:
            entry["values"][a] += float(row.get(a, 0) or 0)

    rows = []
    for code in codes:
        centroid = geo.COUNTRY_CENTROIDS.get(code)
        if centroid is None:
            continue
        entry = by_country.get(code, {"values": {a: 0.0 for a in assets}, "sites": []})
        rows.append({
            "code": code,
            "lat": centroid[0],
            "lon": centroid[1],
            "total": sum(entry["values"].values()),
            "values": entry["values"],
            "site": bool(entry["sites"]),
            "site_names": entry["sites"],
        })

    max_total = max((r["total"] for r in rows), default=0.0)
    for r in rows:
        r["diameter"] = _marker_diameter(r["total"], max_total)
        r["gradient"] = _marker_gradient(r["values"], r["total"])
    return rows


def economics(params: dict, summary: dict) -> dict:
    """CAPEX, annualized CAPEX and operating cost of a finished run, priced with that run's own catalog."""
    import site_investor_planning as hp

    catalog = dict(hp.CANDIDATE_CATALOG)
    for asset, candidates in (params.get("catalog_overrides") or {}).items():
        if asset in catalog and candidates:
            catalog[asset] = [hp.AssetCandidate(**c) for c in candidates]
    cfg = hp.CapexAssumptions(catalog=catalog, discount_rate=params.get("discount_rate_pct", 5) / 100)
    crfs = cfg.capital_recovery_factors()

    capex_by_asset = {a: 0.0 for a in hp.ASSETS}
    by_site: dict[str, dict[str, dict[str, float]]] = {}
    for row in summary.get("units", []):
        asset = row.get("asset")
        if asset not in capex_by_asset:
            continue
        candidate = cfg.catalog[asset][int(row["candidate"])]
        capex = candidate.capex_eur * float(row["units"])
        capex_by_asset[asset] += capex
        site = by_site.setdefault(row.get("site") or row["country"], {}).setdefault(
            asset, {"mw": 0.0, "capex_eur": 0.0, "annualized_eur": 0.0, "products": []})
        site["mw"] += candidate.mw * float(row["units"])
        site["products"].append({"units": int(float(row["units"])), "mw": f"{candidate.mw:g}"})
        site["capex_eur"] += capex
        site["annualized_eur"] += capex * crfs[asset]

    annualized_by_asset = {a: capex_by_asset[a] * crfs[a] for a in hp.ASSETS}
    annualized_total = sum(annualized_by_asset.values())
    raw_capex = summary.get("raw_capex_eur")
    if raw_capex is None:
        raw_capex = sum(capex_by_asset.values())
    objective = summary.get("objective_eur")
    total_mw = sum(summary.get("totals_mw", {}).values())

    return {
        "raw_capex_eur": raw_capex,
        "annualized_capex_eur": annualized_total,
        "operating_cost_eur": (objective - annualized_total) if objective is not None else None,
        "capex_per_mw_eur": (raw_capex / total_mw) if total_mw else None,
        "by_site": by_site,
        "by_asset": {
            a: {"capex_eur": capex_by_asset[a], "annualized_eur": annualized_by_asset[a], "crf": crfs[a]}
            for a in hp.ASSETS
        },
    }


@lru_cache(maxsize=1)
def scenario_probabilities() -> dict[str, float]:
    return dict(planner_module().SCENARIO_PROBS)


@lru_cache(maxsize=1)
def scenario_defaults() -> dict[str, dict]:
    """The planning scenarios (positive default probability) from ``inputs/uncertainty_scenarios.json``,
    the baseline first."""
    path = PROJECT_ROOT / "inputs" / "uncertainty_scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
    planning = {name: sc for name, sc in scenarios.items() if float(sc["probability"]) > 0}
    order = sorted(planning, key=lambda n: (n != BASELINE_SCENARIO, list(scenarios).index(n)))
    return {n: planning[n] for n in order}


@lru_cache(maxsize=1)
def all_scenario_names() -> list[str]:
    """Every scenario defined in ``inputs/uncertainty_scenarios.json``, planning or not."""
    path = PROJECT_ROOT / "inputs" / "uncertainty_scenarios.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["scenarios"])


@lru_cache(maxsize=1)
def default_scenarios() -> list[str]:
    """Scenarios ticked on a new run, from ``default_scenarios`` in ``inputs/uncertainty_scenarios.json``."""
    path = PROJECT_ROOT / "inputs" / "uncertainty_scenarios.json"
    return json.loads(path.read_text(encoding="utf-8")).get("default_scenarios", [BASELINE_SCENARIO])


def build_command(params: dict, output_prefix: Path) -> list[str]:
    cmd = [sys.executable, "plan_capacity.py"]
    if params["all_countries"]:
        cmd.append("--all")
    else:
        cmd += ["--countries", ",".join(params["countries"])]
    if params.get("sites"):
        cmd += ["--sites-file", str(Path(output_prefix).parent / SITES_FILE_NAME)]
    else:
        cmd += ["--n-sites", str(params.get("n_sites", 1))]
    cmd += ["--no-budget"] if params.get("budget") is None else ["--budget", f"{params['budget']:.0f}"]
    cmd += ["--discount-rate", f"{params['discount_rate_pct'] / 100:.6f}",
            "--rep-days-per-month", str(params["rep_days_per_month"]),
            "--gap-tol", str(params["gap_tol"]),
            "--max-iters", str(params["max_iters"]),
            "--master-time-limit", str(params["master_time_limit"]),
            "--workers", str(params["workers"]),
            "--output", str(output_prefix)]
    if params.get("risk_measure") == "cvar":
        cmd += ["--cvar-alpha", str(params["cvar_alpha"])]
    if params.get("max_units_per_candidate"):
        cmd += ["--max-units-per-candidate", str(params["max_units_per_candidate"])]
    if params.get("disabled_assets"):
        cmd += ["--disabled-assets", ",".join(params["disabled_assets"])]
    if params.get("scenarios"):
        cmd += ["--scenarios", ",".join(params["scenarios"])]
    return cmd


def run_dir(run_pk: int) -> Path:
    return RUNS_DIR / f"run_{run_pk}"


def output_prefix_for(run_pk: int) -> Path:
    return run_dir(run_pk) / "plan"


_DONE_RE = re.compile(r"Done in ([\d.]+)s")
_SITES_RE = re.compile(r"Chosen site\(s\): (.*)$", re.MULTILINE)
_CAPEX_RE = re.compile(r"Total raw CAPEX: ([\d,]+) EUR")
_OBJ_RE = re.compile(r"Best objective.*?: (-?[\d,]+)\s*$", re.MULTILINE)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def parse_summary(output_prefix: Path, log: str) -> dict:
    capacities = _read_csv(Path(f"{output_prefix}_capacities.csv"))
    units = _read_csv(Path(f"{output_prefix}_units.csv"))
    green = [{k: _maybe_float(v) for k, v in row.items()} for row in _read_csv(Path(f"{output_prefix}_green_h2.csv"))]
    unmet = [{k: _maybe_float(v) for k, v in row.items()}
             for row in _read_csv(Path(f"{output_prefix}_unmet_demand.csv"))]
    convergence = _read_csv(Path(f"{output_prefix}_convergence.csv"))
    market = _read_csv(Path(f"{output_prefix}_market_costs.csv"))

    assets = list(capex_assumptions_defaults()["assets"])
    totals = {a: 0.0 for a in assets}
    for row in capacities:
        for a in assets:
            totals[a] += float(row.get(a) or 0)

    done = _DONE_RE.search(log)
    capex = _CAPEX_RE.search(log)
    objective = _OBJ_RE.search(log)
    site_rows = _site_rows(capacities)
    sites = [f"{r['site']} → {r['country']}" for r in site_rows]
    return {
        "sites": sites,
        "site_rows": site_rows,
        "elapsed_seconds": float(done.group(1)) if done else None,
        "iterations": len(convergence),
        "converged": bool(re.search(r"converged \(gap", log)),
        "objective_eur": _num(objective.group(1)) if objective else None,
        "raw_capex_eur": _num(capex.group(1)) if capex else None,
        "capacities": capacities,
        "units": units,
        "green_h2": green,
        "green_h2_expected": _expected_green(green),
        "unmet_demand": _expected_unmet(unmet),
        "unmet_labels": [GROUP_LABELS[k[len("unmet_"):-len("_mwh")]] for k in (unmet[0] if unmet else {})
                         if k.startswith("unmet_") and k != "unmet_total_mwh"],
        "convergence": [{k: _maybe_float(v) for k, v in row.items()} for row in convergence],
        "totals_mw": totals,
        "market": _expected_market(market),
    }


def _site_rows(capacities: list[dict]) -> list[dict]:
    """One row per built site: its name, country and capacities. Reads both the per-site results
    (``site`` holds the site name) and older per-country results (``site`` is a 0/1 flag)."""
    rows = []
    for r in capacities:
        flag = r.get("site", "")
        try:
            built = float(flag) > 0.5
            name = r["country"]
        except (TypeError, ValueError):
            built, name = True, flag
        if built:
            rows.append({**r, "site": name})
    return rows


def _expected_green(rows: list[dict]) -> list[dict]:
    """Probability-weighted green hydrogen totals per chosen site."""
    keys = ["h2_demand_mwh", "green_h2_produced_mwh", "green_h2_bought_mwh", "green_share",
            "go_bought_mwh", "go_sold_mwh"]
    out = {}
    for row in rows:
        site = row.get("site") or row["country"]
        acc = out.setdefault(site, {"site": site, "country": row["country"], **{k: 0.0 for k in keys}})
        for k in keys:
            acc[k] += float(row[k]) * float(row["probability"])
    return list(out.values())


MARKET_ITEMS = {
    "electricity_bought": ("Electricity market", "Electricity bought"),
    "grid_fee": ("Electricity market", "Grid import fee"),
    "electricity_sold": ("Electricity market", "Electricity sold"),
    "hydrogen_bought": ("Hydrogen market", "Hydrogen bought"),
    "hydrogen_sold": ("Hydrogen market", "Hydrogen sold"),
    "green_h2_premium": ("Green hydrogen market", "Premium paid for green hydrogen"),
    "certificates_bought": ("Renewable electricity certificates", "Certificates bought"),
    "certificates_sold": ("Renewable electricity certificates", "Certificates sold"),
    "biogas": ("Biogas", "Biogas burned in combined heat and power"),
    "storage_wear": ("Storage", "Storage operating cost"),
    "unmet_penalty": ("Unmet demand", "Heat and cooling not supplied (penalty)"),
}


def _expected_market(rows: list[dict]) -> list[dict]:
    """Probability-weighted yearly volume and cost of each market item per chosen site, leaving out
    items a site doesn't use; unmet heat and cooling is always listed."""
    out = {}
    for row in rows:
        acc = out.setdefault(row["site"], {"site": row["site"], "country": row["country"],
                                           "items": {k: [0.0, 0.0] for k in MARKET_ITEMS}})
        if row["item"] in acc["items"]:
            prob = float(row["probability"])
            acc["items"][row["item"]][0] += float(row["mwh"]) * prob
            acc["items"][row["item"]][1] += float(row["eur"]) * prob
    sites = []
    for acc in out.values():
        items = [{"market": MARKET_ITEMS[k][0], "label": MARKET_ITEMS[k][1], "mwh": mwh, "eur": eur}
                 for k, (mwh, eur) in acc["items"].items() if abs(eur) >= 0.5 or mwh >= 0.5 or k == "unmet_penalty"]
        sites.append({"site": acc["site"], "country": acc["country"], "items": items,
                      "total": sum(i["eur"] for i in items)})
    return sites


def cost_table(summary: dict, economics: dict, labels: dict[str, str], risk_measure: str | None) -> dict:
    """Each chosen site's yearly cost: the annualized capital cost of each technology it built, then
    what it buys and sells on each market, unmet heat and cooling included. The sites' totals add
    up to the run's objective. Under CVaR the operating part of the objective is the average of
    the worst scenarios rather than the probability-weighted sum of the sites, and the difference
    is shown as a risk adjustment."""
    sites = summary.get("market") or []
    if not sites:
        return {}
    by_site = economics.get("by_site", {})
    out = []
    for s in sites:
        invest = [{"group": "Investment", "label": labels.get(a, a), "amount": v["mw"], "unit": "MW",
                   "products": v["products"], "capex": v["capex_eur"], "eur": v["annualized_eur"]}
                  for a, v in by_site.get(s["site"], {}).items() if v["mw"] > 0]
        operate = [{"group": i["market"], "label": i["label"], "amount": i["mwh"], "unit": "MWh", "eur": i["eur"]}
                   for i in s["items"]]
        rows = invest + operate
        out.append({"site": s["site"], "country": s["country"], "rows": rows,
                    "investment": sum(r["eur"] for r in invest), "operation": sum(r["eur"] for r in operate),
                    "capex": sum(r["capex"] for r in invest),
                    "total": sum(r["eur"] for r in rows)})
    total = sum(s["total"] for s in out)
    adjustment = 0.0
    if risk_measure == "cvar" and economics.get("operating_cost_eur") is not None:
        adjustment = economics["operating_cost_eur"] - sum(s["operation"] for s in out)
    investment = sum(s["investment"] for s in out)
    return {"sites": out, "total": total, "capex": sum(s["capex"] for s in out), "investment": investment,
            "operation": sum(s["operation"] for s in out) + adjustment, "risk_adjustment": adjustment if abs(adjustment) >= 1.0 else 0.0,
            "grand_total": total + adjustment}


BALANCE_ORDER = ["electricity", "hydrogen", "space_heat", "space_cool", "process_heat", "steam", "process_cool"]


@lru_cache(maxsize=4)
def _dispatch_table(path: str, mtime: float):
    """A run's ``_dispatch.csv``, cached until the file changes."""
    import pandas as pd
    return pd.read_csv(path)


def _dispatch(output_prefix: str):
    """The run's hourly operation table, or None for a run made before it was written."""
    path = Path(f"{output_prefix}_dispatch.csv")
    return _dispatch_table(str(path), path.stat().st_mtime) if path.exists() else None


def _day_label(day: int) -> str:
    """Day of the model year as a date, e.g. 15 Jan."""
    import datetime
    return (datetime.date(2023, 1, 1) + datetime.timedelta(days=int(day) - 1)).strftime("%d %b").lstrip("0")


def dispatch_options(output_prefix: str, scenario_labels: dict) -> dict:
    """The scenarios and representative days a run's hourly operation can be shown for."""
    df = _dispatch(output_prefix)
    if df is None:
        return {}
    days = sorted(df["day_of_year"].unique())
    return {"scenarios": [{"key": s, "label": scenario_labels.get(s, s)} for s in df["scenario"].unique()],
            "days": [{"day": int(d), "label": _day_label(d)} for d in days]}


def dispatch_data(output_prefix: str, scenario: str, start_day: int, end_day: int) -> dict:
    """Hourly prices and balances of every site from ``start_day`` to ``end_day`` (representative days
    only) in one scenario. A balance with no demand and no supply all year is left out, and so is
    any component that is zero all year at that site."""
    df = _dispatch(output_prefix)
    if df is None:
        return {}
    df = df[df["scenario"] == scenario]
    sites = []
    for (site, country), year in df.groupby(["site", "country"], sort=False):
        window = year[(year["day_of_year"] >= start_day) & (year["day_of_year"] <= end_day)]
        labels = [f"{_day_label(d)} {h:02d}:00" for d, h in zip(window["day_of_year"], window["hour"])]
        balances = []
        for b in BALANCE_ORDER:
            cols = [c for c in year.columns if c.startswith(b + "|") and c != f"{b}|Demand"]
            used = [c for c in cols if year[c].abs().max() > 1e-6]
            if not used and year[f"{b}|Demand"].abs().max() <= 1e-6:
                continue
            balances.append({"key": b, "label": GROUP_LABELS[b],
                             "demand": window[f"{b}|Demand"].round(3).tolist(),
                             "components": [{"label": c.split("|", 1)[1], "values": window[c].round(3).tolist()}
                                            for c in used]})
        sites.append({"site": site, "country": country, "labels": labels,
                      "prices": {"Electricity": window["price|Electricity"].round(2).tolist(),
                                 "Hydrogen": window["price|Hydrogen"].round(2).tolist()},
                      "balances": balances})
    return {"sites": sites}


def _expected_unmet(rows: list[dict]) -> list[dict]:
    """Probability-weighted unmet heat and cooling demand (MWh/yr) per chosen site, one value per demand."""
    services = [s for s in GROUP_LABELS if f"unmet_{s}_mwh" in (rows[0] if rows else {})]
    out = {}
    for row in rows:
        site = row.get("site") or row["country"]
        acc = out.setdefault(site, {"site": site, "country": row["country"], "total": 0.0,
                                    "values": {s: 0.0 for s in services}})
        p = float(row["probability"])
        for s in services:
            acc["values"][s] += float(row[f"unmet_{s}_mwh"]) * p
        acc["total"] += float(row["unmet_total_mwh"]) * p
    return [{**r, "values": [r["values"][s] for s in services]} for r in out.values()] if out else []



def _maybe_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
