"""Industrial Site Investor planning: Benders decomposition CLI placing each site in a candidate country and
choosing its technologies and capacities under a CAPEX budget and wind/PV uncertainty scenarios."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import json

import pandas as pd

import optimize_site_investor as ohp
import site_investor_planning as hp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

CATALOG_OVERRIDES_ENV = "PLANNER_CATALOG_OVERRIDES"


def load_catalog_override(capex_cfg: hp.CapexAssumptions, path: Path) -> None:
    """Replace per-asset candidate lists in-place from a {"catalog": {asset: [candidate, ...]}} JSON file."""
    overrides = json.loads(path.read_text(encoding="utf-8"))["catalog"]
    unknown = [a for a in overrides if a not in hp.ASSETS]
    if unknown:
        raise ValueError(f"catalog override has unknown asset key(s) {unknown} -- choices: {hp.ASSETS}")
    for asset, candidates in overrides.items():
        capex_cfg.catalog[asset] = [hp.AssetCandidate(**c) for c in candidates]
    capex_cfg.lifetime_years = capex_cfg._lifetime_years_from_catalog()


def load_scenario_probs(path: Path | None = None) -> dict[str, float]:
    """Read {scenario: probability} from the saved scenario JSON, keeping only scenarios with a positive
    probability (the rest stay defined for the dispatch runs but are left out of planning)."""
    if path is None:
        data = ohp.load_uncertainty_scenarios()
        source = ROOT / "inputs" / "uncertainty_scenarios.json"
    else:
        data = json.loads(path.read_text())["scenarios"]
        source = path
    probs = {name: float(sc["probability"]) for name, sc in data.items() if float(sc["probability"]) > 0}
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"{source}'s scenario probabilities sum to {total:.6f}, not 1.0")
    return probs


SCENARIO_PROBS = load_scenario_probs()
SCENARIOS = list(SCENARIO_PROBS)
DEFAULT_SCENARIOS = json.loads((ROOT / "inputs" / "uncertainty_scenarios.json").read_text()).get(
    "default_scenarios", [ohp.BASELINE_SCENARIO])


def risk_measure(scenario_probs: dict[str, float], cvar_alpha: float) -> float | None:
    """The CVaR confidence level to plan with: ``None`` (deterministic) for a single scenario,
    ``cvar_alpha`` for two or more."""
    return None if len(scenario_probs) == 1 else cvar_alpha


def eligible_countries() -> list[str]:
    return sorted(hp.candidate_site_zones())


def _snap(value: float, tol: float = 1e-3) -> float:
    """Zero out a core-point coordinate that has decayed below ``tol``; tiny values make the subproblem numerically fragile, and any core point still yields a valid cut."""
    return 0.0 if abs(value) < tol else value


def green_h2_summary(schedule: pd.DataFrame) -> dict[str, float]:
    """Annual green hydrogen and Guarantee of Origin totals (MWh/yr) of one site's representative-day schedule."""
    w = schedule["day_weight"]

    def annual(col: str) -> float:
        return float((schedule[col] * w).sum())

    demand = annual("Site hydrogen demand (MW)")
    produced = annual("Site green H2 produced (MW)")
    bought = annual("Site green H2 bought (MW)")
    return {"h2_demand_mwh": demand, "green_h2_produced_mwh": produced, "green_h2_bought_mwh": bought,
            "green_share": (produced + bought) / demand if demand > 0 else 0.0,
            "go_bought_mwh": annual("Site GOs bought (MWh/h)"), "go_sold_mwh": annual("Site GOs sold (MWh/h)")}


def unmet_demand_summary(schedule: pd.DataFrame) -> dict[str, float]:
    """Annual heat and cooling demand (MWh/yr) of one site's representative-day schedule that the new assets left unmet."""
    w = schedule["day_weight"]
    out = {f"unmet_{svc}_mwh": float((schedule[f"Site unmet {svc} demand (MW)"] * w).sum())
           for svc in hp.THERMAL_SERVICES}
    out["unmet_total_mwh"] = sum(out.values())
    return out


def _unit_id(site: str, country: str) -> str:
    """Label of one site placed in one candidate country."""
    return f"{site}@{country}"


def run_benders(countries: list[str], budget: float | None, max_iters: int, gap_tol: float,
                capex_cfg: hp.CapexAssumptions, rep_days_per_month: int,
                quiet_solver: bool = True, on_iteration=None, cvar_alpha: float | None = None,
                master_time_limit: float = 180.0,
                disabled_assets: list[str] | None = None,
                scenario_probs: dict[str, float] | None = None,
                workers: int | None = None,
                max_units_per_candidate: int | None = None,
                sites: list[hp.SiteSpec] | None = None):
    """Run the Benders loop: master MILP proposes where each site goes and its capacities, joint subproblems (one per scenario) price them and return cuts, repeat to convergence.

    Every site in ``sites`` (default: one default site) is placed in exactly one of ``countries``;
    several sites may share a country. Internally each (site, country) pair is a unit with its own
    capacities and its own on/off placement. Each iteration adds two cuts per scenario: one at the
    trial point itself, so the master can't propose the same plan again without paying its true
    recourse cost, and one at a Pareto core point that starts inside the feasible region (every
    unit a fractional placement, every asset a mid-grid size within its site cap) and moves halfway
    toward each trial point. If HiGHS presolve fails on the master, it is re-solved once with
    presolve off."""
    scenario_probs = scenario_probs if scenario_probs is not None else {s: 1.0 / len(DEFAULT_SCENARIOS)
                                                                          for s in DEFAULT_SCENARIOS}
    if len(scenario_probs) > 1 and cvar_alpha is None:
        raise ValueError("planning over more than one scenario needs a CVaR confidence level (cvar_alpha)")
    sites = sites if sites is not None else hp.default_sites(1)
    cand_mw_c, cand_capex_c, host_zone = hp.build_candidates(countries, capex_cfg)
    crf = capex_cfg.capital_recovery_factors()

    units = {_unit_id(sp.name, c): (host_zone[c], sp) for sp in sites for c in countries}
    unit_ids = list(units)
    unit_country = {_unit_id(sp.name, c): c for sp in sites for c in countries}
    site_units = {sp.name: [_unit_id(sp.name, c) for c in countries] for sp in sites}
    cand_mw = {u: cand_mw_c[unit_country[u]] for u in unit_ids}
    cand_capex = {u: cand_capex_c[unit_country[u]] for u in unit_ids}
    site_caps = {sp.name: hp.site_max_mw(sp, capex_cfg) for sp in sites}
    site_max_mw = {u: dict(site_caps[units[u][1].name]) for u in unit_ids}
    for u in unit_ids:
        for a in disabled_assets or []:
            site_max_mw[u][a] = 0.0

    if budget is not None:
        cheapest_total = len(sites) * sum(min(cand.capex_eur for cand in capex_cfg.catalog[a]) for a in hp.ASSETS
                                          if not (disabled_assets and a in disabled_assets))
        if cheapest_total > budget:
            print(f"NOTE: budget {budget:,.0f} EUR is below the cheapest all-assets-built "
                 f"combination ({cheapest_total:,.0f} EUR) for {len(sites)} site(s) -- expect some "
                 f"assets to come back skipped (0 MW) in the result.")

    print(f"Building enriched price frames for {len(scenario_probs)} capacity scenarios "
         f"(one-time cost, reused by every subproblem solve): {list(scenario_probs)}")
    frames_by_source = {}
    for s in scenario_probs:
        src = ohp.price_scenario(s)
        if src not in frames_by_source:
            frames_by_source[src] = (ohp.enriched_elec_df(scenario=src), ohp.enriched_h2_df(scenario=src))
    price_frames = {s: frames_by_source[ohp.price_scenario(s)] for s in scenario_probs}

    m = hp.build_master(unit_ids, cand_mw, cand_capex, budget, crf, capex_cfg.theta_lower_bound_eur,
                        scenario_probs=scenario_probs, cvar_alpha=cvar_alpha, site_max_mw=site_max_mw,
                        max_units_per_candidate=max_units_per_candidate, site_units=site_units)

    core_cap = {u: {a: min(float(sum([0.0] + list(cand_mw[u][a])) / (len(cand_mw[u][a]) + 1)),
                           0.5 * site_max_mw[u][a])
                    for a in hp.ASSETS} for u in unit_ids}
    core_site = {u: 1.0 / len(countries) for u in unit_ids}

    best_ub, best_capacities, best_capex, best_capex_by_asset = float("inf"), None, None, None
    best_units, best_sites, best_green, best_unmet = None, None, None, None
    log = []
    per_unit_log = {u: [] for u in unit_ids}
    gap = float("inf")

    n_workers = workers if workers is not None else min(4, os.cpu_count() or 1)
    executor = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    if executor is not None:
        print(f"Subproblems will run across {n_workers} worker processes "
             f"(each scenario's trial-point and core-point solves are independent).")
    try:
        for it in range(1, max_iters + 1):
            t_master = time.time()
            status, cond = m.solve(solver_name="highs", output_flag=False, time_limit=master_time_limit,
                                   presolve="on", parallel="on")
            if status != "ok":
                status, cond = m.solve(solver_name="highs", output_flag=False, time_limit=master_time_limit,
                                       presolve="off", parallel="on")
            master_s = time.time() - t_master
            if status != "ok":
                raise RuntimeError(f"master solve failed at iteration {it}: {status}/{cond}")
            cap_star = hp.extract_capacities(m, unit_ids, cand_mw)
            site_star = hp.extract_sites(m, unit_ids)
            lb = float(m.solver_model.getInfo().mip_dual_bound)
            master_mip_gap = float(m.solver_model.getInfo().mip_gap)
            master_cut_off = (cond == "time_limit")

            t_sub = time.time()
            sub_build_s = sub_solve_s = 0.0
            n_cuts = 0

            def solve_all(caps, placed):
                if executor is not None:
                    futures = {s: executor.submit(ohp.solve_joint, units, caps, rep_days_per_month,
                                                  return_duals=True, edf=price_frames[s][0],
                                                  hdf=price_frames[s][1], quiet=quiet_solver, scenario=s,
                                                  sites=placed)
                               for s in scenario_probs}
                    return {s: f.result() for s, f in futures.items()}
                return {s: ohp.solve_joint(units, caps, rep_days_per_month, return_duals=True,
                                           edf=price_frames[s][0], hdf=price_frames[s][1],
                                           quiet=quiet_solver, scenario=s, sites=placed)
                        for s in scenario_probs}

            results = solve_all(cap_star, site_star)
            results_core = solve_all(core_cap, core_site)

            total_Q = 0.0
            Q_total_by_scenario: dict[str, float] = {}
            Q_by_unit = {u: 0.0 for u in unit_ids}
            Q_by_unit_by_scenario: dict[str, dict[str, float]] = {}
            for s, prob in scenario_probs.items():
                result = results[s]
                sub_build_s += result["build_seconds"]
                sub_solve_s += result["solve_seconds"]
                total_Q += prob * float(result["objective"])
                Q_total_by_scenario[s] = float(result["objective"])
                Q_s = {u: result["objective_by_zone"][u] for u in unit_ids}
                Q_by_unit_by_scenario[s] = Q_s
                for u in unit_ids:
                    Q_by_unit[u] += prob * Q_s[u]
                hp.add_optimality_cut(m, unit_ids, it, Q_s, result["cut_coeffs"], cap_star, cand_mw,
                                      scenario=s, lam=result["site_coeffs"], site_star=site_star, tag="trial")
                n_cuts += 1

                result_core = results_core[s]
                sub_build_s += result_core["build_seconds"]
                sub_solve_s += result_core["solve_seconds"]
                hp.add_optimality_cut(m, unit_ids, it, result_core["objective_by_zone"], result_core["cut_coeffs"],
                                      core_cap, cand_mw, scenario=s, lam=result_core["site_coeffs"],
                                      site_star=core_site)
                n_cuts += 1
            sub_s = time.time() - t_sub

            for u in unit_ids:
                for a in hp.ASSETS:
                    core_cap[u][a] = _snap(0.5 * core_cap[u][a] + 0.5 * cap_star[u][a])
                core_site[u] = _snap(0.5 * core_site[u] + 0.5 * site_star[u])

            capex_star = hp.extract_capex(m, unit_ids, cand_capex)
            for u in unit_ids:
                per_unit_log[u].append({"iter": it, "objective": Q_by_unit[u],
                                        "objective_by_scenario": {s: Q_by_unit_by_scenario[s][u]
                                                                  for s in scenario_probs},
                                        "capex": sum(capex_star[u].values()), "capacities": dict(cap_star[u]),
                                        "site": site_star[u]})
            raw_capex = sum(capex_star[u][a] for u in unit_ids for a in hp.ASSETS)
            annualized_capex = sum(capex_star[u][a] * crf[a] for u in unit_ids for a in hp.ASSETS)
            if len(scenario_probs) > 1:
                ub = annualized_capex + hp.cvar_value(Q_total_by_scenario, scenario_probs, cvar_alpha)
            else:
                ub = annualized_capex + total_Q
            if ub < best_ub:
                best_ub, best_capacities, best_capex, best_capex_by_asset = ub, cap_star, raw_capex, capex_star
                best_units = hp.extract_units(m, unit_ids)
                best_sites = site_star
                best_green = [{"site": units[u][1].name, "country": unit_country[u], "scenario": s,
                               "probability": scenario_probs[s],
                               **green_h2_summary(results[s]["schedules"][u])}
                              for u in unit_ids if site_star[u] > 0.5 for s in scenario_probs]
                best_unmet = [{"site": units[u][1].name, "country": unit_country[u], "scenario": s,
                               "probability": scenario_probs[s],
                               **unmet_demand_summary(results[s]["schedules"][u])}
                              for u in unit_ids if site_star[u] > 0.5 for s in scenario_probs]
            gap = (best_ub - lb) / max(abs(best_ub), 1e-6)
            log.append({"iter": it, "lb": lb, "ub": ub, "best_ub": best_ub, "gap": gap,
                        "master_seconds": round(master_s, 2),
                        "subproblems_seconds": round(sub_s, 1),
                        "subproblems_build_seconds": round(sub_build_s, 1),
                        "subproblems_solve_seconds": round(sub_solve_s, 1),
                        "n_cuts": n_cuts,
                        "master_mip_gap": master_mip_gap, "master_cut_off": master_cut_off})
            cutoff_note = " [MASTER CUT OFF AT time_limit]" if master_cut_off else ""
            print(f"iter {it:>3}: LB={lb:>16,.0f}  UB={ub:>16,.0f}  best={best_ub:>16,.0f}  gap={gap:.4f}  "
                 f"master={master_s:.2f}s  subproblems={sub_s:.1f}s (build={sub_build_s:.1f}s, "
                 f"solve={sub_solve_s:.1f}s)  cuts_added={n_cuts}  "
                 f"master_mip_gap={master_mip_gap:.4f}{cutoff_note}")
            if on_iteration is not None:
                on_iteration({"iter": it, "lb": lb, "ub": ub, "best_ub": best_ub, "gap": gap,
                             "best_capacities": best_capacities, "best_capex_by_asset": best_capex_by_asset,
                             "best_sites": best_sites, "units": units, "log": list(log),
                             "per_unit_log": per_unit_log})
            if gap <= gap_tol:
                print(f"converged (gap {gap:.4f} <= tol {gap_tol}) after {it} iteration(s)")
                break
        else:
            print(f"WARNING: reached --max-iters={max_iters} without closing the gap "
                 f"(final gap {gap:.4f}) -- results below are the best FOUND, not proven optimal")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return {"capacities": best_capacities, "capex": best_capex, "objective": best_ub, "log": pd.DataFrame(log),
            "units": units, "unit_country": unit_country, "host_zone": host_zone, "price_frames": price_frames,
            "per_unit_log": per_unit_log, "capex_by_asset": best_capex_by_asset, "unit_counts": best_units,
            "placed": best_sites, "green": best_green, "unmet": best_unmet}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--countries", type=str, default=None,
                       help="comma-separated 2-letter country codes, e.g. DE,FR,PL")
    group.add_argument("--all", action="store_true", help="plan every eligible country")
    ap.add_argument("--sites-file", type=str, default=None,
                    help='JSON file defining the sites to build, each with its own demand peaks, green '
                         'H2 share and flexibility: {"sites": [{"name": "Site 1", "peaks_mw": {...}, '
                         '"green_share": 0.42, "flex_fraction": 0.1}, ...]}; overrides --n-sites, '
                         "--green-h2-share-pct and --demand-flex-pct")
    ap.add_argument("--n-sites", type=int, default=1,
                    help="without --sites-file: how many sites with the default settings to build "
                         "(default 1); the optimizer picks a country for each, and sites may share one")
    ap.add_argument("--green-h2-share-pct", type=float, default=None,
                    help="without --sites-file: minimum green (RFNBO) share of each site's annual "
                         "hydrogen demand, %%, default 42; 0 = no requirement")
    ap.add_argument("--demand-flex-pct", type=float, default=None,
                    help="without --sites-file: hourly demand flexibility of each site, %% of each "
                         "hour's demand (shifts net to zero over each day), default 10; 0 = rigid")
    ap.add_argument("--budget", type=float, default=None,
                    help=f"total CAPEX budget across every site, EUR, raw/unannualized "
                         f"(default {hp.CapexAssumptions().default_budget_eur:,.0f})")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--gap-tol", type=float, default=0.01, help="relative Benders gap, default 0.01 (1%%)")
    ap.add_argument("--discount-rate", type=float, default=None, help="for the capital recovery factor")
    ap.add_argument("--lifetime-years", type=float, default=None,
                    help="capital recovery factor project life, applied uniformly to every asset "
                         "(default: each asset's own CapexAssumptions.lifetime_years entry)")
    ap.add_argument("--output", type=str, default=str(OUT / "plan"), help="output file prefix")
    ap.add_argument("--export-schedules", action="store_true",
                    help="also re-solve at the final chosen capacities and dump each "
                         "site's representative-day schedule")
    ap.add_argument("--rep-days-per-month", type=int, default=7,
                    help="solve every joint subproblem on N days/month (1-29, weighted to "
                         "approximate the full year), default 7")
    ap.add_argument("--cvar-alpha", type=float, default=0.8,
                    help="CVaR confidence level (0-1) used when planning over two or more "
                         "scenarios, default 0.8; a single scenario is planned deterministically")
    ap.add_argument("--master-time-limit", type=float, default=180.0,
                    help="wall-time cap (seconds) per master MILP solve, default 180")
    ap.add_argument("--disabled-assets", type=str, default=None,
                    help="comma-separated asset keys to exclude from every site's candidate "
                         f"selection (max_mw=0), e.g. battery_mw,tank_mw. Choices: {hp.ASSETS}")
    ap.add_argument("--scenarios", type=str, default=None,
                    help="comma-separated capacity-uncertainty scenarios to optimize over, "
                         "probabilities renormalized to sum to 1.0; scenarios beyond the baseline come "
                         "from the PLANNER_SCENARIO_OVERRIDES file the web app writes (default: "
                         f"{','.join(DEFAULT_SCENARIOS)}, the baseline, planned deterministically). "
                         f"Choices: {SCENARIOS}")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes for running each iteration's scenario subproblems in "
                         "parallel (they're mutually independent); default min(4, cpu_count) -- "
                         "raise this if your machine has memory headroom for more. "
                         "Pass 1 to force sequential (e.g. for debugging).")
    ap.add_argument("--max-units-per-candidate", type=int, default=0,
                    help="max buildable units of each individual candidate product per "
                         "site/asset (the default catalog has one product per asset, built as many "
                         "times as needed); default 0 = unbounded. Pass e.g. 3 to cap each product at 3 units.")
    args = ap.parse_args()

    capex_cfg = hp.CapexAssumptions()
    override_path = os.environ.get(CATALOG_OVERRIDES_ENV)
    if override_path:
        load_catalog_override(capex_cfg, Path(override_path))
    if args.discount_rate is not None:
        capex_cfg.discount_rate = args.discount_rate
    if args.lifetime_years is not None:
        capex_cfg.lifetime_years = {a: args.lifetime_years for a in hp.ASSETS}
    budget = args.budget if args.budget is not None else capex_cfg.default_budget_eur

    names = ([s.strip() for s in args.scenarios.split(",") if s.strip()] if args.scenarios
             else list(DEFAULT_SCENARIOS))
    unknown = [s for s in names if s not in SCENARIO_PROBS]
    if unknown:
        raise ValueError(f"scenario(s) {unknown} have no positive probability -- choices: {SCENARIOS}")
    raw = {s: SCENARIO_PROBS[s] for s in names}
    total = sum(raw.values())
    scenario_probs = {s: p / total for s, p in raw.items()}
    cvar_alpha = risk_measure(scenario_probs, args.cvar_alpha)

    disabled_assets = None
    if args.disabled_assets:
        disabled_assets = [a.strip() for a in args.disabled_assets.split(",") if a.strip()]
        unknown = [a for a in disabled_assets if a not in hp.ASSETS]
        if unknown:
            raise ValueError(f"--disabled-assets has unknown key(s) {unknown} -- choices: {hp.ASSETS}")

    if args.all:
        countries = eligible_countries()
    elif args.countries:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    else:
        countries = eligible_countries()
        print(f"No --countries/--all given -- defaulting to all {len(countries)} eligible countries.")

    if args.sites_file:
        sites = hp.load_sites(Path(args.sites_file))
    else:
        sites = hp.default_sites(args.n_sites,
                                 green_share=None if args.green_h2_share_pct is None else args.green_h2_share_pct / 100,
                                 flex_fraction=None if args.demand_flex_pct is None else args.demand_flex_pct / 100)

    print(f"Candidate countries: {countries} | sites to build: {len(sites)} (each in one country; sites may share one)")
    crfs = capex_cfg.capital_recovery_factors()
    crf_str = ", ".join(f"{a}={crfs[a]:.4f}({capex_cfg.lifetime_years[a]:.0f}yr)" for a in hp.ASSETS)
    print(f"Budget: {budget:,.0f} EUR (raw/unannualized) | CRF @ {capex_cfg.discount_rate:.1%} discount: "
         f"{crf_str}")
    units_note = ("unbounded" if args.max_units_per_candidate <= 0
                 else f"max {args.max_units_per_candidate} units/candidate")
    print(f"Candidates ({units_note}, per site):")
    for a in hp.ASSETS:
        cand_str = ", ".join(f"{c.mw:g}MW/{c.capex_eur:,.0f}EUR" for c in capex_cfg.catalog[a])
        print(f"  {a}: {cand_str}")
    print(f"Subproblems: {args.rep_days_per_month} representative day(s)/month "
         f"({args.rep_days_per_month * 12} days solved, weighted to approximate the full year)")
    tech = hp.SITE_TECH
    green = hp.GREEN_H2
    print(f"Green H2: hourly-matched additional renewables | GOs buy {green.go_buy_price_eur_per_mwh:g} / sell "
          f"{green.go_sell_price_eur_per_mwh:g} EUR/MWh | certified green H2 premium "
          f"{green.green_h2_premium_eur_per_mwh:g} EUR/MWh")
    print(f"Sites (annual peaks MW x per-unit curves in {hp.demand.SITE_DEMAND_CSV.relative_to(ROOT)}):")
    for sp in sites:
        annual = hp.annual_demand_mwh(sp.peaks_mw)
        print(f"  {sp.name}: green H2 >= {sp.green_share:.0%}, flexibility +/-{sp.flex_fraction:.0%} | peaks "
              + ", ".join(f"{k}={v:g}" for k, v in sp.peaks_mw.items())
              + " | MWh/yr " + ", ".join(f"{k}={v:,.0f}" for k, v in annual.items()))
    print(f"No existing plant on site: unmet heat/cooling demand penalty "
          f"{tech.unmet_demand_penalty_eur_per_mwh:,.0f} EUR/MWh | grid import fee {tech.grid_import_fee_eur_per_mwh:g} EUR/MWh, "
          f"H2 import fee {tech.h2_import_fee_eur_per_mwh:g} EUR/MWh")
    print(f"Exchange caps: grid=unlimited, H2 pipeline=unlimited")
    if disabled_assets:
        print(f"Disabled assets (max_mw=0 at every site): {disabled_assets}")
    if cvar_alpha is None:
        print(f"Capacity-uncertainty scenario (deterministic): {scenario_probs}")
    else:
        print(f"Capacity-uncertainty scenarios (CVaR_{cvar_alpha:.2f} risk measure): {scenario_probs}")

    t0 = time.time()
    res = run_benders(
        countries, budget, args.max_iters, args.gap_tol, capex_cfg,
        rep_days_per_month=args.rep_days_per_month,
        cvar_alpha=cvar_alpha, master_time_limit=args.master_time_limit,
        disabled_assets=disabled_assets, scenario_probs=scenario_probs, workers=args.workers,
        max_units_per_candidate=(args.max_units_per_candidate
                                 if args.max_units_per_candidate > 0 else None),
        sites=sites)
    elapsed = time.time() - t0
    units, unit_country, host_zone = res["units"], res["unit_country"], res["host_zone"]
    chosen = {units[u][1].name: u for u in units if res["placed"][u] > 0.5}

    print(f"\nDone in {elapsed:.1f}s. Final capacities:")
    rows = []
    for sp in sites:
        u = chosen[sp.name]
        row = {"site": sp.name, "country": unit_country[u], "host_zone": host_zone[unit_country[u]]}
        row.update(res["capacities"][u])
        rows.append(row)
    cap_df = pd.DataFrame(rows)
    print(cap_df.to_string(index=False))
    print(f"\nChosen site(s): {', '.join(f'{sp.name} -> {unit_country[chosen[sp.name]]}' for sp in sites)}")
    print(f"Total raw CAPEX: {res['capex']:,.0f} EUR (budget {budget:,.0f} EUR, "
         f"{res['capex'] / budget:.1%} used)")
    risk_label = ("deterministic" if cvar_alpha is None else f"CVaR_{cvar_alpha:.2f}")
    print(f"Best objective (annualized CAPEX + {risk_label} 1yr operating cost across "
         f"{len(scenario_probs)} scenarios, EUR, lower=better): {res['objective']:,.0f}")

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_csv(f"{out_prefix}_capacities.csv", index=False)
    res["log"].to_csv(f"{out_prefix}_convergence.csv", index=False)
    pd.DataFrame([{"site": sp.name, "green_share": sp.green_share, "flex_fraction": sp.flex_fraction,
                   **{f"peak_{k}_mw": v for k, v in sp.peaks_mw.items()},
                   **{f"annual_{k}_mwh": v for k, v in hp.annual_demand_mwh(sp.peaks_mw).items()}}
                  for sp in sites]).to_csv(f"{out_prefix}_sites.csv", index=False)
    units_rows = []
    for sp in sites:
        u = chosen[sp.name]
        for a in hp.ASSETS:
            for k, (cand, n) in enumerate(zip(capex_cfg.catalog[a], res["unit_counts"][u][a])):
                if n > 0:
                    units_rows.append({"site": sp.name, "country": unit_country[u], "asset": a, "candidate": k,
                                       "candidate_mw": cand.mw, "units": int(n), "total_mw": cand.mw * int(n)})
    pd.DataFrame(units_rows).to_csv(f"{out_prefix}_units.csv", index=False)
    green_df = pd.DataFrame(res["green"])
    green_df.to_csv(f"{out_prefix}_green_h2.csv", index=False)
    targets = {sp.name: sp.green_share for sp in sites}
    for name, g in green_df.groupby("site", sort=False):
        share = float((g["green_share"] * g["probability"]).sum())
        print(f"Green H2 at {name} ({g['country'].iloc[0]}): expected share {share:.1%} of H2 demand "
              f"(target {targets[name]:.0%}), GOs bought {float((g['go_bought_mwh'] * g['probability']).sum()):,.0f} "
              f"/ sold {float((g['go_sold_mwh'] * g['probability']).sum()):,.0f} MWh/yr")
    unmet_df = pd.DataFrame(res["unmet"])
    unmet_df.to_csv(f"{out_prefix}_unmet_demand.csv", index=False)
    for name, g in unmet_df.groupby("site", sort=False):
        total = float((g["unmet_total_mwh"] * g["probability"]).sum())
        print(f"Unmet heat/cooling demand at {name} ({g['country'].iloc[0]}): expected {total:,.0f} MWh/yr")
    print(f"\nwrote {out_prefix}_capacities.csv, {out_prefix}_sites.csv, {out_prefix}_convergence.csv, "
          f"{out_prefix}_units.csv, {out_prefix}_green_h2.csv, {out_prefix}_unmet_demand.csv")

    if args.export_schedules:
        chosen_units = {u: units[u] for u in chosen.values()}
        caps = {u: res["capacities"][u] for u in chosen_units}
        for s, (edf_s, hdf_s) in res["price_frames"].items():
            final = ohp.solve_joint(chosen_units, caps, args.rep_days_per_month,
                                    return_duals=False, edf=edf_s, hdf=hdf_s, quiet=True, scenario=s)
            for u in chosen_units:
                slug = units[u][1].name.replace(" ", "_")
                final["schedules"][u].to_csv(f"{out_prefix}_schedule_{slug}_{unit_country[u]}_{s}.csv", index=False)
        print(f"wrote {out_prefix}_schedule_<site>_<country>_<scenario>.csv for {len(chosen_units)} site(s) "
             f"x {len(res['price_frames'])} scenarios")


if __name__ == "__main__":
    main()
