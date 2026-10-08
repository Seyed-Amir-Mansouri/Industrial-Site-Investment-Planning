"""Industrial Site Investor planning: Benders decomposition CLI choosing site location(s), technologies and
capacities under a CAPEX budget and wind/PV uncertainty scenarios."""
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


def load_scenario_probs(path: Path | None = None) -> dict[str, float]:
    """Read {scenario: probability} from the saved scenario JSON."""
    if path is None:
        data = ohp.load_uncertainty_scenarios()
        source = ROOT / "inputs" / "uncertainty_scenarios.json"
    else:
        data = json.loads(path.read_text())["scenarios"]
        source = path
    probs = {name: float(sc["probability"]) for name, sc in data.items()}
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"{source}'s scenario probabilities sum to {total:.6f}, not 1.0")
    return probs


SCENARIO_PROBS = load_scenario_probs()
SCENARIOS = list(SCENARIO_PROBS)


def _cvar_alpha_arg(value: str) -> float | None:
    if value.lower() in ("off", "none", "expected"):
        return None
    return float(value)


def eligible_countries() -> list[str]:
    return sorted(hp.candidate_site_zones())


def _snap(value: float, tol: float = 1e-3) -> float:
    """Zero out a core-point coordinate that has decayed below ``tol``; tiny values make the subproblem numerically fragile, and any core point still yields a valid cut."""
    return 0.0 if abs(value) < tol else value


def run_benders(countries: list[str], budget: float | None, max_iters: int, gap_tol: float,
                capex_cfg: hp.CapexAssumptions, rep_days_per_month: int,
                quiet_solver: bool = True, on_iteration=None, cvar_alpha: float | None = None,
                master_time_limit: float = 180.0,
                disabled_assets: list[str] | None = None,
                scenario_probs: dict[str, float] | None = None,
                workers: int | None = None,
                max_units_per_candidate: int | None = None,
                n_sites: int = 1,
                flex: dict[str, float] | None = None,
                green_share: float | None = None):
    """Run the Benders loop: master MILP proposes site locations and capacities, joint subproblems (one per scenario) price them and return cuts, repeat to convergence.

    Each iteration adds two cuts per scenario: one at the trial point itself, so the master can't
    propose the same plan again without paying its true recourse cost, and one at a Pareto core
    point that starts inside the feasible region (every country a fractional site, every asset a
    mid-grid size within its site cap) and moves halfway toward each trial point. If HiGHS presolve
    fails on the master, it is re-solved once with presolve off."""
    scenario_probs = scenario_probs if scenario_probs is not None else SCENARIO_PROBS
    site_max_mw, cand_mw, cand_capex, host_zone = hp.build_candidates(countries, capex_cfg)
    crf = capex_cfg.capital_recovery_factors()

    if budget is not None:
        cheapest_total = n_sites * sum(min(cand_capex[countries[0]][a]) for a in hp.ASSETS
                                       if not (disabled_assets and a in disabled_assets))
        if cheapest_total > budget:
            print(f"NOTE: budget {budget:,.0f} EUR is below the cheapest all-assets-built "
                 f"combination ({cheapest_total:,.0f} EUR) for {n_sites} site(s) -- expect some "
                 f"assets to come back skipped (0 MW) in the result.")

    print(f"Building enriched price frames for {len(scenario_probs)} capacity scenarios "
         f"(one-time cost, reused by every subproblem solve): {list(scenario_probs)}")
    price_frames = {s: (ohp.enriched_elec_df(scenario=s), ohp.enriched_h2_df(scenario=s))
                    for s in scenario_probs}

    for c in countries:
        for a in disabled_assets or []:
            site_max_mw[c][a] = 0.0

    m = hp.build_master(countries, cand_mw, cand_capex, budget, crf, capex_cfg.theta_lower_bound_eur,
                        scenario_probs=scenario_probs, cvar_alpha=cvar_alpha, site_max_mw=site_max_mw,
                        max_units_per_candidate=max_units_per_candidate, n_sites=n_sites)

    core_cap = {c: {a: min(float(sum([0.0] + list(cand_mw[c][a])) / (len(cand_mw[c][a]) + 1)),
                           0.5 * site_max_mw[c][a])
                    for a in hp.ASSETS} for c in countries}
    core_site = {c: n_sites / len(countries) for c in countries}

    best_ub, best_capacities, best_capex, best_capex_by_asset = float("inf"), None, None, None
    best_units, best_sites = None, None
    log = []
    per_country_log = {c: [] for c in countries}
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
            cap_star = hp.extract_capacities(m, countries, cand_mw)
            site_star = hp.extract_sites(m, countries)
            lb = float(m.solver_model.getInfo().mip_dual_bound)
            master_mip_gap = float(m.solver_model.getInfo().mip_gap)
            master_cut_off = (cond == "time_limit")

            t_sub = time.time()
            sub_build_s = sub_solve_s = 0.0
            n_cuts = 0
            zones = [host_zone[c] for c in countries]
            caps_by_zone = {host_zone[c]: cap_star[c] for c in countries}
            sites_by_zone = {host_zone[c]: site_star[c] for c in countries}
            caps_core_by_zone = {host_zone[c]: core_cap[c] for c in countries}
            sites_core_by_zone = {host_zone[c]: core_site[c] for c in countries}

            if executor is not None:
                trial_futures = {s: executor.submit(ohp.solve_joint, zones, caps_by_zone, rep_days_per_month,
                                                    return_duals=True, edf=price_frames[s][0],
                                                    hdf=price_frames[s][1], quiet=quiet_solver, scenario=s,
                                                    sites=sites_by_zone, flex=flex, green_share=green_share)
                                 for s in scenario_probs}
                core_futures = {s: executor.submit(ohp.solve_joint, zones, caps_core_by_zone, rep_days_per_month,
                                                   return_duals=True, edf=price_frames[s][0],
                                                   hdf=price_frames[s][1], quiet=quiet_solver, scenario=s,
                                                   sites=sites_core_by_zone, flex=flex, green_share=green_share)
                                for s in scenario_probs}
                results = {s: f.result() for s, f in trial_futures.items()}
                results_core = {s: f.result() for s, f in core_futures.items()}
            else:
                results = {}
                for s in scenario_probs:
                    edf_s, hdf_s = price_frames[s]
                    results[s] = ohp.solve_joint(zones, caps_by_zone, rep_days_per_month,
                                                 return_duals=True, edf=edf_s, hdf=hdf_s,
                                                 quiet=quiet_solver, scenario=s, sites=sites_by_zone,
                                                 flex=flex, green_share=green_share)
                results_core = {}
                for s in scenario_probs:
                    edf_s, hdf_s = price_frames[s]
                    results_core[s] = ohp.solve_joint(zones, caps_core_by_zone, rep_days_per_month,
                                                      return_duals=True, edf=edf_s, hdf=hdf_s,
                                                      quiet=quiet_solver, scenario=s, sites=sites_core_by_zone,
                                                      flex=flex, green_share=green_share)

            for s in scenario_probs:
                sub_build_s += results[s]["build_seconds"]
                sub_solve_s += results[s]["solve_seconds"]

            total_Q = 0.0
            Q_total_by_scenario: dict[str, float] = {}
            Q_by_country = {c: 0.0 for c in countries}
            Q_by_country_by_scenario: dict[str, dict[str, float]] = {}
            for s, prob in scenario_probs.items():
                result = results[s]
                total_Q += prob * float(result["objective"])
                Q_total_by_scenario[s] = float(result["objective"])
                Q_s = {c: result["objective_by_zone"][host_zone[c]] for c in countries}
                Q_by_country_by_scenario[s] = Q_s
                for c in countries:
                    Q_by_country[c] += prob * Q_s[c]

                hp.add_optimality_cut(m, countries, it, Q_s,
                                      {c: result["cut_coeffs"][host_zone[c]] for c in countries},
                                      cap_star, cand_mw, scenario=s,
                                      lam={c: result["site_coeffs"][host_zone[c]] for c in countries},
                                      site_star=site_star, tag="trial")
                n_cuts += 1

                result_core = results_core[s]
                sub_build_s += result_core["build_seconds"]
                sub_solve_s += result_core["solve_seconds"]
                Q_cut = {c: result_core["objective_by_zone"][host_zone[c]] for c in countries}
                mu_cut = {c: result_core["cut_coeffs"][host_zone[c]] for c in countries}
                lam_cut = {c: result_core["site_coeffs"][host_zone[c]] for c in countries}
                hp.add_optimality_cut(m, countries, it, Q_cut, mu_cut, core_cap, cand_mw, scenario=s,
                                      lam=lam_cut, site_star=core_site)
                n_cuts += 1
            sub_s = time.time() - t_sub

            for c in countries:
                for a in hp.ASSETS:
                    core_cap[c][a] = _snap(0.5 * core_cap[c][a] + 0.5 * cap_star[c][a])
                core_site[c] = _snap(0.5 * core_site[c] + 0.5 * site_star[c])

            capex_star = hp.extract_capex(m, countries, cand_capex)
            for c in countries:
                per_country_log[c].append({"iter": it, "objective": Q_by_country[c],
                                           "objective_by_scenario": {s: Q_by_country_by_scenario[s][c]
                                                                     for s in scenario_probs},
                                           "capex": sum(capex_star[c].values()), "capacities": dict(cap_star[c]),
                                           "site": site_star[c]})
            raw_capex = sum(capex_star[c][a] for c in countries for a in hp.ASSETS)
            annualized_capex = sum(capex_star[c][a] * crf[a] for c in countries for a in hp.ASSETS)
            if cvar_alpha is not None:
                ub = annualized_capex + hp.cvar_value(Q_total_by_scenario, scenario_probs, cvar_alpha)
            else:
                ub = annualized_capex + total_Q
            if ub < best_ub:
                best_ub, best_capacities, best_capex, best_capex_by_asset = ub, cap_star, raw_capex, capex_star
                best_units = hp.extract_units(m, countries)
                best_sites = site_star
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
                             "best_sites": best_sites,
                             "host_zone": host_zone, "log": list(log), "per_country_log": per_country_log})
            if gap <= gap_tol:
                print(f"converged (gap {gap:.4f} <= tol {gap_tol}) after {it} iteration(s)")
                break
        else:
            print(f"WARNING: reached --max-iters={max_iters} without closing the gap "
                 f"(final gap {gap:.4f}) -- results below are the best FOUND, not proven optimal")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return (best_capacities, best_capex, best_ub, pd.DataFrame(log), host_zone, price_frames,
           per_country_log, best_capex_by_asset, best_units, best_sites)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--countries", type=str, default=None,
                       help="comma-separated 2-letter country codes, e.g. DE,FR,PL")
    group.add_argument("--all", action="store_true", help="plan every eligible country")
    ap.add_argument("--green-h2-share-pct", type=float, default=None,
                    help="minimum green (RFNBO) share of each site's annual hydrogen demand, %%: own "
                         "electrolyser on hourly-matched additional renewables (own wind/PV or bought "
                         "GOs) or bought certified green H2. Default: GreenH2Params.green_share (42%%); "
                         "0 = no requirement")
    ap.add_argument("--demand-flex-pct", type=float, default=None,
                    help="hourly demand flexibility, %% of each hour's demand every service may shift up "
                         "or down (shifts net to zero over each day); applies to all six demands. "
                         "Default: SiteDemandAssumptions.flex_fraction (10%% each); 0 = rigid demand")
    ap.add_argument("--n-sites", type=int, default=1,
                    help="how many industrial sites to build, each in a different candidate country "
                         "(default 1); the optimizer picks where")
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
                         "chosen site's representative-day schedule")
    ap.add_argument("--rep-days-per-month", type=int, default=7,
                    help="solve every joint subproblem on N days/month (1-29, weighted to "
                         "approximate the full year), default 7")
    ap.add_argument("--cvar-alpha", type=_cvar_alpha_arg, default=0.8,
                    help="risk measure: CVaR at this confidence level (0-1) across "
                         "capacity-uncertainty scenarios, default 0.8; 'off' = expected value")
    ap.add_argument("--master-time-limit", type=float, default=180.0,
                    help="wall-time cap (seconds) per master MILP solve, default 180")
    ap.add_argument("--disabled-assets", type=str, default=None,
                    help="comma-separated asset keys to exclude from every site's candidate "
                         f"selection (max_mw=0), e.g. battery_mw,tank_mw. Choices: {hp.ASSETS}")
    ap.add_argument("--scenarios", type=str, default=None,
                    help="comma-separated subset of capacity-uncertainty scenarios to optimize "
                         f"over, probabilities renormalized to sum to 1.0, e.g. p100 for a single "
                         f"deterministic baseline run (default: all {len(SCENARIO_PROBS)}). "
                         f"Choices: {SCENARIOS}")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes for running each iteration's scenario subproblems in "
                         "parallel (they're mutually independent); default min(4, cpu_count) -- "
                         "raise this if your machine has memory headroom for more. "
                         "Pass 1 to force sequential (e.g. for debugging).")
    ap.add_argument("--max-units-per-candidate", type=int, default=0,
                    help="max buildable units of each individual candidate product per "
                         "country/asset (there are 5 real candidate products per asset in the "
                         "catalog); default 0 = unbounded. Pass e.g. 3 to cap each product at 3 units.")
    args = ap.parse_args()

    capex_cfg = hp.CapexAssumptions()
    if args.discount_rate is not None:
        capex_cfg.discount_rate = args.discount_rate
    if args.lifetime_years is not None:
        capex_cfg.lifetime_years = {a: args.lifetime_years for a in hp.ASSETS}
    budget = args.budget if args.budget is not None else capex_cfg.default_budget_eur

    scenario_probs = SCENARIO_PROBS
    if args.scenarios:
        names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        unknown = [s for s in names if s not in SCENARIO_PROBS]
        if unknown:
            raise ValueError(f"--scenarios has unknown name(s) {unknown} -- choices: {SCENARIOS}")
        raw = {s: SCENARIO_PROBS[s] for s in names}
        total = sum(raw.values())
        scenario_probs = {s: p / total for s, p in raw.items()}

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

    print(f"Candidate site countries: {countries} | sites to build: {args.n_sites}")
    crfs = capex_cfg.capital_recovery_factors()
    crf_str = ", ".join(f"{a}={crfs[a]:.4f}({capex_cfg.lifetime_years[a]:.0f}yr)" for a in hp.ASSETS)
    print(f"Budget: {budget:,.0f} EUR (raw/unannualized) | CRF @ {capex_cfg.discount_rate:.1%} discount: "
         f"{crf_str}")
    n_candidates = len(capex_cfg.catalog[hp.ASSETS[0]])
    units_note = ("unbounded" if args.max_units_per_candidate <= 0
                 else f"max {args.max_units_per_candidate} units/candidate")
    print(f"Candidates: {n_candidates} products/asset ({units_note}, per site):")
    for a in hp.ASSETS:
        cand_str = ", ".join(f"{c.mw:g}MW/{c.capex_eur:,.0f}EUR" for c in capex_cfg.catalog[a])
        print(f"  {a}: {cand_str}")
    print(f"Subproblems: {args.rep_days_per_month} representative day(s)/month "
         f"({args.rep_days_per_month * 12} days solved, weighted to approximate the full year)")
    tech = hp.SITE_TECH
    flex = (hp.SITE_DEMAND.flex_fraction if args.demand_flex_pct is None
            else {svc: args.demand_flex_pct / 100 for svc in hp.SERVICES})
    green = hp.GREEN_H2
    green_share = green.green_share if args.green_h2_share_pct is None else args.green_h2_share_pct / 100
    print(f"Green H2: >= {green_share:.0%} of H2 demand, hourly-matched additional renewables | GOs buy "
          f"{green.go_buy_price_eur_per_mwh:g} / sell {green.go_sell_price_eur_per_mwh:g} EUR/MWh | "
          f"certified green H2 premium {green.green_h2_premium_eur_per_mwh:g} EUR/MWh")
    print("Demand flexibility (+/- share of each hour's demand, net zero per day): "
          + ", ".join(f"{k}={v:.0%}" for k, v in flex.items()))
    print(f"Site demand (MWh/yr) from {hp.demand.SITE_DEMAND_CSV.relative_to(ROOT)}:")
    for c in countries:
        print(f"  {c}: " + ", ".join(f"{k}={v:,.0f}" for k, v in hp.annual_demand_mwh(c).items()))
    print(f"Backup: gas boiler heat {tech.gas_heat_cost_eur_per_mwh_th:.1f} EUR/MWh_th, legacy chiller "
          f"COP {tech.legacy_chiller_cop:g} | grid import fee {tech.grid_import_fee_eur_per_mwh:g} EUR/MWh, "
          f"H2 import fee {tech.h2_import_fee_eur_per_mwh:g} EUR/MWh")
    print(f"Exchange caps: grid=unlimited, H2 pipeline=unlimited")
    if disabled_assets:
        print(f"Disabled assets (max_mw=0 at every site): {disabled_assets}")
    if args.cvar_alpha is None:
        print(f"Capacity-uncertainty scenarios (expected-value risk measure): {scenario_probs}")
    else:
        print(f"Capacity-uncertainty scenarios (CVaR_{args.cvar_alpha:.2f} risk measure): "
             f"{scenario_probs}")

    t0 = time.time()
    (best_capacities, best_capex, best_ub, log_df, host_zone, price_frames,
    per_country_log, best_capex_by_asset, best_units, best_sites) = run_benders(
        countries, budget, args.max_iters, args.gap_tol, capex_cfg,
        rep_days_per_month=args.rep_days_per_month,
        cvar_alpha=args.cvar_alpha, master_time_limit=args.master_time_limit,
        disabled_assets=disabled_assets, scenario_probs=scenario_probs, workers=args.workers,
        max_units_per_candidate=(args.max_units_per_candidate
                                 if args.max_units_per_candidate > 0 else None),
        n_sites=args.n_sites, flex=flex, green_share=green_share)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s. Final capacities:")
    rows = []
    for c in countries:
        row = {"country": c, "host_zone": host_zone[c], "site": int(best_sites[c])}
        row.update(best_capacities[c])
        rows.append(row)
    cap_df = pd.DataFrame(rows)
    print(cap_df.to_string(index=False))
    print(f"\nChosen site(s): {', '.join(c for c in countries if best_sites[c])}")
    print(f"Total raw CAPEX: {best_capex:,.0f} EUR (budget {budget:,.0f} EUR, "
         f"{best_capex / budget:.1%} used)")
    risk_label = (f"EXPECTED" if args.cvar_alpha is None else f"CVaR_{args.cvar_alpha:.2f}")
    print(f"Best objective (annualized CAPEX + {risk_label} 1yr operating cost across "
         f"{len(scenario_probs)} scenarios, EUR, lower=better): {best_ub:,.0f}")

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_csv(f"{out_prefix}_capacities.csv", index=False)
    log_df.to_csv(f"{out_prefix}_convergence.csv", index=False)
    units_rows = []
    for c in countries:
        for a in hp.ASSETS:
            for k, (cand, n) in enumerate(zip(capex_cfg.catalog[a], best_units[c][a])):
                if n > 0:
                    units_rows.append({"country": c, "asset": a, "candidate": k, "candidate_mw": cand.mw,
                                       "units": int(n), "total_mw": cand.mw * int(n)})
    pd.DataFrame(units_rows).to_csv(f"{out_prefix}_units.csv", index=False)
    print(f"\nwrote {out_prefix}_capacities.csv, {out_prefix}_convergence.csv, {out_prefix}_units.csv")

    if args.export_schedules:
        site_countries = [c for c in countries if best_sites[c]]
        zones = [host_zone[c] for c in site_countries]
        caps_by_zone = {host_zone[c]: best_capacities[c] for c in site_countries}
        for s, (edf_s, hdf_s) in price_frames.items():
            final = ohp.solve_joint(zones, caps_by_zone, args.rep_days_per_month,
                                    return_duals=False, edf=edf_s, hdf=hdf_s, quiet=True, scenario=s,
                                    flex=flex, green_share=green_share)
            for c in site_countries:
                final["schedules"][host_zone[c]].to_csv(f"{out_prefix}_schedule_{c}_{s}.csv", index=False)
        print(f"wrote {out_prefix}_schedule_<country>_<scenario>.csv for {len(site_countries)} site(s) "
             f"x {len(price_frames)} scenarios")


if __name__ == "__main__":
    main()
