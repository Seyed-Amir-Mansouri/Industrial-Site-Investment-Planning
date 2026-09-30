"""H2 Producer capacity planning: Benders decomposition CLI over a system-wide CAPEX budget and wind/PV uncertainty scenarios."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import json

import pandas as pd

import optimize_h2_producer as ohp
import h2_planning as hp
from h2_planning.candidates import default_sizing_and_zones

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def load_scenario_probs(path: Path | None = None) -> dict[str, float]:
    """Read {scenario: probability} from the saved scenario JSON."""
    if path is None:
        path = ROOT / "inputs" / "uncertainty_scenarios_11.json"
    data = json.loads(path.read_text())["scenarios"]
    probs = {name: float(sc["probability"]) for name, sc in data.items()}
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"{path}'s scenario probabilities sum to {total:.6f}, not 1.0")
    return probs


SCENARIO_PROBS = load_scenario_probs()
SCENARIOS = list(SCENARIO_PROBS)


def eligible_countries() -> list[str]:
    sizing, _ = default_sizing_and_zones()
    return sorted(sizing)


def run_benders(countries: list[str], budget: float | None, max_iters: int, gap_tol: float,
                capex_cfg: hp.CapexAssumptions, rep_days_per_month: int,
                quiet_solver: bool = True, on_iteration=None, cvar_alpha: float | None = None,
                master_time_limit: float = 180.0, use_pareto_cuts: bool = False):
    """Run the Benders loop: master MILP proposes capacities, joint subproblems (one per scenario) price them and return cuts, repeat to convergence."""
    default_mw, cand_mw, cand_capex, host_zone = hp.build_candidates(countries, capex_cfg)
    crf = capex_cfg.capital_recovery_factors()

    if budget is not None:
        cheapest_total = sum(min(cand_capex[c][a]) for c in countries for a in hp.ASSETS)
        if cheapest_total > budget:
            print(f"NOTE: budget {budget:,.0f} EUR is below the cheapest all-assets-built "
                 f"combination ({cheapest_total:,.0f} EUR) for {countries} -- expect some "
                 f"assets to come back skipped (0 MW) in the result.")

    print(f"Building enriched price frames for {len(SCENARIO_PROBS)} capacity scenarios "
         f"(one-time cost, reused by every subproblem solve): {list(SCENARIO_PROBS)}")
    price_frames = {s: (ohp.enriched_elec_df(scenario=s), ohp.enriched_h2_df(scenario=s))
                    for s in SCENARIO_PROBS}

    m = hp.build_master(countries, cand_mw, cand_capex, budget, crf, capex_cfg.theta_lower_bound_eur,
                        scenario_probs=SCENARIO_PROBS, cvar_alpha=cvar_alpha)

    core_cap = {c: {a: float(sum([0.0] + list(cand_mw[c][a])) / (len(cand_mw[c][a]) + 1))
                    for a in hp.ASSETS} for c in countries}

    best_ub, best_capacities, best_capex, best_capex_by_asset = float("inf"), None, None, None
    log = []
    per_country_log = {c: [] for c in countries}
    gap = float("inf")
    for it in range(1, max_iters + 1):
        status, cond = m.solve(solver_name="highs", output_flag=False, time_limit=master_time_limit,
                               presolve="on", parallel="on")
        if status != "ok":
            raise RuntimeError(f"master solve failed at iteration {it}: {status}/{cond}")
        cap_star = hp.extract_capacities(m, countries, cand_mw)
        lb = float(m.solver_model.getInfo().mip_dual_bound)
        master_mip_gap = float(m.solver_model.getInfo().mip_gap)
        master_cut_off = (cond == "time_limit")

        t_sub = time.time()
        zones = [host_zone[c] for c in countries]
        caps_by_zone = {host_zone[c]: cap_star[c] for c in countries}

        caps_core_by_zone = {host_zone[c]: core_cap[c] for c in countries}
        results = {}
        for s in SCENARIO_PROBS:
            edf_s, hdf_s = price_frames[s]
            results[s] = ohp.solve_joint(zones, caps_by_zone, rep_days_per_month,
                                         return_duals=not use_pareto_cuts, edf=edf_s, hdf=hdf_s,
                                         quiet=quiet_solver)

        total_Q = 0.0
        Q_total_by_scenario: dict[str, float] = {}
        Q_by_country = {c: 0.0 for c in countries}
        Q_by_country_by_scenario: dict[str, dict[str, float]] = {}
        for s, prob in SCENARIO_PROBS.items():
            result = results[s]
            total_Q += prob * float(result["objective"])
            Q_total_by_scenario[s] = float(result["objective"])
            Q_s = {c: result["objective_by_zone"][host_zone[c]] for c in countries}
            Q_by_country_by_scenario[s] = Q_s
            for c in countries:
                Q_by_country[c] += prob * Q_s[c]

            if use_pareto_cuts:
                edf_s, hdf_s = price_frames[s]
                result_core = ohp.solve_joint(zones, caps_core_by_zone, rep_days_per_month,
                                              return_duals=True, edf=edf_s, hdf=hdf_s,
                                              quiet=quiet_solver)
                Q_cut = {c: result_core["objective_by_zone"][host_zone[c]] for c in countries}
                mu_cut = {c: result_core["cut_coeffs"][host_zone[c]] for c in countries}
                cap_for_cut = core_cap
            else:
                mu_s = {c: result["cut_coeffs"][host_zone[c]] for c in countries}
                Q_cut, mu_cut, cap_for_cut = Q_s, mu_s, cap_star
            hp.add_optimality_cut(m, countries, it, Q_cut, mu_cut, cap_for_cut, cand_mw, scenario=s)
        sub_s = time.time() - t_sub

        if use_pareto_cuts:
            for c in countries:
                for a in hp.ASSETS:
                    core_cap[c][a] = 0.5 * core_cap[c][a] + 0.5 * cap_star[c][a]

        capex_star = hp.extract_capex(m, countries, cand_capex)
        for c in countries:
            per_country_log[c].append({"iter": it, "objective": Q_by_country[c],
                                       "objective_by_scenario": {s: Q_by_country_by_scenario[s][c]
                                                                 for s in SCENARIO_PROBS},
                                       "capex": sum(capex_star[c].values()), "capacities": dict(cap_star[c])})
        raw_capex = sum(capex_star[c][a] for c in countries for a in hp.ASSETS)
        annualized_capex = sum(capex_star[c][a] * crf[a] for c in countries for a in hp.ASSETS)
        if cvar_alpha is not None:
            ub = annualized_capex + hp.cvar_value(Q_total_by_scenario, SCENARIO_PROBS, cvar_alpha)
        else:
            ub = annualized_capex + total_Q
        if ub < best_ub:
            best_ub, best_capacities, best_capex, best_capex_by_asset = ub, cap_star, raw_capex, capex_star
        gap = (best_ub - lb) / max(abs(best_ub), 1e-6)
        log.append({"iter": it, "lb": lb, "ub": ub, "best_ub": best_ub, "gap": gap,
                    "subproblems_seconds": round(sub_s, 1), "master_mip_gap": master_mip_gap,
                    "master_cut_off": master_cut_off})
        cutoff_note = " [MASTER CUT OFF AT time_limit]" if master_cut_off else ""
        print(f"iter {it:>3}: LB={lb:>16,.0f}  UB={ub:>16,.0f}  best={best_ub:>16,.0f}  "
             f"gap={gap:.4f}  (subproblems {sub_s:.1f}s, master_mip_gap={master_mip_gap:.4f}){cutoff_note}")
        if on_iteration is not None:
            on_iteration({"iter": it, "lb": lb, "ub": ub, "best_ub": best_ub, "gap": gap,
                         "best_capacities": best_capacities, "best_capex_by_asset": best_capex_by_asset,
                         "host_zone": host_zone, "log": list(log), "per_country_log": per_country_log})
        if gap <= gap_tol:
            print(f"converged (gap {gap:.4f} <= tol {gap_tol}) after {it} iteration(s)")
            break
    else:
        print(f"WARNING: reached --max-iters={max_iters} without closing the gap "
             f"(final gap {gap:.4f}) -- results below are the best FOUND, not proven optimal")

    return (best_capacities, best_capex, best_ub, pd.DataFrame(log), host_zone, price_frames,
           per_country_log, best_capex_by_asset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--countries", type=str, default=None,
                       help="comma-separated 2-letter country codes, e.g. DE,FR,PL")
    group.add_argument("--all", action="store_true", help="plan every eligible country")
    ap.add_argument("--budget", type=float, default=None,
                    help=f"total system-wide CAPEX budget, EUR, raw/unannualized "
                         f"(default {hp.CapexAssumptions().default_budget_eur:,.0f})")
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--gap-tol", type=float, default=0.01, help="relative Benders gap, default 0.01 (1%%)")
    ap.add_argument("--discount-rate", type=float, default=None, help="for the capital recovery factor")
    ap.add_argument("--lifetime-years", type=float, default=None,
                    help="capital recovery factor project life, applied uniformly to every asset "
                         "(default: each asset's own CapexAssumptions.lifetime_years entry)")
    ap.add_argument("--output", type=str, default=str(OUT / "plan"), help="output file prefix")
    ap.add_argument("--export-schedules", action="store_true",
                    help="also re-solve at the final chosen capacities and dump each "
                         "included country's representative-day schedule")
    ap.add_argument("--rep-days-per-month", type=int, default=7,
                    help="solve every joint subproblem on N days/month (1-29, weighted to "
                         "approximate the full year), default 7")
    ap.add_argument("--cvar-alpha", type=float, default=None,
                    help="switch the risk measure from the default expected value across "
                         "capacity-uncertainty scenarios to CVaR at this confidence level "
                         "(0-1, e.g. 0.95)")
    ap.add_argument("--master-time-limit", type=float, default=180.0,
                    help="wall-time cap (seconds) per master MILP solve, default 180")
    ap.add_argument("--pareto-cuts", action="store_true",
                    help="build optimality cuts from a moving core point (Papadakos-style "
                         "Pareto-optimal cuts) instead of the trial point")
    args = ap.parse_args()

    capex_cfg = hp.CapexAssumptions()
    if args.discount_rate is not None:
        capex_cfg.discount_rate = args.discount_rate
    if args.lifetime_years is not None:
        capex_cfg.lifetime_years = {a: args.lifetime_years for a in hp.ASSETS}
    budget = args.budget if args.budget is not None else capex_cfg.default_budget_eur

    if args.all:
        countries = eligible_countries()
    elif args.countries:
        countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    else:
        countries = eligible_countries()
        print(f"No --countries/--all given -- defaulting to all {len(countries)} eligible countries.")

    print(f"Planning countries: {countries}")
    crfs = capex_cfg.capital_recovery_factors()
    crf_str = ", ".join(f"{a}={crfs[a]:.4f}({capex_cfg.lifetime_years[a]:.0f}yr)" for a in hp.ASSETS)
    print(f"Budget: {budget:,.0f} EUR (raw/unannualized) | CRF @ {capex_cfg.discount_rate:.1%} discount: "
         f"{crf_str}")
    print(f"Subproblems: {args.rep_days_per_month} representative day(s)/month "
         f"({args.rep_days_per_month * 12} days solved, weighted to approximate the full year)")
    print("Joint mode: ALL countries solved together per iteration, merchant electricity "
         "+ hydrogen trading -- no downstream demand modeled")
    print(f"Exchange caps: grid=unlimited, H2 pipeline=unlimited")
    if args.cvar_alpha is None:
        print(f"Capacity-uncertainty scenarios (expected-value risk measure, equal-weighted): "
             f"{SCENARIO_PROBS}")
    else:
        print(f"Capacity-uncertainty scenarios (CVaR_{args.cvar_alpha:.2f} risk measure, "
             f"equal-weighted scenario probabilities): {SCENARIO_PROBS}")

    t0 = time.time()
    (best_capacities, best_capex, best_ub, log_df, host_zone, price_frames,
    per_country_log, best_capex_by_asset) = run_benders(
        countries, budget, args.max_iters, args.gap_tol, capex_cfg,
        rep_days_per_month=args.rep_days_per_month,
        cvar_alpha=args.cvar_alpha, master_time_limit=args.master_time_limit,
        use_pareto_cuts=args.pareto_cuts)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s. Final capacities:")
    rows = []
    for c in countries:
        row = {"country": c, "host_zone": host_zone[c]}
        row.update(best_capacities[c])
        rows.append(row)
    cap_df = pd.DataFrame(rows)
    print(cap_df.to_string(index=False))
    print(f"\nTotal raw CAPEX: {best_capex:,.0f} EUR (budget {budget:,.0f} EUR, "
         f"{best_capex / budget:.1%} used)")
    risk_label = (f"EXPECTED" if args.cvar_alpha is None else f"CVaR_{args.cvar_alpha:.2f}")
    print(f"Best objective (annualized CAPEX + {risk_label} 1yr operating cost across "
         f"{len(SCENARIO_PROBS)} scenarios, EUR, lower=better): {best_ub:,.0f}")

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_csv(f"{out_prefix}_capacities.csv", index=False)
    log_df.to_csv(f"{out_prefix}_convergence.csv", index=False)
    print(f"\nwrote {out_prefix}_capacities.csv, {out_prefix}_convergence.csv")

    if args.export_schedules:
        zones = [host_zone[c] for c in countries]
        caps_by_zone = {host_zone[c]: best_capacities[c] for c in countries}
        for s, (edf_s, hdf_s) in price_frames.items():
            final = ohp.solve_joint(zones, caps_by_zone, args.rep_days_per_month,
                                    return_duals=False, edf=edf_s, hdf=hdf_s, quiet=True)
            for c in countries:
                final["schedules"][host_zone[c]].to_csv(f"{out_prefix}_schedule_{c}_{s}.csv", index=False)
        print(f"wrote {out_prefix}_schedule_<country>_<scenario>.csv for {len(countries)} countries "
             f"x {len(price_frames)} scenarios")


if __name__ == "__main__":
    main()
