"""H2 Producer capacity PLANNING: Benders decomposition CLI.

Chooses each requested country's Hydrogen Producer capacity (electrolyser, wind, PV,
battery, H2 tank -- each independently discrete/binary, see ``h2_planning/config.py``)
under one system-wide CAPEX budget, by iterating between:

* a MILP MASTER (``h2_planning/master.py``) that picks one candidate MW value per
  asset per country, minimizing annualized CAPEX + a per-country recourse variable
  (``theta``), subject to one-hot candidate selection and the raw (unannualized)
  budget constraint;
* per-country LP SUBPROBLEMS -- ``optimize_h2_producer.solve(capacities=...,
  return_duals=True)`` -- that price each trial capacity vector using that country's
  OWN trained proxy (Formulation.md SS1), by default over the FULL calendar year (SS2.5's
  documented single-day-scope pitfall), or on ``--rep-days-per-month`` representative
  days/month instead (weighted to approximate the full year, SS2.6) for a much faster
  but approximate subproblem, and hand back duals used to tighten theta's cut for the
  next master iteration.

Subproblems are solved SEQUENTIALLY, one LP per included country per iteration (~45-50s
each for a full-year DE00 solve, well under 1s for a few representative days/month) -- a
full 13-country run with the exact full-year subproblem can take 15-30+ minutes
depending on how many iterations it takes to close the gap;
``--rep-days-per-month`` cuts that roughly in proportion to how many fewer hours each
subproblem solves. See ``Formulation.md`` SS4 for the full derivation and known
limitations (placeholder CAPEX figures, LP-degeneracy-driven weak-but-valid cuts).

Usage:
    python plan_h2_capacity.py --countries DE,FR,PL --budget 500000000
    python plan_h2_capacity.py --all --budget 1000000000 --max-iters 40
    python plan_h2_capacity.py --countries DE,FR --rep-days-per-month 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import optimize_h2_producer as ohp
import h2_planning as hp
from h2_planning.candidates import default_sizing_and_zones

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def eligible_countries() -> list[str]:
    sizing, _ = default_sizing_and_zones()
    return sorted(sizing)


def run_benders(countries: list[str], budget: float | None, max_iters: int, gap_tol: float,
                capex_cfg: hp.CapexAssumptions, quiet_solver: bool = True,
                rep_days_per_month: int | None = None, downstream_load_mw: float | None = None,
                downstream_load_flex_pct: float | None = None,
                require_electrolyser_for_others: bool = False):
    default_mw, cand_mw, host_zone = hp.build_candidates(countries, capex_cfg)
    unit_cost = capex_cfg.effective_unit_cost_eur_per_mw()
    crf = capex_cfg.capital_recovery_factors()

    if budget is not None:
        cheapest_total = sum(min(cand_mw[c][a]) * unit_cost[a] for c in countries for a in hp.ASSETS)
        if cheapest_total > budget:
            print(f"NOTE: budget {budget:,.0f} EUR is below the cheapest all-assets-built "
                 f"combination ({cheapest_total:,.0f} EUR) for {countries} -- expect some "
                 f"assets to come back skipped (0 MW) in the result.")

    print("Building enriched price frames (one-time cost, reused by every subproblem solve)...")
    edf = ohp.enriched_elec_df()
    hdf = ohp.enriched_h2_df()

    m = hp.build_master(countries, cand_mw, unit_cost, budget, crf, capex_cfg.theta_lower_bound_eur,
                        downstream_load_mw=downstream_load_mw,
                        require_electrolyser_for_others=require_electrolyser_for_others)

    best_ub, best_capacities, best_capex = float("inf"), None, None
    log = []
    gap = float("inf")
    for it in range(1, max_iters + 1):
        status, cond = m.solve(solver_name="highs", output_flag=False)
        if status != "ok":
            raise RuntimeError(f"master solve failed at iteration {it}: {status}/{cond}")
        cap_star = hp.extract_capacities(m, countries, cand_mw)
        lb = float(m.objective.value)

        total_Q = 0.0
        t_sub = time.time()
        for c in countries:
            if rep_days_per_month is not None:
                out = ohp.solve(host_zone[c], rep_days_per_month=rep_days_per_month,
                                capacities=cap_star[c], return_duals=True,
                                downstream_load_mw=downstream_load_mw,
                                downstream_load_flex_pct=downstream_load_flex_pct,
                                edf=edf, hdf=hdf, quiet=quiet_solver)
            else:
                out = ohp.solve(host_zone[c], 1, 364, capacities=cap_star[c], return_duals=True,
                                downstream_load_mw=downstream_load_mw,
                                downstream_load_flex_pct=downstream_load_flex_pct,
                                edf=edf, hdf=hdf, quiet=quiet_solver)
            total_Q += out.attrs["objective"]
            hp.add_optimality_cut(m, c, it, out.attrs["objective"], out.attrs["cut_coeffs"],
                                  cap_star[c], cand_mw)
        sub_s = time.time() - t_sub

        raw_capex = sum(cap_star[c][a] * unit_cost[a] for c in countries for a in hp.ASSETS)
        annualized_capex = sum(cap_star[c][a] * unit_cost[a] * crf[a]
                               for c in countries for a in hp.ASSETS)
        ub = annualized_capex + total_Q
        if ub < best_ub:
            best_ub, best_capacities, best_capex = ub, cap_star, raw_capex
        gap = (best_ub - lb) / max(abs(best_ub), 1e-6)
        log.append({"iter": it, "lb": lb, "ub": ub, "best_ub": best_ub, "gap": gap,
                    "subproblems_seconds": round(sub_s, 1)})
        print(f"iter {it:>3}: LB={lb:>16,.0f}  UB={ub:>16,.0f}  best={best_ub:>16,.0f}  "
             f"gap={gap:.4f}  (subproblems {sub_s:.1f}s)")
        if gap <= gap_tol:
            print(f"converged (gap {gap:.4f} <= tol {gap_tol}) after {it} iteration(s)")
            break
    else:
        print(f"WARNING: reached --max-iters={max_iters} without closing the gap "
             f"(final gap {gap:.4f}) -- results below are the best FOUND, not proven optimal")

    return best_capacities, best_capex, best_ub, pd.DataFrame(log), host_zone, edf, hdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--countries", type=str, default=None,
                       help="comma-separated 2-letter country codes, e.g. DE,FR,PL")
    group.add_argument("--all", action="store_true", help="plan every eligible country")
    ap.add_argument("--budget", type=float, default=None,
                    help=f"total system-wide CAPEX budget, EUR, raw/unannualized "
                         f"(default {hp.CapexAssumptions().default_budget_eur:,.0f}). "
                         f"Ignored if --no-budget is given")
    ap.add_argument("--no-budget", action="store_true",
                    help="deactivate the budget constraint entirely -- every country sizes "
                         "purely off subproblem economics and annualized CAPEX, no system-wide "
                         "spend cap (h2_planning/master.py's budget=None)")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--gap-tol", type=float, default=0.01, help="relative Benders gap, default 0.01 (1%%)")
    ap.add_argument("--discount-rate", type=float, default=None, help="for the capital recovery factor")
    ap.add_argument("--lifetime-years", type=float, default=None,
                    help="capital recovery factor project life, applied uniformly to every asset "
                         "(default: each asset's own CapexAssumptions.lifetime_years entry, 20yr for "
                         "all today -- edit that dict directly for per-asset lifetimes)")
    ap.add_argument("--output", type=str, default=str(OUT / "plan"), help="output file prefix")
    ap.add_argument("--export-schedules", action="store_true",
                    help="also re-solve each included country at the final chosen "
                         "capacities and dump its full-year (or, with --rep-days-per-month, "
                         "representative-day) schedule")
    ap.add_argument("--rep-days-per-month", type=int, default=None,
                    help="solve each Benders subproblem on N representative days/month "
                         "(1-29, weighted to approximate the full year) instead of the full "
                         "364-day year -- much faster per iteration, at the cost of the "
                         "subproblem's objective/cut coefficients being an approximation "
                         "rather than exact (see optimize_h2_producer.solve's docstring and "
                         "Formulation.md SS2.6/SS4.5). Default: full 364-day year, unchanged.")
    ap.add_argument("--downstream-load-mw", type=float, default=None,
                    help="fix every country's H2 Producer downstream demand baseline to this "
                         "exact MW figure, decoupled from electrolyser capacity (default: "
                         "baseline scales with electrolyser capacity, the original behavior). "
                         "Also adds a master constraint forcing electrolyser capacity >= this "
                         "value for every country (h2_planning/master.py's "
                         "downstream_load_mw) -- see optimize_h2_producer.solve's docstring.")
    ap.add_argument("--downstream-load-flex-pct", type=float, default=None,
                    help="demand flexibility as a fraction of --downstream-load-mw (e.g. 0.2 "
                         "= +/-20%%); only used together with --downstream-load-mw (default: "
                         "reuses RunConfig's own h2_producer_demand_flex_pct)")
    ap.add_argument("--require-electrolyser-for-others", action="store_true",
                    help="wind/PV/battery/tank can only be selected for a country that also "
                         "selected a positive electrolyser candidate -- skipping the electrolyser "
                         "forces every other asset to be skipped too (h2_planning/master.py's "
                         "require_electrolyser_for_others). Vacuous/redundant whenever "
                         "--downstream-load-mw is also given, since that already forces the "
                         "electrolyser on everywhere -- only bites when the electrolyser itself "
                         "is optional.")
    args = ap.parse_args()

    capex_cfg = hp.CapexAssumptions()
    if args.discount_rate is not None:
        capex_cfg.discount_rate = args.discount_rate
    if args.lifetime_years is not None:
        capex_cfg.lifetime_years = {a: args.lifetime_years for a in hp.ASSETS}
    budget = None if args.no_budget else (args.budget if args.budget is not None else capex_cfg.default_budget_eur)

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
    budget_str = "DEACTIVATED (--no-budget: no system-wide spend cap)" if budget is None else f"{budget:,.0f} EUR (raw/unannualized)"
    print(f"Budget: {budget_str} | CRF @ {capex_cfg.discount_rate:.1%} discount: "
         f"{crf_str} -- see Formulation.md SS4.3 for why CAPEX is annualized in the objective but not "
         f"the budget")

    if args.rep_days_per_month is not None:
        print(f"Subproblems: {args.rep_days_per_month} representative day(s)/month "
             f"({args.rep_days_per_month * 12} days solved, weighted to approximate the full year) "
             f"-- faster but approximate; omit --rep-days-per-month for the exact full-year subproblem")
    if args.downstream_load_mw is not None:
        flex_note = f"+/-{args.downstream_load_flex_pct:.0%}" if args.downstream_load_flex_pct is not None else "default flex"
        print(f"Downstream load: fixed at {args.downstream_load_mw:g} MW ({flex_note}) for every "
             f"country, decoupled from electrolyser sizing; electrolyser capacity forced >= "
             f"{args.downstream_load_mw:g} MW per country (master constraint)")

    t0 = time.time()
    best_capacities, best_capex, best_ub, log_df, host_zone, edf, hdf = run_benders(
        countries, budget, args.max_iters, args.gap_tol, capex_cfg,
        rep_days_per_month=args.rep_days_per_month, downstream_load_mw=args.downstream_load_mw,
        downstream_load_flex_pct=args.downstream_load_flex_pct,
        require_electrolyser_for_others=args.require_electrolyser_for_others)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s. Final capacities:")
    rows = []
    for c in countries:
        row = {"country": c, "host_zone": host_zone[c]}
        row.update(best_capacities[c])
        rows.append(row)
    cap_df = pd.DataFrame(rows)
    print(cap_df.to_string(index=False))
    if budget is None:
        print(f"\nTotal raw CAPEX: {best_capex:,.0f} EUR (no budget cap)")
    else:
        print(f"\nTotal raw CAPEX: {best_capex:,.0f} EUR (budget {budget:,.0f} EUR, "
             f"{best_capex / budget:.1%} used)")
    print(f"Best objective (annualized CAPEX + 1yr operating cost, EUR, lower=better): {best_ub:,.0f}")

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_csv(f"{out_prefix}_capacities.csv", index=False)
    log_df.to_csv(f"{out_prefix}_convergence.csv", index=False)
    print(f"\nwrote {out_prefix}_capacities.csv, {out_prefix}_convergence.csv")

    if args.export_schedules:
        for c in countries:
            if args.rep_days_per_month is not None:
                out = ohp.solve(host_zone[c], rep_days_per_month=args.rep_days_per_month,
                                capacities=best_capacities[c], quiet=True,
                                downstream_load_mw=args.downstream_load_mw,
                                downstream_load_flex_pct=args.downstream_load_flex_pct,
                                edf=edf, hdf=hdf)
            else:
                out = ohp.solve(host_zone[c], 1, 364, capacities=best_capacities[c], quiet=True,
                                downstream_load_mw=args.downstream_load_mw,
                                downstream_load_flex_pct=args.downstream_load_flex_pct,
                                edf=edf, hdf=hdf)
            out.to_csv(f"{out_prefix}_schedule_{c}.csv", index=False)
        print(f"wrote {out_prefix}_schedule_<country>.csv for {len(countries)} countries")


if __name__ == "__main__":
    main()
