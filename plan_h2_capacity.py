"""H2 Producer capacity PLANNING: Benders decomposition CLI.

Chooses each requested country's Hydrogen Producer capacity (electrolyser, wind, PV,
battery, H2 tank -- each independently discrete/binary, see ``h2_planning/config.py``)
under one system-wide CAPEX budget, by iterating between:

* a MILP MASTER (``h2_planning/master.py``) that picks one candidate MW value per
  asset per country, minimizing annualized CAPEX + a per-country recourse variable
  (``theta``), subject to one-hot candidate selection and the raw (unannualized)
  budget constraint;
* ONE JOINT SUBPROBLEM per iteration (``optimize_h2_producer.solve_joint``) -- every
  included country solved TOGETHER in a single linopy model, sharing one downstream-
  demand POOL: each country's flat BASELINE share (``demand_base``, no hour index) is a
  free variable the subproblem decides (bounded by that country's own electrolyser
  capacity), with the baselines across all countries summing to a fixed total
  (``--joint-pool-mw``) -- redistributed toward whichever built electrolyser sees the
  most favorable price signal. On top of that baseline, each country's REALIZED hourly
  demand may additionally shift +/-20% (``solve_joint``'s ``demand_flex_pct``, default
  0.20), net-zero within each representative day -- so the realized cross-country total
  at any single hour is no longer pinned exactly to ``--joint-pool-mw``, only each
  country's own baseline and its own per-day net shift are. Duals from this joint solve
  become Benders optimality-cut coefficients, one cut per country per iteration.

wind/PV/battery/tank can only be selected for a country that ALSO selected a positive
electrolyser candidate that iteration (``h2_planning.build_master``'s
``require_electrolyser_for_others``, unconditionally on here) -- a country skipping the
electrolyser gets every other asset forced to 0 MW too, since this facility is a
Hydrogen Producer, not a standalone merchant power plant.

The electricity grid and H2 pipeline exchange connections (every country's ``x_grid``/
``x_h2``) are UNLIMITED here -- this planning pipeline always passes
``grid_cap_mw=h2_cap_mw=float("inf")`` into ``solve_joint``, rather than leaving it at
``None`` (which would fall back to RunConfig's real 40/20 MW physical caps).
``optimize_h2_producer.solve``/``solve_joint`` called directly (e.g. Formulation.md
SS2/SS3's backtest) are unaffected -- their own default is still the real 40/20 MW cap.

``--rep-days-per-month`` is required (``solve_joint`` has no full-year contiguous
mode) -- subproblems are solved on N representative days/month (weighted to
approximate the full year, Formulation.md SS2.6) rather than every one of the 8,760
hours. See ``Formulation.md`` SS4 for the full derivation and known limitations
(candidate-catalog CAPEX sourcing, LP-degeneracy-driven weak-but-valid cuts).

Usage:
    python plan_h2_capacity.py --countries DE,FR,PL --rep-days-per-month 1 --joint-pool-mw 100
    python plan_h2_capacity.py --all --no-budget --rep-days-per-month 1 --joint-pool-mw 300
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import optimize_h2_producer as ohp
import h2_planning as hp
from economic_dispatch.config import RunConfig
from h2_planning.candidates import default_sizing_and_zones

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

UNLIMITED_EXCHANGE_MW = float("inf")


def eligible_countries() -> list[str]:
    sizing, _ = default_sizing_and_zones()
    return sorted(sizing)


def run_benders(countries: list[str], budget: float | None, max_iters: int, gap_tol: float,
                capex_cfg: hp.CapexAssumptions, joint_pool_mw: float, rep_days_per_month: int,
                quiet_solver: bool = True):
    """Every iteration solves ALL of ``countries`` TOGETHER in one joint linopy model
    (``optimize_h2_producer.solve_joint``), sharing a fixed ``joint_pool_mw`` MW/hr
    downstream-demand pool -- see module docstring. Automatically floors the master's
    aggregate electrolyser capacity at ``joint_pool_mw /
    RunConfig().h2_producer_electrolyser_efficiency`` (``h2_planning.build_master``'s
    ``min_total_electrolyser_mw``): without it, iteration 1's all-skip master proposal
    would make the joint subproblem infeasible (a 0-capacity fleet can't supply any
    pool demand at all). Also unconditionally passes
    ``require_electrolyser_for_others=True`` -- wind/PV/battery/tank can only be
    selected for a country that also selected a positive electrolyser candidate that
    iteration."""
    default_mw, cand_mw, cand_capex, host_zone = hp.build_candidates(countries, capex_cfg)
    crf = capex_cfg.capital_recovery_factors()

    if budget is not None:
        cheapest_total = sum(min(cand_capex[c][a]) for c in countries for a in hp.ASSETS)
        if cheapest_total > budget:
            print(f"NOTE: budget {budget:,.0f} EUR is below the cheapest all-assets-built "
                 f"combination ({cheapest_total:,.0f} EUR) for {countries} -- expect some "
                 f"assets to come back skipped (0 MW) in the result.")

    print("Building enriched price frames (one-time cost, reused by every subproblem solve)...")
    edf = ohp.enriched_elec_df()
    hdf = ohp.enriched_h2_df()

    eta_ely = RunConfig().h2_producer_electrolyser_efficiency
    min_total_electrolyser_mw = joint_pool_mw / eta_ely
    max_grid_total = sum(max(cand_mw[c]["electrolyser_mw"]) for c in countries)
    if min_total_electrolyser_mw > max_grid_total:
        raise ValueError(f"joint_pool_mw={joint_pool_mw:g} needs >= {min_total_electrolyser_mw:.1f} MW "
                         f"of aggregate electrolyser capacity (at {eta_ely:.0%} efficiency), but the "
                         f"candidate grids across {countries} only go up to {max_grid_total:.1f} MW total "
                         f"-- widen the electrolyser candidate grid or lower --joint-pool-mw")
    print(f"Joint shared-pool mode: {joint_pool_mw:g} MW/hr across {countries}, "
         f"min_total_electrolyser_mw floor = {min_total_electrolyser_mw:.1f} MW")

    m = hp.build_master(countries, cand_mw, cand_capex, budget, crf, capex_cfg.theta_lower_bound_eur,
                        min_total_electrolyser_mw=min_total_electrolyser_mw,
                        require_electrolyser_for_others=True)

    best_ub, best_capacities, best_capex, best_capex_by_asset = float("inf"), None, None, None
    log = []
    per_country_log = {c: [] for c in countries}
    gap = float("inf")
    for it in range(1, max_iters + 1):
        status, cond = m.solve(solver_name="highs", output_flag=False)
        if status != "ok":
            raise RuntimeError(f"master solve failed at iteration {it}: {status}/{cond}")
        cap_star = hp.extract_capacities(m, countries, cand_mw)
        lb = float(m.objective.value)

        t_sub = time.time()
        zones = [host_zone[c] for c in countries]
        caps_by_zone = {host_zone[c]: cap_star[c] for c in countries}
        result = ohp.solve_joint(zones, caps_by_zone, rep_days_per_month, joint_pool_mw,
                                 return_duals=True, grid_cap_mw=UNLIMITED_EXCHANGE_MW,
                                 h2_cap_mw=UNLIMITED_EXCHANGE_MW, edf=edf, hdf=hdf, quiet=quiet_solver)
        total_Q = float(result["objective"])
        Q_by_country = {c: result["objective_by_zone"][host_zone[c]] for c in countries}
        mu_by_country = {c: result["cut_coeffs"][host_zone[c]] for c in countries}
        hp.add_optimality_cut(m, countries, it, Q_by_country, mu_by_country, cap_star, cand_mw)
        sub_s = time.time() - t_sub

        capex_star = hp.extract_capex(m, countries, cand_capex)
        for c in countries:
            per_country_log[c].append({"iter": it, "objective": result["objective_by_zone"][host_zone[c]],
                                       "capex": sum(capex_star[c].values()), "capacities": dict(cap_star[c])})
        raw_capex = sum(capex_star[c][a] for c in countries for a in hp.ASSETS)
        annualized_capex = sum(capex_star[c][a] * crf[a] for c in countries for a in hp.ASSETS)
        ub = annualized_capex + total_Q
        if ub < best_ub:
            best_ub, best_capacities, best_capex, best_capex_by_asset = ub, cap_star, raw_capex, capex_star
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

    return (best_capacities, best_capex, best_ub, pd.DataFrame(log), host_zone, edf, hdf,
           per_country_log, best_capex_by_asset)


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
                         "(default: each asset's own CapexAssumptions.lifetime_years entry, sourced "
                         "from CANDIDATE_CATALOG's 2030 column -- 25/30/40/20/30yr for "
                         "electrolyser/wind/PV/battery/tank today -- edit capex_cfg.lifetime_years "
                         "directly for a different per-asset lifetime)")
    ap.add_argument("--output", type=str, default=str(OUT / "plan"), help="output file prefix")
    ap.add_argument("--export-schedules", action="store_true",
                    help="also re-solve at the final chosen capacities and dump each "
                         "included country's representative-day schedule")
    ap.add_argument("--rep-days-per-month", type=int, required=True,
                    help="REQUIRED -- solve every joint subproblem on N representative "
                         "days/month (1-29, weighted to approximate the full year) -- "
                         "optimize_h2_producer.solve_joint has no full-year contiguous mode "
                         "(see optimize_h2_producer.solve's docstring and Formulation.md "
                         "SS2.6/SS4.5).")
    ap.add_argument("--joint-pool-mw", type=float, required=True,
                    help="REQUIRED -- total downstream H2 demand (MW/hr) shared across every "
                         "included country every hour -- optimize_h2_producer.solve_joint's "
                         "pool_mw. Each country's own SHARE of this fixed total is a free "
                         "variable the subproblem decides (bounded by that country's own "
                         "electrolyser capacity); the total itself never changes hour to hour.")
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
    print(f"Subproblems: {args.rep_days_per_month} representative day(s)/month "
         f"({args.rep_days_per_month * 12} days solved, weighted to approximate the full year)")
    print(f"Joint mode: ALL countries solved together per iteration, sharing a "
         f"{args.joint_pool_mw:g} MW/hr downstream-demand pool (fixed baseline total, "
         f"free per-country split, +/-20% hourly flex net-zero per day)")
    print(f"Exchange caps: grid=unlimited, H2 pipeline=unlimited")
    print("Require-electrolyser-for-others: ON -- wind/PV/battery/tank only buildable "
         "alongside a positive electrolyser candidate")

    t0 = time.time()
    (best_capacities, best_capex, best_ub, log_df, host_zone, edf, hdf,
    per_country_log, best_capex_by_asset) = run_benders(
        countries, budget, args.max_iters, args.gap_tol, capex_cfg,
        joint_pool_mw=args.joint_pool_mw, rep_days_per_month=args.rep_days_per_month)
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
        zones = [host_zone[c] for c in countries]
        caps_by_zone = {host_zone[c]: best_capacities[c] for c in countries}
        final = ohp.solve_joint(zones, caps_by_zone, args.rep_days_per_month, args.joint_pool_mw,
                                return_duals=False, grid_cap_mw=UNLIMITED_EXCHANGE_MW,
                                h2_cap_mw=UNLIMITED_EXCHANGE_MW, edf=edf, hdf=hdf, quiet=True)
        for c in countries:
            final["schedules"][host_zone[c]].to_csv(f"{out_prefix}_schedule_{c}.csv", index=False)
        print(f"wrote {out_prefix}_schedule_<country>.csv for {len(countries)} countries")


if __name__ == "__main__":
    main()
