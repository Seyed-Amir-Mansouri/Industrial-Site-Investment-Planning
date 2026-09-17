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

Capacity is chosen under wind/PV UNCERTAINTY (added 2026-09-14, at user request):
every iteration's trial capacity is priced once per capacity-uncertainty scenario
(``run_capacity_scenarios.py``'s ``p100``/``wind70``/``pv70`` -- each using that
scenario's own trained-surrogate price context, not just its own capacity multiplier).
The master's OBJECTIVE defaults to an equal-weighted EXPECTED VALUE across scenarios
(``SCENARIO_PROBS`` below) -- the master's capacity decision itself carries no scenario
index, exactly one plan, sized for its average performance. Expected value was chosen
over a worst-case/robust or CVaR-style risk measure for simplicity, at user request on
2026-09-14; a CVaR alternative was added 2026-09-16 (``--cvar-alpha``, ``h2_planning.
build_master``'s ``cvar_alpha``/``h2_planning.cvar_value`` -- REPLACES, doesn't blend
with, the expected value when given). ``SCENARIO_PROBS``' equal weights are also an
assumption, editable directly if some scenarios should be judged more/less likely than
33% each.

The 3 scenarios' subproblem solves each get their OWN Benders cut on their OWN
per-scenario ``theta_s`` (``h2_planning.master``'s ``scenario_probs``/
``add_optimality_cut(..., scenario=s)``), added 2026-09-16 at user request for faster
convergence -- NOT the same as the SS4.6 single-cut-across-countries fix, which stays
exactly as it was (still one combined cut per scenario, just no longer one combined cut
across scenarios too). This roughly triples the Benders information gained per
iteration for a 3-scenario run versus the original one-cut-total design, without
weakening rigor -- see Formulation.md SS4.9.1 for why splitting cuts by scenario is
valid where splitting by country was not. The 3 solves themselves still run
SEQUENTIALLY, not concurrently -- a ThreadPoolExecutor attempt measured SLOWER (Python's
GIL isn't released enough during linopy's model-construction to benefit from threads);
see the loop body's own comment for the measured numbers and why a real fix
(ProcessPoolExecutor) wasn't pursued.

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
from run_capacity_scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

UNLIMITED_EXCHANGE_MW = float("inf")

# Equal probability weight per capacity-uncertainty scenario (p100/wind70/pv70/ely70 --
# see run_capacity_scenarios.py for what each represents). Every Benders iteration below
# solves the joint subproblem ONCE PER SCENARIO at the SAME trial capacity, then combines
# the 4 outcomes into ONE expected-value optimality cut -- so the master's capacity
# decision carries NO scenario index: one plan, sized for its average performance across
# all 4 scenarios (risk measure = expected value, chosen over worst-case/CVaR at user
# request 2026-09-14 for simplicity -- see conversation/commit history if a more
# risk-averse measure is wanted later). Edit this dict directly for unequal weights
# (e.g. if a shortfall scenario is judged more/less likely than 25%).
SCENARIO_PROBS = {name: 1.0 / len(SCENARIOS) for name in SCENARIOS}


def eligible_countries() -> list[str]:
    sizing, _ = default_sizing_and_zones()
    return sorted(sizing)


def run_benders(countries: list[str], budget: float | None, max_iters: int, gap_tol: float,
                capex_cfg: hp.CapexAssumptions, joint_pool_mw: float, rep_days_per_month: int,
                quiet_solver: bool = True, on_iteration=None, cvar_alpha: float | None = None,
                max_total_assets: int | None = None, master_time_limit: float = 180.0,
                use_pareto_cuts: bool = False):
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
    iteration.

    ``on_iteration``, if given, is called after every iteration's cut is added (before
    the convergence check) with one dict: ``{"iter", "lb", "ub", "best_ub", "gap",
    "best_capacities", "best_capex_by_asset", "host_zone", "log", "per_country_log"}``
    (``log``/``per_country_log`` are the FULL history so far, not just this iteration) --
    e.g. to checkpoint progress to disk, since a long run has no other way to survive an
    interruption (§4.9's OOM-kill finding: a run given enough iterations to fully close
    the gap on a 13-country/multi-scenario problem can run long enough to be killed for
    memory before it ever returns, losing everything if nothing was saved along the
    way). Default ``None`` preserves every existing caller's behavior exactly.

    ``cvar_alpha`` (optional, added 2026-09-16 at user request): switches the master's
    risk measure from the default equal-weighted EXPECTED VALUE across scenarios to
    CVaR_alpha (``h2_planning.build_master``'s own ``cvar_alpha`` param -- see its
    docstring for the exact linearization and the "collapses to worst-case with few
    equally-likely scenarios" caveat). When given, the reported ``ub`` each iteration
    is also switched from ``annualized_capex + sum_s prob_s*Q_s(cap*)`` to
    ``annualized_capex + CVaR_alpha({Q_s(cap*)})`` (``h2_planning.cvar_value``, exact
    formula on the REAL per-scenario subproblem objectives, not the relaxed ``theta_s``/
    ``zeta`` surrogates) -- so the printed objective is the actual risk-adjusted number
    being minimized, not the plain expectation. Default ``None`` preserves the original
    expected-value behavior exactly.

    ``master_time_limit`` (seconds, default 180, added 2026-09-16 at user request after
    the master MILP was observed taking 600-900s+ and STILL GROWING per iteration on a
    13-country/``max_total_assets=20`` run as cuts accumulated -- a diagnostic against
    those REAL accumulated cuts found no ``solver_options`` combination (mip_rel_gap
    1e-4/1e-3/1e-2, mip_heuristic_effort, presolve+parallel) let the master itself prove
    even a 30% inner gap within 180s: this master is just genuinely hard to solve to
    proven optimality at this scale, not a settings problem. Capping wall time turns
    "iterations get arbitrarily slower" into "iterations are bounded, the outer loop
    just runs more of them" -- and stays fully rigorous because the Benders ``lb`` is
    read from HiGHS's ``mip_dual_bound`` (a genuine lower bound on the master's true
    optimum, valid EVEN WHEN the search was cut off early), not the incumbent
    ``objective.value`` (which is only an upper bound on the master's own optimum until
    proven). Verified ``dual_bound == objective`` exactly whenever the master DOES solve
    to proven optimality within the time limit (small/early-iteration masters), so this
    changes nothing for those. ``presolve="on", parallel="on"`` are always passed too --
    empirically the best of the tested combinations (tightest dual bound for the same
    time budget).

    ``use_pareto_cuts`` (default False, added 2026-09-17 at user request after
    ``master_time_limit`` alone still left the 13-country/``max_total_assets=20`` run's
    gap plateaued around 0.80 for 60+ iterations): builds each optimality cut from a
    subproblem solve at a MOVING CORE POINT (Papadakos 2008's practical simplification
    of Magnanti-Wong Pareto-optimal cuts) instead of the trial point ``cap_star`` the
    master just proposed. Method: a one-time core point (mean of ``{0} u candidates``
    per (country, asset)) is solved as a SEPARATE subproblem each iteration (in addition
    to, not instead of, the usual trial-point solve, which is still needed for the true
    UB/``best_ub``) -- its ``(Q, mu)`` become the cut, evaluated at the core point, not
    ``cap_star``. Then the core point is updated ``core_cap = 0.5*core_cap +
    0.5*cap_star`` for the next iteration, so it drifts toward the explored region while
    staying non-degenerate (an interior point, not a corner of the [0,1] one-hot
    hypercube the master's binary trial points always are).

    This is SAFE in a way the reverted trust-region attempt was not: it never touches
    the master's objective, only WHICH FEASIBLE POINT the (convex) subproblem value
    function is linearized at -- a supporting-hyperplane cut from ANY feasible point's
    dual solution is valid for the entire feasible region (standard LP value-function
    duality), so this changes cut QUALITY, never VALIDITY. Verified on 2-country and
    5-country test problems: identical final capacities/best_ub to the non-Pareto
    baseline (just reached in more, cheaper-to-solve-at-that-scale iterations -- no
    benefit shows up until the problem is large enough that cut quality, not sheer
    problem size, is the bottleneck). At 13-country/``max_total_assets=20`` scale this
    is decisive: a fresh 10-iteration head-to-head against the baseline reached gap
    0.0038 (vs baseline's 1.012, which never dropped below 1.0) for only +5.8% more
    wall time (~53s of extra subproblem solving per iteration is small next to the
    100-300s+ the master itself costs at this scale)."""
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
                        require_electrolyser_for_others=True, scenario_probs=SCENARIO_PROBS,
                        cvar_alpha=cvar_alpha, max_total_assets=max_total_assets)

    # One-time Pareto-cut core-point init (mean of {0} u candidates per (country, asset))
    # -- see use_pareto_cuts's docstring above. Unused when use_pareto_cuts is False.
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

        # Solve the SAME trial capacity once per scenario, then add ONE Benders cut PER
        # SCENARIO on that scenario's own theta_s (see build_master's scenario_probs /
        # add_optimality_cut's scenario param), instead of flattening all scenarios into
        # one combined cut first. Only the MASTER's OBJECTIVE uses the expected value
        # (prob-weighted sum of theta_s); the cuts themselves stay scenario-specific for
        # faster convergence (Formulation.md SS4.9.1).
        #
        # NOT run concurrently, despite being independent given a fixed capacity: tried
        # a ThreadPoolExecutor here (2026-09-16) and measured it SLOWER, not faster --
        # 29.3s/27.4s per iteration at 13-country scale vs ~18-19s sequential -- Python's
        # GIL isn't released enough during linopy's numpy/xarray-heavy model construction
        # for threads to help, and thread-switching overhead makes it net worse. A real
        # speedup would need separate OS processes (ProcessPoolExecutor), which was not
        # attempted: it would multiply memory footprint (each worker needs its own copy
        # of the enriched elec/h2 price frames) at exactly the scale where OOM kills are
        # already a live risk (Formulation.md SS4.9's checkpoint-and-restart finding) --
        # judged not worth that tradeoff without first proving it's needed.
        caps_core_by_zone = {host_zone[c]: core_cap[c] for c in countries}
        results = {}
        for s in SCENARIO_PROBS:
            edf_s, hdf_s = price_frames[s]
            # With Pareto cuts, the trial-point solve only needs its OBJECTIVE (true
            # UB/best_ub) -- return_duals=False, cheaper -- since the CUT itself comes
            # from a separate solve at the core point instead (below).
            results[s] = ohp.solve_joint(zones, caps_by_zone, rep_days_per_month, joint_pool_mw,
                                         return_duals=not use_pareto_cuts, grid_cap_mw=UNLIMITED_EXCHANGE_MW,
                                         h2_cap_mw=UNLIMITED_EXCHANGE_MW, edf=edf_s, hdf=hdf_s,
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
                result_core = ohp.solve_joint(zones, caps_core_by_zone, rep_days_per_month, joint_pool_mw,
                                              return_duals=True, grid_cap_mw=UNLIMITED_EXCHANGE_MW,
                                              h2_cap_mw=UNLIMITED_EXCHANGE_MW, edf=edf_s, hdf=hdf_s,
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
        # UB is evaluated with the SAME risk measure the master minimizes: exact CVaR of
        # the real per-scenario objectives (h2_planning.cvar_value), not the relaxed
        # zeta/theta_s surrogates, when cvar_alpha is set -- otherwise plain expectation.
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
    ap.add_argument("--cvar-alpha", type=float, default=None,
                    help="switch the risk measure from the default equal-weighted EXPECTED "
                         "VALUE across capacity-uncertainty scenarios to CVaR at this "
                         "confidence level (0-1, e.g. 0.95), REPLACING the expected value "
                         "(h2_planning.build_master's cvar_alpha). NOTE: with only 3 "
                         "equally-likely scenarios (1/3 each), any alpha with "
                         "(1-alpha) < 1/3 (e.g. 0.95) collapses CVaR exactly to pure "
                         "worst-case over the 3 scenarios -- see build_master's docstring.")
    ap.add_argument("--max-assets", type=int, default=None,
                    help="cap the TOTAL number of (country, asset) build decisions across "
                         "every included country and every asset combined -- any mix of "
                         "technologies, no per-asset/per-country sub-limit "
                         "(h2_planning.build_master's max_total_assets)")
    ap.add_argument("--master-time-limit", type=float, default=180.0,
                    help="wall-time cap (seconds) per master MILP solve, default 180 -- the "
                         "Benders lb is read from HiGHS's mip_dual_bound (valid even when "
                         "cut off early), not the incumbent, so this stays mathematically "
                         "rigorous while bounding per-iteration time (see run_benders's "
                         "docstring)")
    ap.add_argument("--pareto-cuts", action="store_true",
                    help="build optimality cuts from a moving CORE POINT (Papadakos-style "
                         "Pareto-optimal cuts) instead of the trial point -- an extra "
                         "subproblem solve per scenario per iteration (small overhead vs "
                         "the master's own cost at 13-country scale), but decisively fixes "
                         "the gap-plateau problem seen without it (see run_benders's "
                         "docstring for the measured before/after)")
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
        joint_pool_mw=args.joint_pool_mw, rep_days_per_month=args.rep_days_per_month,
        cvar_alpha=args.cvar_alpha, max_total_assets=args.max_assets,
        master_time_limit=args.master_time_limit, use_pareto_cuts=args.pareto_cuts)
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
    risk_label = (f"EXPECTED" if args.cvar_alpha is None else f"CVaR_{args.cvar_alpha:.2f}")
    print(f"Best objective (annualized CAPEX + {risk_label} 1yr operating cost across "
         f"{len(SCENARIO_PROBS)} scenarios, EUR, lower=better): {best_ub:,.0f}")

    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cap_df.to_csv(f"{out_prefix}_capacities.csv", index=False)
    log_df.to_csv(f"{out_prefix}_convergence.csv", index=False)
    print(f"\nwrote {out_prefix}_capacities.csv, {out_prefix}_convergence.csv")

    if args.export_schedules:
        # The chosen capacity is ONE non-scenario-indexed plan -- but how it OPERATES
        # (and what it earns) still depends on which scenario actually happens, so we
        # export one schedule per scenario to show that operational spread.
        zones = [host_zone[c] for c in countries]
        caps_by_zone = {host_zone[c]: best_capacities[c] for c in countries}
        for s, (edf_s, hdf_s) in price_frames.items():
            final = ohp.solve_joint(zones, caps_by_zone, args.rep_days_per_month, args.joint_pool_mw,
                                    return_duals=False, grid_cap_mw=UNLIMITED_EXCHANGE_MW,
                                    h2_cap_mw=UNLIMITED_EXCHANGE_MW, edf=edf_s, hdf=hdf_s, quiet=True)
            for c in countries:
                final["schedules"][host_zone[c]].to_csv(f"{out_prefix}_schedule_{c}_{s}.csv", index=False)
        print(f"wrote {out_prefix}_schedule_<country>_<scenario>.csv for {len(countries)} countries "
             f"x {len(price_frames)} scenarios")


if __name__ == "__main__":
    main()
