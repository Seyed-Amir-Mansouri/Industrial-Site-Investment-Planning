# Price Models & H2 Producer Capacity Planning

Two pieces of work, both built on an upstream LP economic-dispatch engine's output for
the 20-zone Central-European CORE region (NT2030 scenario):

## 1. Price models (`price_model/`)

Per-zone demand → price models (electricity and hydrogen), one gradient-boosted
model per bidding zone, trained on `inputs/hourly_balance_{elec,h2}.csv`.

```python
from price_model import electricity_price, hydrogen_price

electricity_price("DE00", 55000)   # price at 55 GW demand
hydrogen_price("AT00", 500)        # H2 price at 500 MWH2 demand
```

Retrain with `python train_model.py`.

## 2. H2 Producer capacity planning (`h2_planning/`, `optimize_h2_producer.py`)

Sizes each country's Hydrogen Producer (electrolyser, wind, PV, battery, H2 tank)
using the price models above as a price-taker market signal, instead of the full
network coupling the upstream dispatch engine solves.

- `optimize_h2_producer.py` — standalone LP for one country's fixed-capacity
  schedule (`solve`), or a joint multi-country solve sharing one demand pool
  (`solve_joint`).
- `h2_planning/` + `plan_h2_capacity.py` — turns the five asset sizes into a
  discrete/binary choice under a CAPEX budget, solved via Benders decomposition
  (MILP master + per-country LP subproblems).

```bash
python optimize_h2_producer.py --zone DE00 --day 5
python plan_h2_capacity.py --countries DE,FR,PL --rep-days-per-month 1 --joint-pool-mw 100
```

`optimize_h2_producer.py`'s standalone `--day`/`--start-day`/`--end-day` CLI supports
`--rep-days-per-month N` too, to solve on N representative days/month (weighted to
approximate the full year) instead of a contiguous day range. `plan_h2_capacity.py`
always solves every included country TOGETHER (`solve_joint`) on representative days —
`--rep-days-per-month` is REQUIRED there, not optional (`solve_joint` has no full-year
contiguous mode).

CAPEX/lifetime figures in `h2_planning/config.py::CANDIDATE_CATALOG` come from
`Help/Candidates (Edited).docx`'s 2030 candidate-product table (four real-world MW
sizes per asset, each own its own absolute CAPEX and lifetime) — a real cited source,
not a vendor quote.

### `plan_h2_capacity.py` flags

Full, current list also always available via `python plan_h2_capacity.py --help`.

**Scope**
| Flag | Default | What it does |
|---|---|---|
| `--countries CC,CC,...` | — | Comma-separated 2-letter country codes, e.g. `DE,FR,PL` (mutually exclusive with `--all`) |
| `--all` | — | Plan every eligible country |

**Budget / CAPEX**
| Flag | Default | What it does |
|---|---|---|
| `--budget EUR` | 500,000,000 | System-wide raw/unannualized CAPEX budget, shared across every included country — the SOLE discretionary installation limit (on top of the always-on structural constraints below) |
| `--no-budget` | off | Deactivate the budget constraint entirely — every country sizes purely off subproblem economics and annualized CAPEX |
| `--discount-rate R` | 0.05 | Discount rate for the capital recovery factor |
| `--lifetime-years N` | catalog's own (25/30/40/20/30yr for electrolyser/wind/PV/battery/tank) | Overrides EVERY asset's lifetime uniformly (edit `CapexAssumptions.lifetime_years` directly for a per-asset override instead) |

A per-country downstream-load floor, a forced-off asset list, and a system-wide
build-count cap were all tried and **removed** 2026-09-22 at user request, leaving
`--budget` as the only discretionary lever on what gets built — see `Formulation.md`
§4.9.3 if reviving any of them.

**Capacity-uncertainty scenarios / risk measure**
| Flag | Default | What it does |
|---|---|---|
| `--cvar-alpha A` | off (expected value) | Switch the risk measure across the 3 wind/PV capacity-uncertainty scenarios (`p100`/`wind70`/`pv70`, equal-weighted) from the default expected value to CVaR at confidence level `A` (0–1), REPLACING the expected value. With only 3 equally-likely scenarios, any `A` with `1-A < 1/3` (e.g. 0.95) collapses CVaR exactly to worst-case over the 3 scenarios — see `Formulation.md` §4.9.3 |

**Subproblem / demand (both REQUIRED)**
| Flag | What it does |
|---|---|
| `--rep-days-per-month N` | Solve the joint subproblem on N representative days/month (1–29, weighted to approximate the full year) — `solve_joint` has no full-year contiguous mode |
| `--joint-pool-mw MW` | Total downstream H2 demand baseline (MW/hr), summed across every included country's own flat baseline (`demand_base`). Each country's SHARE of that fixed total is a free variable the joint subproblem decides (bounded by its own electrolyser capacity). Each country's ACTUAL hourly demand can then deviate ±20% from its own baseline every representative hour (`optimize_h2_producer.solve_joint`'s `demand_flex_pct`, default 0.20), with total upward and downward shifts netting to exactly zero PER COUNTRY PER REPRESENTATIVE DAY (a 24h cycle, same horizon as the battery/tank storage cyclic constraints — not an annual aggregate) — so the system-wide hourly total is no longer pinned to exactly `--joint-pool-mw` at every single hour, only each country's own baseline and its own per-day net shift are |

Every country's subproblem is always solved TOGETHER, in one joint linopy model
(`optimize_h2_producer.solve_joint`) — there's no independent-per-country mode anymore.
wind/PV/battery/tank can only be built for a country that ALSO builds a positive
electrolyser candidate that iteration (`h2_planning.build_master`'s
`require_electrolyser_for_others`, unconditionally on) — skipping the electrolyser
forces every other asset to 0 MW too, since this facility is a Hydrogen Producer, not a
standalone merchant power plant. The electricity grid and H2 pipeline exchange
connections are always UNLIMITED in this pipeline (`solve_joint`'s `grid_cap_mw`/
`h2_cap_mw` are hardcoded to `float("inf")` here) — `optimize_h2_producer.solve`/
`solve_joint` called directly (outside this pipeline) are unaffected and still default
to `RunConfig`'s real 40/20 MW caps.

**Solve control / output**
| Flag | Default | What it does |
|---|---|---|
| `--max-iters N` | 30 | Benders iteration cap |
| `--gap-tol G` | 0.01 (1%) | Relative Benders convergence gap |
| `--master-time-limit S` | 180 | Wall-time cap (seconds) per master MILP solve — the master gets genuinely hard to solve to proven optimality as cuts accumulate at large scale (no `solver_options` tuning fixes this, see `Formulation.md` §4.9.3), so this bounds it instead; the Benders lower bound is read from HiGHS's own proven dual bound, so this stays mathematically rigorous even when the search is cut off early |
| `--pareto-cuts` | off | Build each optimality cut from a subproblem solve at a moving CORE POINT (Papadakos-style Pareto-optimal cuts) instead of the trial point — a small extra cost per iteration, but the difference between genuinely converging and plateauing well above the gap tolerance at large scale (13-country, budget-only test: converged cleanly, gap monotonically decreasing every iteration, in 6 iterations / ~7.4 min — see `Formulation.md` §4.9.4). **Recommended for any run beyond a handful of countries.** |
| `--output PREFIX` | `outputs/plan` | Output file prefix for `_capacities.csv`/`_convergence.csv`/(with `--export-schedules`) `_schedule_<country>.csv` |
| `--export-schedules` | off | Also re-solve at the final chosen capacities and dump each included country's representative-day schedule |

### How the Benders solve works

Sizing all five assets (electrolyser, wind, PV, battery, H2 tank) for every country at
once, as one MILP with a full year of hourly LP dispatch variables per country, doesn't
scale — so `plan_h2_capacity.py` splits it into a master problem and one JOINT
subproblem (covering every included country together), iterating between them:

1. **Master (MILP, `h2_planning/master.py`)** — picks one candidate MW value per
   asset per country (binary one-hot over `CANDIDATE_CATALOG`'s 4-product-per-asset
   catalog — real MW sizes, each with its own absolute CAPEX and lifetime, see below),
   subject to the annualized, system-wide CAPEX budget. Its objective is annualized
   CAPEX plus a per-country profit stand-in (`theta`) that starts unconstrained and
   gets tightened every round by the cuts below.
2. **Joint subproblem (LP, every country in ONE linopy model, `optimize_h2_producer.
   solve_joint`)** — for the master's chosen capacities, solves the shared-pool
   representative-day dispatch (`--rep-days-per-month`/`--joint-pool-mw`, both
   required) and returns each country's own realized operating profit plus the dual
   values (shadow prices) on its own capacity constraints.
3. **Cut generation** — each country's profit and duals are turned into its own
   Benders optimality cut: a linear upper bound on that country's `theta`, expressed in
   the master's binary capacity-choice variables (duals become the cut's coefficients,
   including a cross term for how electrolyser sizing affects the others). The cut is
   added back into the master.
4. **Loop** — resolve the master with the new cuts, resolve the joint subproblem at the
   updated capacities, and repeat until the master's `theta` upper bounds and the
   subproblems' actual profits converge (or an iteration cap is hit).

This avoids ever building one giant MILP over all countries' asset choices AND all
8,760 hours at once — each iteration is a small MILP plus one joint, representative-day
LP, which is what makes the discrete/binary capacity search tractable. See
`Formulation.md` §4 for the full derivation of the cut coefficients (including the
electrolyser cross-coupling term) and known limitations of the current implementation.

## Structure

```
price_model/            demand -> price models
h2_planning/             Benders master + candidate grids + CAPEX assumptions
optimize_h2_producer.py  standalone / joint H2 Producer LP
plan_h2_capacity.py      Benders CLI driver
day_sampling.py          representative-day sampling
economic_dispatch/       LP dispatch engine, vendored locally
inputs/                  sample parquets, balance CSVs, zone/network DBs
outputs/                 trained models, plans (git-ignored)
```

See `Formulation.md` for the full math.

## Install

```bash
pip install -r requirements.txt
```
