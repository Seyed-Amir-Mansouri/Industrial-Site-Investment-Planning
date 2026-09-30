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
  schedule (`solve`, models real downstream demand), or a joint multi-country
  MERCHANT solve (`solve_joint`, buys/sells electricity and hydrogen at market
  prices, NO downstream demand modeled since 2026-09-22 -- "there is no demand").
- `h2_planning/` + `plan_h2_capacity.py` — turns the five asset sizes into a
  discrete/binary choice under a CAPEX budget, solved via Benders decomposition
  (MILP master + one JOINT LP subproblem covering every included country together,
  `solve_joint` — not independent per-country subproblems).

```bash
python optimize_h2_producer.py --zone DE00 --day 5
python plan_h2_capacity.py --countries DE,FR,PL --rep-days-per-month 1
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
| `--budget EUR` | 500,000,000 | System-wide raw/unannualized CAPEX budget, shared across every included country — ALWAYS active, the ONLY constraint the master applies beyond one-hot candidate selection |
| `--discount-rate R` | 0.05 | Discount rate for the capital recovery factor |
| `--lifetime-years N` | catalog's own (25/30/40/20/30yr for electrolyser/wind/PV/battery/tank) | Overrides EVERY asset's lifetime uniformly (edit `CapexAssumptions.lifetime_years` directly for a per-asset override instead) |

**2026-09-22, two rounds of simplification, at user request:**
1. A per-country downstream-load floor, a forced-off asset list, and a system-wide
   build-count cap were tried and removed, leaving `--budget` as the only
   discretionary lever (`Formulation.md` §4.9.4).
2. Then `--no-budget` was removed too (budget can no longer be deactivated from this
   CLI), along with `require_electrolyser_for_others` (wind/PV/battery/tank no longer
   need a positive electrolyser in the same country) and `min_total_electrolyser_mw`
   — and `solve_joint` lost its ENTIRE downstream H2 demand mechanism (`--joint-pool-mw`
   is gone; there is no demand target anymore at all, `Formulation.md` §4.9.5). Every
   asset is now an independent one-hot choice, coupled to the others ONLY by budget.

**Capacity-uncertainty scenarios / risk measure**
| Flag | Default | What it does |
|---|---|---|
| `--cvar-alpha A` | off (expected value) | Switch the risk measure across the 3 wind/PV capacity-uncertainty scenarios (`p100`/`wind70`/`pv70`, equal-weighted) from the default expected value to CVaR at confidence level `A` (0–1), REPLACING the expected value. With only 3 equally-likely scenarios, any `A` with `1-A < 1/3` (e.g. 0.95) collapses CVaR exactly to worst-case over the 3 scenarios — see `Formulation.md` §4.9.3 |

**Subproblem**
| Flag | What it does |
|---|---|
| `--rep-days-per-month N` | REQUIRED. Solve the joint subproblem on N representative days/month (1–29, weighted to approximate the full year) — `solve_joint` has no full-year contiguous mode |

Every country's subproblem is always solved TOGETHER, in one joint linopy model
(`optimize_h2_producer.solve_joint`) — there's no independent-per-country mode anymore.
**No downstream H2 demand is modeled** (removed 2026-09-22, "there is no demand").
**General-investor model, no shared site balance, no GC market** (2026-09-28): PV and
wind each sell independently into the electricity market; the electrolyser buys
electricity and sells hydrogen; the battery buys/sells electricity; the H2 tank
buys/sells hydrogen — each bounded only by its own installed capacity, like five
separate one-asset investments rather than one co-located microgrid with a shared
site balance. There's no more shared grid/pipeline connection cap either (`x_grid`/
`x_h2`/`grid_cap_mw`/`h2_cap_mw` are gone from `solve_joint`'s signature entirely, not
just defaulted to unlimited), and the RED III renewable-h2 quota — previously ACTIVE
by default at 42% — is no longer enforceable or enforced at all, since it depended on
the removed on-site renewable-coverage accounting. Every asset is still a fully
independent one-hot choice, so a country CAN build wind/PV/battery/tank with zero
electrolyser (a standalone merchant power+storage plant), or vice versa. See
`Formulation.md` §2.7 for the full math. `optimize_h2_producer.solve` (outside this
pipeline, the standalone CLI backtest) follows the same redesign — its own site
connection cap (previously a real 40/20 MW default) is gone too.

**Solve control / output**
| Flag | Default | What it does |
|---|---|---|
| `--max-iters N` | 100 | Benders iteration cap |
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
   CAPEX plus a recourse stand-in (`theta`) that starts unconstrained and gets
   tightened every round by the cuts below — a SINGLE `theta` SHARED across every
   included country (never one per country, `Formulation.md` §4.6), and in practice
   ALWAYS one `theta_s` per capacity-uncertainty scenario (3, equal-weighted, or
   CVaR-weighted under `--cvar-alpha`) rather than one plain scalar, `Formulation.md`
   §4.9/§4.9.3.
2. **Joint subproblem (LP, every country in ONE linopy model, `optimize_h2_producer.
   solve_joint`)** — for the master's chosen capacities, solves the merchant
   representative-day dispatch (`--rep-days-per-month`), once per capacity-uncertainty
   scenario, and returns each country's own realized operating profit (buying/selling
   electricity and hydrogen at market prices, no demand target) plus the dual values
   (shadow prices) on its own capacity constraints.
3. **Cut generation** — every country's profit and duals for one scenario are combined
   into ONE Benders optimality cut covering all of them together (a design kept from
   when it was REQUIRED for correctness, `Formulation.md` §4.6 — now only OPTIONAL
   since demand's removal made the subproblem separable again, but left unchanged and
   still valid): a linear
   upper bound on that scenario's `theta_s`, expressed in the master's binary
   capacity-choice variables (duals become the cut's coefficients — no cross term
   between assets anymore now that demand, the source of the electrolyser's old
   cross-coupling term, is gone, `Formulation.md` §4.5/§4.9.5). One such combined cut is
   added per scenario per iteration. Under `--pareto-cuts`, the duals instead come
   from a solve at a moving core point rather than the trial point (`Formulation.md`
   §4.9.3).
4. **Loop** — resolve the master with the new cuts, resolve the joint subproblem at the
   updated capacities, and repeat until the master's lower bound (HiGHS's own proven
   dual bound, `Formulation.md` §4.6) and the best-found upper bound converge (or an
   iteration cap is hit).

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
