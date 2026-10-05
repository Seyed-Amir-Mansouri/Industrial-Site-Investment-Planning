# Machine-Learning-Surrogate-Assisted Investment Planning of Coupled European Electricity and Hydrogen Systems

Two pieces of work, both built on an upstream LP economic-dispatch engine's output for
the 20-zone Central-European CORE region (NT2030 scenario):

## 1. Price models (`price_model/`)

Per-zone demand → price models (electricity and hydrogen), one gradient-boosted
model per bidding zone, trained on feature tables extracted from the upstream
dispatch engine's per-scenario output (`elec_samples.parquet`/`h2_samples.parquet`
under `data_exchange/01_dispatch_output__train_input/`, built by `price_model/extract.py`
from each scenario's `hourly_balance_{elec,h2}.csv`).

```python
from price_model import electricity_price, hydrogen_price

electricity_price("DE00", 55000)   # price at 55 GW demand
hydrogen_price("AT00", 500)        # H2 price at 500 MWH2 demand
```

Retrain with `python train_model.py`.

## 2. General Investor capacity planning (`g_investor_planning/`, `optimize_g_investor.py`)

Sizes each country's General Investor (electrolyser, wind, PV, battery, H2 tank)
using the price models above as a price-taker market signal, instead of the full
network coupling the upstream dispatch engine solves.

- `optimize_g_investor.py` — a merchant LP: each asset buys or sells electricity
  and hydrogen at the modeled market price, bounded only by its own installed
  capacity, with no downstream demand obligation. `solve` runs one country on its
  own, either over a contiguous day range or a representative-day sample.
  `solve_joint` runs every requested country's LP together in a single model
  (still independent problems, just solved in one call) and is the only mode
  `plan_capacity.py` uses.
- `g_investor_planning/` + `plan_capacity.py` — turns the five asset sizes into a
  discrete/binary choice under a CAPEX budget, solved via Benders decomposition
  (MILP master + one joint LP subproblem covering every included country together,
  `solve_joint` — not independent per-country subproblems).

```bash
python optimize_g_investor.py --zone DE00 --day 5
python plan_capacity.py --countries DE,FR,PL --rep-days-per-month 1
```

`optimize_g_investor.py`'s standalone `--day`/`--start-day`/`--end-day` CLI also
accepts `--rep-days-per-month N`, to solve on N representative days/month (weighted
to approximate the full year) instead of a contiguous day range. `plan_capacity.py`
always solves every included country together (`solve_joint`) on representative days,
so `--rep-days-per-month` is effectively required there too — `solve_joint` has no
contiguous-range mode.

CAPEX/lifetime figures in `g_investor_planning/config.py::CANDIDATE_CATALOG` come from
`Help/Candidates (Edited).docx`'s 2030 candidate-product table, trimmed down to the
single smallest real-world MW size per asset (electrolyser/wind/PV/battery/tank), each
with its own absolute CAPEX and lifetime — a real cited source, not a vendor quote.

### `plan_capacity.py` flags

Full, current list also always available via `python plan_capacity.py --help`.

**Scope**
| Flag | Default | What it does |
|---|---|---|
| `--countries CC,CC,...` | — | Comma-separated 2-letter country codes, e.g. `DE,FR,PL` (mutually exclusive with `--all`) |
| `--all` | — | Plan every eligible country |

**Budget / CAPEX**
| Flag | Default | What it does |
|---|---|---|
| `--budget EUR` | 500,000,000 | System-wide raw/unannualized CAPEX budget, shared across every included country. It's the only constraint the master applies beyond picking one candidate size per asset per country. |
| `--discount-rate R` | 0.05 | Discount rate for the capital recovery factor |
| `--lifetime-years N` | catalog's own (25/30/40/20/30yr for electrolyser/wind/PV/battery/tank) | Overrides every asset's lifetime uniformly (edit `CapexAssumptions.lifetime_years` directly for a per-asset override instead) |

Every asset choice is independent, coupled to the others only through the shared
budget — there's no minimum-electrolyser requirement, no per-country demand floor,
and no separate cap on how many assets a country can build.

**Capacity-uncertainty scenarios / risk measure**
| Flag | Default | What it does |
|---|---|---|
| `--cvar-alpha A` | 0.8 | Risk measure: CVaR at confidence level `A` (0–1) across the 11 capacity-uncertainty scenarios in `inputs/uncertainty_scenarios.json` (`p100` + `unc01`–`unc10`, non-uniform probabilities — 6 scenarios at ~3.33% each summing to 20%, 5 at 16% each, so the α=0.8 tail lands exactly on those 6). |
| `--scenarios S,S,...` | all 11 | Restrict to a subset of scenarios (probabilities renormalized to sum to 1), e.g. `--scenarios p100` for a single deterministic baseline run ("on-plan"). |
| `--disabled-assets A,A,...` | none | Exclude asset keys from every country's candidate selection (max MW = 0), e.g. `battery_mw,tank_mw`. |

**Subproblem**
| Flag | Default | What it does |
|---|---|---|
| `--rep-days-per-month N` | 7 | Solves the joint subproblem on N representative days/month (1–29, weighted to approximate the full year) — `solve_joint` has no contiguous-range mode |

Every country's subproblem is solved together, in one joint linopy model
(`optimize_g_investor.solve_joint`); there's no independent-per-country mode.
No downstream hydrogen demand is modeled anywhere in this pipeline — each asset is a
merchant participant, trading purely at the modeled market prices. PV and wind each
sell independently into the electricity market; the electrolyser buys electricity and
sells hydrogen; the battery buys/sells electricity; the H2 tank buys/sells hydrogen —
each bounded only by its own installed capacity, more like five separate one-asset
investments than one co-located microgrid with a shared site balance or a shared
grid/pipeline connection limit. A country can freely build wind/PV/battery/tank with
zero electrolyser (a standalone merchant power-and-storage plant), or the reverse.
`optimize_g_investor.solve`, the standalone single-country CLI, follows the same
merchant model.

**Solve control / output**
| Flag | Default | What it does |
|---|---|---|
| `--max-iters N` | 30 | Benders iteration cap |
| `--gap-tol G` | 0.01 (1%) | Relative Benders convergence gap |
| `--master-time-limit S` | 180 | Wall-time cap (seconds) per master MILP solve. The master gets genuinely hard to solve to proven optimality as cuts accumulate at large scale, so this bounds it instead; the Benders lower bound is still read from HiGHS's own proven dual bound, so the result stays mathematically rigorous even when the search is cut off early. |
| `--output PREFIX` | `outputs/plan` | Output file prefix for `_capacities.csv`/`_convergence.csv`/(with `--export-schedules`) `_schedule_<country>.csv` |
| `--export-schedules` | off | Also re-solve at the final chosen capacities and dump each included country's representative-day schedule |

Every optimality cut is built from a subproblem solve at a moving core point
(Papadakos-style Pareto-optimal cuts) rather than the raw trial point — this isn't
configurable; a plain trial-point cut converges far slower (LP dual degeneracy at the
all-zero starting point gives valid but misleading cuts) and was removed entirely once
Pareto-optimal cuts proved reliably fast.

### How the Benders solve works

Sizing all five assets (electrolyser, wind, PV, battery, H2 tank) for every country at
once, as one MILP with a full year of hourly LP dispatch variables per country, doesn't
scale — so `plan_capacity.py` splits it into a master problem and one joint
subproblem (covering every included country together), iterating between them:

1. **Master (MILP, `g_investor_planning/master.py`)** — picks one candidate MW value per
   asset per country (a binary one-hot choice over `CANDIDATE_CATALOG`'s single
   real-world candidate per asset, with its own absolute CAPEX and lifetime, unbounded
   above — no per-candidate unit cap), subject to the annualized, system-wide CAPEX
   budget. Its objective is annualized CAPEX plus a recourse stand-in (`theta`) that
   starts unconstrained and gets tightened every round by the cuts below. There's one
   `theta_s` per capacity-uncertainty scenario, combined via CVaR at `--cvar-alpha`
   (default 0.8) rather than a single shared scalar.
2. **Joint subproblem (LP, every country in one linopy model, `optimize_g_investor.
   solve_joint`)** — for the master's chosen capacities, solves the merchant
   representative-day dispatch (`--rep-days-per-month`), once per capacity-uncertainty
   scenario, and returns each country's own realized operating profit (buying and
   selling electricity and hydrogen at market prices) plus the dual values (shadow
   prices) on its own capacity constraints.
3. **Cut generation** — each scenario's per-country profits and duals are combined
   into one Benders optimality cut covering every included country together: a linear
   upper bound on that scenario's `theta_s`, expressed in the master's binary
   capacity-choice variables, with the duals as the cut's coefficients, evaluated at a
   moving Pareto-optimal core point rather than the raw trial point (see above). One
   such combined cut is added per scenario per iteration.
4. **Loop** — resolve the master with the new cuts, resolve the joint subproblem at the
   updated capacities, and repeat until the master's lower bound (HiGHS's own proven
   dual bound) and the best-found upper bound converge, or the iteration cap is hit.

This avoids ever building one giant MILP over every country's asset choices and every
hour of the model year at once — each iteration is a small MILP plus one joint,
representative-day LP, which is what makes the discrete/binary capacity search
tractable.

## Structure

```
price_model/              demand -> price models
g_investor_planning/      Benders master + candidate grids + CAPEX assumptions
optimize_g_investor.py    standalone / joint General Investor LP
plan_capacity.py          Benders CLI driver
economic_dispatch/        LP dispatch engine, vendored locally
data_exchange/            hand-off directory between pipeline stages: dispatch
                          output feeding price-model training, then trained
                          models feeding the Benders planner
inputs/                   zone/network DBs, adjacency and scenario definitions
                          (including uncertainty_scenarios.json)
outputs/                  plan CSVs (git-ignored)
```

## Install

```bash
pip install -r requirements.txt
```
