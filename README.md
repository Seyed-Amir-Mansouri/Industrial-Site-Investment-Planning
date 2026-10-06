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

## From scratch to a running app

This section walks through everything you need to go from a fresh copy of the repo to
the web planner running. Run all commands from the project root unless a step says
otherwise.

### What you need before you start

- **Python 3.12** (64-bit). Check with `py -3 --version` or `python --version`.
- **Internet access** for the first run of `webuipp.bat`, which downloads the two
  large sample files (about 285 MB in total). See the table below.
- **Git LFS is not needed.** Everything in the table except the two sample files is
  already in git.

### Required inputs

| File | Folder | Where it comes from | What uses it |
|---|---|---|---|
| `zones_2030.parquet`, `networks_2030.parquet`, `marginal_price_electricity_2030.parquet`, `marginal_price_hydrogen_2030.parquet`, `crossborder_electricity_2030.parquet`, `crossborder_hydrogen_2030.parquet`, `hydro_*_2030.parquet`, `smr_production_2030.parquet` | `inputs/` | In git | Dispatch engine and planner |
| `uncertainty_scenarios.json` | `inputs/` | In git | Planner (scenario probabilities and country error factors) |
| `elec_adjacency.json`, `h2_adjacency.json` | `inputs/` | In git (rewritten by `build_dataset.py`) | Price models |
| `electricity_model.joblib`, `hydrogen_model.joblib` and the `*_metrics.csv` files | `data_exchange/02_train_output__benders_input/` | In git (written by `train_model.py`) | Planner |
| `elec_samples.parquet`, `h2_samples.parquet` | `data_exchange/01_dispatch_output__train_input/` | Downloaded automatically by `webuipp.bat` from the project's Google Drive. You can also build them with steps 1 and 2. | Planner (`optimize_g_investor.py`) and `train_model.py` |

The `scenarios/` folder inside `data_exchange/01_dispatch_output__train_input/` is
only needed if you rebuild the sample files with steps 1 and 2. You don't need it to
run the app.

### Step 0: create the virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`webui\app.bat` also creates `.venv` if it's missing, so this step is only needed for
the commands below.

### Step 1: run the dispatch scenarios

```bash
python run_capacity_scenarios.py
```

This runs the dispatch engine once for the baseline and for every training capacity
scenario. Each run writes two files:

```
data_exchange/01_dispatch_output__train_input/scenarios/<name>/hourly_balance_elec.csv
data_exchange/01_dispatch_output__train_input/scenarios/<name>/hourly_balance_h2.csv
```

It takes a long time, because each scenario is a full-year LP. To run only some of
them, pass a comma-separated list:

```bash
python run_capacity_scenarios.py --scenarios p100,unc01
```

### Step 2: build the feature tables

```bash
python build_dataset.py
```

This pools every scenario folder from step 1 and writes the two sample tables that the
price models and the Benders planner read:

```
data_exchange/01_dispatch_output__train_input/elec_samples.parquet
data_exchange/01_dispatch_output__train_input/h2_samples.parquet
```

It also rewrites the zone adjacency files in `inputs/`. To pool only some scenarios,
use `--scenarios p100,unc01`.

### Step 3: train the price models

```bash
python train_model.py
```

This trains one gradient-boosted model per bidding zone, for electricity and for
hydrogen. The results go to `data_exchange/02_train_output__benders_input/`:
`electricity_model.joblib`, `hydrogen_model.joblib`, and the matching `*_metrics.csv`
files. To retrain just one commodity, use `python train_model.py --only electricity`.

### Step 4: a quick planning check (optional)

```bash
python plan_capacity.py --countries DE,FR --scenarios p100 --rep-days-per-month 1 --max-iters 5
```

This is a small, fast run that checks the whole chain works before you start a full
run. The output goes to `outputs/plan_*.csv`. For the full run, drop `--scenarios`,
use `--all`, and raise `--rep-days-per-month` (see the flags table above).

### Step 5: start the web planner

Once steps 0–3 are done, start the app.

**Recommended: double-click `app.bat`.** It's in the `webui` folder inside the project,
next to `manage.py`. On this machine the full path is:

```
G:\My Drive\Temp\BSRO\Project 5\webui\app.bat
```

A console window opens and shows each step as it runs. When the server is up, your
browser opens `http://localhost:9000/` by itself. The server runs in its own window
titled **Capacity Planner**. Keep it open while you use the planner. Close that window
to stop the server.

**From a terminal (optional).** Open a terminal in the project folder and run:

```bat
webui\app.bat
```

The script does the rest, in this order:

1. **Downloads the sample files.** If `elec_samples.parquet` or `h2_samples.parquet` is
   missing from `data_exchange/01_dispatch_output__train_input/`, the script downloads it
   from the project's Google Drive. The links are in `webuipp.bat`. The first run takes
   a while because of the file sizes. Files that already exist are left alone. A download
   under 1 MB is treated as a failed download: the script deletes it and stops.
2. **Finds Python** (`py -3`, then `python`), creates `.venv` if it's missing, and installs
   anything missing from `requirements.txt`.
3. **Applies the Django migrations.**
4. **Starts the server** on port 9000 and opens `http://localhost:9000/` once the server
   answers. You don't need to run `manage.py` yourself.

If the browser doesn't open, go to `http://localhost:9000/` by hand. If the download step
fails, check your internet connection, or download the two files by hand into the folder
above and run the script again.

### Troubleshooting

- **The app exits straight away with an illegal instruction (exit code 132).** Your CPU
  doesn't support AVX2. `requirements.txt` already asks for `polars[rtcompat]`, which
  avoids this. If you installed packages by hand, install that version instead.
- **Port 9000 is in use or reserved.** Stop the other process, or change the port in
  `webui\app.bat` and in `CSRF_TRUSTED_ORIGINS` in `webui\planner_ui\settings.py`.
