# Machine-Learning-Surrogate-Assisted Industrial Site Investment Planning

A strategic industrial site investment planning framework for optimizing site locations,
technology selection, and capacity sizing across Europe's Core Capacity Calculation Region.
It embeds machine-learning price surrogates into a decomposition-based optimization model to
support scalable, data-driven investment decisions for industrial energy systems.

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

## 2. Industrial Site Investor planning (`site_investor_planning/`, `optimize_site_investor.py`)

You define one or more industrial **sites**, each with its own internal demand for
electricity, heat, cooling and hydrogen, its own minimum green hydrogen share and its own demand
flexibility. For every site the planner decides **where** to build it (which candidate country;
several sites may share one), **which technologies** to install there, and **how big** each one
should be, using the price models above as a price-taker market signal.

| Demand | Main asset(s) | Backup when the new assets don't cover it |
|---|---|---|
| Electricity | Solar PV, wind, battery | Grid import |
| Space heating | Heat pump | Existing gas boiler |
| Low/medium-temperature process heat | Industrial heat pump | Existing gas boiler |
| High-temperature process heat / steam | Electric boiler | Existing gas boiler |
| Cooling | Electric chiller | Existing legacy chiller (on site electricity) |
| Hydrogen | Electrolyser, hydrogen storage | Hydrogen market import |

- `optimize_site_investor.py` is the site's operating LP. Every hour, six separate balances
  must hold, one per demand, each served by its own asset(s). The heat pumps, boiler and chiller draw electricity at their COP.
  The site buys any electricity or hydrogen deficit at the modeled market price plus an
  import fee, and sells any surplus at the market price. Gas boilers and a legacy chiller
  are already on site (no CAPEX, unlimited capacity), so every demand is always met, whatever
  the new assets are. `solve` runs one site on its own, either over a contiguous day range or
  a representative-day sample. `solve_joint` runs every candidate country's site LP
  together in one model (still independent problems, just solved in one call). It's the only
  mode `plan_capacity.py` uses.
- `site_investor_planning/` + `plan_capacity.py` turn the site choice and the nine asset
  sizes into discrete/binary choices under a CAPEX budget, solved via Benders decomposition
  (MILP master + one joint LP subproblem covering every candidate country together).

```bash
python optimize_site_investor.py --zone DE00 --day 5
python plan_capacity.py --countries DE,FR,PL --n-sites 1 --rep-days-per-month 1
python plan_capacity.py --countries DE,FR,PL --sites-file sites.json --rep-days-per-month 1
```

`optimize_site_investor.py`'s standalone `--day`/`--start-day`/`--end-day` CLI also
accepts `--rep-days-per-month N`, to solve on N representative days/month (weighted to
approximate the full year) instead of a contiguous day range. It sizes the site with
`DEFAULT_SITE_CAPACITIES`. `plan_capacity.py` always solves every candidate country together
(`solve_joint`) on representative days, so `--rep-days-per-month` is effectively required there
too, since `solve_joint` has no contiguous-range mode.

### Site demand

A site's hourly demand is one per-unit curve per demand times the site's daily peak:

- `inputs/site_demand.csv` holds the **per-unit curves** for the whole year: 8736 rows (`hour`
  0–8735), each value that hour's demand as a share of the site's daily peak (1.0 = a normal
  day's peak hour; a day can be set higher or lower). The curves are shared by every site. The
  shipped file repeats one daily curve every day, so edit individual days to make them differ:

  ```
  hour,electricity,space_heat,process_heat,steam,cooling,hydrogen
  ```

- Each site's **daily peaks** (MW of heat or cooling for the thermal demands, MW_LHV for
  hydrogen), **minimum green hydrogen share** and **flexibility** belong to the site, not the
  country: a site keeps its demand wherever it is built. New sites start from
  `DEFAULT_SITE_PEAKS_MW`, a 42% green share and ±10% flexibility (`SiteSpec` in
  `site_investor_planning/config.py`).

Sites are given to the planner in a JSON file (`--sites-file`), one entry per site; anything left
out falls back to the defaults:

```json
{"sites": [
  {"name": "Food plant", "peaks_mw": {"electricity": 6.6, "steam": 4.0}, "green_share": 0.42, "flex_fraction": 0.10},
  {"name": "Chemicals", "peaks_mw": {"steam": 9.0, "hydrogen": 3.0}, "green_share": 0.80, "flex_fraction": 0.0}
]}
```

Without a file, `--n-sites N` builds N sites with the default settings, and
`--green-h2-share-pct` / `--demand-flex-pct` change the green share and flexibility of all of
them. The planner prints each site's settings and annual totals at the start of a run and writes
them to `_sites.csv`; the web New run page edits every site on its own card.

`process_heat` is low/medium-temperature process heat and `steam` is high-temperature process
heat / steam. The shipped curves are synthetic: electricity, process heat, steam and hydrogen
run at full load from 6:00 to 22:00 and 60% at night; space heating at full load from 6:00 to
20:00 and 70% otherwise; cooling peaks at 14:00 and drops to about 54% at night. The default
peaks were set so the annual totals match the earlier synthetic demand (50 GWh electricity,
40 GWh process heat, 30 GWh steam, 10 GWh hydrogen, 8 GWh space heating and 13.2 GWh cooling per
year, the German values of the earlier per-country demand). There are no seasons or weekends in
the shipped curves: every day is the same until you edit it.

Every demand is also **flexible**: each hour it may move up or down by up to the site's
flexibility share of its original value (10% by default), as long as the shifts net to zero over
each day, so the day's total energy is unchanged. The site uses this to move load into cheap or
high-renewable hours. A site with 0% flexibility has rigid demand. The exported schedules
include each demand's hourly shift.

### Green hydrogen and the certificate market

At least the site's own share of its annual hydrogen demand must be green (RFNBO), by default
42% (the RED III 2030 target for hydrogen used in industry). Green hydrogen comes from two sources:

- **The site's electrolyser, under the EU RFNBO rules.** Its green electricity must be
  *additional* and *matched every hour* (the temporal-correlation rule that applies from 2030).
  Additional electricity is the site's own new wind/PV, or Guarantees of Origin (GOs) bought from
  additional plants (8 EUR/MWh by default).
- **Certified green hydrogen bought on the market**, at a premium over the hydrogen price
  (120 EUR/MWh, about EUR 4/kg, by default). This keeps the requirement reachable whatever the
  master proposes.

The site also sells GOs (6 EUR/MWh by default) for the wind/PV it exports, except for output it
has already claimed for green hydrogen. The prices live in `GreenH2Params` in
`site_investor_planning/config.py`; each site sets its own share (`0` removes the requirement for
that site; GO sales stay on). The exported schedules include the electrolyser's
green load, the wind/PV claimed for it, GOs bought and sold, and green hydrogen bought.

### Technology and cost assumptions

`site_investor_planning/config.py` holds:

- `CANDIDATE_CATALOG`: one real-world product per asset, with its own absolute CAPEX and
  lifetime. The optimizer builds it as many times as needed (up to the site cap), so capacity
  comes in steps of that product's size. More products can be added per run on the New run
  page. Wind, PV, battery, electrolyser and H2 tank come from
  `Help/Candidates (Edited).docx`'s 2030 candidate-product table. The heat pump, industrial
  heat pump, electric boiler and electric chiller entries are indicative 2030 costs in the
  range of public technology catalogues (e.g. the Danish Energy Agency's). Replace them with
  vendor quotes for a real site. Thermal assets are sized in MW of heat or cooling output.
- `SiteTechParams`: COPs and efficiencies (heat pump 3.0, industrial heat pump 2.5, electric
  boiler 0.99, chiller 4.5), the backup gas boiler's cost (gas 35 EUR/MWh + CO2 90 EUR/t at 90%
  efficiency, about 59 EUR/MWh of heat), the legacy chiller's COP (3.0) and the grid and
  hydrogen import fees (15 and 0 EUR/MWh).
- `CapexAssumptions.site_max_mw`: the most MW of wind, PV, battery, electrolyser and H2 tank
  one site may host. Heat pumps, boilers and chillers are instead capped at 1.25 times the
  site's own peak demand for their service, since their output can't be sold.

### `plan_capacity.py` flags

Full, current list also always available via `python plan_capacity.py --help`.

**Scope**
| Flag | Default | What it does |
|---|---|---|
| `--countries CC,CC,...` | — | Candidate site countries, comma-separated 2-letter codes, e.g. `DE,FR,PL` (mutually exclusive with `--all`) |
| `--all` | — | Every eligible country is a candidate |
| `--sites-file FILE` | — | JSON file defining each site to build (name, daily peaks, green share, flexibility); see *Site demand*. Overrides the three flags below. |
| `--n-sites N` | 1 | Without `--sites-file`: how many default sites to build. The optimizer picks a country for each; sites may share one. |
| `--green-h2-share-pct P` | 42 | Without `--sites-file`: minimum green (RFNBO) share of every site's annual hydrogen demand, in %. `0` = no requirement. |
| `--demand-flex-pct P` | 10 | Without `--sites-file`: hourly demand flexibility of every site, in % of each hour's demand; shifts net to zero over each day. `0` = rigid demand. |

**Budget / CAPEX**
| Flag | Default | What it does |
|---|---|---|
| `--budget EUR` | 1,500,000,000 | Raw/unannualized CAPEX budget, shared across every site. |
| `--discount-rate R` | 0.05 | Discount rate for the capital recovery factor |
| `--lifetime-years N` | catalog's own (30/40/20yr wind/PV/battery, 20/25/25/20yr heat pump/industrial heat pump/electric boiler/chiller, 25/30yr electrolyser/tank) | Overrides every asset's lifetime uniformly (edit `CapexAssumptions.lifetime_years` directly for a per-asset override instead) |

The master places every site in exactly one candidate country, and several sites may share a
country. Internally each (site, country) pair is a separate candidate with its own capacities;
only the chosen pair of each site may host capacity, up to that site's cap for each asset (see
*Technology and cost assumptions*). Sites are otherwise coupled only through the shared budget.
Each site is a price-taker, so two sites in one country don't affect each other's prices. With
more sites, each iteration solves more site LPs (sites × candidate countries), so runs take
longer.

**Capacity-uncertainty scenarios / risk measure**
| Flag | Default | What it does |
|---|---|---|
| `--scenarios S,S,...` | `p100` (the Baseline) | Scenarios to plan over, probabilities renormalized to sum to 1, e.g. `--scenarios p100,unc01,unc04`. One scenario is planned deterministically; two or more use CVaR. |
| `--cvar-alpha A` | 0.8 | CVaR confidence level (0–1), used only with two or more scenarios. |
| `--disabled-assets A,A,...` | none | Exclude asset keys at every site (max MW = 0), e.g. `battery_mw,tank_mw`. Keys: `wind_mw`, `pv_mw`, `battery_mw`, `heat_pump_mw`, `industrial_heat_pump_mw`, `electric_boiler_mw`, `electric_chiller_mw`, `electrolyser_mw`, `tank_mw`. |

**Subproblem**
| Flag | Default | What it does |
|---|---|---|
| `--rep-days-per-month N` | 7 | Solves the joint subproblem on N representative days/month (1–29, weighted to approximate the full year) — `solve_joint` has no contiguous-range mode |

Every candidate country's site LP is solved together, in one joint linopy model
(`optimize_site_investor.solve_joint`); there's no independent-per-country mode. A country
without a site has no demand and no capacity, so it costs nothing.

**Planning scenarios**

By default a run plans over the **Baseline** (`p100`, no capacity shortfall) alone, with 100%
probability, and is solved **deterministically**. To plan under uncertainty, add scenarios; with
two or more the risk measure is **CVaR** at `--cvar-alpha` (default 0.8). There is no
expected-value option.

`inputs/uncertainty_scenarios.json` defines 11 capacity-uncertainty scenarios (`p100` plus
`unc01`–`unc10`), all kept for the dispatch runs that build the price data, and lists the
default (`default_scenarios`: `p100`). Four of them have a positive probability and can be added
to a run. The 10 `unc` scenarios fall into three groups by how much they raise average
electricity prices over `p100` in the candidate countries (hydrogen prices move by at most
EUR 1.6/MWh), and one scenario stands for each group, carrying the group's combined probability
when all four are combined:

| Scenario | Stands for | Electricity price vs `p100` | Probability |
|---|---|---|---|
| `p100` | no capacity shortfall | — | 3.33% |
| `unc09` | `unc09` | +10.5 EUR/MWh | 3.33% |
| `unc01` | `unc01`, `unc02`, `unc03`, `unc05`, `unc08`, `unc10` | +20 to +24 EUR/MWh | 45.33% |
| `unc04` | `unc04`, `unc06`, `unc07` | +29 to +30 EUR/MWh | 48.00% |

With all four combined and CVaR at α = 0.8, the 20% worst tail falls inside `unc04`. To make
another defined scenario available, give it a positive probability in the file.

You can also add **your own scenarios** on the New run page: each has a name, a probability and a
wind and a solar error in % compared with the Baseline, the same in every country. A custom
scenario uses the Baseline's market prices and derates only the site's own wind/PV output by
exactly those errors (the defined `unc` scenarios are rescaled so their worst case is 50%; custom
ones are applied as entered).

**Solve control / output**
| Flag | Default | What it does |
|---|---|---|
| `--max-iters N` | 30 | Benders iteration cap |
| `--gap-tol G` | 0.01 (1%) | Relative Benders convergence gap |
| `--master-time-limit S` | 180 | Wall-time cap (seconds) per master MILP solve. The master gets genuinely hard to solve to proven optimality as cuts accumulate at large scale, so this bounds it instead; the Benders lower bound is still read from HiGHS's own proven dual bound, so the result stays mathematically rigorous even when the search is cut off early. |
| `--output PREFIX` | `outputs/plan` | Output file prefix for `_capacities.csv` (one row per site: its chosen country and capacities)/`_sites.csv` (each site's settings and annual demand)/`_units.csv`/`_convergence.csv`/`_green_h2.csv` (each site's green hydrogen and GO totals per scenario)/(with `--export-schedules`) `_schedule_<site>_<country>_<scenario>.csv` |
| `--export-schedules` | off | Also re-solve at the final chosen capacities and dump each chosen site's representative-day schedule: every demand, asset output, grid and H2 import/export, and backup use |

Each iteration adds two optimality cuts per scenario: one from the subproblem solved at the
master's proposed plan (tight there, so the master can't propose the same plan again without
paying its true cost), and one from a solve at a moving core point (Papadakos-style
Pareto-optimal cuts), which carry information about the rest of the plan space. This isn't
configurable. Core-point values that decay below 1e-3 are set to zero, which keeps the
subproblem numerically stable. If HiGHS fails on the master or a subproblem, it's re-solved once
with presolve off.

### How the Benders solve works

Choosing site locations and sizing all nine assets at once, as one MILP with a full year of
hourly LP dispatch variables per candidate country, doesn't scale. So `plan_capacity.py`
splits it into a master problem and one joint subproblem (covering every candidate country
together), iterating between them:

1. **Master (MILP, `site_investor_planning/master.py`)** picks a country for each site (one
   binary per site and country, exactly one per site) and how many units of each catalog product
   each site builds, subject to the CAPEX budget and the per-site caps. Its objective is annualized
   CAPEX plus a recourse stand-in (`theta`) that starts unconstrained and gets tightened every
   round by the cuts below. There's one `theta_s` per capacity-uncertainty scenario, combined
   via CVaR at `--cvar-alpha` (default 0.8); with a single scenario it is just that scenario's
   recourse (deterministic).
2. **Joint subproblem (LP, every country in one linopy model, `optimize_site_investor.
   solve_joint`)** takes the master's site choice and capacities and solves the
   representative-day site operation (`--rep-days-per-month`), once per capacity-uncertainty
   scenario. It returns each country's operating cost (electricity and hydrogen purchases
   minus sales, backup gas, GOs bought minus sold, green hydrogen premium) and the dual values
   (shadow prices) on its capacity constraints, demand balances and green hydrogen requirement.
3. **Cut generation.** Each scenario's per-country costs and duals become a Benders
   optimality cut: a linear lower bound on that scenario's `theta_s` in the master's capacity
   and site variables. Capacity duals give each asset's coefficient. The duals on everything
   that scales with the site (demand balances, flexibility bands, green hydrogen requirement)
   give the site's coefficient: what switching that site on costs.
4. **Loop.** Re-solve the master with the new cuts, re-solve the subproblem at the updated
   plan, and repeat until the master's lower bound (HiGHS's own proven dual bound) and the
   best-found upper bound converge, or the iteration cap is hit.

This avoids ever building one giant MILP over every country's site and asset choices and
every hour of the model year at once. Each iteration is a small MILP plus one joint,
representative-day LP, which is what makes the discrete/binary search tractable.

## Structure

```
price_model/              demand -> price models
site_investor_planning/   Benders master, candidate sites and grids, site demand,
                          technology and CAPEX assumptions
optimize_site_investor.py standalone / joint site operating LP
plan_capacity.py          Benders CLI driver
economic_dispatch/        LP dispatch engine, vendored locally
data_exchange/            hand-off directory between pipeline stages: dispatch
                          output feeding price-model training, then trained
                          models feeding the Benders planner
inputs/                   zone/network DBs, adjacency and scenario definitions
                          (including uncertainty_scenarios.json)
outputs/                  plan CSVs (git-ignored)
```

## User interface

The web planner is a small Django app in `webui/`. You start it with `webui\app.bat`
(see [From scratch to a running app](#from-scratch-to-a-running-app)), and it opens at
`http://localhost:9000/`. The top bar has four pages.

- **Runs** is the home page. It lists the 100 most recent planning runs with their
  status (running, completed or failed), the candidate countries and technologies in scope,
  the risk measure, the budget, the objective and the time taken. Click a run to open it.
- **New run** is where you set up a planning run. The form has six numbered sections:
  1. **Problem definition:** run name, total CAPEX budget, candidate countries, the number of
     sites to build (up to 6), excluded technologies and the cap on units per product.
     Countries are shown in a table, each with its flag, full name and code, and you click
     one to select it. *Select all* and *Clear* sit above them. Ticking *All eligible
     countries* turns the country picks off.
  2. **Sites:** one card per site to build, with its name, minimum green hydrogen share,
     flexibility and daily peak demand (MW) for each of the six demands, filled with the
     defaults. Only as many cards as *Sites to build* are shown, and each has a **Reset site**
     button. The sites are written to `sites.json` in the run's output folder and passed to the
     planner with `--sites-file`.
  3. **Candidate catalog:** one card per technology listing the products the optimizer may
     build: size (MW), CAPEX and, for the battery and H2 storage, energy (MWh), plus one
     lifetime per technology. You can edit any value, add products with **+ Add candidate**
     (up to 8 per technology) or remove them by clearing their row, and each card has a
     **Reset to default** button. Technologies may have different numbers of products. The
     edits apply to this run only (written to `catalog_overrides.json` in the run's output
     folder and passed to the planner through `PLANNER_CATALOG_OVERRIDES`); the defaults in
     `site_investor_planning/config.py` stay as they are. The run page prices its cost
     breakdown with the run's own catalog.
  4. **Uncertainty scenarios:** only the **Baseline** is ticked by default, at 100%. One card
     per defined scenario has an include tick box, the scenario's probability in percent and a
     short description; open a card to see the wind and solar error % for every country.
     **+ Add scenario** adds your own scenario with a name, probability and one wind and one
     solar error % compared with the Baseline (up to 4; **Remove** takes one out again). Error %
     is the share of the site's own nominal output that is lost, so 0% means no loss. The
     probabilities of the ticked scenarios must add up to 100%; a badge at the top shows the
     total in green when it's right and red when it isn't. Changes apply to this run only.
  5. **Economics:** discount rate, the risk measure and the CVaR confidence level. The risk
     measure follows the scenarios: **Deterministic** with one scenario, **CVaR** with two or
     more. Lifetimes are set per technology in the candidate catalog.
  6. **Solver settings:** representative days per month, optimality gap, maximum
     Benders iterations, master time limit and parallel workers.

  Each section has its own **Reset this section** button. Each scenario card has a
  **Reset scenario** button. **Reset all settings** at the bottom returns the whole form
  to its defaults. **Run plan** starts the run, and the page opens the run's details.
- **Run detail** shows one finished run. It has each site with its chosen country, the headline numbers
  (objective, raw CAPEX, risk measure), installed capacity at each site in MW, the product
  units built, the green hydrogen results (green share reached, green hydrogen produced and
  bought, GOs bought and sold, per scenario and probability-weighted), a map of the candidate
  countries (colored by which demand the capacity serves), the convergence chart of the Benders iterations, the inputs the run used and the
  solver log.
- **Compare** puts two completed runs side by side. Pick them from the lists at the top.
  It shows the installed MW by country for each run and the difference between them.
- **Catalog** shows the site demand, the operating assumptions (COPs, backup costs, import
  fees), the green hydrogen and certificate settings, the demand flexibility, the candidate products and their CAPEX and lifetime assumptions, the discount rate
  and budget defaults, and the uncertainty scenarios with their default probabilities (one
  column per scenario).

Only one run can be in progress at a time. While one is running, the **Run plan** button
on the New run page is disabled and a notice explains why. You can still fill in the form
for the next run.

## Install

You don't need to install anything by hand. `webui\app.bat` creates `.venv` in the
project root and installs everything in `requirements.txt` the first time it runs.

## From scratch to a running app

There are two ways to use this project. Pick the one that matches what you want to do.

**Option A: run the planner with the pretrained models (quick start).** The files the
planner needs are already in this repo: the `inputs/` data, the trained price models in
`data_exchange/02_train_output__benders_input/`, and the scenario definitions. The two
large sample files are downloaded automatically the first time you start the app. So
you only need Python and *Step 5*. Skip steps 0 to 4, which are for Option B.

**Option B: build your own scenarios and train your own models (full rebuild).** Use this
when you change the scenarios, add new dispatch runs, or want to retrain the price
models on your own data. It needs the dispatch engine to run, so steps 0 to 4 are
required in order, and it takes much longer.

This section walks through both. Run all commands from the project root unless a step
says otherwise.

### What you need before you start

- **Python 3.12** (64-bit). Check with `py -3 --version` or `python --version`.
- **Internet access** for the first run of `webui\app.bat`, which downloads the two
  large sample files (about 285 MB in total). See the table below.
- **Git LFS is not needed.** Everything in the table except the two sample files is
  already in git.

### Required inputs

| File | Folder | Where it comes from | What uses it |
|---|---|---|---|
| `zones_2030.parquet`, `networks_2030.parquet`, `marginal_price_electricity_2030.parquet`, `marginal_price_hydrogen_2030.parquet`, `crossborder_electricity_2030.parquet`, `crossborder_hydrogen_2030.parquet`, `hydro_*_2030.parquet`, `smr_production_2030.parquet` | `inputs/` | In git | Dispatch engine and planner |
| `uncertainty_scenarios.json` | `inputs/` | In git | Planner (scenario probabilities and country error factors) |
| `site_demand.csv` | `inputs/` | In git | Planner (per-unit hourly demand curves for the year, shared by all sites; each site's daily peaks come with the site) |
| `elec_adjacency.json`, `h2_adjacency.json` | `inputs/` | In git (rewritten by `build_dataset.py`) | Price models |
| `electricity_model.joblib`, `hydrogen_model.joblib` and the `*_metrics.csv` files | `data_exchange/02_train_output__benders_input/` | In git (written by `train_model.py`) | Planner |
| `elec_samples.parquet`, `h2_samples.parquet` | `data_exchange/01_dispatch_output__train_input/` | Downloaded automatically by `webui\app.bat` from the project's Google Drive. You can also build them with steps 1 and 2. | Planner (`optimize_site_investor.py`) and `train_model.py` |

The `scenarios/` folder inside `data_exchange/01_dispatch_output__train_input/` is
only needed if you rebuild the sample files with steps 1 and 2. You don't need it to
run the app.

### Step 0: set up the environment

Nothing to install by hand. `webui\app.bat` creates `.venv` and installs
`requirements.txt` the first time it runs. For Option B, run `webui\app.bat` once so the
environment exists, then close the **Site Investment Planner** window.

After that, activate the environment from the project root in a terminal:

```bat
.venv\Scripts\activate
```

The commands in steps 1 to 4 use this environment.

### Step 1: run the dispatch scenarios (Option B only)

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

### Step 2: build the feature tables (Option B only)

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

### Step 3: train the price models (Option B only)

```bash
python train_model.py
```

This trains one gradient-boosted model per bidding zone, for electricity and for
hydrogen. The results go to `data_exchange/02_train_output__benders_input/`:
`electricity_model.joblib`, `hydrogen_model.joblib`, and the matching `*_metrics.csv`
files. To retrain just one commodity, use `python train_model.py --only electricity`.

### Step 4: a quick planning check (optional)

```bash
python plan_capacity.py --countries DE,FR --scenarios p100 --rep-days-per-month 1 --max-iters 10
```

This is a small, fast run that checks the whole chain works before you start a full
run. The output goes to `outputs/plan_*.csv`. For the full run, use `--all`, add the scenarios
you want (e.g. `--scenarios p100,unc01,unc04`), and raise `--rep-days-per-month` (see the flags
table above).

### Step 5: start the web planner

Once steps 0–3 are done, start the app.

**Recommended: double-click `app.bat`.** It's in the `webui` folder inside the project,
next to `manage.py`. On this machine the full path is:

```
G:\My Drive\Temp\BSRO\Project 5\webui\app.bat
```

A console window opens and shows each step as it runs. When the server is up, your
browser opens `http://localhost:9000/` by itself. The server runs in its own window
titled **Site Investment Planner**. Keep it open while you use the planner. Close that window
to stop the server.

**From a terminal (optional).** Open a terminal in the project folder and run:

```bat
webui\app.bat
```

The script does the rest, in this order:

1. **Downloads the sample files.** If `elec_samples.parquet` or `h2_samples.parquet` is
   missing from `data_exchange/01_dispatch_output__train_input/`, the script downloads it
   from the project's Google Drive. The links are in `webui\app.bat`. The first run takes
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
