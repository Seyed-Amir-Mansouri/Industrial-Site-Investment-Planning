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
electricity, heating, cooling and hydrogen, its own minimum green hydrogen share and its own demand
flexibility. For every site the planner decides **where** to build it (which candidate country;
several sites may share one), **which technologies** to install there, and **how big** each one
should be, using the price models above as a price-taker market signal.

| Demand | Main asset(s) | If the new assets don't cover it |
|---|---|---|
| Electricity | Solar PV, wind, battery, biogas CHP | Grid import |
| Space heating (keeping buildings warm) | Heat pump, hot-water storage (CHP, solar thermal and heat-recovery chiller heat can also serve it) | Unmet, at the unmet-demand penalty |
| Space cooling (air conditioning of buildings) | AC chiller, chilled-water / ice storage (the absorption chiller can also serve it) | Unmet, at the unmet-demand penalty |
| Process heat (up to 150 °C: washing, drying, pasteurising) | Industrial heat pump, solar thermal (CHP and heat-recovery chiller heat can also serve it) | Unmet, at the unmet-demand penalty |
| Steam (above 150 °C: steam networks, sterilising, reactors) | Electric boiler, thermal battery | Unmet, at the unmet-demand penalty |
| Process cooling (machines, products, cold stores) | Process chiller, heat-recovery chiller, absorption chiller | Unmet, at the unmet-demand penalty |
| Hydrogen | Electrolyser, hydrogen storage | Hydrogen market import |

- `optimize_site_investor.py` is the site's operating LP. Every hour, seven separate balances
  must hold, one per demand, each served by its own asset(s). The heat pumps, boiler and chillers draw electricity at their COP.
  Some assets link balances: the biogas CHP burns bought biogas for electricity plus space or
  process heat, solar thermal serves space or process heat, the heat-recovery chiller cools the
  process and sends its rejected heat to space or process heat, the absorption chiller turns
  process heat into space or process cooling, and the thermal battery charges with electricity
  and discharges steam. Hot-water and chilled-water / ice storage shift space heating and space
  cooling within each day. Heat that the CHP, solar thermal or heat-recovery chiller can't use is
  released.
  The site buys any electricity or hydrogen deficit at the modeled market price plus an
  import fee, and sells any surplus at the market price. There is no existing plant on
  site: heat or cooling the new assets can't supply is left unmet and charged a penalty
  (1,000,000 EUR/MWh by default), so every hour stays solvable whatever the new assets are. The
  penalty is far above the yearly cost of any asset per MW, so the planner never leaves demand
  unmet to save on building when the site caps allow enough capacity. Each run reports any unmet demand per site in
  `_unmet_demand.csv`. `solve` runs one site on its own, either over a contiguous day range or
  a representative-day sample. `solve_joint` runs every candidate country's site LP
  together in one model (still independent problems, just solved in one call). It's the only
  mode `plan_capacity.py` uses.
- `site_investor_planning/` + `plan_capacity.py` turn the site choice and the 17 asset
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

A site's hourly demand is one per-unit curve per demand times the site's annual peak:

- `inputs/site_demand.csv` holds the **per-unit curves** for the whole year: 8736 rows (`hour`
  0–8735), each value that hour's demand as a share of the site's annual peak, so every curve's
  highest hour is 1.0. The curves are shared by every site:

  ```
  hour,electricity,space_heat,space_cool,process_heat,steam,process_cool,hydrogen
  ```

- Each site's **peaks** (MW of heat or cooling for the thermal demands, MW_LHV for
  hydrogen), **minimum green hydrogen share**, **flexibility of each demand**, **active assets**
  and optional **maximum MW per asset** belong to the site, not the country: a site keeps them
  wherever it is built. New sites start from `DEFAULT_SITE_PEAKS_MW`, a 42% green share, ±10%
  flexibility on every demand, every asset active and the default caps (`SiteSpec` in
  `site_investor_planning/config.py`). An inactive asset gets a 0 MW cap at that site, and a
  maximum MW replaces the default cap.

Sites are given to the planner in a JSON file (`--sites-file`), one entry per site; anything left
out falls back to the defaults:

```json
{"sites": [
  {"name": "Food plant", "peaks_mw": {"electricity": 6.6, "steam": 4.0}, "green_share": 0.42,
   "flex_fraction": {"electricity": 0.20, "steam": 0.0}},
  {"name": "Chemicals", "peaks_mw": {"steam": 9.0, "hydrogen": 3.0}, "green_share": 0.80, "flex_fraction": 0.0,
   "active_assets": ["pv_mw", "electric_boiler_mw", "thermal_battery_mw", "electrolyser_mw", "tank_mw"],
   "max_mw": {"pv_mw": 15}}
]}
```

Without a file, `--n-sites N` builds N sites with the default settings, and
`--green-h2-share-pct` / `--demand-flex-pct` change the green share and flexibility of all of
them. `flex_fraction` is either one share for every demand or a share per demand (missing
demands keep 10%). The planner prints each site's settings and annual totals at the start of a run
and writes them to `_sites.csv`; the web New run page edits every site on its own card.

`process_heat` is process heat up to about 150 °C, `steam` is heat or steam above it,
`space_cool` is air conditioning of buildings and `process_cool` is cooling of machines, products
and cold stores. The shipped curves are a synthetic Central European year starting on a Monday.
Electricity, process heat, steam, process cooling and hydrogen follow two shifts on weekdays,
with quieter weekends and public holidays, a slowdown over Christmas and a two-week maintenance
stop in August (steam and hydrogen run more continuously). Space heating follows the outdoor
temperature, high in winter and close to zero in summer; space cooling appears only on warm days,
mostly on summer afternoons. The default annual peaks are 8 MW electricity, 4 MW space heating,
3 MW space cooling, 7 MW process heat, 5 MW steam, 2 MW process cooling and 1.5 MW hydrogen,
giving about 50 GWh electricity, 8.1 GWh space heating, 1.4 GWh space cooling, 39.3 GWh process
heat, 30 GWh steam, 11.3 GWh process cooling and 9.2 GWh hydrogen per year.

Every demand is also **flexible**: each hour it may move up or down by up to the site's
flexibility share for that demand (10% by default), as long as the shifts net to zero over each
day, so the day's total energy is unchanged. The site uses this to move load into cheap or
high-renewable hours. A demand with 0% flexibility is rigid. The exported schedules
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
  `Help/Candidates (Edited).docx`'s 2030 candidate-product table. The heat pump, AC chiller,
  industrial heat pump, electric boiler, process chiller, biogas CHP (1 MW_el, EUR 1.2M), hot-water
  storage (1 MW / 6 MWh, EUR 150k), chilled-water / ice storage (0.5 MW / 3 MWh, EUR 120k), solar
  thermal (1 MW_th peak, EUR 450k), thermal battery (1 MW / 8 MWh, EUR 700k), heat-recovery chiller
  (0.5 MW, EUR 300k) and absorption chiller (0.5 MW, EUR 200k) entries are indicative 2030 costs in the
  range of public technology catalogues (e.g. the Danish Energy Agency's). Replace them with
  vendor quotes for a real site. Thermal assets are sized in MW of heat or cooling output, the
  stores in MW of charge and discharge, and the CHP in MW of electricity. Each storage product
  also has its energy (MWh): a site's store holds the MWh of the products built (battery 2 h,
  hot-water and chilled-water / ice 6 h, thermal battery 8 h and H2 tank 16.6 h in the defaults),
  so products with different durations can be mixed.
- `SiteTechParams`: COPs and efficiencies (heat pump 3.0, AC chiller 3.5, industrial heat pump
  2.5, electric boiler 0.99, process chiller 4.5, heat-recovery chiller 3.0 with all rejected heat
  usable, absorption chiller 0.7 per MWh of process heat), the biogas CHP (40% electric, up to 45%
  heat, biogas at 60 EUR/MWh), the thermal stores' efficiency (hot-water, chilled-water / ice and
  thermal battery 95%; solar thermal follows the site's PV profile), the unmet heat/cooling demand
  penalty (1,000,000 EUR/MWh) and the grid and hydrogen import fees (15 and 0 EUR/MWh).
- `CapexAssumptions.cap_rules` (`CAP_RULES`): the default most MW of each asset one site may host,
  large enough for the asset to cover on its own every peak it can serve: a factor times the
  site's weighted annual peaks of the demands it serves, rounded up to a whole number of its
  smallest product and then to a whole MW. Heat and cooling assets and their stores get 1.25
  times the peak they serve (25% headroom); the biogas CHP 1.25 times the larger of the
  electricity peak and the heat for space heating, process heat and the absorption chiller's
  drive (both cooling peaks divided by its 0.7 COP), in MW of electricity (x 0.40 / 0.45); the
  industrial heat pump 1.25 times process heat plus that drive heat; the thermal battery 1.25
  times the steam peak divided by its 95% efficiency; solar thermal 1.25 times space heating plus
  process heat; the absorption chiller 1.25 times both cooling peaks; wind and PV 7.5 times and
  the battery 5 times the electricity peak (they also sell to the market); the electrolyser 1.25
  times the hydrogen peak divided by its 68% efficiency (it is sized in MW of electricity) and the
  H2 tank 1.25 times the hydrogen peak. At the default peaks that is 60 MW wind and PV, 40 MW
  battery, 21 MW CHP, 18 MW industrial heat pump, 3 MW electrolyser and 2 MW tank. With these
  defaults every demand an active asset firmly serves is covered and the plan leaves nothing
  unmet. A demand with a 0 MW peak gives its assets a 0 MW default. A site's own `max_mw`
  replaces the rule (`default_max_mw` and `site_max_mw` in `candidates.py`).

### `plan_capacity.py` flags

Full, current list also always available via `python plan_capacity.py --help`.

**Scope**
| Flag | Default | What it does |
|---|---|---|
| `--countries CC,CC,...` | — | Candidate site countries, comma-separated 2-letter codes, e.g. `DE,FR,PL` (mutually exclusive with `--all`) |
| `--all` | — | Every eligible country is a candidate |
| `--sites-file FILE` | — | JSON file defining each site to build (name, annual peaks, green share, flexibility per demand, active assets, maximum MW); see *Site demand*. Overrides the three flags below. |
| `--n-sites N` | 1 | Without `--sites-file`: how many default sites to build. The optimizer picks a country for each; sites may share one. |
| `--green-h2-share-pct P` | 42 | Without `--sites-file`: minimum green (RFNBO) share of every site's annual hydrogen demand, in %. `0` = no requirement. |
| `--demand-flex-pct P` | 10 | Without `--sites-file`: hourly flexibility of every demand at every site, in % of each hour's demand; shifts net to zero over each day. `0` = rigid demand. |

**Budget / CAPEX**
| Flag | Default | What it does |
|---|---|---|
| `--budget EUR` | 1,500,000,000 | Raw/unannualized CAPEX budget, shared across every site. |
| `--no-budget` | off | Plan without any CAPEX budget limit (can't be combined with `--budget`). |
| `--discount-rate R` | 0.05 | Discount rate for the capital recovery factor |
| `--lifetime-years N` | catalog's own (30/40/20yr wind/PV/battery, 20/20/25/25/20yr heat pump/AC chiller/industrial heat pump/electric boiler/process chiller, 20/30/25/25/25/20/25yr CHP/hot-water storage/chilled-water storage/solar thermal/thermal battery/heat-recovery chiller/absorption chiller, 25/30yr electrolyser/tank) | Overrides every asset's lifetime uniformly (edit `CapexAssumptions.lifetime_years` directly for a per-asset override instead) |

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
| `--scenarios S,S,...` | `p100` (the Baseline) | Scenarios to plan over, probabilities renormalized to sum to 1. Scenarios other than the Baseline are defined in the `PLANNER_SCENARIO_OVERRIDES` file the web app writes. One scenario is planned deterministically; two or more use CVaR. |
| `--cvar-alpha A` | 0.8 | CVaR confidence level (0–1), used only with two or more scenarios. |
| `--disabled-assets A,A,...` | none | Exclude asset keys at every site (max MW = 0), e.g. `battery_mw,tank_mw`. Keys: `wind_mw`, `pv_mw`, `battery_mw`, `heat_pump_mw`, `ac_chiller_mw`, `industrial_heat_pump_mw`, `electric_boiler_mw`, `electric_chiller_mw` (process chiller), `chp_mw`, `hot_water_storage_mw`, `cold_storage_mw`, `solar_thermal_mw`, `thermal_battery_mw`, `heat_recovery_chiller_mw`, `absorption_chiller_mw`, `electrolyser_mw`, `tank_mw`. |

**Subproblem**
| Flag | Default | What it does |
|---|---|---|
| `--rep-days-per-month N` | 7 | Solves the joint subproblem on N representative days/month (1–29, weighted to approximate the full year) — `solve_joint` has no contiguous-range mode |

Every candidate country's site LP is solved together, in one joint linopy model
(`optimize_site_investor.solve_joint`); there's no independent-per-country mode. A country
without a site has no demand and no capacity, so it costs nothing.

**Planning scenarios**

By default a run plans over the **Baseline** (`p100`, capacities as planned) alone, with 100%
probability, and is solved **deterministically**. To plan under uncertainty, add scenarios; with
two or more the risk measure is **CVaR** at `--cvar-alpha` (default 0.8). There is no
expected-value option.

`inputs/uncertainty_scenarios.json` defines only the Baseline (`p100`) and lists it as the
default (`default_scenarios`).

Further scenarios are your own, added on the New run page: each has a name, a probability and a
change in installed wind and in installed solar capacity, in % compared with the Baseline, for
each country: positive for more capacity than planned, negative for less. A scenario is priced
in two steps:

1. It takes the price-model inputs of the trained dispatch scenario whose country capacities are
   closest to its own.
2. It rescales each country's wind and solar generation, and the residual load that follows,
   to the scenario's capacities (+20% gives 120% of the Baseline, -20% gives 80%).

The price models then predict the electricity and hydrogen prices from those inputs. The site's
own generation and storage are added to its zone's inputs in every scenario, and its demand and
capacity factors are the same in every scenario, so a scenario changes the plan only through
prices. Each country's change must stay within the range
the price models were trained on: from the lowest level in the training scenarios (about -69% to
-80% depending on the country) up to 130% of the Baseline's capacity (+30%). The New run page
shows that range under every box; it is read from the training samples (`elec_samples.parquet`). The web app passes them to the planner in the run's scenario overrides file
(`PLANNER_SCENARIO_OVERRIDES`, entries marked `"custom": true`).

**Solve control / output**
| Flag | Default | What it does |
|---|---|---|
| `--max-iters N` | 100 | Benders iteration cap |
| `--gap-tol G` | 0.01 (1%) | Relative Benders convergence gap |
| `--master-time-limit S` | 180 | Wall-time cap (seconds) per master MILP solve. The master gets genuinely hard to solve to proven optimality as cuts accumulate at large scale, so this bounds it instead; the Benders lower bound is still read from HiGHS's own proven dual bound, so the result stays mathematically rigorous even when the search is cut off early. |
| `--core-cuts` / `--no-core-cuts` | on | Also dispatch each placed site at a moving Pareto core point every iteration and add that cut too (two cuts per scenario and site instead of one). `--no-core-cuts` keeps only the trial-point cut. |
| `--output PREFIX` | `outputs/plan` | Output file prefix for `_capacities.csv` (one row per site: its chosen country, MW per asset and MWh per storage asset)/`_sites.csv` (each site's settings and annual demand)/`_units.csv`/`_convergence.csv`/`_green_h2.csv` (each site's green hydrogen and GO totals per scenario)/`_unmet_demand.csv` (each site's unmet heat and cooling demand per scenario)/(with `--export-schedules`) `_schedule_<site>_<country>_<scenario>.csv` |
| `--export-schedules` | off | Also re-solve at the final chosen capacities and dump each chosen site's representative-day schedule: every demand, asset output, grid and H2 import/export, and any unmet demand |

Each iteration adds one optimality cut per scenario and site, for the country the site is
placed in, from the subproblem solved at the master's proposed plan (tight there, so the master
can't propose the same plan again without paying its true cost). Countries a site isn't placed
in aren't dispatched and keep the cuts from earlier visits. By default each placed site also
gets a cut from a solve at a moving core point (Papadakos-style Pareto-optimal cuts);
core-point values that decay below 1e-3 are set to zero, which keeps the subproblem numerically
stable. If HiGHS fails on the master or a subproblem, it's re-solved once with presolve off; a
master that still fails is re-solved with a looser integer feasibility tolerance (1e-4), which
the steep cuts of the large unmet penalty can need. For the same reason the master works in
million EUR: in EUR its cut slopes reach about 1e11 per MW and HiGHS returns wrong bounds.

### How the Benders solve works

Choosing site locations and sizing all nine assets at once, as one MILP with a full year of
hourly LP dispatch variables per candidate country, doesn't scale. So `plan_capacity.py`
splits it into a master problem and one subproblem per site and scenario (the site's dispatch in
the country the master placed it in), iterating between them:

1. **Master (MILP, `site_investor_planning/master.py`)** picks a country for each site (one
   binary per site and country, exactly one per site) and how many units of each catalog product
   each site builds, subject to the CAPEX budget and the per-site caps. Its objective is annualized
   CAPEX plus a recourse stand-in (`theta`) per site, country and scenario that gets tightened
   every round by the cuts below. Before the first round each site is dispatched once in every
   country with the site off and nothing built (one LP per site, country and scenario). That costs
   nothing, and its cut keeps a site's `theta` at or above 0 in every country it isn't placed in,
   while its slopes give the master a first estimate of placing the site there. Each site is also
   dispatched once in every country at its maximum capacities; that cut bounds how much extra
   capacity can lower the site's cost, so the master can't overbuild against the steep slopes of
   the first one. The master also holds hourly heat and cooling balance rows for each site and
   country: a relaxed dispatch of the heat and cooling assets (storage limited only by its MW)
   that must meet each service's demand, less its flexibility band, in a set of hours. They
   start at the site's peak hours and grow by every hour a trial plan left demand unmet, but
   only hours the site's maximum capacities cover in every scenario. The 1,000,000 EUR/MWh
   unmet penalty makes the cuts so steep that the master would otherwise take many rounds to
   learn how much to build; with these rows its first plans already leave nothing unmet. If
   they make the master infeasible (a budget too small to meet demand) they are dropped and the
   run goes on with cuts only. The scenarios' totals are combined
   via CVaR at `--cvar-alpha` (default 0.8); with a single scenario it is just that scenario's
   recourse (deterministic).
2. **Subproblems (LP, one HiGHS model per site, country and scenario, `optimize_site_investor.
   SiteLP`)** take the master's capacities for each site in its chosen country and solve the
   representative-day site operation (`--rep-days-per-month`), once per capacity-uncertainty
   scenario. Each returns the site's operating cost (electricity and hydrogen purchases
   minus sales, unmet-demand penalty, GOs bought minus sold, green hydrogen premium) and the
   sensitivities of that cost to each asset's MW and to switching the site on. Each LP is built
   once, with every asset's MW and the site's on/off as columns fixed at the values being solved;
   a new plan only changes those bounds and the electricity and hydrogen prices (the price proxy
   sees the site's own capacity), and HiGHS restarts from the previous solution. The sensitivities
   are the reduced costs of those fixed columns. Each site, country and scenario always runs in
   the same worker process, which keeps its LP between iterations. Sites don't interact in the
   dispatch, so solving them separately gives the same costs as one joint model, at a fraction of
   the memory.
3. **Cut generation.** Each placed site's cost and duals in each scenario become a Benders
   optimality cut: a linear lower bound on that site's `theta` in its chosen country, in the
   master's capacity and site variables for that country. Each asset's sensitivity gives its
   coefficient, and the site's sensitivity (it scales the demand balances, flexibility bands and
   green hydrogen requirement) gives the site's coefficient: what switching that site on costs.
   Sites with identical settings (peaks, green share, flexibility, active assets and maximum MW) have the same cost in a
   country, so they share one LP there, a point both need is solved once, and every cut one of
   them gets is also given to the others. The master also places identical sites in country order
   (the second in the same or a later country than the first), so it doesn't search mirror-image
   plans.
4. **Loop.** Re-solve the master with the new cuts, re-solve the subproblem at the updated
   plan, and repeat until the master's lower bound (HiGHS's own proven dual bound) and the
   best-found upper bound converge, or the iteration cap is hit.

This avoids ever building one giant MILP over every country's site and asset choices and
every hour of the model year at once. Each iteration is a small MILP plus one
representative-day LP per site and scenario, which is what makes the discrete/binary search
tractable.

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
- **New run** is where you set up a planning run. The form has seven numbered sections:
  1. **Problem definition:** run name and the CAPEX budget. **Limited budget** is ticked by
     default; untick it to plan without a budget limit (`--no-budget`).
  2. **Sites:** one card per site, starting with one; **+ Add site** adds more (up to 6) and
     **Remove site** deletes one. Each card has the site's name, minimum green hydrogen share, a
     table with the annual peak (MW), the flexibility (± %) and the coverage of each of the seven
     demands, and its **active assets**. Every peak and flexibility starts at 0 and every asset
     unticked; enter the site's demand and tick the assets it may build. Each asset shows the demands it serves and a *Max MW* box, filled from the default
     rule (whole MW) and following the peaks until you type your own value (decimals are fine).
     The **Coverage** row colors each demand: green *Covered* when the active assets' firm
     capacity reaches every peak at once, at the most their Max MW lets the catalog products
     build and counting the CHP's heat and the absorption chiller once across all the demands
     they serve (a covered demand is never left unmet in the plan), amber *Partly* when it falls short or only solar thermal,
     recovered heat or storage serves it, red *Unmet* when nothing active serves it (all of it
     would be charged the unmet-demand penalty), blue *Market* for electricity or hydrogen bought
     on the market, and grey *No demand*; hover a chip for the numbers. A red or amber demand
     also puts a warning at the top of the card and a dashed outline on the inactive assets that
     could cover it. Each site needs one or more active assets. Each card has a **Reset site**
     button, and *What the coverage colors mean* below the cards explains the colors; the Max MW
     rule is explained on the Catalog page. The sites are
     written to `sites.json` in the run's output folder and passed to the planner with
     `--sites-file`.
  3. **Countries:** the candidate countries, shown in a table, each with its flag, full name
     and code; click one to select it. *Select all* and *Clear* sit above them. Ticking *All
     eligible countries* turns the country picks off.
  4. **Candidate catalog:** one card per technology active at one or more sites, listing the
     products the optimizer may build: size (MW), CAPEX and, for the storage assets, energy (MWh, the
     energy the store holds), plus one
     lifetime per technology. You can edit any value, add products with **+ Add candidate**
     (up to 8 per technology; a new row starts from the technology's default product) or
     remove them with **Remove** (each technology keeps at least one), and each card has a
     **Reset to default** button. Technologies may have different numbers of products. The
     edits apply to this run only (written to `catalog_overrides.json` in the run's output
     folder and passed to the planner through `PLANNER_CATALOG_OVERRIDES`); the defaults in
     `site_investor_planning/config.py` stay as they are. The run page prices its cost
     breakdown with the run's own catalog.
  5. **Uncertainty scenarios:** the **Baseline** card, ticked at 100% by default, and
     **+ Add scenario**, which adds as many scenarios of your own as you like. Each added
     scenario has a name, a probability and a table of the change in installed wind and solar
     capacity (%) compared with the Baseline for every selected country (positive for more,
     negative for less), each with its trained range under it and turning red outside it; the *All
     countries* row fills a whole column at once. **Remove** deletes it. The probabilities must add up to 100%; a badge at the top shows the total in
     green when it's right and red when it isn't. **Reset this section** goes back to the
     Baseline alone. Changes apply to this run only.
  6. **Economics:** discount rate, the risk measure and the CVaR confidence level. The risk
     measure follows the scenarios: **Deterministic** with one scenario, **CVaR** with two or
     more. Lifetimes are set per technology in the candidate catalog.
  7. **Solver settings:** days per month (representative days solved per month), optimality gap, maximum
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
- **Catalog** shows the example site's demand (the command-line default), how each asset's
  default maximum MW per site follows the site's peaks (with the example site's values), the operating assumptions (COPs, unmet-demand penalty, import
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
| `uncertainty_scenarios.json` | `inputs/` | In git | Planner (the Baseline scenario and its probability) |
| `site_demand.csv` | `inputs/` | In git | Planner (per-unit hourly demand curves for the year, shared by all sites; each site's annual peaks come with the site) |
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
python run_capacity_scenarios.py --scenarios p100,wind70
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
use `--scenarios p100,wind70`.

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
run. The output goes to `outputs/plan_*.csv`. For the full run, use `--all` and raise `--rep-days-per-month` (see the flags
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
