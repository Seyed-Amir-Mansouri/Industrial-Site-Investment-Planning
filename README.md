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
python plan_h2_capacity.py --countries DE,FR,PL --budget 500000000
```

Both support `--rep-days-per-month N` to solve on N representative days/month
(weighted to approximate the full year) instead of the full 364-day year.

CAPEX figures in `h2_planning/config.py` are placeholders, not real quotes.

### How the Benders solve works

Sizing all five assets (electrolyser, wind, PV, battery, H2 tank) for every country
jointly, as one MILP with a full year of hourly LP dispatch variables per country,
doesn't scale — so `plan_h2_capacity.py` splits it into a master problem and one
subproblem per country, iterating between them:

1. **Master (MILP, `h2_planning/master.py`)** — picks one candidate MW value per
   asset per country (binary one-hot over a 5-point grid), subject to the annualized,
   system-wide CAPEX budget. Its objective is annualized CAPEX plus a per-country
   profit stand-in (`theta`) that starts unconstrained and gets tightened every round
   by the cuts below.
2. **Subproblems (LP, one per country, reusing `optimize_h2_producer.solve`)** — for
   the master's chosen capacities, solve the full-year hourly dispatch and return the
   realized operating profit plus the dual values (shadow prices) on the capacity
   constraints.
3. **Cut generation** — each subproblem's profit and duals are turned into a Benders
   optimality cut: a linear upper bound on that country's `theta`, expressed in the
   master's binary capacity-choice variables (duals become the cut's coefficients,
   including a cross term for how electrolyser sizing affects the others). The cut is
   added back into the master.
4. **Loop** — resolve the master with the new cut, resolve subproblems at the updated
   capacities, and repeat until the master's `theta` upper bounds and the subproblems'
   actual profits converge (or an iteration cap is hit).

This avoids ever building one giant LP over all countries and all 8,760 hours at once
— each iteration is a small MILP plus independent per-country LPs, which is what makes
the discrete/binary capacity search tractable. See `Formulation.md` §4 for the full
derivation of the cut coefficients (including the electrolyser cross-coupling term)
and known limitations of the current implementation.

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
