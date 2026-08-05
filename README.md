# Project 5 — Price Models & H2 Producer Capacity Planning

Two pieces of work, both built on Project 3's dispatch output for the 20-zone
Central-European CORE region (NT2030 scenario):

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
using the price models above as a price-taker market signal, instead of Project 3's
full network coupling.

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

## Structure

```
price_model/            demand -> price models
h2_planning/             Benders master + candidate grids + CAPEX assumptions
optimize_h2_producer.py  standalone / joint H2 Producer LP
plan_h2_capacity.py      Benders CLI driver
day_sampling.py          representative-day sampling
economic_dispatch/       Project 3's dispatch engine, vendored locally
inputs/                  sample parquets, balance CSVs, zone/network DBs
outputs/                 trained models, plans (git-ignored)
```

See `Formulation.md` for the full math.

## Install

```bash
pip install -r requirements.txt
```
