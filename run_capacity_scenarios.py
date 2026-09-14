"""Regenerate ``hourly_balance_{elec,h2}.csv`` under 3 capacity-uncertainty scenarios --
a 100% ("as-planned NT2030") baseline plus 2 one-factor-at-a-time (OFAT) shortfall
scenarios, each reducing exactly ONE of {wind, solar (PV+rooftop+thermal)} installed
capacity to 70% while holding the other at 100%, uniformly across every zone. Superseded
the original single combined "all factors move together" axis (100/70/50% tried first)
at user request: that design makes wind/pv capacity perfectly collinear in the training
data (they only ever changed together), so a query like "PV at -10%, wind at plan" is
asking the model to extrapolate off a combination it never saw -- OFAT gives each
technology genuinely independent variation so single-factor queries interpolate within
seen data instead. A full factorial (every combination of levels across all factors, to
also capture interaction effects) would be more statistically complete but costs far more
compute; OFAT was chosen as the practical middle ground given each scenario is an
expensive full-year re-solve (see below).

Electrolyser capacity was ALSO a scenario factor (``ely70``) until 2026-09-14, when it
was dropped from the scenario set and from both price models' feature lists entirely, at
user request ("just wind and pv") -- see ``price_model/config.py``. The underlying
``economic_dispatch`` capacity-scaling machinery (``RunConfig.capacity_scale``,
``CAPACITY_SCALE_KEYS["electrolyser"]``) still supports scaling electrolyser capacity
directly if a future scenario needs it again; only this project's own scenario set and
the two trained models' feature lists dropped it.

Each scenario reruns the full 20-zone/8736-hour joint dispatch LP
(``economic_dispatch.pipeline.solve_scenario``) with ``capacity_scale`` applied and
``use_plexos_renewable_override=False`` -- the vendored dispatch model otherwise replaces
wind/solar/ROR/other-RES generation with Project 3's own FIXED historical PLEXOS-realized
curves (not sensitive to installed capacity at all), so that override must be disabled for
the capacity scaling to have any effect on those technologies. This means even the
baseline (100%, "as NT2030") scenario here is NOT byte-identical to the originally-
committed ``inputs/hourly_balance_elec.csv`` (which used the PLEXOS-curve path) -- every
scenario is regenerated fresh on the same capacity-driven path for a fair, internally-
consistent comparison.

Writes, per scenario, under ``inputs/scenarios/<name>/``:
    hourly_balance_elec.csv, hourly_balance_h2.csv  -- same format build_dataset.py reads
    capacities.csv                                   -- per-zone actual scaled capacity
        (columns: zone, wind_capacity_mw, pv_capacity_mw, electrolyser_capacity_mw --
        the last one still recorded for reference even though no longer a scenario
        factor or a trained feature, since it costs nothing to keep), read back by
        price_model/extract.py::attach_capacity_features to tag every row of that
        scenario's samples with these as new model features (wind/pv only, see above).

Usage:
    python run_capacity_scenarios.py --scenarios p100   # one at a time, isolated process
                                                          # per run -- see CLAUDE.md on
                                                          # why (LP degeneracy from
                                                          # disabling the PLEXOS override
                                                          # makes each solve ~8-16 min and
                                                          # memory-heavy; running several
                                                          # in one process risked an OOM
                                                          # kill on a 24GB machine)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from economic_dispatch.config import RunConfig
from economic_dispatch.data_loader import CAPACITY_SCALE_KEYS
from economic_dispatch.pipeline import solve_scenario
from economic_dispatch.report import write_hourly_balance

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "inputs" / "scenarios"

# scenario name -> per-group capacity multiplier (wind/solar independently; electrolyser
# dropped 2026-09-14, see module docstring -- kept at 1.0/no-op here for any zone whose
# capacities.csv manifest still records it for reference)
SCENARIOS = {
    "p100": {"wind": 1.0, "solar": 1.0, "electrolyser": 1.0},
    "wind70": {"wind": 0.7, "solar": 1.0, "electrolyser": 1.0},
    "pv70": {"wind": 1.0, "solar": 0.7, "electrolyser": 1.0},
}


def _capacities_manifest(zdata: dict) -> pd.DataFrame:
    """One row per zone: actual (post-scale) installed MW summed within each of the 3
    scenario technology groups (see ``CAPACITY_SCALE_KEYS``)."""
    rows = []
    for z, zd in zdata.items():
        wind = sum(zd.capacities.get(k, 0.0) for k in CAPACITY_SCALE_KEYS["wind"])
        pv = sum(zd.capacities.get(k, 0.0) for k in CAPACITY_SCALE_KEYS["solar"])
        ely = sum(zd.capacities.get(k, 0.0) for k in CAPACITY_SCALE_KEYS["electrolyser"])
        rows.append({"zone": z, "wind_capacity_mw": wind, "pv_capacity_mw": pv,
                    "electrolyser_capacity_mw": ely})
    return pd.DataFrame(rows)


def run_one(name: str, scale: dict[str, float]) -> None:
    t0 = time.time()
    print(f"=== scenario {name!r}: {scale} ===")
    cfg = RunConfig(
        start_day=1, end_day=364,
        capacity_scale=scale,
        use_plexos_renewable_override=False,
        out_tag=f"scenario_{name}",
    )
    build, zdata = solve_scenario(cfg, return_zdata=True)
    out_dir = OUT / name
    write_hourly_balance(build, out_dir)
    _capacities_manifest(zdata).to_csv(out_dir / "capacities.csv", index=False)
    print(f"[{name}] wrote {out_dir} in {time.time() - t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", default=",".join(SCENARIOS),
                   help="comma-separated subset of: " + ",".join(SCENARIOS))
    args = p.parse_args()
    names = args.scenarios.split(",")
    for name in names:
        run_one(name, SCENARIOS[name])


if __name__ == "__main__":
    main()
