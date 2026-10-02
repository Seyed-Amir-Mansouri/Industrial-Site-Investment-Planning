"""Regenerate ``hourly_balance_{elec,h2}.csv`` for every training capacity scenario, including the baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from economic_dispatch.config import RunConfig
from economic_dispatch.data_loader import (
    CAPACITY_SCALE_KEYS, BATTERY_CHAR_TECH, BATTERY_CHAR_MW_COLS, H2_STORAGE_ASSET_KEYS,
)
from economic_dispatch.pipeline import solve_scenario
from economic_dispatch.report import write_hourly_balance

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_exchange" / "01_dispatch_output__train_input" / "scenarios"
UNCERTAINTY_JSON = ROOT / "inputs" / "uncertainty_scenarios.json"

_OFAT_PCTS = (110, 120, 130)
_SINGLE_ASSET_TAGS = {"wind": "wind", "solar": "pv", "battery": "batt",
                     "electrolyser": "ely", "tank": "tank"}
_TRAIN_PCTS = (105, 110, 115, 120, 125, 130)
_MIX_PCTS = (110, 120, 130)
_MIX_GROUPS = {
    "storage": ("battery", "tank"),
    "h2asset": ("electrolyser", "battery", "tank"),
    "all": ("wind", "solar", "electrolyser", "battery", "tank"),
}


def _uncertainty_scenarios() -> dict[str, dict]:
    """wind/solar per-country ``capacity_scale`` dicts for every scenario in ``uncertainty_scenarios.json``."""
    if not UNCERTAINTY_JSON.exists():
        return {}
    scenarios = json.loads(UNCERTAINTY_JSON.read_text())["scenarios"]
    return {name: {"wind": s["wind"], "solar": s["solar"]} for name, s in scenarios.items()}


def build_scenarios() -> dict[str, dict]:
    """Every named training scenario's ``RunConfig(capacity_scale=...)`` kwargs."""
    s: dict[str, dict] = {
        "p100": {"capacity_scale": {"wind": 1.0, "solar": 1.0}},
        "wind70": {"capacity_scale": {"wind": 0.7}},
        "pv70": {"capacity_scale": {"solar": 0.7}},
    }
    for group, tag in _SINGLE_ASSET_TAGS.items():
        for pct in _OFAT_PCTS:
            s[f"{tag}{pct}"] = {"capacity_scale": {group: pct / 100}}
    for pct in _TRAIN_PCTS:
        s[f"train{pct}"] = {"capacity_scale": {"wind": pct / 100, "solar": pct / 100}}
    for tag, groups in _MIX_GROUPS.items():
        for pct in _MIX_PCTS:
            s[f"{tag}{pct}"] = {"capacity_scale": {g: pct / 100 for g in groups}}
    for name, scale in _uncertainty_scenarios().items():
        s[name] = {"capacity_scale": scale}
    return s


SCENARIOS = build_scenarios()


def _capacities_manifest(zdata: dict) -> pd.DataFrame:
    """One row per zone: actual (post-scale) installed MW per scenario technology group."""
    rows = []
    for z, zd in zdata.items():
        wind = sum(zd.capacities.get(k, 0.0) for k in CAPACITY_SCALE_KEYS["wind"])
        pv = sum(zd.capacities.get(k, 0.0) for k in CAPACITY_SCALE_KEYS["solar"])
        ely = sum(zd.capacities.get(k, 0.0) for k in CAPACITY_SCALE_KEYS["electrolyser"])
        batt = zd.char_val(BATTERY_CHAR_TECH, BATTERY_CHAR_MW_COLS[0], 0.0)
        tank = zd.h2_assets.get(H2_STORAGE_ASSET_KEYS[0], 0.0)
        rows.append({"zone": z, "wind_capacity_mw": wind, "pv_capacity_mw": pv,
                    "electrolyser_capacity_mw": ely, "battery_capacity_mw": batt,
                    "tank_capacity_mw": tank})
    return pd.DataFrame(rows)


def run_one(name: str, kwargs: dict) -> None:
    t0 = time.time()
    print(f"=== scenario {name!r}: {kwargs} ===")
    cfg = RunConfig(start_day=1, end_day=364, use_plexos_renewable_override=False,
                    out_tag=f"scenario_{name}", **kwargs)
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
