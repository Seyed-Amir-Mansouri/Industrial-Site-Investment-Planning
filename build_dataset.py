"""Build the electricity and hydrogen feature-table parquets from the upstream dispatch output, pooling capacity-uncertainty scenarios if present."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from price_model.extract import (
    extract_electricity, extract_hydrogen, extract_adjacency, attach_capacity_features,
    DEFAULT_ELEC_CSV, DEFAULT_H2_CSV, DEFAULT_NETWORKS_PARQUET,
)
from price_model.config import COMMODITIES

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUT = ROOT / "data_exchange" / "01_dispatch_output__train_input"
SCENARIOS_DIR = OUT / "scenarios"


def _report(name, df, demand, price):
    """Print a one-block summary of an extracted feature table."""
    feat = [c for c in df.columns if c not in ("zone", "hour", "datetime")]
    active = df[df[demand].fillna(0) > 0]
    print(
        f"[{name}] {len(df):,} rows | {df['zone'].nunique()} zones | "
        f"{df['hour'].nunique()} hours\n"
        f"    columns: {', '.join(feat)}\n"
        f"    {len(active):,} rows with {demand} > 0 | "
        f"price {df[price].min():.2f} .. {df[price].max():.2f}"
    )


def _scenario_dirs(names: list[str] | None = None) -> list[Path]:
    """All valid scenario dirs under SCENARIOS_DIR, or just the ones named in ``names`` if given."""
    if not SCENARIOS_DIR.is_dir():
        return []
    all_dirs = sorted(p for p in SCENARIOS_DIR.iterdir()
                      if p.is_dir() and (p / "hourly_balance_elec.csv").exists())
    if names is None:
        return all_dirs
    by_name = {p.name: p for p in all_dirs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"--scenarios named {missing} not found under {SCENARIOS_DIR} "
                       f"-- available: {sorted(by_name)}")
    return [by_name[n] for n in names]


def _build_pooled(scenario_dirs: list[Path]):
    """Extract and concatenate every scenario dir's electricity/hydrogen tables, tagged with a scenario column."""
    print(f"Pooling {len(scenario_dirs)} capacity scenarios: "
          f"{', '.join(p.name for p in scenario_dirs)}")
    elec_frames, h2_frames = [], []
    for sdir in scenario_dirs:
        e = extract_electricity(sdir / "hourly_balance_elec.csv")
        e = attach_capacity_features(e, sdir / "capacities.csv")
        e.insert(0, "scenario", sdir.name)
        elec_frames.append(e)

        h = extract_hydrogen(sdir / "hourly_balance_h2.csv", elec_df=e)
        h = attach_capacity_features(h, sdir / "capacities.csv")
        h.insert(0, "scenario", sdir.name)
        h2_frames.append(h)
    return pd.concat(elec_frames, ignore_index=True), pd.concat(h2_frames, ignore_index=True)


def main(elec_csv=DEFAULT_ELEC_CSV, h2_csv=DEFAULT_H2_CSV,
         networks_parquet=DEFAULT_NETWORKS_PARQUET, scenarios: list[str] | None = None) -> None:
    """Build and write both commodities' sample parquets plus zone-adjacency JSONs."""
    OUT.mkdir(parents=True, exist_ok=True)
    INPUTS.mkdir(parents=True, exist_ok=True)

    scenario_dirs = _scenario_dirs(scenarios)
    if scenario_dirs:
        elec, h2 = _build_pooled(scenario_dirs)
    else:
        print(f"Reading {Path(elec_csv).name} / {Path(h2_csv).name} ...")
        elec = extract_electricity(elec_csv)
        h2 = extract_hydrogen(h2_csv, elec_df=elec)

    elec.to_parquet(OUT / COMMODITIES["electricity"]["samples"], index=False,
                    compression="zstd")
    _report("electricity", elec, "demand", "price_eur_mwh")

    h2.to_parquet(OUT / COMMODITIES["hydrogen"]["samples"], index=False,
                  compression="zstd")
    _report("hydrogen", h2, "h2_demand", "h2_price")

    if not Path(networks_parquet).exists():
        print(f"WARNING: {networks_parquet} not found -- skipping adjacency "
              f"(no neighbour/system-total features will be trained).")
        return

    elec_zones = set(elec["zone"])
    elec_adj = extract_adjacency(networks_parquet, "electricity", elec_zones)
    (INPUTS / COMMODITIES["electricity"]["adjacency"]).write_text(json.dumps(elec_adj, indent=1))
    print(f"[electricity] adjacency: {len(elec_adj)} zones -> "
          f"{COMMODITIES['electricity']['adjacency']}")

    h2_zones = set(h2["zone"])
    h2_adj = extract_adjacency(networks_parquet, "hydrogen", h2_zones)
    (INPUTS / COMMODITIES["hydrogen"]["adjacency"]).write_text(json.dumps(h2_adj, indent=1))
    print(f"[hydrogen] adjacency: {len(h2_adj)} zones -> "
          f"{COMMODITIES['hydrogen']['adjacency']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, default=None,
                    help="comma-separated subset of data_exchange/01_dispatch_output__train_input/"
                         "scenarios/<name> dirs to pool (default: every dir found)")
    args = ap.parse_args()
    main(scenarios=args.scenarios.split(",") if args.scenarios else None)
