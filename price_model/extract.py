"""Extract per-zone, per-hour feature tables from the upstream dispatch model's output."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
DISPATCH_TRAIN = ROOT / "data_exchange" / "01_dispatch_output__train_input"
DEFAULT_BASELINE_SCENARIO = "p100"
DEFAULT_ELEC_CSV = DISPATCH_TRAIN / "scenarios" / DEFAULT_BASELINE_SCENARIO / "hourly_balance_elec.csv"
DEFAULT_H2_CSV = DISPATCH_TRAIN / "scenarios" / DEFAULT_BASELINE_SCENARIO / "hourly_balance_h2.csv"
DEFAULT_NETWORKS_PARQUET = INPUTS / "networks_2030.parquet"

HEADER_ROWS = 3


def _classify_elec(cat: str):
    """Map an electricity balance-CSV category name to (feature group, sign)."""
    c = cat.strip()
    if c.startswith("Marginal Price"):        return ("price_eur_mwh", 1)
    if c.startswith("Demand"):                return ("demand", -1)
    if c.startswith("DSR"):                   return ("dsr", 1)
    if c.startswith("Wind"):                  return ("wind", 1)
    if c.startswith("Solar"):                 return ("solar", 1)
    if c.startswith("Hydro "):                return ("hydro", 1)
    if c.startswith("Battery"):               return ("battery", 1)
    if c.startswith(("Net line import", "External exchange")): return ("balance", 1)
    if c.startswith("Load shedding"):         return ("ens", 1)
    if c.startswith("Dumped"):                return ("dumped", -1)
    if c.startswith(("Nuclear", "Lignite", "Hard Coal", "Gas ", "Light Oil", "Heavy oil",
                     "Hydrogen (ccgt)", "Hydrogen (fc)", "Other Non-RES")):
        return ("thermal", 1)
    return (None, 0)


def _classify_h2(cat: str):
    """Map a hydrogen balance-CSV category name to (feature group, sign)."""
    c = cat.strip()
    if c.startswith("Marginal Price"):        return ("h2_price", 1)
    if c.startswith("Demand"):                return ("h2_demand", -1)
    if c.startswith("Electrolyser production"): return ("electrolyser_gen", 1)
    if c.startswith("SMR production"):        return ("smr", 1)
    if c.startswith("H2 storage"):            return ("storage", 1)
    if c.startswith(("Terminal import", "Net pipeline import", "External exchange")):
        return ("h2_net_trade", 1)
    if c.startswith("Load shedding"):         return ("hns", 1)
    if c.startswith("Dumped"):                return ("dumped", -1)
    if c.startswith("H2 plant consumption"):  return ("h2_plant_consumption", -1)
    return (None, 0)


ELEC_GROUPS = ["price_eur_mwh", "demand", "dsr", "wind", "solar", "hydro",
               "battery", "balance", "ens", "dumped", "thermal"]
H2_GROUPS = ["h2_price", "h2_demand", "electrolyser_gen", "smr", "storage",
             "h2_net_trade", "h2_plant_consumption", "hns", "dumped"]

PRICE_GROUPS = {"price_eur_mwh", "h2_price"}


def _read_balance_csv(csv_path: str | Path):
    """Return (zones, categories, values) for a ``hourly_balance_*.csv`` file."""
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        zones = next(r)[1:]
        cats = next(r)[1:]
    raw = pd.read_csv(csv_path, skiprows=HEADER_ROWS, header=None, dtype=float)
    values = raw.iloc[:, 1:].to_numpy(dtype=float)
    return zones, cats, values


def _aggregate(zones, cats, values, classify, groups):
    """Sum each zone's matching columns into the named ``groups``."""
    zone_list = sorted(set(zones))
    nrows = values.shape[0]
    accum: dict[str, dict[str, np.ndarray]] = {
        z: {g: np.zeros(nrows) for g in groups} for z in zone_list
    }
    for j, (z, c) in enumerate(zip(zones, cats)):
        g, sign = classify(c)
        if g is None:
            continue
        col = values[:, j]
        if g in PRICE_GROUPS:
            accum[z][g] = col
        else:
            accum[z][g] += sign * np.nan_to_num(col, nan=0.0)
    return zone_list, nrows, accum


def _assemble(zones, nrows, accum, groups, year):
    dt = pd.date_range(f"{year}-01-01", periods=nrows, freq="h")
    month = dt.month.to_numpy()
    season = (dt.month.to_numpy() % 12) // 3
    frames = []
    for z in zones:
        d = {g: accum[z][g][:nrows] for g in groups}
        df = pd.DataFrame(d)
        df.insert(0, "zone", z)
        df.insert(1, "hour", np.arange(nrows, dtype="int32"))
        df.insert(2, "datetime", dt)
        df.insert(3, "month", month)
        df.insert(4, "season", season)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_electricity(csv_path: str | Path = DEFAULT_ELEC_CSV, year: int = 2030) -> pd.DataFrame:
    """Return the per-(zone, hour) electricity feature table."""
    zones, cats, values = _read_balance_csv(csv_path)
    zone_list, nrows, accum = _aggregate(zones, cats, values, _classify_elec, ELEC_GROUPS)
    feat = _assemble(zone_list, nrows, accum, ELEC_GROUPS, year)
    feat["vre"] = feat["wind"] + feat["solar"]
    feat["residual_load"] = feat["demand"] - feat["vre"]
    return feat


def extract_hydrogen(csv_path: str | Path = DEFAULT_H2_CSV, year: int = 2030,
                      elec_df: pd.DataFrame | None = None,
                      elec_csv: str | Path = DEFAULT_ELEC_CSV) -> pd.DataFrame:
    """Return the per-(zone, hour) hydrogen feature table, with ``elec_price`` merged in."""
    zones, cats, values = _read_balance_csv(csv_path)
    zone_list, nrows, accum = _aggregate(zones, cats, values, _classify_h2, H2_GROUPS)
    feat = _assemble(zone_list, nrows, accum, H2_GROUPS, year)

    if elec_df is None:
        elec_df = extract_electricity(elec_csv, year)
    price_map = elec_df[["zone", "hour", "price_eur_mwh"]].rename(
        columns={"price_eur_mwh": "elec_price"})
    feat = feat.merge(price_map, on=["zone", "hour"], how="left")
    return feat


def attach_capacity_features(df: pd.DataFrame, capacities_csv: str | Path) -> pd.DataFrame:
    """Left-merge a per-zone capacity manifest (wind/pv/electrolyser/battery/tank MW) onto every row of ``df``."""
    caps = pd.read_csv(capacities_csv)
    df = df.merge(caps, on="zone", how="left")
    for col in ("wind_capacity_mw", "pv_capacity_mw", "electrolyser_capacity_mw",
               "battery_capacity_mw", "tank_capacity_mw"):
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def extract_adjacency(parquet_path: str | Path, carrier: str,
                       zones: set[str] | None = None) -> dict[str, list[str]]:
    """Undirected zone adjacency for ``carrier`` from the network topology parquet."""
    net = pd.read_parquet(parquet_path)
    net = net[net["carrier"] == carrier]
    net = net[(net["cap_from_to_mw"] > 0) | (net["cap_to_from_mw"] > 0)]
    if zones is not None:
        net = net[net["frm"].isin(zones) & net["to"].isin(zones)]
    adj: dict[str, set[str]] = {}
    for a, b in zip(net["frm"], net["to"]):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return {z: sorted(n) for z, n in adj.items()}


if __name__ == "__main__":
    e = extract_electricity()
    h = extract_hydrogen()
    print("electricity:", e.shape, e["zone"].nunique(), "zones")
    print("hydrogen:   ", h.shape, h["zone"].nunique(), "zones")
