"""System-total and per-zone neighbour/correlated-price demand features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_adjacency(path: str | Path) -> dict[str, list[str]]:
    """Load a zone adjacency map, or {} if the file doesn't exist."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _join_keys(df: pd.DataFrame) -> list[str]:
    """Join key(s) for merging per-zone features back onto ``df``."""
    return ["scenario", "hour"] if "scenario" in df.columns else ["hour"]


def _key_index(g: pd.DataFrame, keys: list[str]) -> pd.Index:
    """Index of ``g`` over ``keys``, for reindexing a pivoted wide frame."""
    return pd.MultiIndex.from_frame(g[keys]) if len(keys) > 1 else pd.Index(g[keys[0]].to_numpy())


def add_neighbor_features(
    df: pd.DataFrame,
    demand_col: str,
    adjacency: dict[str, list[str]],
    net_demand_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (df with system-total + per-zone neighbour columns merged in, {zone: [extra feature column names]})."""
    keys = _join_keys(df)
    value_cols = [("demand", demand_col)]
    if net_demand_col and net_demand_col != demand_col:
        value_cols.append(("net_demand", net_demand_col))

    wides = {label: df.pivot_table(index=keys, columns="zone", values=col, aggfunc="first")
             for label, col in value_cols}
    system_totals = {label: w.sum(axis=1) for label, w in wides.items()}

    def prefix(label: str) -> str:
        return "neighbor_demand_" if label == "demand" else "neighbor_net_demand_"

    def total_name(label: str) -> str:
        return "neighbor_demand_total" if label == "demand" else "neighbor_net_demand_total"

    def system_name(label: str) -> str:
        return "demand_system_total" if label == "demand" else "net_demand_system_total"

    extra_frames = []
    zone_extra: dict[str, list[str]] = {}
    for zone, g in df.groupby("zone", sort=True):
        idx = _key_index(g, keys)
        cols = {k: g[k].to_numpy() for k in keys}
        cols["zone"] = zone
        extra_cols = []
        for label, wide in wides.items():
            neighbor_cols = []
            for n in adjacency.get(zone, []):
                if n not in wide.columns:
                    continue
                colname = f"{prefix(label)}{n}"
                cols[colname] = wide[n].reindex(idx).to_numpy()
                neighbor_cols.append(colname)
            total = (np.zeros(len(idx)) if not neighbor_cols
                     else np.nansum([cols[c] for c in neighbor_cols], axis=0))
            cols[total_name(label)] = total
            cols[system_name(label)] = system_totals[label].reindex(idx).to_numpy()
            extra_cols += neighbor_cols + [total_name(label), system_name(label)]
        zone_extra[zone] = extra_cols
        extra_frames.append(pd.DataFrame(cols))

    extra_df = pd.concat(extra_frames, ignore_index=True)
    enriched = df.merge(extra_df, on=["zone"] + keys, how="left")
    return enriched, zone_extra


def add_candidate_neighbor_prices(
    df: pd.DataFrame,
    target_col: str,
    adjacency: dict[str, list[str]],
    top_n: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]]]:
    """Add each zone's top-``top_n`` price-correlated zones' ``price_<zone>`` columns.

    Returns (df with the needed ``price_<zone>`` columns merged in, {zone: {"top_n": [...col names...]}}).
    """
    keys = _join_keys(df)
    wide = df.pivot_table(index=keys, columns="zone", values=target_col, aggfunc="first")

    extra_frames = []
    zone_candidates: dict[str, dict[str, list[str]]] = {}
    for zone, g in df.groupby("zone", sort=True):
        idx = _key_index(g, keys)
        top_n_list: list[str] = []
        if zone in wide.columns:
            corrs = wide.corrwith(wide[zone]).drop(zone).dropna().sort_values(ascending=False)
            top_n_list = corrs.head(top_n).index.tolist()
        cols = {k: g[k].to_numpy() for k in keys}
        cols["zone"] = zone
        for n in top_n_list:
            cols[f"price_{n}"] = wide[n].reindex(idx).to_numpy()
        zone_candidates[zone] = {"top_n": [f"price_{n}" for n in top_n_list]}
        extra_frames.append(pd.DataFrame(cols))

    extra_df = pd.concat(extra_frames, ignore_index=True)
    enriched = df.merge(extra_df, on=["zone"] + keys, how="left")
    return enriched, zone_candidates
