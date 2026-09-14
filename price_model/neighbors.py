"""System-total and per-zone neighbour demand features -- both raw and net.

Derived from a commodity's own (zone, hour, demand) samples plus its zone adjacency map
(``price_model/extract.py::extract_adjacency``, persisted as
``inputs/<commodity>_adjacency.json`` by ``build_dataset.py``). Two parallel feature
groups, raw demand and *net* demand (demand net of renewables -- the ``residual_load``
concept: demand - wind - solar; pass ``net_demand_col`` to name that column, e.g.
electricity's existing ``residual_load``):

* ``demand_system_total`` / ``net_demand_system_total`` -- sum across every zone, same
  value for every zone at a given hour. Shared features, added to every zone's list.
* ``neighbor_demand_<N>`` / ``neighbor_net_demand_<N>`` -- one column per zone actually
  interconnected to a given zone (named by the neighbour's own code), plus
  ``neighbor_demand_total`` / ``neighbor_net_demand_total`` (their sums). Zones differ in
  how many neighbours they have (median ~3, up to 9 for a hub like DE00), so each zone
  gets its own feature list rather than one shared set.

If ``net_demand_col`` is omitted or equal to ``demand_col`` (hydrogen has no renewables
column tied to H2 zones, so there's no distinct net-demand quantity to compute), only the
raw-demand set is produced.

If ``df`` has a ``scenario`` column (capacity-uncertainty scenarios pooled together, see
``run_capacity_scenarios.py``/``build_dataset.py``), both functions pivot/join on
``(scenario, hour)`` instead of ``hour`` alone -- every scenario reuses the same 0..8735
``hour`` range, so joining on ``hour`` alone would silently mix one scenario's neighbour
values into another's rows (``pivot_table``'s ``aggfunc="first"`` picking whichever
scenario happened to sort first). Absent a ``scenario`` column, behaviour is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_adjacency(path: str | Path) -> dict[str, list[str]]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _join_keys(df: pd.DataFrame) -> list[str]:
    return ["scenario", "hour"] if "scenario" in df.columns else ["hour"]


def _key_index(g: pd.DataFrame, keys: list[str]) -> pd.Index:
    return pd.MultiIndex.from_frame(g[keys]) if len(keys) > 1 else pd.Index(g[keys[0]].to_numpy())


def add_neighbor_features(
    df: pd.DataFrame,
    demand_col: str,
    adjacency: dict[str, list[str]],
    net_demand_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return (df with system-total + per-zone neighbour columns merged in,
    {zone: [extra feature column names]})."""
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
    """Add ``price_<zone>`` columns for two candidate feature sets per zone: (a) *every*
    directly-interconnected neighbour's price ("declared"), and (b) the ``top_n``
    system-wide most price-correlated zones regardless of interconnection ("top_n") --
    many zones move together with zones they aren't directly wired to (shared weather,
    fuel costs, or transitive interconnection through a hub). Neither candidate is
    uniformly better than the other across zones, and both can be *worse* than no
    neighbour-price feature at all for zones with rare extreme-price hours (cross-
    validation punishes a single badly-extrapolated spike hard, however good the
    feature). So this only computes the columns; the caller decides per zone by trying
    baseline vs. both candidates and keeping whichever CV R^2 is highest.

    Returns (df with every needed ``price_<zone>`` column merged in,
    {zone: {"declared": [...col names...], "top_n": [...col names...]}}).
    """
    keys = _join_keys(df)
    wide = df.pivot_table(index=keys, columns="zone", values=target_col, aggfunc="first")

    extra_frames = []
    zone_candidates: dict[str, dict[str, list[str]]] = {}
    for zone, g in df.groupby("zone", sort=True):
        idx = _key_index(g, keys)
        declared = [n for n in adjacency.get(zone, []) if n in wide.columns]
        top_n_list: list[str] = []
        if zone in wide.columns:
            corrs = wide.corrwith(wide[zone]).drop(zone).dropna().sort_values(ascending=False)
            top_n_list = corrs.head(top_n).index.tolist()
        needed = sorted(set(declared) | set(top_n_list))
        cols = {k: g[k].to_numpy() for k in keys}
        cols["zone"] = zone
        for n in needed:
            cols[f"price_{n}"] = wide[n].reindex(idx).to_numpy()
        zone_candidates[zone] = {
            "declared": [f"price_{n}" for n in declared],
            "top_n": [f"price_{n}" for n in top_n_list],
        }
        extra_frames.append(pd.DataFrame(cols))

    extra_df = pd.concat(extra_frames, ignore_index=True)
    enriched = df.merge(extra_df, on=["zone"] + keys, how="left")
    return enriched, zone_candidates
