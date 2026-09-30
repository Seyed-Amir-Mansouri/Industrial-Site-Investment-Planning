"""Derive fixed cross-border electricity/hydrogen exchange for the selected zones from the crossborder-flow parquet databases."""
from __future__ import annotations

import numpy as np
import pandas as pd


def country(zone: str) -> str:
    """Country code = first two characters of a zone code (AT00->AT, BEOF->BE)."""
    return zone[:2]


def _flow_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if isinstance(c, str) and "->" in c]


def _num(series: pd.Series) -> np.ndarray:
    return np.nan_to_num(series.to_numpy(dtype=float))


def elec_border_legs(selected: list[str], edf: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Per-(zone, external neighbour) signed net flow (MW, + = zone exports), full-year arrays."""
    sel = set(selected)
    out: dict[tuple[str, str], np.ndarray] = {}
    for f in _flow_cols(edf):
        a, b = f.split("->")
        if a in sel and b not in sel:
            out[(a, b)] = out.get((a, b), 0.0) + _num(edf[f])
        elif b in sel and a not in sel:
            out[(b, a)] = out.get((b, a), 0.0) - _num(edf[f])
    return out


def _cc(node: str) -> str:
    """Country code of an H2 node header (strip the ``_H2`` suffix)."""
    return node[:-3] if node.endswith("_H2") else node


def _h2_edges(hdf: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """Resolve ``IB*`` interconnector hubs and return the flattened (from_node, to_node, array) edge list."""
    flows = _flow_cols(hdf)
    hub_sinks: dict[str, list[str]] = {}
    for f in flows:
        l, r = f.split("->")
        if l.startswith("IB") and l.endswith("_H2") and not (r.startswith("IB") and r.endswith("_H2")):
            hub_sinks.setdefault(l, []).append(r)
    edges: list[tuple[str, str, np.ndarray]] = []
    for f in flows:
        l, r = f.split("->")
        if l.startswith("IB") and l.endswith("_H2"):
            continue
        arr = _num(hdf[f])
        if r.startswith("IB") and r.endswith("_H2"):
            for sink in hub_sinks.get(r, []):
                edges.append((l, sink, arr))
        else:
            edges.append((l, r, arr))
    return edges


def smr_injection(selected: list[str], main_map: dict[str, str],
                  smr: pd.DataFrame) -> dict[str, np.ndarray]:
    """Steam-Methane-Reformer output as a per-main-zone H2 injection (MW, + = domestic supply)."""
    out: dict[str, np.ndarray] = {}
    for C in {country(z) for z in selected}:
        M = main_map.get(C)
        if M is not None and C in smr.columns:
            out[M] = out.get(M, 0.0) + _num(smr[C])
    return out


def exogenous_h2_injection(selected: list[str], main_map: dict[str, str],
                           hdf: pd.DataFrame) -> dict[str, np.ndarray]:
    """Non-cross-border H2 supply from virtual source nodes (MW, + = supply added to the H2 balance)."""
    sel_c = {country(z) for z in selected}
    out: dict[str, np.ndarray] = {}
    for l, r, arr in _h2_edges(hdf):
        if l.endswith("_H2") and r.endswith("_H2"):
            continue
        real, virtual = (r, l) if r.endswith("_H2") else (l, r)
        if virtual.endswith("_H2"):
            continue
        C = _cc(real)
        if C not in sel_c:
            continue
        M = main_map.get(C)
        if M is None:
            continue
        sign = 1.0 if real == r else -1.0
        out[M] = out.get(M, 0.0) + sign * arr
    return out


def h2_border_legs(selected: list[str], main_map: dict[str, str],
                   hdf: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Per-(main zone, external neighbour country) signed net H2 flow (MW, + = exports), full-year."""
    sel = set(selected)
    sel_c = {country(z) for z in selected}
    out: dict[tuple[str, str], np.ndarray] = {}
    for l, r, arr in _h2_edges(hdf):
        if not (l.endswith("_H2") and r.endswith("_H2")):
            continue
        xc, yc = _cc(l), _cc(r)
        if xc in sel_c and yc not in sel_c:
            C, N, sign = xc, yc, 1.0
        elif yc in sel_c and xc not in sel_c:
            C, N, sign = yc, xc, -1.0
        else:
            continue
        M = main_map.get(C)
        if M is None or M not in sel:
            continue
        key = (M, N)
        out[key] = out.get(key, 0.0) + sign * arr
    return out
