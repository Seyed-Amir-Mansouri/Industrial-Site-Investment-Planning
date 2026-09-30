"""Transport lines + global price scalars, from the networks parquet database."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_NETWORKS_DB

_CO2 = "CO2 Price (EUR/ton)"


@dataclass
class Line:
    frm: str
    to: str
    cap_ft: float
    cap_tf: float
    loss: float


@dataclass
class NetworkData:
    elec: list[Line]
    hydrogen: list[Line]
    co2_price: float


def load_networks(zones: list[str], db_path: Path = DEFAULT_NETWORKS_DB) -> NetworkData:
    """Read the networks parquet database, filtered to lines between the given zones."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Networks database not found: {db_path}.")
    df = pd.read_parquet(db_path)
    zset = set(zones)

    def lines_for(carrier: str) -> list[Line]:
        sub = df[df["carrier"] == carrier]
        out: list[Line] = []
        for frm, to, ft, tf, loss in zip(sub["frm"], sub["to"], sub["cap_from_to_mw"],
                                          sub["cap_to_from_mw"], sub["loss_fraction"]):
            if frm in zset and to in zset and frm != to:
                out.append(Line(frm, to, float(ft), float(tf), float(loss)))
        return out

    prices = df[df["carrier"] == "prices"].set_index("frm")["cap_from_to_mw"].to_dict()
    return NetworkData(lines_for("electricity"), lines_for("hydrogen"),
                       float(prices.get(_CO2, 0.0)))


def border_line_caps(carrier: str = "electricity", db_path: Path = DEFAULT_NETWORKS_DB) -> dict[tuple, tuple]:
    """{(frm, to): (cap_from_to_mw, cap_to_from_mw)} for every line of the given carrier, unfiltered by zone."""
    df = pd.read_parquet(db_path)
    sub = df[df["carrier"] == carrier]
    return {(frm, to): (float(ft), float(tf)) for frm, to, ft, tf in
            zip(sub["frm"], sub["to"], sub["cap_from_to_mw"], sub["cap_to_from_mw"])}
