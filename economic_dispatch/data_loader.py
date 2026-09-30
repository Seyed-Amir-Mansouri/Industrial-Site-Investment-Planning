"""Zone data structures and per-zone capacity/classification helpers for the dispatch model."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

CAT_COMMIT = "committable"
CAT_VRES = "vres"
CAT_ROR = "ror"
CAT_PROFILE = "profile_gen"
CAT_IGNORE = "ignore"


@dataclass
class ZoneData:
    code: str
    capacities: dict[str, float]
    storage_energy: dict[str, float]
    reserves: dict[str, float]
    h2_assets: dict[str, float]
    char: pd.DataFrame
    profiles: pd.DataFrame

    def char_val(self, tech: str, col: str, default: float = 0.0) -> float:
        try:
            v = self.char.at[tech, col]
        except KeyError:
            return default
        if pd.isna(v):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def must_run_pct(self, tech: str, month: int) -> float:
        """Must-run share for the given 0-based month, as % of installed capacity."""
        try:
            raw = self.char.at[tech, "Must Run (%)"]
        except KeyError:
            return 0.0
        return _month_value(raw, month)


def _month_value(raw, month: int) -> float:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    if not parts:
        return 0.0
    idx = min(month, len(parts) - 1)
    try:
        return float(parts[idx])
    except ValueError:
        return 0.0


def _zone_from_db(zdf: pd.DataFrame, code: str, h0: int, h1: int) -> ZoneData:
    """Reconstruct a ZoneData from this zone's slice of the long parquet table."""
    def scalar(section):
        d = zdf[zdf["section"] == section]
        return dict(zip(d["item"], d["value_num"]))

    c = zdf[zdf["section"] == "characteristics"].copy()
    ti = c["item"].str.split("||", n=1, expand=True, regex=False)
    c["tech"], c["attr"] = ti[0], ti[1]
    c["value"] = c["value_str"].where(c["value_str"].notna(), c["value_num"])
    char = c.pivot(index="tech", columns="attr", values="value")
    char.index.name = "Technology"
    char.columns.name = None

    p = zdf[zdf["section"] == "profiles"]
    prof = p.pivot(index="hour", columns="item", values="value_num")
    prof.columns.name = None
    prof = prof.iloc[h0:h1].reset_index(drop=True)

    return ZoneData(code, scalar("capacities"), scalar("storage_energy"),
                    scalar("reserves"), scalar("h2_assets"), char, prof)


def load_zones_from_db(codes: list[str], db_path: Path,
                       hour_start: int, hour_end: int) -> dict[str, ZoneData]:
    """Load ZoneData for the given zones from ``zones_2030.parquet`` (one read)."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Zone database not found: {db_path}. Build it with `python build_zones_db.py`.")
    db = pd.read_parquet(db_path, filters=[("zone", "in", list(codes))])
    present = set(db["zone"].unique())
    missing = [z for z in codes if z not in present]
    if missing:
        raise KeyError(f"zones not in {db_path.name}: {missing}")
    by_zone = {z: g for z, g in db.groupby("zone", sort=False)}
    return {z: _zone_from_db(by_zone[z], z, hour_start, hour_end) for z in codes}


CAPACITY_SCALE_KEYS: dict[str, list[str]] = {
    "wind": ["Wind (onshore) (MW)", "Wind (offshore) (MW)"],
    "solar": ["Solar (MW)", "Solar (rooftop) (MW)",
             "Solar (thermal) (MW)", "Solar (thermal_with_storage) (MW)"],
    "electrolyser": ["Electrolyser (MW)"],
}

BATTERY_CHAR_TECH = "Battery (MWh)"
BATTERY_CHAR_MW_COLS = ["Net maximum capacity - generation perspective (MW)",
                        "Net maximum capacity - demand perspective (MW)"]
H2_STORAGE_ASSET_KEYS = ["Withdraw (Hydrogen) (MW)", "Injection (Hydrogen) (MW)"]


def apply_capacity_scale(zdata: dict[str, ZoneData],
                         scale: dict[str, float | dict[str, float]]) -> dict[str, ZoneData]:
    """Return a copy of ``zdata`` with each zone's installed capacity scaled per ``scale``
    (group -> factor, or group -> {country: factor}); empty ``scale`` is a no-op."""
    if not scale:
        return zdata
    out = {}
    for z, zd in zdata.items():
        caps = dict(zd.capacities)
        char = zd.char.copy()
        storage_energy = dict(zd.storage_energy)
        h2_assets = dict(zd.h2_assets)
        for group, factor in scale.items():
            f = factor.get(z[:2], 1.0) if isinstance(factor, dict) else factor
            if group == "battery":
                if BATTERY_CHAR_TECH in char.index:
                    for col in BATTERY_CHAR_MW_COLS:
                        if col in char.columns and pd.notna(char.at[BATTERY_CHAR_TECH, col]):
                            char.at[BATTERY_CHAR_TECH, col] = char.at[BATTERY_CHAR_TECH, col] * f
                if BATTERY_CHAR_TECH in storage_energy:
                    storage_energy[BATTERY_CHAR_TECH] = storage_energy[BATTERY_CHAR_TECH] * f
                continue
            if group == "tank":
                for key in H2_STORAGE_ASSET_KEYS:
                    if key in h2_assets:
                        h2_assets[key] = h2_assets[key] * f
                continue
            for key in CAPACITY_SCALE_KEYS.get(group, []):
                if key in caps:
                    caps[key] = caps[key] * f
        out[z] = replace(zd, capacities=caps, char=char, storage_energy=storage_energy, h2_assets=h2_assets)
    return out


def classify(tech: str) -> tuple[str, bool]:
    """Map a Technology-Capacities row name to (category, is_h2_fuel)."""
    t = tech
    if (t.startswith("Nuclear") or t.startswith("Hard Coal") or t.startswith("Lignite")
            or t.startswith("Gas (") or t.startswith("Light Oil")
            or t.startswith("Heavy oil") or t.startswith("Oil shale")
            or t.startswith("Hydrogen (fc)") or t.startswith("Hydrogen (ccgt)")):
        return CAT_COMMIT, False
    if t.startswith("Wind (") or t.startswith("Solar ("):
        return CAT_VRES, False
    if t.startswith("Hydro (river)"):
        return CAT_ROR, False
    if t.startswith("Other RES") or t.startswith("Other Non-RES") or t.startswith("DSR"):
        return CAT_PROFILE, False
    return CAT_IGNORE, False


VRES_PROFILE = {
    "Wind (onshore) (MW)": "Wind_Onshore Profile",
    "Wind (offshore) (MW)": "Wind_Offshore Profile",
    "Solar (MW)": "Solar Profile",
    "Solar (rooftop) (MW)": "Solar_Rooftop Profile",
    "Solar (thermal) (MW)": "CSP_noStorage Profile",
    "Solar (thermal_with_storage) (MW)": "CSP_withStorage_D Profile",
}


def profile_gen_column(tech: str) -> str:
    """Profile column (MW series) for an Other RES / Other Non-RES / DSR tech."""
    return tech.replace("(MW)", "(MW/h)")
