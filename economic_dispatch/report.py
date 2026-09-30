"""Extract and export the solved dispatch as hourly per-technology balance tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import BuildResult


def _sol(build: BuildResult, name: str) -> pd.DataFrame:
    """Return a variable's solution as a (dim0 x hour) DataFrame, or empty."""
    if name not in build.model.variables:
        return pd.DataFrame()
    da = build.model.solution[name]
    return da.to_pandas()


def _to_pandas(da) -> pd.DataFrame:
    """DataArray -> pandas, evaluating an unresolved linopy ``LinearExpression`` first."""
    if hasattr(da, "solution"):
        da = da.solution
    return da.to_pandas()


def extract(build: BuildResult) -> dict[str, pd.DataFrame]:
    """Pull every dispatch/H2-Producer variable's solution into a dict of DataFrames."""
    prod_wind_p = _sol(build, "prod_wind_p")
    prod_pv_p = _sol(build, "prod_pv_p")
    prod_batt_dis = _sol(build, "prod_batt_dis")
    prod_batt_ch = _sol(build, "prod_batt_ch")
    prod_ely_p = _sol(build, "prod_ely_p")
    prod_tank_dis = _sol(build, "prod_tank_dis")
    prod_tank_ch = _sol(build, "prod_tank_ch")
    if not prod_wind_p.empty:
        ely_eff = build.cfg.h2_producer_electrolyser_efficiency
        prod_grid_net = prod_wind_p + prod_pv_p + prod_batt_dis - prod_batt_ch - prod_ely_p
        prod_h2_net = ely_eff * prod_ely_p + prod_tank_dis - prod_tank_ch
    else:
        prod_grid_net = pd.DataFrame()
        prod_h2_net = pd.DataFrame()
    return {
        "gen_p": _sol(build, "gen_p"),
        "dis": _sol(build, "dis"),
        "ch": _sol(build, "ch"),
        "soc": _sol(build, "soc"),
        "spill": _sol(build, "spill"),
        "ely_p": _sol(build, "ely_p"),
        "term_h2": _sol(build, "term_h2"),
        "shed_e": _sol(build, "shed_e"),
        "shed_h": _sol(build, "shed_h"),
        "dump_e": _sol(build, "dump_e"),
        "dump_h": _sol(build, "dump_h"),
        "smr_gen": _sol(build, "smr_gen"),
        "prod_wind_p": prod_wind_p,
        "prod_pv_p": prod_pv_p,
        "prod_batt_dis": prod_batt_dis,
        "prod_batt_ch": prod_batt_ch,
        "prod_ely_p": prod_ely_p,
        "prod_tank_dis": prod_tank_dis,
        "prod_tank_ch": prod_tank_ch,
        "prod_grid_net": prod_grid_net,
        "prod_h2_net": prod_h2_net,
    }


def _ids_on_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Orient so that 'zone|...' resource ids are the row index."""
    if any(isinstance(i, str) and "|" in i for i in df.index):
        return df
    if any(isinstance(c, str) and "|" in c for c in df.columns):
        return df.T
    return df


def _zones_on_rows(df: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    """Orient a (zone x hour) frame so zones are the row index."""
    if set(zones) & set(df.index):
        return df
    if set(zones) & set(df.columns):
        return df.T
    return df


def _prod_on_rows(df: pd.DataFrame, prod_idx) -> pd.DataFrame:
    """Orient a (country x hour) Hydrogen-Producer solution frame so country ids are the row index."""
    if set(prod_idx) & set(df.index):
        return df
    return df.T


def _prod_zone_sum(df: pd.DataFrame, build: BuildResult, zones: list[str], H: int) -> pd.DataFrame:
    """Sum a (country x hour) Hydrogen-Producer solution frame into (zone x hour)."""
    prod = build.h2_producer
    if df.empty or prod.empty:
        return pd.DataFrame(0.0, index=zones, columns=range(H))
    df = _prod_on_rows(df, prod.index)
    grp = df.groupby(lambda c: prod.loc[c, "zone"]).sum()
    return grp.reindex(zones).fillna(0.0)


def _ely_production(build: BuildResult, sol) -> pd.DataFrame:
    """H2 produced per zone = eff * elec consumed (electrolyser efficiency)."""
    z = build.zones
    ely = sol["ely_p"]
    ely = _zones_on_rows(ely, z).reindex(z).fillna(0.0) if not ely.empty \
        else pd.DataFrame(0.0, index=z, columns=range(len(build.hours)))
    eff = getattr(build, "_ely_eff", pd.Series(0.68, index=z)).reindex(z).fillna(0.68)
    return ely.mul(eff, axis=0)


def _lines_on_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Orient a flow frame so line ids (containing '->') are the row index."""
    if any(isinstance(i, str) and "->" in i for i in df.index):
        return df
    return df.T


def _net_import_from_solution(build: BuildResult, tag: str) -> pd.DataFrame:
    z = build.zones
    lines = build.elines if tag == "e" else build.hlines
    H = len(build.hours)
    out = pd.DataFrame(0.0, index=z, columns=range(H))
    if not lines:
        return out
    fpos = _lines_on_rows(build.model.solution[f"f{tag}_pos"].to_pandas())
    fneg = _lines_on_rows(build.model.solution[f"f{tag}_neg"].to_pandas())
    for i, l in enumerate(lines):
        key = f"{tag}{i}:{l.frm}->{l.to}"
        p = fpos.loc[key].to_numpy(dtype=float)
        n = fneg.loc[key].to_numpy(dtype=float)
        out.loc[l.to] = out.loc[l.to].to_numpy() + p * (1 - l.loss) - n
        out.loc[l.frm] = out.loc[l.frm].to_numpy() + n * (1 - l.loss) - p
    return out


def write_hourly_balance(build: BuildResult, out_dir: Path) -> None:
    """Write the hourly per-technology balance tables (elec & H2) to CSV."""
    write_balance_tables(hourly_balance_tables(build), Path(out_dir))


def write_balance_tables(tables: dict, out_dir: Path) -> None:
    """Clean-slate write the two balance tables to ``out_dir`` as CSVs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        try:
            old.unlink()
        except OSError as e:
            print(f"  warning: could not remove {old.name} ({e.strerror})")
    tables["elec"].to_csv(out_dir / "hourly_balance_elec.csv")
    tables["h2"].to_csv(out_dir / "hourly_balance_h2.csv")


def hourly_balance_tables(build: BuildResult) -> dict:
    """PLEXOS-style hourly per-technology balances as two wide (zone, category)-columned DataFrames."""
    z = build.zones
    H = len(build.hours)
    sol = extract(build)

    def zrows(name):
        df = sol[name]
        if df.empty:
            return pd.DataFrame(0.0, index=z, columns=range(H))
        return _zones_on_rows(df, z).reindex(z).fillna(0.0)

    def da_rows(da):
        return _zones_on_rows(_to_pandas(da), z).reindex(z).fillna(0.0)

    gp = _ids_on_rows(sol["gen_p"]) if not sol["gen_p"].empty else pd.DataFrame()
    dis_ids = _ids_on_rows(sol["dis"]) if not sol["dis"].empty else pd.DataFrame()
    ch_ids = _ids_on_rows(sol["ch"]) if not sol["ch"].empty else pd.DataFrame()

    def storage_kind_cols(zone: str, carrier: str):
        """Per-device discharge/charge columns, e.g. 'Hydro reservoir discharge (MW)'."""
        st = build.storage
        if st.empty:
            return []
        out = []
        rows = st[(st["zone"] == zone) & (st["carrier"] == carrier)]
        for sid, row in rows.iterrows():
            kind = row["kind"]
            dis_s = dis_ids.loc[sid].to_numpy(dtype=float) if sid in dis_ids.index else np.zeros(H)
            ch_s = ch_ids.loc[sid].to_numpy(dtype=float) if sid in ch_ids.index else np.zeros(H)
            out.append((f"{kind} discharge (MW)", dis_s))
            out.append((f"{kind} charge (-) (MW)", -ch_s))
        return out

    ely, term = zrows("ely_p"), zrows("term_h2")
    shed_e, shed_h = zrows("shed_e"), zrows("shed_h")
    dmp_e, dmp_h = zrows("dump_e"), zrows("dump_h")
    smr = zrows("smr_gen")
    net_e, net_h = _net_import_from_solution(build, "e"), _net_import_from_solution(build, "h")
    dem_e, dem_h = da_rows(build.demand_e), da_rows(build.demand_h)
    ext_e, ext_h = da_rows(build.external_e), da_rows(build.external_h2)
    ely_prod = _ely_production(build, sol)
    voll = build.cfg.voll_eur_per_mwh
    price_e = price_h = None
    if getattr(build, "price_e", None) is not None:
        price_e = da_rows(build.price_e).mask(
            (da_rows(build.price_e).abs() >= 0.99 * voll) & (shed_e.abs() <= 1e-6))
    if getattr(build, "price_h", None) is not None:
        price_h = da_rows(build.price_h).mask(
            (da_rows(build.price_h).abs() >= 0.99 * voll) & (shed_h.abs() <= 1e-6))

    h2 = build.gens[build.gens["h2_fuel"]]
    h2_cons = pd.DataFrame(0.0, index=z, columns=range(H))
    if not h2.empty and not gp.empty:
        for gid, row in h2.iterrows():
            if gid in gp.index:
                h2_cons.loc[row["zone"]] += gp.loc[gid].to_numpy() / row["eff"]

    prod_wind = _prod_zone_sum(sol["prod_wind_p"], build, z, H)
    prod_pv = _prod_zone_sum(sol["prod_pv_p"], build, z, H)
    prod_batt_dis = _prod_zone_sum(sol["prod_batt_dis"], build, z, H)
    prod_batt_ch = _prod_zone_sum(sol["prod_batt_ch"], build, z, H)
    prod_ely = _prod_zone_sum(sol["prod_ely_p"], build, z, H)
    prod_tank_dis = _prod_zone_sum(sol["prod_tank_dis"], build, z, H)
    prod_tank_ch = _prod_zone_sum(sol["prod_tank_ch"], build, z, H)
    prod_grid_net = _prod_zone_sum(sol["prod_grid_net"], build, z, H)
    prod_h2_net = _prod_zone_sum(sol["prod_h2_net"], build, z, H)
    prod_ely_h2 = prod_ely * build.cfg.h2_producer_electrolyser_efficiency

    def build_table(per_zone_cols):
        data = {}
        for zone in z:
            for cat, series in per_zone_cols(zone):
                data[(zone, cat)] = np.asarray(series, dtype=float)
        df = pd.DataFrame(data, index=pd.Index(range(H), name="hour"))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["zone", "category"])
        return df.round(3)

    def elec_cols(zone):
        out = []
        if not gp.empty:
            for gid in gp.index:
                if gid.split("|", 1)[0] == zone:
                    out.append((gid.split("|", 1)[1], gp.loc[gid].to_numpy()))
        out += storage_kind_cols(zone, "electricity")
        out += [
            ("Electrolyser load (-)", -ely.loc[zone]),
            ("Net line import", net_e.loc[zone]),
            ("External exchange", ext_e.loc[zone]),
            ("Load shedding", shed_e.loc[zone]),
            ("Dumped/curtailed (-)", -dmp_e.loc[zone]),
            ("Demand (-)", -dem_e.loc[zone]),
            ("H2 Producer wind (MW)", prod_wind.loc[zone]),
            ("H2 Producer pv (MW)", prod_pv.loc[zone]),
            ("H2 Producer battery discharge (MW)", prod_batt_dis.loc[zone]),
            ("H2 Producer battery charge (-) (MW)", -prod_batt_ch.loc[zone]),
            ("H2 Producer electrolyser load (-) (MW)", -prod_ely.loc[zone]),
            ("H2 Producer grid exchange (MW)", prod_grid_net.loc[zone]),
        ]
        if price_e is not None:
            out.append(("Marginal Price (EUR/MWh)", price_e.loc[zone]))
        return out

    def h2_cols(zone):
        out = [
            ("Electrolyser production", ely_prod.loc[zone]),
            ("Terminal import", term.loc[zone]),
            ("SMR production", smr.loc[zone]),
            ("Net pipeline import", net_h.loc[zone]),
            ("External exchange", ext_h.loc[zone]),
        ]
        out += storage_kind_cols(zone, "hydrogen")
        out += [
            ("Load shedding", shed_h.loc[zone]),
            ("Dumped/curtailed (-)", -dmp_h.loc[zone]),
            ("H2 plant consumption (-)", -h2_cons.loc[zone]),
            ("Demand (-)", -dem_h.loc[zone]),
            ("H2 Producer electrolyser production (MW)", prod_ely_h2.loc[zone]),
            ("H2 Producer tank discharge (MW)", prod_tank_dis.loc[zone]),
            ("H2 Producer tank charge (-) (MW)", -prod_tank_ch.loc[zone]),
            ("H2 Producer pipeline exchange (MW)", prod_h2_net.loc[zone]),
        ]
        if price_h is not None:
            out.append(("Marginal Price (EUR/MWh)", price_h.loc[zone]))
        return out

    return {"elec": build_table(elec_cols), "h2": build_table(h2_cols)}
