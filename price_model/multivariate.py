"""Commodity-agnostic per-zone price model: train, cross-validate, and predict."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict

from .neighbors import add_candidate_neighbor_prices, add_neighbor_features

MIN_SAMPLES = 200


def _new_estimator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=20,
        l2_regularization=1.0, random_state=0,
    )


def _cv_scores(X: np.ndarray, y: np.ndarray, k: int = 5):
    """Cross-validated R^2 and RMSE via out-of-fold predictions (shuffled folds)."""
    kf = KFold(n_splits=min(k, len(y)), shuffle=True, random_state=0)
    pred = cross_val_predict(_new_estimator(), X, y, cv=kf)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    return r2, rmse


def demand_only_row(features: list[str], demand_col: str, medians: dict[str, float],
                    demand_values, context: dict | None = None) -> pd.DataFrame:
    """Build a feature row with demand (+ overrides) set and everything else at its zone median."""
    context = context or {}
    n = len(demand_values)
    row = {f: np.full(n, medians[f], dtype=float) for f in features}
    row[demand_col] = np.asarray(demand_values, dtype=float)
    for k, v in context.items():
        row[k] = np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, float)
    if "residual_load" in features and "residual_load" not in context:
        row["residual_load"] = row[demand_col] - row.get("wind", 0) - row.get("solar", 0)
    return pd.DataFrame(row)[features]


def train_zone(df_zone: pd.DataFrame, target: str, features: list[str], demand: str):
    """Train + CV-score one zone on its active hours (demand > 0)."""
    d = df_zone.dropna(subset=features + [target])
    d = d[d[demand] > 0]
    if len(d) < MIN_SAMPLES:
        return None
    X = d[features].to_numpy()
    y = d[target].to_numpy()

    cv_r2, cv_rmse = _cv_scores(X, y)
    model = _new_estimator().fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=3, random_state=0)
    medians = {f: float(d[f].median()) for f in features}

    X_demand_only = demand_only_row(features, demand, medians, d[demand].to_numpy()).to_numpy()
    pred_demand_only = model.predict(X_demand_only)
    ss_res = float(np.sum((y - pred_demand_only) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    demand_only_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    demand_only_rmse = float(np.sqrt(ss_res / len(y)))

    return {
        "model": model,
        "cv_r2": cv_r2, "cv_rmse": cv_rmse,
        "demand_only_r2": demand_only_r2, "demand_only_rmse": demand_only_rmse,
        "n": int(len(d)),
        "importances": dict(zip(features, (float(v) for v in perm.importances_mean))),
        "medians": medians,
        "features": features,
    }


def train_all(df: pd.DataFrame, commodity: str, target: str, features: list[str],
              demand: str, unit: str, adjacency: dict[str, list[str]] | None = None,
              net_demand_col: str | None = None, max_price: float | None = None) -> dict:
    """Train a model per zone, adding neighbour/system-total and top-5 correlated-price features. Returns a joblib-serialisable bundle."""
    zone_extra: dict[str, list[str]] = {}
    if adjacency:
        df, zone_extra = add_neighbor_features(df, demand, adjacency, net_demand_col)

    neighbor_price_candidates: dict[str, dict[str, list[str]]] = {}
    if adjacency:
        df, neighbor_price_candidates = add_candidate_neighbor_prices(df, target, adjacency)

    zones = {}
    for zone, g in df.groupby("zone", sort=True):
        if max_price is not None:
            g = g[g[target] <= max_price]
        zone_features = features + zone_extra.get(zone, [])
        cols = neighbor_price_candidates.get(zone, {}).get("top_n", [])
        if cols:
            zone_features = zone_features + cols
        best = train_zone(g, target, zone_features, demand)
        if best is None:
            continue
        if cols:
            best["neighbor_price_method"] = "top_n"
        zones[zone] = best
    return {"commodity": commodity, "target": target, "features": features,
            "demand": demand, "unit": unit, "zones": zones}


def predict(bundle: dict, zone: str, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict price for ``zone`` from a frame/array of that zone's own features."""
    entry = bundle["zones"][zone]
    feats = entry.get("features", bundle["features"])
    if isinstance(X, pd.DataFrame):
        X = X[feats].to_numpy()
    return entry["model"].predict(np.asarray(X, float))
