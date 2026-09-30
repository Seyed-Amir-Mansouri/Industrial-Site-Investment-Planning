"""Per-commodity config: demand/target columns, feature list, and I/O filenames."""
from __future__ import annotations

COMMODITIES = {
    "electricity": {
        "unit": "EUR/MWh",
        "demand": "demand",
        "target": "price_eur_mwh",
        "features": ["demand", "residual_load", "wind", "solar", "battery", "month", "season", "hour"],
        "samples": "elec_samples.parquet",
        "adjacency": "elec_adjacency.json",
        "net_demand_col": "residual_load",
        "max_price": 500,
        "model": "electricity_model.joblib",
        "metrics": "electricity_metrics.csv",
    },
    "hydrogen": {
        "unit": "EUR/MWhH2",
        "demand": "h2_demand",
        "target": "h2_price",
        "features": ["h2_demand", "smr", "electrolyser_gen", "storage", "month", "season", "hour"],
        "samples": "h2_samples.parquet",
        "adjacency": "h2_adjacency.json",
        "net_demand_col": None,
        "model": "hydrogen_model.joblib",
        "metrics": "hydrogen_metrics.csv",
    },
}
