"""Per-zone hourly series (marginal prices, DSR-implicit and other adjustments) read back from parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_EXPORTS_DIR, DEFAULT_MARGINAL_PRICE_ELEC_DB, DEFAULT_MARGINAL_PRICE_H2_DB

DEFAULT_DSR_IMPLICIT_DB = DEFAULT_EXPORTS_DIR / "dsr_implicit_electricity_2030.parquet"
DEFAULT_WIND_ONSHORE_DB = DEFAULT_EXPORTS_DIR / "wind_onshore_electricity_2030.parquet"
DEFAULT_WIND_OFFSHORE_DB = DEFAULT_EXPORTS_DIR / "wind_offshore_electricity_2030.parquet"
DEFAULT_ROR_DB = DEFAULT_EXPORTS_DIR / "ror_electricity_2030.parquet"
DEFAULT_SOLAR_PV_DB = DEFAULT_EXPORTS_DIR / "solar_pv_electricity_2030.parquet"
DEFAULT_SOLAR_THERMAL_DB = DEFAULT_EXPORTS_DIR / "solar_thermal_electricity_2030.parquet"
DEFAULT_OTHER_RES_DB = DEFAULT_EXPORTS_DIR / "other_res_electricity_2030.parquet"
DEFAULT_HYDRO_RESERVOIR_DB = DEFAULT_EXPORTS_DIR / "hydro_reservoir_electricity_2030.parquet"
DEFAULT_HYDRO_PONDAGE_DB = DEFAULT_EXPORTS_DIR / "hydro_pondage_electricity_2030.parquet"
DEFAULT_HYDRO_OPEN_PS_DB = DEFAULT_EXPORTS_DIR / "hydro_open_ps_electricity_2030.parquet"


def load_zone_series(names: list[str], hours: pd.Index,
                     db_path: Path = DEFAULT_MARGINAL_PRICE_ELEC_DB) -> pd.DataFrame:
    """Read back a subset of zones/countries over an hour window from a wide parquet database."""
    df = pd.read_parquet(db_path)
    h0, h1 = int(hours[0]), int(hours[-1]) + 1
    out = pd.DataFrame(index=hours)
    for n in names:
        out[n] = df[n].to_numpy()[h0:h1] if n in df.columns else np.zeros(len(hours))
    return out
