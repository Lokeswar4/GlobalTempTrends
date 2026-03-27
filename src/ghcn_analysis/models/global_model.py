"""Tier 3: Global LightGBM via MLForecast — one model for all stations."""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalMinMaxScaler

from ghcn_analysis import config

__all__ = ["build_long_format", "train_global_model", "predict_global"]

logger = logging.getLogger(__name__)


def build_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide per-station DataFrame to MLForecast long format.

    Input: wide DataFrame with VALUE1–12 per station-year row.
    Output: long DataFrame with columns:
      unique_id: station ID (str)
      ds:        monthly date (pd.Timestamp, YYYY-MM-01)
      y:         temperature value (float)
      LATITUDE, LONGITUDE, STNELEV: repeated static covariates
    """
    rows = []
    static_cols = [c for c in ["LATITUDE", "LONGITUDE", "STNELEV"] if c in df.columns]

    for _, row in df.iterrows():
        for month_idx, col in enumerate(config.VALUE_COLS, start=1):
            val = row[col]
            if pd.isna(val):
                continue
            rows.append({
                "unique_id": row["ID"],
                "ds": pd.Timestamp(year=int(row["YEAR"]), month=month_idx, day=1),
                "y": float(val),
                **{c: row[c] for c in static_cols},
            })

    long_df = pd.DataFrame(rows)
    long_df["ds"] = pd.to_datetime(long_df["ds"])
    return long_df


def train_global_model(df: pd.DataFrame) -> MLForecast:
    """Train MLForecast LightGBM on all stations (train rows only).

    MLForecast config:
      models = [LGBMRegressor(n_estimators=200, random_state=RANDOM_SEED)]
      freq = "MS"  (month start)
      lags = [1, 2, 3, 6, 12]
      date_features = ["month", "year"]
      target_transforms = [LocalMinMaxScaler()]
      static_features = ["LATITUDE", "LONGITUDE", "STNELEV"]

    Filters to ds.year < TRAIN_CUTOFF_YEAR before fitting.
    Saves to MODELS_DIR/global/global_lgbm.pkl.
    Returns fitted MLForecast object.
    """
    long_df = build_long_format(df)
    train_df = long_df[long_df["ds"].dt.year < config.TRAIN_CUTOFF_YEAR]
    logger.info(
        "Training global model on %d rows (%d stations)",
        len(train_df), train_df["unique_id"].nunique(),
    )

    static_cols = [c for c in ["LATITUDE", "LONGITUDE", "STNELEV"] if c in train_df.columns]

    mlf = MLForecast(
        models=[LGBMRegressor(n_estimators=200, random_state=config.RANDOM_SEED)],
        freq="MS",
        lags=[1, 2, 3, 6, 12],
        date_features=["month", "year"],
        target_transforms=[LocalMinMaxScaler()],
    )
    mlf.fit(train_df, static_features=static_cols)

    out_dir = config.MODELS_DIR / "global"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "global_lgbm.pkl"
    joblib.dump(mlf, out)
    logger.info("Saved global model → %s", out)

    return mlf


def predict_global(mlf: MLForecast, h: int = 24 * 12) -> pd.DataFrame:
    """Run mlf.predict(h=h). Returns long-format predictions DataFrame.

    h: number of future monthly steps (default 24 years × 12 = 288 months)
    """
    return mlf.predict(h=h)
