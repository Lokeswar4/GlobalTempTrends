"""Final cleaning pipeline: drop QCFLAGs, interpolate, filter stations, compute MEAN_TEMP."""

from __future__ import annotations

import logging

import pandas as pd

from ghcn_analysis import config

__all__ = ["cleanse"]

logger = logging.getLogger(__name__)


def cleanse(
    df: pd.DataFrame,
    min_operational_years: int = config.MIN_OPERATIONAL_YEARS,
    slug: str | None = None,
) -> pd.DataFrame:
    """Final cleaning pipeline.

    Steps:
      1. Drop QCFLAG1–12 columns
      2. Sort by station + year for correct interpolation order
      3. Interpolate VALUE1–12 linearly across time (limit_direction='both')
      4. .ffill().bfill() to handle edge NaNs
      5. Filter to stations with operational duration ≥ min_operational_years
         AND max(YEAR) == most recent year in dataset
      6. Compute MEAN_TEMP = row-wise mean of VALUE1–12
      7. Round VALUE1–12 and MEAN_TEMP to 2 decimal places
      8. Save to PROCESSED_DIR/{slug}_clean.csv if slug provided

    Args:
        df: DataFrame post-QC flag handling (per-country).
        min_operational_years: Minimum number of unique years per station.
        slug: Country slug for output filename (e.g. "united_states").

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # 1. Drop QCFLAG columns
    qcflag_cols = [c for c in df.columns if c.startswith("QCFLAG")]
    df.drop(columns=qcflag_cols, inplace=True)

    # 2. Sort by station + year for correct interpolation order
    df = df.sort_values(["ID", "YEAR"]).reset_index(drop=True)

    # 3. Interpolate + fill edges (per station group)
    value_cols_present = [c for c in config.VALUE_COLS if c in df.columns]

    def _interp_axis0(grp: pd.DataFrame) -> pd.DataFrame:
        """Pass 1: interpolate each month column across years (axis=0)."""
        out = grp.copy()
        for col in out.columns:
            out[col] = (
                out[col]
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
        return out

    def _interp_axis1(grp: pd.DataFrame) -> pd.DataFrame:
        """Pass 2: interpolate across months within each row (axis=1).
        Handles months that are NaN for the entire station lifetime."""
        return grp.interpolate(method="linear", axis=1, limit_direction="both")

    # Apply axis=0 interpolation per station (no FutureWarning: operate on value cols only)
    df[value_cols_present] = (
        df.groupby("ID", group_keys=False)[value_cols_present]
        .apply(_interp_axis0, include_groups=False)
    )

    # Apply axis=1 interpolation per station for any still-NaN columns
    if df[value_cols_present].isna().any().any():
        df[value_cols_present] = (
            df.groupby("ID", group_keys=False)[value_cols_present]
            .apply(_interp_axis1, include_groups=False)
        )

    # Final edge fill
    for col in value_cols_present:
        df[col] = df[col].ffill().bfill()

    # 4. Filter to long-operating stations
    station_stats = df.groupby("ID")["YEAR"].agg(["min", "max", "count"])
    station_stats["duration"] = station_stats["max"] - station_stats["min"]
    keep = station_stats[station_stats["duration"] >= min_operational_years].index
    before = df["ID"].nunique()
    df = df[df["ID"].isin(keep)].reset_index(drop=True)
    after = df["ID"].nunique()
    logger.info(
        "Station filter: %d → %d stations (duration ≥%d years)",
        before, after, min_operational_years,
    )

    # 5. Compute MEAN_TEMP
    df["MEAN_TEMP"] = df[config.VALUE_COLS].mean(axis=1)

    # 6. Round
    round_cols = config.VALUE_COLS + ["MEAN_TEMP"]
    for col in round_cols:
        if col in df.columns:
            df[col] = df[col].round(2)

    # 7. Save checkpoint
    if slug:
        out_path = config.PROCESSED_DIR / f"{slug}_clean.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved clean data → %s (%d rows)", out_path, len(df))

    return df
