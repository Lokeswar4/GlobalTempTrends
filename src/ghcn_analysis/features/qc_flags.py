"""All 10 GHCN v4 QC flag handlers.

Each handler:
  - Accepts a DataFrame (does NOT mutate original — uses .copy() internally)
  - Returns a DataFrame with flagged VALUE{i} corrected per NOAA spec
  - Iterates range(1, 13) — includes December (VALUE12)

Bug fix from original notebook: range(1, 12) was used, silently skipping December.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

__all__ = [
    "apply_all_qc_flags",
    "handle_e_flag", "handle_w_flag", "handle_d_flag", "handle_k_flag",
    "handle_r_flag", "handle_l_flag", "handle_o_flag", "handle_t_flag",
    "handle_s_flag", "handle_m_flag", "handle_i_flag",
    "biweight_mean", "biweight_std",
]

logger = logging.getLogger(__name__)


def apply_all_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch all 10 handlers in sequence. Returns cleaned DataFrame."""
    df = handle_e_flag(df)
    df = handle_w_flag(df)
    df = handle_d_flag(df)
    df = handle_k_flag(df)
    df = handle_r_flag(df)
    df = handle_l_flag(df)
    df = handle_o_flag(df)
    df = handle_t_flag(df)
    df = handle_s_flag(df)
    df = handle_m_flag(df)
    df = handle_i_flag(df)
    return df


def _null_flagged(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Generic helper: set VALUE{i} → NaN where QCFLAG{i} == flag."""
    df = df.copy()
    for i in range(1, 13):
        mask = df[f"QCFLAG{i}"] == flag
        df.loc[mask, f"VALUE{i}"] = float("nan")
    return df


def handle_e_flag(df: pd.DataFrame) -> pd.DataFrame:
    """E flag: duplicate-station check — flagged VALUE → NaN."""
    return _null_flagged(df, "E")


def handle_w_flag(df: pd.DataFrame) -> pd.DataFrame:
    """W flag: world record exceedance — flagged VALUE → NaN."""
    return _null_flagged(df, "W")


def handle_d_flag(df: pd.DataFrame) -> pd.DataFrame:
    """D flag: duplicate record — flagged VALUE → NaN."""
    return _null_flagged(df, "D")


def handle_k_flag(df: pd.DataFrame) -> pd.DataFrame:
    """K flag: repeated-value run (≥5 consecutive identical months).

    Apply centred rolling mean (window=7, min_periods=1) to smooth flagged VALUES.
    """
    df = df.copy()
    for i in range(1, 13):
        mask = df[f"QCFLAG{i}"] == "K"
        if not mask.any():
            continue
        smoothed = (
            df[f"VALUE{i}"]
            .rolling(window=7, center=True, min_periods=1)
            .mean()
        )
        df.loc[mask, f"VALUE{i}"] = smoothed[mask]
    return df


def handle_r_flag(df: pd.DataFrame) -> pd.DataFrame:
    """R flag: range check failure — flagged VALUE → NaN."""
    return _null_flagged(df, "R")


def handle_l_flag(df: pd.DataFrame) -> pd.DataFrame:
    """L flag: isolated in time — flagged VALUE → NaN."""
    return _null_flagged(df, "L")


def handle_o_flag(df: pd.DataFrame) -> pd.DataFrame:
    """O flag: outlier (≥5 biweight std deviations from biweight mean).

    For each flagged VALUE{i}, compute biweight statistics row-wise across all
    12 monthly values of that row, then cap the flagged value at
    biweight_mean + 5 * biweight_std.

    Row-wise computation is used because temperature outliers are defined
    relative to a station's annual cycle, not relative to other stations.
    When biweight_std is 0 (near-flat row), fall back to mean + 5 * std
    (NumPy). If std is also 0, skip capping.
    """
    df = df.copy()
    value_cols = [f"VALUE{i}" for i in range(1, 13)]
    for i in range(1, 13):
        col = f"VALUE{i}"
        mask = df[f"QCFLAG{i}"] == "O"
        if not mask.any():
            continue
        for idx in df.index[mask]:
            row_vals = df.loc[idx, value_cols].dropna()
            if row_vals.empty:
                continue
            bm = biweight_mean(row_vals)
            bs = biweight_std(row_vals)
            if bs == 0:
                # Fall back to median + 5*MAD; if MAD is 0, cap at median
                x = np.asarray(row_vals, dtype=float)
                median = float(np.median(x))
                mad = float(np.median(np.abs(x - median)))
                cap = median + 5 * mad
            else:
                cap = bm + 5 * bs
            if df.loc[idx, col] > cap:
                df.loc[idx, col] = cap
    return df


def handle_t_flag(df: pd.DataFrame) -> pd.DataFrame:
    """T flag: temperature z-score vs IDW neighbour z-score ≥ 3.0 — NaN."""
    return _null_flagged(df, "T")


def handle_s_flag(df: pd.DataFrame) -> pd.DataFrame:
    """S flag: spatial consistency failure — NaN."""
    return _null_flagged(df, "S")


def handle_m_flag(df: pd.DataFrame) -> pd.DataFrame:
    """M flag: manually flagged as erroneous by NOAA — NaN. [NEW]"""
    return _null_flagged(df, "M")


def handle_i_flag(df: pd.DataFrame) -> pd.DataFrame:
    """I flag: internal consistency (TMIN > TMAX) — NaN. [NEW]"""
    return _null_flagged(df, "I")


# ---------------------------------------------------------------------------
# Biweight estimators (used by handle_o_flag only)
# ---------------------------------------------------------------------------

def biweight_mean(series: pd.Series) -> float:
    """Tukey's biweight location estimator. Robust to outliers."""
    x = np.asarray(series.dropna(), dtype=float)
    if len(x) == 0:
        return float("nan")
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) or 1.0
    u = (x - median) / (6.0 * mad)
    mask = np.abs(u) < 1.0
    if not mask.any():
        return median
    weights = (1 - u[mask] ** 2) ** 2
    return float(np.sum(weights * x[mask]) / np.sum(weights))


def biweight_std(series: pd.Series) -> float:
    """Tukey's biweight scale estimator. Robust to outliers."""
    x = np.asarray(series.dropna(), dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) or 1.0
    u = (x - median) / (6.0 * mad)
    mask = np.abs(u) < 1.0
    if not mask.any():
        return 0.0
    num = np.sum((x[mask] - median) ** 2 * (1 - u[mask] ** 2) ** 4)
    denom = np.abs(np.sum((1 - u[mask] ** 2) * (1 - 5 * u[mask] ** 2)))
    return float(np.sqrt(n * num) / denom) if denom > 0 else 0.0
