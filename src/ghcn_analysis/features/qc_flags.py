"""All 11 GHCN v4 QC flag handlers.

Each handler:
  - Accepts a DataFrame (does NOT mutate original — uses .copy() internally)
  - Returns a DataFrame with flagged VALUE{i} set to NaN per NOAA spec
  - Iterates range(1, 13) — includes December (VALUE12)

Handler dispatch order matches NOAA spec: E→D→R→K→W→I→L→O→S→T→M

Bug fix from original notebook: range(1, 12) was used, silently skipping December.
"""

from __future__ import annotations

import logging

import pandas as pd

__all__ = [
    "apply_all_qc_flags",
    "handle_e_flag", "handle_d_flag", "handle_r_flag", "handle_k_flag",
    "handle_w_flag", "handle_i_flag", "handle_l_flag", "handle_o_flag",
    "handle_s_flag", "handle_t_flag", "handle_m_flag",
]

logger = logging.getLogger(__name__)


def apply_all_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch all 11 handlers in NOAA spec order. Returns cleaned DataFrame.

    Order: E→D→R→K→W→I→L→O→S→T→M
    """
    df = handle_e_flag(df)
    df = handle_d_flag(df)
    df = handle_r_flag(df)
    df = handle_k_flag(df)
    df = handle_w_flag(df)
    df = handle_i_flag(df)
    df = handle_l_flag(df)
    df = handle_o_flag(df)
    df = handle_s_flag(df)
    df = handle_t_flag(df)
    df = handle_m_flag(df)
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
    """W flag: duplicated from prior month (regional/spatial check, 2000-present) — VALUE → NaN."""
    return _null_flagged(df, "W")


def handle_d_flag(df: pd.DataFrame) -> pd.DataFrame:
    """D flag: duplicate record — flagged VALUE → NaN."""
    return _null_flagged(df, "D")


def handle_k_flag(df: pd.DataFrame) -> pd.DataFrame:
    """K flag: run of 5+ consecutive identical non-missing values — flagged VALUE → NaN.

    Per NOAA spec: identifies suspect repeated-value runs. Treated identically
    to other QC flags — value nulled, not smoothed.
    """
    return _null_flagged(df, "K")


def handle_r_flag(df: pd.DataFrame) -> pd.DataFrame:
    """R flag: exceeds known world TAVG extremes — flagged VALUE → NaN."""
    return _null_flagged(df, "R")


def handle_l_flag(df: pd.DataFrame) -> pd.DataFrame:
    """L flag: isolated in time — flagged VALUE → NaN."""
    return _null_flagged(df, "L")


def handle_o_flag(df: pd.DataFrame) -> pd.DataFrame:
    """O flag: outlier >= 5 biweight std deviations from biweight mean — VALUE → NaN."""
    return _null_flagged(df, "O")


def handle_t_flag(df: pd.DataFrame) -> pd.DataFrame:
    """T flag: temperature z-score vs IDW neighbour z-score ≥ 3.0 — NaN."""
    return _null_flagged(df, "T")


def handle_s_flag(df: pd.DataFrame) -> pd.DataFrame:
    """S flag: spatial consistency failure — NaN."""
    return _null_flagged(df, "S")


def handle_m_flag(df: pd.DataFrame) -> pd.DataFrame:
    """M flag: manually flagged as erroneous by NOAA — NaN."""
    return _null_flagged(df, "M")


def handle_i_flag(df: pd.DataFrame) -> pd.DataFrame:
    """I flag: internal consistency (TMIN > TMAX) — NaN."""
    return _null_flagged(df, "I")
