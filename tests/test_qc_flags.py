"""Tests for src/ghcn_analysis/features/qc_flags.py — 13 tests.

Key regression guard: VALUE12 (December) must be processed by all handlers.
The original notebook used range(1, 12), skipping December.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ghcn_analysis.features.qc_flags import (
    apply_all_qc_flags,
    handle_d_flag,
    handle_e_flag,
    handle_i_flag,
    handle_k_flag,
    handle_l_flag,
    handle_m_flag,
    handle_o_flag,
    handle_r_flag,
    handle_s_flag,
    handle_t_flag,
    handle_w_flag,
)


def _make_df(flag_col: str, flag_month: int, value: float = 20.0) -> pd.DataFrame:
    """One-row DataFrame with a single flagged month."""
    row = {"ID": "TEST0000001", "YEAR": 2000}
    for i in range(1, 13):
        row[f"VALUE{i}"] = value
        row[f"QCFLAG{i}"] = flag_col if i == flag_month else ""
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Simple null-on-flag handlers (E, W, D, R, L, T, S, M, I)
# ---------------------------------------------------------------------------

def test_e_flag_nulls_value():
    df = _make_df("E", flag_month=5)
    result = handle_e_flag(df)
    assert pd.isna(result.iloc[0]["VALUE5"])
    assert not pd.isna(result.iloc[0]["VALUE1"])  # unflagged month unchanged


def test_w_flag_nulls_value():
    df = _make_df("W", flag_month=3)
    result = handle_w_flag(df)
    assert pd.isna(result.iloc[0]["VALUE3"])
    assert not pd.isna(result.iloc[0]["VALUE4"])


def test_d_flag_nulls_value():
    df = _make_df("D", flag_month=7)
    result = handle_d_flag(df)
    assert pd.isna(result.iloc[0]["VALUE7"])


def test_r_flag_value12():
    """VALUE12 nulled when QCFLAG12='R' — regression guard for range(1,12) bug."""
    df = _make_df("R", flag_month=12)
    result = handle_r_flag(df)
    assert pd.isna(result.iloc[0]["VALUE12"])


def test_l_flag_value12():
    df = _make_df("L", flag_month=12)
    result = handle_l_flag(df)
    assert pd.isna(result.iloc[0]["VALUE12"])


def test_t_flag_value12():
    df = _make_df("T", flag_month=12)
    result = handle_t_flag(df)
    assert pd.isna(result.iloc[0]["VALUE12"])


def test_s_flag_value12():
    df = _make_df("S", flag_month=12)
    result = handle_s_flag(df)
    assert pd.isna(result.iloc[0]["VALUE12"])


def test_m_flag_nulls_value():
    """M flag (new in refactor): manually flagged → NaN."""
    df = _make_df("M", flag_month=6)
    result = handle_m_flag(df)
    assert pd.isna(result.iloc[0]["VALUE6"])


def test_i_flag_nulls_value():
    """I flag (new in refactor): internal consistency failure → NaN."""
    df = _make_df("I", flag_month=11)
    result = handle_i_flag(df)
    assert pd.isna(result.iloc[0]["VALUE11"])


# ---------------------------------------------------------------------------
# Special handlers
# ---------------------------------------------------------------------------

def test_k_flag_nulls():
    """K flag: repeated-value run — VALUE → NaN per NOAA spec."""
    df = _make_df("K", flag_month=6, value=20.0)
    result = handle_k_flag(df)
    assert pd.isna(result.iloc[0]["VALUE6"])
    assert not pd.isna(result.iloc[0]["VALUE1"])  # unflagged month unchanged


def test_o_flag_nulls():
    """O flag: outlier value → NaN per NOAA spec."""
    df = _make_df("O", flag_month=3)
    result = handle_o_flag(df)
    assert pd.isna(result.iloc[0]["VALUE3"])
    assert not pd.isna(result.iloc[0]["VALUE1"])  # unflagged month unchanged


def test_apply_all_qc_flags_nulls_each():
    """apply_all_qc_flags: each flag character nulls the corresponding VALUE."""
    flags = list("EDRKWILOSTM")
    row = {"ID": "TEST0000001", "YEAR": 2000}
    for i, flag in enumerate(flags, start=1):
        row[f"VALUE{i}"] = float(i * 5)
        row[f"QCFLAG{i}"] = flag
    for i in range(len(flags) + 1, 13):
        row[f"VALUE{i}"] = float(i * 5)
        row[f"QCFLAG{i}"] = ""
    df = pd.DataFrame([row])
    result = apply_all_qc_flags(df)
    for i in range(1, len(flags) + 1):
        assert pd.isna(result.iloc[0][f"VALUE{i}"]), f"VALUE{i} should be NaN (flag={flags[i-1]})"


def test_unflagged_unchanged():
    """Rows with blank QCFLAG untouched by all handlers."""
    row = {"ID": "TEST0000001", "YEAR": 2000}
    for i in range(1, 13):
        row[f"VALUE{i}"] = float(i * 5)
        row[f"QCFLAG{i}"] = ""
    df = pd.DataFrame([row])

    for handler in [
        handle_e_flag, handle_w_flag, handle_d_flag, handle_k_flag,
        handle_r_flag, handle_l_flag, handle_o_flag, handle_t_flag,
        handle_s_flag, handle_m_flag, handle_i_flag,
    ]:
        result = handler(df)
        for i in range(1, 13):
            assert result.iloc[0][f"VALUE{i}"] == pytest.approx(float(i * 5))
