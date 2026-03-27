"""Tests for src/ghcn_analysis/features/cleanse.py — 6 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ghcn_analysis.features.cleanse import cleanse


def _make_df(n_years: int, nan_month: int | None = None) -> pd.DataFrame:
    """Station with n_years of data. Optionally inject NaN in one VALUE col."""
    rows = []
    for year in range(2024 - n_years, 2024):
        row = {"ID": "US000000001", "YEAR": year, "NAME": "TEST", "LATITUDE": 40.0,
               "LONGITUDE": -75.0, "STNELEV": 100.0, "Country Name": "United States"}
        for i in range(1, 13):
            row[f"VALUE{i}"] = float(i)
            row[f"QCFLAG{i}"] = ""
        if nan_month is not None:
            row[f"VALUE{nan_month}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _make_two_station_df(long_years: int, short_years: int) -> pd.DataFrame:
    """Two stations: one long-operating, one short-operating."""
    rows_long = _make_df(long_years)
    rows_short = _make_df(short_years).copy()
    rows_short["ID"] = "US000000002"
    # Shift short station years to be recent but short
    rows_short["YEAR"] = rows_short["YEAR"] + (long_years - short_years)
    return pd.concat([rows_long, rows_short], ignore_index=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_qcflag_cols_dropped():
    df = _make_df(100)
    result = cleanse(df, min_operational_years=80)
    assert not any(c.startswith("QCFLAG") for c in result.columns)


def test_no_nan_after_interpolation():
    """No NaN in VALUE1–12 after cleanse for a station with enough data."""
    df = _make_df(100, nan_month=6)  # inject NaN in June
    result = cleanse(df, min_operational_years=80)
    from ghcn_analysis import config
    for col in config.VALUE_COLS:
        assert not result[col].isna().any(), f"{col} has NaN after cleanse"


def test_ffill_bfill_not_deprecated():
    """Verify .ffill()/.bfill() used — no DeprecationWarning from fillna(method=...)."""
    import warnings
    df = _make_df(100, nan_month=1)  # NaN at start (edge case for ffill)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cleanse(df, min_operational_years=80)  # should not raise


def test_min_years_filter():
    """Station with 50 years removed; station with 100 years retained."""
    df = _make_two_station_df(long_years=100, short_years=50)
    result = cleanse(df, min_operational_years=80)
    retained_ids = result["ID"].unique()
    assert "US000000001" in retained_ids
    assert "US000000002" not in retained_ids


def test_mean_temp_computed():
    df = _make_df(100)
    result = cleanse(df, min_operational_years=80)
    assert "MEAN_TEMP" in result.columns
    # MEAN_TEMP should equal row-wise mean of VALUE1..12
    from ghcn_analysis import config
    expected = result[config.VALUE_COLS].mean(axis=1)
    pd.testing.assert_series_equal(
        result["MEAN_TEMP"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
        atol=0.01,
    )


def test_rounding():
    """VALUE1–12 should have max 2 decimal places."""
    df = _make_df(100)
    # Inject a value with many decimal places
    df.loc[df.index[0], "VALUE1"] = 12.3456789
    result = cleanse(df, min_operational_years=80)
    val = result.iloc[0]["VALUE1"]
    assert round(val, 2) == val
