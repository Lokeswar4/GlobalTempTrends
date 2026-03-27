"""Tests for models/split.py — 10 leakage prevention + shape tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ghcn_analysis.models.split import temporal_split, to_supervised, to_supervised_lstm


def _make_station_df(start: int = 1900, end: int = 2024) -> pd.DataFrame:
    """Synthetic single-station DataFrame with sequential years."""
    rows = []
    for year in range(start, end + 1):
        row = {"ID": "US000000001", "YEAR": year}
        for i in range(1, 13):
            row[f"VALUE{i}"] = float(year - 1900 + i) / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# temporal_split tests
# ---------------------------------------------------------------------------

def test_train_test_year_boundary():
    df = _make_station_df()
    train, test, _ = temporal_split(df, cutoff_year=2000)
    assert train["YEAR"].max() < 2000
    assert test["YEAR"].min() >= 2000


def test_no_leakage_scaler_mean():
    """scaler.mean_ must match train VALUE_COLS mean, not full-dataset mean."""
    from ghcn_analysis import config
    df = _make_station_df()
    train_raw = df[df["YEAR"] < 2000][config.VALUE_COLS].values
    _, _, scaler = temporal_split(df, cutoff_year=2000)
    np.testing.assert_allclose(
        scaler.data_min_,
        train_raw.min(axis=0),
        rtol=1e-4,
        err_msg="Scaler min_ derived from full dataset (data leakage!)",
    )


def test_no_leakage_scaler_scale():
    """scaler.scale_ derived from train data only."""
    from ghcn_analysis import config
    df = _make_station_df()
    train_raw = df[df["YEAR"] < 2000][config.VALUE_COLS].values
    _, _, scaler = temporal_split(df, cutoff_year=2000)
    # If scaler saw test data, scale_ would differ
    full_range = df[config.VALUE_COLS].values.max(axis=0) - df[config.VALUE_COLS].values.min(axis=0)
    train_range = train_raw.max(axis=0) - train_raw.min(axis=0)
    # Scale should match train range, not full-data range
    np.testing.assert_allclose(scaler.scale_, train_range, rtol=1e-3)


def test_already_scaled_output():
    """Returned train_df VALUES are in [0, 1]; raw values not present."""
    from ghcn_analysis import config
    df = _make_station_df()
    train, test, _ = temporal_split(df, cutoff_year=2000)
    for col in config.VALUE_COLS:
        assert train[col].min() >= -1e-6
        assert train[col].max() <= 1.0 + 1e-6


def test_scaler_inverse():
    """scaler.inverse_transform(train_df[VALUE_COLS]) recovers original values."""
    from ghcn_analysis import config
    df = _make_station_df()
    train_original = df[df["YEAR"] < 2000][config.VALUE_COLS].values.copy()
    train, _, scaler = temporal_split(df, cutoff_year=2000)
    recovered = scaler.inverse_transform(train[config.VALUE_COLS].values)
    np.testing.assert_allclose(recovered, train_original, rtol=1e-4)


def test_empty_train_raises():
    """ValueError when no rows before cutoff_year."""
    df = _make_station_df(start=2010, end=2024)
    with pytest.raises(ValueError, match="empty"):
        temporal_split(df, cutoff_year=2000)


# ---------------------------------------------------------------------------
# to_supervised tests
# ---------------------------------------------------------------------------

def test_to_supervised_shape():
    """X.shape == (N-1, 12), y.shape == (N-1, 12) for n_input=1."""
    from ghcn_analysis import config
    df = _make_station_df(start=1900, end=1950)
    train, _, _ = temporal_split(df, cutoff_year=2000)
    X, y = to_supervised(train)
    n = len(train)
    assert X.shape == (n - 1, 12)
    assert y.shape == (n - 1, 12)


def test_to_supervised_content():
    """X[i] == VALUE1..12 of year i; y[i] == VALUE1..12 of year i+1."""
    from ghcn_analysis import config
    df = _make_station_df(start=1900, end=1910)
    train, _, _ = temporal_split(df, cutoff_year=2000)
    X, y = to_supervised(train)
    # First input row should equal first row of train VALUE cols
    np.testing.assert_allclose(X[0], train[config.VALUE_COLS].iloc[0].values, rtol=1e-5)
    # First target row should equal second row of train VALUE cols
    np.testing.assert_allclose(y[0], train[config.VALUE_COLS].iloc[1].values, rtol=1e-5)


# ---------------------------------------------------------------------------
# to_supervised_lstm test
# ---------------------------------------------------------------------------

def test_to_supervised_lstm_shape():
    """X_lstm.shape == (N-1, 12, 1) for n_input=1."""
    df = _make_station_df(start=1900, end=1950)
    train, _, _ = temporal_split(df, cutoff_year=2000)
    X, _ = to_supervised(train)
    from ghcn_analysis.models.split import to_supervised_lstm
    X_lstm = to_supervised_lstm(train)
    n = len(train)
    assert X_lstm.shape == (n - 1, 12, 1)


# ---------------------------------------------------------------------------
# Cluster label coverage (uses fit_clusters — requires cluster.py)
# ---------------------------------------------------------------------------
# test_cluster_labels_coverage and test_cluster_label_range are in test_cluster.py (Part 4)
