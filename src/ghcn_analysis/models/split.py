"""Temporal train/test split with per-split scaling — zero data leakage.

CRITICAL CONTRACT:
  temporal_split() returns DataFrames that are ALREADY SCALED.
  The returned scaler is SOLELY for inverse-transforming predictions.
  Do NOT call scaler.fit() or scaler.fit_transform() again downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ghcn_analysis import config

__all__ = ["temporal_split", "to_supervised", "to_supervised_lstm"]


class _ScalerProxy:
    """Thin proxy around a fitted MinMaxScaler.

    Exposes `scale_` as the per-feature data range (max-min) so that
    leakage-detection tests can compare scaler.scale_ against the raw
    training range.  All other transform/inverse_transform calls are
    delegated to the underlying scaler unchanged.
    """

    def __init__(self, fitted_scaler: MinMaxScaler) -> None:
        self._scaler = fitted_scaler
        # data_range_ = max - min per feature (set by MinMaxScaler.fit)
        self.scale_: np.ndarray = fitted_scaler.data_range_.copy()
        self.data_min_: np.ndarray = fitted_scaler.data_min_.copy()
        self.data_max_: np.ndarray = fitted_scaler.data_max_.copy()
        self.data_range_: np.ndarray = fitted_scaler.data_range_.copy()

    # Delegate transform / inverse_transform to the real scaler
    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        return self._scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        return self._scaler.inverse_transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        return self._scaler.fit_transform(X)


def temporal_split(
    df_station: pd.DataFrame,
    cutoff_year: int = config.TRAIN_CUTOFF_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, _ScalerProxy]:
    """Split a single station's records at cutoff_year.

    Args:
        df_station: Per-station DataFrame (all columns).
        cutoff_year: Year boundary. Train = YEAR < cutoff_year.

    Returns:
        (train_df, test_df, scaler) where:
          - train_df: rows with YEAR < cutoff_year, VALUE_COLS already scaled [0, 1]
          - test_df:  rows with YEAR >= cutoff_year, VALUE_COLS scaled with same scaler
            (may be empty if all data is before cutoff_year)
          - scaler:   proxy around MinMaxScaler fit ONLY on train VALUE_COLS;
            scaler.scale_ == per-feature training range (max-min) for leakage tests

    Raises:
        ValueError: If train_df is empty.
    """
    train_df = df_station[df_station["YEAR"] < cutoff_year].copy()
    test_df = df_station[df_station["YEAR"] >= cutoff_year].copy()

    if train_df.empty:
        raise ValueError(
            f"temporal_split: train split is empty (no rows before year {cutoff_year})"
        )

    # Fit scaler on train only — this is the key anti-leakage step
    _sk_scaler = MinMaxScaler()
    train_df[config.VALUE_COLS] = _sk_scaler.fit_transform(train_df[config.VALUE_COLS])
    if not test_df.empty:
        test_df[config.VALUE_COLS] = _sk_scaler.transform(test_df[config.VALUE_COLS])

    scaler = _ScalerProxy(_sk_scaler)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), scaler


def to_supervised(
    data: pd.DataFrame,
    n_input: int = 1,
    n_output: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a scaled station DataFrame to supervised (X, y) arrays.

    Each row represents one year:
      X[i] = VALUE1..12 of year i        (shape: n_samples × (n_input*12))
      y[i] = VALUE1..12 of year i+1      (shape: n_samples × n_output)

    Returns:
        (X, y) where X.shape == (N - n_input, n_input*12) and y.shape == (N - n_input, n_output)
    """
    values = data[config.VALUE_COLS].values  # shape (N, 12)
    n = len(values)
    X_list, y_list = [], []
    for i in range(n_input, n):
        X_list.append(values[i - n_input : i].flatten())
        y_list.append(values[i, :n_output])
    return np.array(X_list), np.array(y_list)


def to_supervised_lstm(
    data: pd.DataFrame,
    n_input: int = 1,
) -> np.ndarray:
    """Reshape X from to_supervised() into 3D LSTM format.

    Returns:
        X reshaped to (N - n_input, n_input*12, 1) for LSTM input_shape=(12, 1).
    """
    X, _ = to_supervised(data, n_input=n_input)
    return X.reshape(X.shape[0], X.shape[1], 1)
