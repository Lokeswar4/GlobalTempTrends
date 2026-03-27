"""Tier 1: Per-station RF/GB/MLP/LSTM with RandomizedSearchCV + keras_tuner.

Uses from tensorflow.keras (not from keras — deprecated).
Parallelised with joblib.Parallel(n_jobs=-1) over stations.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

from ghcn_analysis import config
from ghcn_analysis.models.split import temporal_split, to_supervised, to_supervised_lstm

__all__ = ["train_local_models"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------
_SKLEARN_PARAMS = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=config.RANDOM_SEED),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=config.RANDOM_SEED),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
        },
    },
    "NeuralNetwork": {
        "model": MLPRegressor(max_iter=1000, random_state=config.RANDOM_SEED),
        "params": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "learning_rate_init": [0.001, 0.01, 0.1],
        },
    },
}

_CV = TimeSeriesSplit(n_splits=5)
_N_ITER = 10
_SCORING = "neg_mean_squared_error"


def _train_sklearn_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, dict]:
    """Run RandomizedSearchCV for RF, GB, MLP. Returns dict keyed by model name."""
    results = {}
    for name, spec in _SKLEARN_PARAMS.items():
        search = RandomizedSearchCV(
            spec["model"],
            spec["params"],
            n_iter=_N_ITER,
            cv=_CV,
            scoring=_SCORING,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        avg_rmse = float(np.sqrt(-search.best_score_))
        results[name] = {"model": search.best_estimator_, "cv_rmse": avg_rmse}
    return results


def _train_lstm(
    X_train_lstm: np.ndarray,
    y_train: np.ndarray,
    station_id: str,
) -> tuple[object, float]:
    """Tune + train LSTM via keras_tuner.RandomSearch.

    Returns (best_model, avg_cv_rmse).
    Import from tensorflow.keras (not from keras).
    """
    import keras_tuner as kt
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    def build_model(hp: kt.HyperParameters) -> Sequential:
        units = hp.Int("units", 32, 256, step=32)
        use_second = hp.Boolean("use_second_lstm")
        optimizer = hp.Choice("optimizer", ["adam", "rmsprop"])

        model = Sequential()
        model.add(LSTM(units=units, activation="relu",
                       input_shape=(X_train_lstm.shape[1], 1),
                       return_sequences=use_second))
        if use_second:
            units2 = hp.Int("units_2", 32, 256, step=32)
            model.add(LSTM(units=units2, activation="relu"))
        model.add(Dense(y_train.shape[1]))
        model.compile(optimizer=optimizer, loss="mse")
        return model

    tuner_dir = config.MODELS_DIR / "local" / station_id / "keras_tuner"
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=1,
        directory=str(tuner_dir),
        project_name="lstm",
        overwrite=True,
    )
    split_idx = int(len(X_train_lstm) * 0.8)
    tuner.search(
        X_train_lstm[:split_idx], y_train[:split_idx],
        validation_data=(X_train_lstm[split_idx:], y_train[split_idx:]),
        epochs=20, batch_size=32, verbose=0,
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    val_loss = tuner.oracle.get_best_trials(1)[0].score
    avg_cv_rmse = float(np.sqrt(abs(val_loss))) if val_loss is not None else float("inf")
    return best_model, avg_cv_rmse


def _train_one_station(station_id: str, station_df: pd.DataFrame) -> tuple[str, dict] | None:
    """Train all models for one station. Returns (station_id, result_dict) or None."""
    train, test, scaler = temporal_split(station_df)
    if len(train) < 5:
        logger.warning("Skipping %s: fewer than 5 training years", station_id)
        return None

    X_train, y_train = to_supervised(train)
    X_test, y_test = to_supervised(test)
    X_train_lstm = to_supervised_lstm(train)

    if len(X_train) < 2:
        logger.warning("Skipping %s: insufficient supervised samples", station_id)
        return None

    # Sklearn models
    sklearn_results = _train_sklearn_models(X_train, y_train)

    # LSTM
    try:
        lstm_model, lstm_rmse = _train_lstm(X_train_lstm, y_train, station_id)
        sklearn_results["LSTM"] = {"model": lstm_model, "cv_rmse": lstm_rmse}
    except Exception as exc:
        logger.warning("LSTM failed for %s: %s", station_id, exc)

    # Select best
    best_name = min(sklearn_results, key=lambda k: sklearn_results[k]["cv_rmse"])
    best = sklearn_results[best_name]

    # Generate test predictions (inverse-transformed)
    if best_name == "LSTM":
        X_test_lstm = to_supervised_lstm(test)
        raw_pred = best["model"].predict(X_test_lstm).flatten()
    else:
        raw_pred = best["model"].predict(X_test).flatten()

    # Inverse transform: reshape to (n, 12) for scaler
    n_samples = len(raw_pred) // 12
    raw_matrix = raw_pred[: n_samples * 12].reshape(n_samples, 12)
    test_preds = scaler.inverse_transform(raw_matrix).flatten()
    test_actuals = scaler.inverse_transform(y_test[:n_samples]).flatten()

    # Save model + scaler
    out_dir = config.MODELS_DIR / "local" / station_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_name == "LSTM":
        best["model"].save(str(out_dir / "best_model.keras"))
    else:
        joblib.dump(best["model"], out_dir / "best_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    return station_id, {
        "best_model_name": best_name,
        "best_model": best["model"],
        "scaler": scaler,
        "cv_results": {k: v["cv_rmse"] for k, v in sklearn_results.items()},
        "test_predictions": test_preds,
        "test_actuals": test_actuals,
    }


def train_local_models(
    df: pd.DataFrame,
    station_id: str | None = None,
) -> dict[str, dict]:
    """Train RF/GB/MLP/LSTM for one or all stations.

    Args:
        df: Full multi-station DataFrame.
        station_id: Train only this station if provided; otherwise train all.

    Returns:
        {station_id: {"best_model_name", "best_model", "scaler",
                      "cv_results", "test_predictions", "test_actuals"}}
    """
    if station_id:
        station_ids = [station_id]
    else:
        station_ids = df["ID"].unique().tolist()

    logger.info("Training local models for %d stations ...", len(station_ids))

    results_list = joblib.Parallel(n_jobs=-1, verbose=5)(
        joblib.delayed(_train_one_station)(
            sid, df[df["ID"] == sid].copy()
        )
        for sid in station_ids
    )

    return {sid: res for sid, res in results_list if res is not None}
