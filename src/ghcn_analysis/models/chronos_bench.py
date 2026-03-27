"""Tier 4: Chronos zero-shot benchmark.

Runs Amazon Chronos T5 pretrained model on a random sample of CHRONOS_SAMPLE_N stations.
No training. Directly generates 12-month forecasts from the test split context window.

Install: pip install chronos-forecasting
Source: https://github.com/amazon-science/chronos-forecasting
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from ghcn_analysis import config

__all__ = ["run_chronos_benchmark"]

logger = logging.getLogger(__name__)


def run_chronos_benchmark(
    df: pd.DataFrame,
    sample_n: int = config.CHRONOS_SAMPLE_N,
    model_name: str = "amazon/chronos-t5-small",
) -> dict[str, dict]:
    """Run Chronos zero-shot benchmark on sample_n randomly selected stations.

    Args:
        df: Full multi-station DataFrame.
        sample_n: Number of stations to sample.
        model_name: HuggingFace model identifier for Chronos.

    Returns:
        {station_id: {
            "predictions": np.ndarray,  # shape (n_test_years, 12)
            "actuals":     np.ndarray,
            "model":       model_name,
        }}

    Side effect:
        Saves results to MODELS_DIR/chronos/chronos_results.pkl.
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:
        raise ImportError(
            "chronos-forecasting not installed. Run: pip install chronos-forecasting"
        ) from exc

    rng = np.random.default_rng(config.RANDOM_SEED)
    all_station_ids = df["ID"].unique()
    sampled_ids = rng.choice(
        all_station_ids,
        size=min(sample_n, len(all_station_ids)),
        replace=False,
    ).tolist()

    logger.info("Loading Chronos model: %s ...", model_name)
    pipeline = ChronosPipeline.from_pretrained(model_name, device_map="cpu")

    results: dict[str, dict] = {}

    for station_id in sampled_ids:
        sdf = df[df["ID"] == station_id].sort_values("YEAR")
        train_df = sdf[sdf["YEAR"] < config.TRAIN_CUTOFF_YEAR]
        test_df = sdf[sdf["YEAR"] >= config.TRAIN_CUTOFF_YEAR]

        if train_df.empty or test_df.empty:
            logger.warning("Skipping %s: missing train or test data", station_id)
            continue

        # Build context tensor: flatten annual→monthly (train years)
        context_vals = train_df[config.VALUE_COLS].values.flatten()
        context_tensor = torch.tensor(context_vals, dtype=torch.float32).unsqueeze(0)

        predictions_list = []
        actuals_list = []

        for _, test_row in test_df.iterrows():
            try:
                forecast = pipeline.predict(
                    context=context_tensor,
                    prediction_length=12,
                    num_samples=20,
                )
                # median forecast across samples
                median_forecast = np.quantile(
                    forecast[0].numpy(), 0.5, axis=0
                )
                predictions_list.append(median_forecast)
                actuals_list.append(test_row[config.VALUE_COLS].values.astype(float))
                # Extend context with actual observed values (rolling context)
                context_vals = np.append(context_vals, actuals_list[-1])
                context_tensor = torch.tensor(
                    context_vals, dtype=torch.float32
                ).unsqueeze(0)
            except Exception as exc:
                logger.warning(
                    "Chronos predict failed for %s year %s: %s",
                    station_id, test_row["YEAR"], exc,
                )

        if predictions_list:
            results[station_id] = {
                "predictions": np.array(predictions_list),
                "actuals": np.array(actuals_list),
                "model": model_name,
            }

    out_dir = config.MODELS_DIR / "chronos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "chronos_results.pkl"
    joblib.dump(results, out)
    logger.info("Saved Chronos results (%d stations) → %s", len(results), out)

    return results
