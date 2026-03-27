"""Phase 6: Evaluation — per-station RMSE/MAE for all 4 tiers + comparison figures."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ghcn_analysis import config

__all__ = [
    "generate_local_predictions",
    "generate_cluster_predictions",
    "generate_global_predictions",
    "compute_metrics",
    "evaluate_all_tiers",
    "summarise_tiers",
    "plot_tier_comparison",
    "plot_global_vs_local",
    "plot_chronos_vs_trained",
    "plot_rmse_by_cluster",
]

logger = logging.getLogger(__name__)


def generate_local_predictions(
    df: pd.DataFrame,
    local_results: dict,
) -> pd.DataFrame:
    """Generate test predictions for all stations with local models.

    Returns tidy DataFrame: [station_id, year, month, predicted, actual].
    Predictions in original (inverse-transformed) scale.
    """
    rows = []
    for station_id, res in local_results.items():
        preds = res["test_predictions"]
        actuals = res["test_actuals"]
        n = min(len(preds), len(actuals))
        for idx in range(n):
            rows.append({
                "station_id": station_id,
                "predicted": float(preds[idx]),
                "actual": float(actuals[idx]),
            })
    return pd.DataFrame(rows)


def generate_cluster_predictions(
    df: pd.DataFrame,
    cluster_models: dict,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """Generate cluster-model predictions for all stations.

    Returns tidy DataFrame same format as generate_local_predictions.
    """
    from ghcn_analysis.models.cluster import predict_for_station
    from ghcn_analysis.models.split import temporal_split, to_supervised

    rows = []
    for station_id, cluster_id in cluster_labels.items():
        if cluster_id not in cluster_models:
            continue
        sdf = df[df["ID"] == station_id]
        if sdf.empty:
            continue
        try:
            _, test, scaler = temporal_split(sdf)
            _, y_test = to_supervised(test)
            preds = predict_for_station(sdf, cluster_id, cluster_models)
            actuals = scaler.inverse_transform(y_test).flatten()
            n = min(len(preds), len(actuals))
            for idx in range(n):
                rows.append({
                    "station_id": station_id,
                    "predicted": float(preds[idx]),
                    "actual": float(actuals[idx]),
                })
        except Exception as exc:
            logger.warning("Cluster prediction failed for %s: %s", station_id, exc)

    return pd.DataFrame(rows)


def generate_global_predictions(
    df: pd.DataFrame,
    mlf: object,
) -> pd.DataFrame:
    """Run mlf.predict() for test period; join with actuals.

    Returns tidy DataFrame: [station_id, year, month, predicted, actual].
    """
    from ghcn_analysis.models.global_model import build_long_format

    long_df = build_long_format(df)
    test_long = long_df[long_df["ds"].dt.year >= config.TRAIN_CUTOFF_YEAR]

    h = int(test_long["ds"].dt.year.max() - config.TRAIN_CUTOFF_YEAR + 1) * 12
    preds_df = mlf.predict(h=h)

    # Rename LGBMRegressor column to 'predicted'
    pred_col = [c for c in preds_df.columns if c not in ("unique_id", "ds")][0]
    preds_df = preds_df.rename(columns={pred_col: "predicted", "unique_id": "station_id"})

    actual_df = test_long.rename(columns={"unique_id": "station_id", "y": "actual"})
    merged = preds_df.merge(
        actual_df[["station_id", "ds", "actual"]], on=["station_id", "ds"], how="inner"
    )
    return merged[["station_id", "predicted", "actual"]]


def compute_metrics(
    predictions_df: pd.DataFrame,
    tier_name: str,
) -> pd.DataFrame:
    """Compute per-station RMSE and MAE from tidy predictions DataFrame.

    Returns DataFrame: [station_id, tier, rmse, mae, n_test_years].
    """
    rows = []
    for station_id, grp in predictions_df.groupby("station_id"):
        if grp.empty:
            continue
        rmse = float(np.sqrt(mean_squared_error(grp["actual"], grp["predicted"])))
        mae = float(mean_absolute_error(grp["actual"], grp["predicted"]))
        n_test_years = len(grp) // 12
        rows.append({
            "station_id": station_id,
            "tier": tier_name,
            "rmse": rmse,
            "mae": mae,
            "n_test_years": n_test_years,
        })
    return pd.DataFrame(rows)


def evaluate_all_tiers(
    df: pd.DataFrame,
    local_results: dict,
    cluster_models: dict,
    cluster_labels: pd.Series,
    mlf: object,
    chronos_results: dict,
) -> pd.DataFrame:
    """Master evaluation: compute metrics for all 4 tiers.

    Returns combined tidy DataFrame saved to outputs/evaluation_results.csv.
    """
    parts = []

    logger.info("Evaluating Tier 1 (Local) ...")
    local_preds = generate_local_predictions(df, local_results)
    if not local_preds.empty:
        parts.append(compute_metrics(local_preds, "Local"))

    logger.info("Evaluating Tier 2 (Cluster) ...")
    cluster_preds = generate_cluster_predictions(df, cluster_models, cluster_labels)
    if not cluster_preds.empty:
        parts.append(compute_metrics(cluster_preds, "Cluster"))

    logger.info("Evaluating Tier 3 (Global) ...")
    global_preds = generate_global_predictions(df, mlf)
    if not global_preds.empty:
        parts.append(compute_metrics(global_preds, "Global"))

    logger.info("Evaluating Tier 4 (Chronos) ...")
    if chronos_results:
        chron_rows = []
        for sid, res in chronos_results.items():
            for pred_row, act_row in zip(res["predictions"], res["actuals"]):
                for p, a in zip(pred_row, act_row):
                    chron_rows.append({"station_id": sid, "predicted": p, "actual": a})
        if chron_rows:
            chron_df = pd.DataFrame(chron_rows)
            parts.append(compute_metrics(chron_df, "Chronos"))

    combined = pd.concat(parts, ignore_index=True)

    # Add country column from df
    country_map = df.groupby("ID")["Country Name"].first()
    combined["country"] = combined["station_id"].map(country_map)

    out = config.MODELS_DIR.parent / "evaluation_results.csv"
    combined.to_csv(out, index=False)
    logger.info("Saved evaluation results → %s", out)

    print(summarise_tiers(combined).to_string())
    return combined


def summarise_tiers(results_df: pd.DataFrame) -> pd.DataFrame:
    """Group by tier; compute mean/median/std RMSE and MAE."""
    return (
        results_df.groupby("tier")[["rmse", "mae"]]
        .agg(["mean", "median", "std"])
        .round(4)
    )


def plot_tier_comparison(results_df: pd.DataFrame, save: bool = True) -> Figure:
    """Box plot of per-station RMSE, one box per tier."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["Local", "Cluster", "Global", "Chronos"]
    present = [t for t in order if t in results_df["tier"].values]
    sns.boxplot(data=results_df, x="tier", y="rmse", order=present, ax=ax)
    ax.set_title("RMSE Distribution per Model Tier")
    ax.set_xlabel("Tier")
    ax.set_ylabel("RMSE (°C)")
    footnote = f"* Chronos based on {config.CHRONOS_SAMPLE_N}-station sample"
    if "Chronos" in present:
        fig.text(0.01, 0.01, footnote, fontsize=8, color="gray")
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / "tier_comparison_boxplot.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_global_vs_local(results_df: pd.DataFrame, save: bool = True) -> Figure:
    """Scatter: x=local RMSE, y=global RMSE, one point per station.

    Diagonal reference line (y=x): above = global better, below = local better.
    """
    local = results_df[results_df["tier"] == "Local"][["station_id", "rmse"]].rename(
        columns={"rmse": "local_rmse"}
    )
    global_ = results_df[results_df["tier"] == "Global"][["station_id", "rmse"]].rename(
        columns={"rmse": "global_rmse"}
    )
    merged = local.merge(global_, on="station_id")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged["local_rmse"], merged["global_rmse"], alpha=0.5, s=15)
    lim_max = float(max(merged["local_rmse"].max(), merged["global_rmse"].max())) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "r--", label="y=x (equal)")
    ax.set_xlabel("Local Model RMSE (°C)")
    ax.set_ylabel("Global Model RMSE (°C)")
    ax.set_title("Global vs Local RMSE per Station")
    ax.legend()
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / "global_vs_local_scatter.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_chronos_vs_trained(results_df: pd.DataFrame, save: bool = True) -> Figure:
    """Bar chart: Chronos RMSE vs best trained RMSE for CHRONOS_SAMPLE_N stations."""
    chronos = results_df[results_df["tier"] == "Chronos"][["station_id", "rmse"]].rename(
        columns={"rmse": "chronos_rmse"}
    )
    trained = (
        results_df[results_df["tier"].isin(["Local", "Cluster"])]
        .groupby("station_id")["rmse"]
        .min()
        .reset_index()
        .rename(columns={"rmse": "trained_rmse"})
    )
    merged = chronos.merge(trained, on="station_id").sort_values("trained_rmse")

    x = np.arange(len(merged))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, merged["chronos_rmse"], width, label="Chronos (zero-shot)")
    ax.bar(x + width / 2, merged["trained_rmse"], width, label="Best trained (Local/Cluster)")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["station_id"], rotation=90, fontsize=6)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title(f"Chronos vs Best Trained Model ({config.CHRONOS_SAMPLE_N}-station sample)")
    ax.legend()
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / "chronos_vs_best_trained.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_rmse_by_cluster(
    results_df: pd.DataFrame,
    cluster_labels: pd.Series,
    save: bool = True,
) -> Figure:
    """Box plot of RMSE per cluster, grouped by tier."""
    import seaborn as sns

    cluster_df = results_df.copy()
    cluster_df["cluster_id"] = cluster_df["station_id"].map(cluster_labels)
    cluster_df = cluster_df.dropna(subset=["cluster_id"])
    cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(int)

    tiers = ["Local", "Cluster", "Global"]
    plot_df = cluster_df[cluster_df["tier"].isin(tiers)]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=plot_df, x="cluster_id", y="rmse", hue="tier",
                order=sorted(plot_df["cluster_id"].unique()), ax=ax)
    ax.set_title("RMSE per Climate Cluster by Tier")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("RMSE (°C)")
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / "rmse_by_cluster.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig
