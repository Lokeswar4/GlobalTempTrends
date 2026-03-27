"""EDA visualisations for GHCN temperature data.

All functions:
  - Accept df: pd.DataFrame (loaded from _clean.csv) and country: str label
  - Return the matplotlib Figure (for inline notebook display with save=False)
  - Save to FIGURES_DIR/{country}_{name}.png when save=True
  - Call plt.close(fig) after saving to prevent memory leaks in pipeline mode
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from ghcn_analysis import config

__all__ = [
    "plot_missing_heatmap",
    "plot_monthly_trends",
    "plot_correlation_heatmap",
    "plot_boxplot_by_month",
    "plot_feature_importance",
    "plot_kmeans_clusters",
    "plot_station_stats",
    "print_summary",
]

logger = logging.getLogger(__name__)


def plot_missing_heatmap(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> Figure:
    """Seaborn heatmap of isnull() on VALUE1–12."""
    fig, ax = plt.subplots(figsize=(14, 6))
    missing = df[config.VALUE_COLS].isnull().astype(int)
    sns.heatmap(missing, ax=ax, cbar=False, cmap="Reds", yticklabels=False)
    ax.set_title(f"{country} — Missing Values (VALUE1–12)".strip(" — "))
    ax.set_xticklabels(config.MONTH_NAMES, rotation=45)
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_missing_heatmap.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_monthly_trends(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> Figure:
    """Line plot: yearly mean per month (VALUE1–12 over time)."""
    yearly = df.groupby("YEAR")[config.VALUE_COLS].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(config.VALUE_COLS):
        ax.plot(yearly.index, yearly[col], label=config.MONTH_NAMES[i], alpha=0.8)
    ax.set_title(f"{country} — Monthly Temperature Trends".strip(" — "))
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_monthly_trends.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> Figure:
    """Pearson correlation matrix of VALUE1–12."""
    corr = df[config.VALUE_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        xticklabels=config.MONTH_NAMES, yticklabels=config.MONTH_NAMES,
    )
    ax.set_title(f"{country} — Inter-Month Correlations".strip(" — "))
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_correlation_heatmap.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def _remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove rows outside Q1-1.5*IQR / Q3+1.5*IQR per column (for visualisation only)."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
    return df[mask]


def plot_boxplot_by_month(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> Figure:
    """Box plot of VALUE1–12 after IQR outlier removal (1.5×IQR rule)."""
    clean = _remove_outliers_iqr(df, config.VALUE_COLS)
    melted = clean[config.VALUE_COLS].melt(var_name="Month", value_name="Temp")
    melted["Month"] = melted["Month"].map(
        {f"VALUE{i+1}": config.MONTH_NAMES[i] for i in range(12)}
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=melted, x="Month", y="Temp", order=config.MONTH_NAMES, ax=ax)
    ax.set_title(f"{country} — Temperature Distribution by Month (IQR filtered)".strip(" — "))
    ax.set_ylabel("Temperature (°C)")
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_boxplot_by_month.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_feature_importance(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> Figure:
    """RF feature importance: LATITUDE, LONGITUDE, STNELEV → VALUE1–12."""
    feature_cols = ["LATITUDE", "LONGITUDE", "STNELEV"]
    available = [c for c in feature_cols if c in df.columns]
    target = df[config.VALUE_COLS].mean(axis=1)
    X = df[available].fillna(df[available].median())

    rf = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf.fit(X, target)

    importances = pd.Series(rf.feature_importances_, index=available).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot.bar(ax=ax)
    ax.set_title(f"{country} — Feature Importance (RF)".strip(" — "))
    ax.set_ylabel("Importance")
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_feature_importance.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_kmeans_clusters(
    df: pd.DataFrame,
    kmeans: KMeans,
    country: str = "",
    save: bool = True,
) -> Figure:
    """Geographic scatter: LONGITUDE vs LATITUDE, coloured by cluster label."""
    station_df = df.groupby("ID").agg(
        LATITUDE=("LATITUDE", "first"),
        LONGITUDE=("LONGITUDE", "first"),
    ).reset_index()

    # Build profiles for prediction (same feature order as fit_clusters)
    profiles = df.groupby("ID")[config.VALUE_COLS].mean()
    station_df = station_df.set_index("ID").join(profiles).reset_index()
    profile_cols = config.VALUE_COLS + ["LATITUDE", "LONGITUDE"]
    available = [c for c in profile_cols if c in station_df.columns]
    labels = kmeans.predict(station_df[available].fillna(0))

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        station_df["LONGITUDE"], station_df["LATITUDE"],
        c=labels, cmap="tab10", alpha=0.7, s=20,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{country} — KMeans Climate Clusters".strip(" — "))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_kmeans_clusters.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def plot_station_stats(
    df: pd.DataFrame,
    country: str = "",
    top_n: int = 20,
    save: bool = True,
) -> Figure:
    """Bar plot of per-station mean MEAN_TEMP for top_n stations by record count."""
    stats = (
        df.groupby("ID")
        .agg(mean_temp=("MEAN_TEMP", "mean"), records=("YEAR", "count"), name=("NAME", "first"))
        .nlargest(top_n, "records")
        .sort_values("mean_temp", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(stats)), stats["mean_temp"])
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["name"], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{country} — Top {top_n} Stations by Mean Temperature".strip(" — "))
    ax.set_ylabel("Mean Temperature (°C)")
    fig.tight_layout()
    if save:
        out = config.FIGURES_DIR / f"{country}_station_stats.png"
        fig.savefig(out, dpi=100)
        logger.info("Saved %s", out)
        plt.close(fig)
    return fig


def print_summary(df: pd.DataFrame, country: str) -> None:
    """Print summary stats to stdout."""
    print(f"\n{'='*60}")
    print(f"Country: {country}")
    print(f"  Rows: {len(df):,}  |  Stations: {df['ID'].nunique():,}")
    print(f"  Year range: {df['YEAR'].min()}–{df['YEAR'].max()}")
    print(df[config.VALUE_COLS].describe().round(2).to_string())
    print(f"  Missing VALUE cols: {df[config.VALUE_COLS].isna().sum().sum()}")
    print(f"{'='*60}\n")
