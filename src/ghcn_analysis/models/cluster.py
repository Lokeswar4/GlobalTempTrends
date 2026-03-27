"""KMeans climate zone clustering.

Phase 4 (fit step): build_station_profiles, select_k, fit_clusters
Phase 5 (train step): train_cluster_models, predict_for_station  (added in Part 4)
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ghcn_analysis import config

__all__ = ["build_station_profiles", "select_k", "fit_clusters"]

logger = logging.getLogger(__name__)


def build_station_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one row per station with clustering features.

    Features: mean of VALUE1..12 (12 dims) + LATITUDE + LONGITUDE = 14 dims.
    Index = station ID.
    """
    monthly_means = df.groupby("ID")[config.VALUE_COLS].mean()
    coords = df.groupby("ID")[["LATITUDE", "LONGITUDE"]].first()
    profiles = monthly_means.join(coords)
    return profiles


def select_k(profiles: pd.DataFrame, k_range: range = range(2, 15)) -> int:
    """Compute silhouette score for each k; return k with highest score.

    Plots silhouette curve to FIGURES_DIR/silhouette_scores.png.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    X = scaler.fit_transform(profiles.fillna(0))

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) > 1:
            scores[k] = silhouette_score(X, labels)
        else:
            scores[k] = -1.0

    best_k = max(scores, key=lambda k: scores[k])
    logger.info("Best k=%d (silhouette=%.4f)", best_k, scores[best_k])

    # Save silhouette plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(scores.keys()), list(scores.values()), marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax.set_title("Silhouette Score vs Number of Clusters")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    fig.tight_layout()
    out = config.FIGURES_DIR / "silhouette_scores.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    logger.info("Saved %s", out)

    return best_k


def fit_clusters(
    df: pd.DataFrame,
    n_clusters: int = config.N_CLUSTERS,
) -> tuple[KMeans, pd.Series]:
    """Fit KMeans on station temperature profiles.

    Args:
        df: Full multi-station DataFrame.
        n_clusters: Number of clusters (default config.N_CLUSTERS = 8).

    Returns:
        (kmeans, cluster_labels) where cluster_labels is a pd.Series
        mapping station_id → cluster_id (int).

    Side effect:
        Saves fitted KMeans to MODELS_DIR/kmeans.pkl.
    """
    profiles = build_station_profiles(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(profiles.fillna(0))

    km = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10)
    km.fit(X)

    cluster_labels = pd.Series(km.labels_, index=profiles.index, name="cluster_id")

    out = config.MODELS_DIR / "kmeans.pkl"
    joblib.dump(km, out)
    logger.info("Saved KMeans model → %s", out)

    return km, cluster_labels
