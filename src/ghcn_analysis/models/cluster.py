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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ghcn_analysis import config

__all__ = [
    "build_station_profiles", "select_k", "fit_clusters",
    "train_cluster_models", "predict_for_station",
]

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


# ---------------------------------------------------------------------------
# Phase 5: Cluster model training
# ---------------------------------------------------------------------------

_CLUSTER_PARAMS = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=config.RANDOM_SEED),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=config.RANDOM_SEED),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    },
}


def train_cluster_models(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict[int, dict]:
    """Pool training data per cluster; train RF + GB per cluster.

    Returns:
        {cluster_id: {
            "best_model_name": str,
            "best_model": fitted model,
            "scaler": MinMaxScaler,
            "cv_results": {name: avg_rmse},
            "station_ids": list[str],
        }}
    """
    from ghcn_analysis.models.split import temporal_split, to_supervised

    results: dict[int, dict] = {}
    cv = TimeSeriesSplit(n_splits=5)

    for cluster_id in sorted(cluster_labels.unique()):
        station_ids = cluster_labels[cluster_labels == cluster_id].index.tolist()
        cluster_df = df[df["ID"].isin(station_ids)]

        # Pool all train rows for the cluster
        train_rows = []
        scalers = []
        for sid in station_ids:
            sdf = cluster_df[cluster_df["ID"] == sid]
            if sdf.empty or len(sdf) < 5:
                continue
            try:
                train, _, scaler = temporal_split(sdf)
                X, y = to_supervised(train)
                if len(X) > 0:
                    train_rows.append((X, y))
                    scalers.append(scaler)
            except ValueError:
                continue

        if not train_rows:
            logger.warning("Cluster %d: no usable stations, skipping", cluster_id)
            continue

        X_all = np.vstack([t[0] for t in train_rows])
        y_all = np.vstack([t[1] for t in train_rows])

        # Use first station's scaler as cluster scaler
        cluster_scaler = scalers[0] if scalers else MinMaxScaler()

        model_results = {}
        for name, spec in _CLUSTER_PARAMS.items():
            search = RandomizedSearchCV(
                spec["model"], spec["params"],
                n_iter=10, cv=cv, scoring="neg_mean_squared_error",
                random_state=config.RANDOM_SEED, n_jobs=-1, refit=True,
            )
            search.fit(X_all, y_all)
            model_results[name] = {
                "model": search.best_estimator_,
                "cv_rmse": float(np.sqrt(-search.best_score_)),
            }

        best_name = min(model_results, key=lambda k: model_results[k]["cv_rmse"])

        out_dir = config.MODELS_DIR / "cluster" / str(cluster_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_results[best_name]["model"], out_dir / "best_model.pkl")
        joblib.dump(cluster_scaler, out_dir / "scaler.pkl")

        results[cluster_id] = {
            "best_model_name": best_name,
            "best_model": model_results[best_name]["model"],
            "scaler": cluster_scaler,
            "cv_results": {k: v["cv_rmse"] for k, v in model_results.items()},
            "station_ids": station_ids,
        }
        logger.info(
            "Cluster %d: best=%s (RMSE=%.4f), %d stations",
            cluster_id, best_name, model_results[best_name]["cv_rmse"], len(station_ids),
        )

    return results


def predict_for_station(
    station_df: pd.DataFrame,
    cluster_id: int,
    cluster_models: dict[int, dict],
) -> np.ndarray:
    """Apply the cluster model to a single station's test data.

    Returns predictions in original (inverse-transformed) scale.
    """
    from ghcn_analysis.models.split import temporal_split, to_supervised

    cluster = cluster_models[cluster_id]
    _, test, _ = temporal_split(station_df)
    X_test, _ = to_supervised(test)

    raw = cluster["best_model"].predict(X_test)
    # Inverse transform: reshape to (n, 12)
    n = len(raw) // 12
    reshaped = raw[: n * 12].reshape(n, 12)
    return cluster["scaler"].inverse_transform(reshaped).flatten()
