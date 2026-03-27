"""Model-only runner. Assumes cleaned data exists in data/processed/.

Usage:
  uv run python scripts/train_models.py --tier all
  uv run python scripts/train_models.py --tier local --station-id USC00026796
  uv run python scripts/train_models.py --tier cluster
  uv run python scripts/train_models.py --tier global
  uv run python scripts/train_models.py --tier chronos
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GHCN Model Training")
    p.add_argument(
        "--tier", default="all",
        choices=["local", "cluster", "global", "chronos", "all"],
    )
    p.add_argument("--station-id", default=None, help="Single station ID for local tier")
    p.add_argument(
        "--countries", nargs="+",
        default=["United States", "Canada", "Australia", "Russia", "Germany"],
    )
    return p.parse_args()


def load_combined(countries: list[str]) -> object:
    import pandas as pd

    from ghcn_analysis import config

    dfs = []
    for country in countries:
        slug = country.lower().replace(" ", "_")
        p = config.PROCESSED_DIR / f"{slug}_clean.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            logger.warning("Clean CSV not found for %s: %s", country, p)
    if not dfs:
        raise FileNotFoundError(
            "No clean CSVs found. Run run_pipeline.py first (without --skip-train)."
        )
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    args = parse_args()
    df = load_combined(args.countries)

    from ghcn_analysis.models.cluster import fit_clusters
    km, cluster_labels = fit_clusters(df)

    if args.tier in ("local", "all"):
        from ghcn_analysis.models.local import train_local_models
        logger.info("Training local models ...")
        train_local_models(df, station_id=args.station_id)

    if args.tier in ("cluster", "all"):
        from ghcn_analysis.models.cluster import train_cluster_models
        logger.info("Training cluster models ...")
        train_cluster_models(df, cluster_labels)

    if args.tier in ("global", "all"):
        from ghcn_analysis.models.global_model import train_global_model
        logger.info("Training global model ...")
        train_global_model(df)

    if args.tier in ("chronos", "all"):
        from ghcn_analysis.models.chronos_bench import run_chronos_benchmark
        logger.info("Running Chronos benchmark ...")
        run_chronos_benchmark(df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
