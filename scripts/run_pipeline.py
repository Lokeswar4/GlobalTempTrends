"""Full GHCN analysis pipeline.

Usage:
  uv run python scripts/run_pipeline.py [options]

Options:
  --countries US CA AU RU DE   FIPS codes to process (default: all 5)
  --variant qcu|qcf|qfe        Dataset variant (default: qcu)
  --skip-download              Skip if data/raw/ contains .dat and .inv
  --skip-prepare               Skip if data/interim/{country}_prepared.csv exists
  --skip-qc                    Skip if data/interim/{country}_qc.csv exists
  --skip-cleanse               Skip if data/processed/{country}_clean.csv exists
  --skip-eda                   Skip EDA figure generation
  --skip-train                 Skip all model training
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Map FIPS code argument to country name
_FIPS_TO_COUNTRY = {
    "US": "United States",
    "CA": "Canada",
    "AU": "Australia",
    "RU": "Russia",
    "DE": "Germany",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GHCN Analysis Pipeline")
    p.add_argument("--countries", nargs="+", default=list(_FIPS_TO_COUNTRY.keys()))
    p.add_argument("--variant", default="qcu", choices=["qcu", "qcf", "qfe"])
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-qc", action="store_true")
    p.add_argument("--skip-cleanse", action="store_true")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    countries = [_FIPS_TO_COUNTRY[c] for c in args.countries if c in _FIPS_TO_COUNTRY]

    from ghcn_analysis import config
    from ghcn_analysis.data.download import (
        download_dataset,
        download_support_files,
        extract_archive,
    )
    from ghcn_analysis.data.parse import parse_data, parse_inventory
    from ghcn_analysis.data.prepare import prepare
    from ghcn_analysis.features.cleanse import cleanse
    from ghcn_analysis.features.qc_flags import apply_all_qc_flags

    # -------------------------------------------------------------------------
    # Phase 1: Download + Parse
    # -------------------------------------------------------------------------
    if not args.skip_download:
        logger.info("=== Phase 1: Download ===")
        archive = download_dataset(variant=args.variant)
        download_support_files()
        parsed_paths = extract_archive(archive)
    else:
        dat_files = list(config.RAW_DIR.glob(f"**/*.{args.variant}.dat"))
        inv_files = list(config.RAW_DIR.glob(f"**/*.{args.variant}.inv"))
        if not dat_files or not inv_files:
            logger.error("--skip-download set but no .dat/.inv found in %s", config.RAW_DIR)
            sys.exit(1)
        from ghcn_analysis.data.download import ParsedPaths
        parsed_paths = ParsedPaths(dat=dat_files[0], inv=inv_files[0])
        logger.info("Using cached files: %s", parsed_paths)

    logger.info("Parsing .inv and .dat files ...")
    metadata_df = parse_inventory(parsed_paths.inv)
    data_df = parse_data(parsed_paths.dat)
    countries_path = config.RAW_DIR / "ghcnm-countries.txt"

    # -------------------------------------------------------------------------
    # Phase 2: Prepare + QC + Cleanse (per country)
    # -------------------------------------------------------------------------
    import pandas as pd

    country_dfs: dict[str, object] = {}

    for country in countries:
        slug = country.lower().replace(" ", "_")
        clean_path = config.PROCESSED_DIR / f"{slug}_clean.csv"

        if not args.skip_cleanse and clean_path.exists():
            logger.info("Using cached clean data for %s", country)
            country_dfs[country] = pd.read_csv(clean_path)
            continue

        if not args.skip_prepare:
            logger.info("=== Phase 2a: Prepare — %s ===", country)
            prepared = prepare(metadata_df, data_df, countries_path, countries=[country])
            if country not in prepared:
                logger.warning("No data for %s, skipping", country)
                continue
            df = prepared[country]
        else:
            p = config.INTERIM_DIR / f"{slug}_prepared.csv"
            if not p.exists():
                logger.error("--skip-prepare set but %s not found", p)
                sys.exit(1)
            df = pd.read_csv(p)

        if not args.skip_qc:
            logger.info("=== Phase 2b: QC Flags — %s ===", country)
            df = apply_all_qc_flags(df)
            qc_out = config.INTERIM_DIR / f"{slug}_qc.csv"
            df.to_csv(qc_out, index=False)
        else:
            p = config.INTERIM_DIR / f"{slug}_qc.csv"
            if not p.exists():
                logger.error("--skip-qc set but %s not found", p)
                sys.exit(1)
            df = pd.read_csv(p)

        if not args.skip_cleanse:
            logger.info("=== Phase 2c: Cleanse — %s ===", country)
            df = cleanse(df, slug=slug)

        country_dfs[country] = df

    if not country_dfs:
        logger.error("No countries processed. Exiting.")
        sys.exit(1)

    # Combine for multi-country modeling
    combined_df = pd.concat(country_dfs.values(), ignore_index=True)

    # -------------------------------------------------------------------------
    # Phase 3: EDA
    # -------------------------------------------------------------------------
    if not args.skip_eda:
        logger.info("=== Phase 3: EDA ===")
        from ghcn_analysis.models.cluster import fit_clusters
        from ghcn_analysis.viz import eda, maps

        km, cluster_labels = fit_clusters(combined_df)

        for country, df in country_dfs.items():
            slug = country.lower().replace(" ", "_")
            eda.print_summary(df, country)
            eda.plot_missing_heatmap(df, country=slug)
            eda.plot_monthly_trends(df, country=slug)
            eda.plot_correlation_heatmap(df, country=slug)
            eda.plot_boxplot_by_month(df, country=slug)
            eda.plot_feature_importance(df, country=slug)
            eda.plot_kmeans_clusters(df, kmeans=km, country=slug)
            eda.plot_station_stats(df, country=slug)
            maps.make_cluster_map(df, country=slug)
            maps.make_temperature_map(df, country=slug)
    else:
        from ghcn_analysis.models.cluster import fit_clusters
        km, cluster_labels = fit_clusters(combined_df)

    # -------------------------------------------------------------------------
    # Phase 4+5: Model Training
    # -------------------------------------------------------------------------
    if not args.skip_train:
        logger.info("=== Phase 5: Model Training ===")
        from ghcn_analysis.models.chronos_bench import run_chronos_benchmark
        from ghcn_analysis.models.cluster import train_cluster_models
        from ghcn_analysis.models.global_model import train_global_model
        from ghcn_analysis.models.local import train_local_models

        local_results = train_local_models(combined_df)
        cluster_models = train_cluster_models(combined_df, cluster_labels)
        mlf = train_global_model(combined_df)
        chronos_results = run_chronos_benchmark(combined_df)

        # -------------------------------------------------------------------------
        # Phase 6: Evaluation
        # -------------------------------------------------------------------------
        logger.info("=== Phase 6: Evaluation ===")
        from ghcn_analysis.models.evaluate import evaluate_all_tiers

        evaluate_all_tiers(
            combined_df, local_results, cluster_models,
            cluster_labels, mlf, chronos_results,
        )
        logger.info("Pipeline complete. Evaluation results saved.")
    else:
        logger.info("Skipping model training (--skip-train). Pipeline complete.")


if __name__ == "__main__":
    main()
