"""Merge, filter, and sentinel-clean raw GHCN DataFrames."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ghcn_analysis import config

__all__ = ["load_countries", "prepare"]

logger = logging.getLogger(__name__)


def load_countries(countries_path: Path) -> pd.DataFrame:
    """Parse ghcnm-countries.txt → DataFrame[Country Code, Country Name].

    Format: '<2-char-code> <name>\\n' — split on first space only.
    """
    rows = []
    for line in countries_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        code, _, name = line.partition(" ")
        rows.append({"Country Code": code.strip(), "Country Name": name.strip()})
    return pd.DataFrame(rows)


def prepare(
    metadata_df: pd.DataFrame,
    data_df: pd.DataFrame,
    countries_path: Path,
    countries: list[str] = config.COUNTRIES,
) -> dict[str, pd.DataFrame]:
    """Merge, filter, and sentinel-clean the raw DataFrames.

    Steps:
      1. Load countries lookup (FIPS code → country name)
      2. Extract FIPS code from first 2 chars of station ID
      3. Merge metadata with countries lookup
      4. Merge result with data_df on ID (inner join)
      5. Replace VALUE sentinel -99.99 → NaN (safety net)
      6. Replace STNELEV sentinel -999.0 → NaN
      7. Drop DMFLAG1–12 columns
      8. Drop DSFLAG1–12 columns
      9. Filter to target countries
      10. Split into per-country dict, save each to INTERIM_DIR

    Returns:
        dict mapping country name → DataFrame.
    """
    countries_df = load_countries(countries_path)

    # FIPS code from first 2 chars of station ID
    meta = metadata_df.copy()
    meta["Country Code"] = meta["ID"].str[:2]
    meta = meta.merge(countries_df, on="Country Code", how="left")

    # Merge with temperature data
    merged = meta.merge(data_df, on="ID", how="inner")

    # Sentinel replacements
    for col in config.VALUE_COLS:
        if col in merged.columns:
            merged[col] = merged[col].replace(-99.99, float("nan"))
    if "STNELEV" in merged.columns:
        merged["STNELEV"] = merged["STNELEV"].replace(-999.0, float("nan"))

    # Drop flag columns not needed downstream
    drop_cols = (
        [c for c in merged.columns if c.startswith("DMFLAG")]
        + [c for c in merged.columns if c.startswith("DSFLAG")]
    )
    merged.drop(columns=drop_cols, inplace=True)

    # Filter and split
    result: dict[str, pd.DataFrame] = {}
    for country in countries:
        subset = merged[merged["Country Name"] == country].copy()
        if subset.empty:
            logger.warning("No stations found for country: %s", country)
            continue
        slug = country.lower().replace(" ", "_")
        out_path = config.INTERIM_DIR / f"{slug}_prepared.csv"
        subset.to_csv(out_path, index=False)
        logger.info("Saved %d rows for %s → %s", len(subset), country, out_path)
        result[country] = subset

    return result
