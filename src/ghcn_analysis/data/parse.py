"""GHCN v4 fixed-width file parsers.

Uses pd.read_fwf() for both .inv and .dat files — C-level I/O, ~10-100x faster
than Python-level line iteration.

All functions are pure (no side effects beyond reading the input file).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ghcn_analysis import config

__all__ = ["parse_inventory", "parse_data"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .inv colspecs (0-indexed half-open intervals for pd.read_fwf)
# ---------------------------------------------------------------------------
_INV_COLSPECS = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 68)]
_INV_NAMES = ["ID", "LATITUDE", "LONGITUDE", "STNELEV", "NAME"]
_INV_DTYPES: dict[str, object] = {
    "ID": str,
    "LATITUDE": "float32",
    "LONGITUDE": "float32",
    "STNELEV": "float32",
    "NAME": str,
}

# ---------------------------------------------------------------------------
# .dat colspecs — built programmatically
# ---------------------------------------------------------------------------
_DAT_HEADER_COLSPECS = [(0, 11), (11, 15), (15, 19)]
_DAT_HEADER_NAMES = ["ID", "YEAR", "ELEMENT"]


def _build_dat_colspecs() -> tuple[list[tuple[int, int]], list[str]]:
    """Build (colspecs, names) for all 12 monthly blocks in .dat file.

    Block formula (0-indexed):
      offset_i = 19 + (i - 1) * 8    for month i in 1..12
      VALUE{i}  = [offset_i : offset_i+5]
      DMFLAG{i} = [offset_i+5 : offset_i+6]
      QCFLAG{i} = [offset_i+6 : offset_i+7]
      DSFLAG{i} = [offset_i+7 : offset_i+8]
    """
    colspecs = list(_DAT_HEADER_COLSPECS)
    names = list(_DAT_HEADER_NAMES)
    for i in range(1, 13):
        offset = 19 + (i - 1) * 8
        colspecs += [
            (offset, offset + 5),      # VALUE{i}
            (offset + 5, offset + 6),  # DMFLAG{i}
            (offset + 6, offset + 7),  # QCFLAG{i}
            (offset + 7, offset + 8),  # DSFLAG{i}
        ]
        names += [f"VALUE{i}", f"DMFLAG{i}", f"QCFLAG{i}", f"DSFLAG{i}"]
    return colspecs, names  # 51 columns total (3 header + 48 monthly)


_DAT_COLSPECS, _DAT_NAMES = _build_dat_colspecs()

_FLAG_COLS = (
    [f"DMFLAG{i}" for i in range(1, 13)]
    + [f"QCFLAG{i}" for i in range(1, 13)]
    + [f"DSFLAG{i}" for i in range(1, 13)]
)


def parse_inventory(inv_path: Path) -> pd.DataFrame:
    """Parse GHCN .inv file into station metadata DataFrame.

    Args:
        inv_path: Path to the .inv file.

    Returns:
        DataFrame with columns [ID, LATITUDE, LONGITUDE, STNELEV, NAME].
        STNELEV -999.0 sentinel retained as-is (replaced in Phase 2).

    Raises:
        FileNotFoundError: If inv_path does not exist.
    """
    if not inv_path.exists():
        raise FileNotFoundError(inv_path)

    df = pd.read_fwf(
        inv_path,
        colspecs=_INV_COLSPECS,
        names=_INV_NAMES,
        header=None,
        encoding="latin-1",
        dtype=_INV_DTYPES,
    )
    df["NAME"] = df["NAME"].fillna("").str.strip()

    # Drop rows with unparseable lat/lon
    bad = df["LATITUDE"].isna() | df["LONGITUDE"].isna()
    if bad.any():
        for sid in df.loc[bad, "ID"]:
            logger.warning("Dropping station %s: bad lat/lon", sid)
        df = df[~bad].reset_index(drop=True)

    logger.info("Parsed inventory: %d stations", len(df))
    return df


def parse_data(dat_path: Path) -> pd.DataFrame:
    """Parse GHCN .dat file into temperature records DataFrame.

    Args:
        dat_path: Path to the .dat file.

    Returns:
        DataFrame with 50 columns (ELEMENT dropped after filter):
        ID, YEAR, VALUE1–12, DMFLAG1–12, QCFLAG1–12, DSFLAG1–12.
        Raw -9999 → NaN before /100. VALUE cols = float32. Flag cols = Categorical.

    Raises:
        FileNotFoundError: If dat_path does not exist.
    """
    if not dat_path.exists():
        raise FileNotFoundError(dat_path)

    df = pd.read_fwf(
        dat_path,
        colspecs=_DAT_COLSPECS,
        names=_DAT_NAMES,
        header=None,
        encoding="latin-1",
    )

    # --- Filter to TAVG, warn on unexpected elements ---
    unexpected = df.loc[df["ELEMENT"] != "TAVG", "ELEMENT"].unique()
    if len(unexpected) > 0:
        logger.warning(
            "Unexpected ELEMENT values found: %s — filtered out",
            list(unexpected),
        )
    df = df[df["ELEMENT"] == "TAVG"].copy()
    df.drop(columns=["ELEMENT"], inplace=True)

    # --- Dtypes ---
    df["YEAR"] = df["YEAR"].astype("int16")

    # --- VALUE cols: replace -9999 sentinel → NaN, then divide by 100 ---
    for col in config.VALUE_COLS:
        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce")
            raw = raw.where(raw != -9999, other=float("nan"))
            df[col] = (raw / 100.0).astype("float32")

    # --- Flag cols → Categorical ---
    for col in _FLAG_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("category")

    logger.info(
        "Parsed %d TAVG records for %d stations (%d–%d)",
        len(df),
        df["ID"].nunique(),
        int(df["YEAR"].min()),
        int(df["YEAR"].max()),
    )
    return df
