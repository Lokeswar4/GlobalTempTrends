"""Central configuration: paths, constants, country list, random seed."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_VARIANT: str = "qcu"  # "qcu" | "qcf" | "qfe"
TRAIN_CUTOFF_YEAR: int = 2000
# 80 years: includes non-US stations (Canada, AU, RU, DE have sparser pre-1900
# coverage; 150 would leave only US stations, defeating multi-country scope)
MIN_OPERATIONAL_YEARS: int = 80
N_CLUSTERS: int = 8
CHRONOS_SAMPLE_N: int = 30

# ---------------------------------------------------------------------------
# Countries (English names as they appear in ghcnm-countries.txt)
# ---------------------------------------------------------------------------
COUNTRIES: list[str] = [
    "United States",
    "Canada",
    "Australia",
    "Russia",
    "Germany",
]

# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
VALUE_COLS: list[str] = [f"VALUE{i}" for i in range(1, 13)]
QCFLAG_COLS: list[str] = [f"QCFLAG{i}" for i in range(1, 13)]
DMFLAG_COLS: list[str] = [f"DMFLAG{i}" for i in range(1, 13)]
DSFLAG_COLS: list[str] = [f"DSFLAG{i}" for i in range(1, 13)]
MONTH_NAMES: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ---------------------------------------------------------------------------
# Paths (all relative to project root — resolved at runtime)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent  # GlobalTempTrends/

RAW_DIR: Path = _ROOT / "data" / "raw"
INTERIM_DIR: Path = _ROOT / "data" / "interim"
PROCESSED_DIR: Path = _ROOT / "data" / "processed"
MODELS_DIR: Path = _ROOT / "outputs" / "models"
FIGURES_DIR: Path = _ROOT / "outputs" / "figures"
MAPS_DIR: Path = _ROOT / "outputs" / "maps"

# Ensure directories exist on import
for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, MAPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NOAA URLs
# ---------------------------------------------------------------------------
NOAA_BASE_URL: str = "https://www.ncei.noaa.gov/pub/data/ghcn/v4"
SUPPORT_FILES: list[str] = ["ghcnm-countries.txt", "ghcnm-flags.txt", "readme.txt"]
