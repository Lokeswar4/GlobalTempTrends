"""End-to-end ingestion test: parse → prepare → qc_flags → cleanse — 3 tests.

Exercises the full data-ingestion pipeline with synthetic data to catch
regressions that only appear when the four stages are composed together.
"""

from __future__ import annotations

from pathlib import Path

from ghcn_analysis import config
from ghcn_analysis.data.parse import parse_data, parse_inventory
from ghcn_analysis.data.prepare import prepare
from ghcn_analysis.features.cleanse import cleanse
from ghcn_analysis.features.qc_flags import apply_all_qc_flags

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COUNTRIES_TXT = "US United States\n"

_INV_LINE = (
    "US000000001  40.7128  -74.0060  10.0 NEW YORK                        \n"
)


def _make_dat_row(station: str, year: int, values: list[int]) -> str:
    """Build one fixed-width TAVG row matching the GHCN .dat spec."""
    row = f"{station}{year:4d}TAVG"
    for v in values:
        row += f"{v:5d}   "  # VALUE(5) + DMFLAG(1) QCFLAG(1) DSFLAG(1) blank
    return row


def _write_fixtures(tmp_path: Path, n_years: int = 100) -> tuple[Path, Path, Path]:
    """Write .inv, .dat, and countries files. Returns (inv, dat, countries)."""
    inv = tmp_path / "test.qcu.inv"
    inv.write_text(_INV_LINE, encoding="latin-1")

    dat_rows = [
        _make_dat_row(
            "US000000001",
            1924 + yr,
            [2000 + yr * 5] * 12,  # gentle warming trend
        )
        for yr in range(n_years)
    ]
    dat = tmp_path / "test.qcu.dat"
    dat.write_text("\n".join(dat_rows) + "\n", encoding="latin-1")

    countries = tmp_path / "ghcnm-countries.txt"
    countries.write_text(_COUNTRIES_TXT)

    return inv, dat, countries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_stages_produce_expected_shapes(tmp_path):
    """parse_inventory and parse_data return the expected row/column counts."""
    inv, dat, _ = _write_fixtures(tmp_path, n_years=100)

    meta = parse_inventory(inv)
    data = parse_data(dat)

    assert len(meta) == 1, "Inventory should have 1 station"
    assert len(data) == 100, "Data should have 100 rows (one per year)"
    # 50 cols: ID, YEAR + 12 × (VALUE + DMFLAG + QCFLAG + DSFLAG)
    assert len(data.columns) == 50


def test_prepare_merges_and_drops_flags(tmp_path):
    """prepare() merges metadata, drops DMFLAG/DSFLAG, keeps QCFLAG columns."""
    inv, dat, countries = _write_fixtures(tmp_path)
    meta = parse_inventory(inv)
    data = parse_data(dat)

    result = prepare(meta, data, countries, countries=["United States"])
    df = result["United States"]

    # Metadata columns present
    assert "LATITUDE" in df.columns
    assert "Country Name" in df.columns

    # DMFLAG and DSFLAG dropped; QCFLAG retained for the QC stage
    assert not any(c.startswith("DMFLAG") for c in df.columns)
    assert not any(c.startswith("DSFLAG") for c in df.columns)
    assert "QCFLAG1" in df.columns


def test_full_pipeline_clean_output(tmp_path):
    """parse → prepare → qc_flags → cleanse produces NaN-free, rounded output."""
    inv, dat, countries = _write_fixtures(tmp_path, n_years=100)
    meta = parse_inventory(inv)
    data = parse_data(dat)

    # All four stages
    prepared = prepare(meta, data, countries, countries=["United States"])
    df = prepared["United States"]
    df = apply_all_qc_flags(df)
    df = cleanse(df, min_operational_years=80)

    assert len(df) > 0, "Cleansed output should have rows"

    # No NaN in any VALUE column
    for col in config.VALUE_COLS:
        assert not df[col].isna().any(), f"{col} has NaN after full pipeline"

    # MEAN_TEMP computed and all QCFLAG columns dropped
    assert "MEAN_TEMP" in df.columns
    assert not any(c.startswith("QCFLAG") for c in df.columns)

    # Values rounded to ≤2 decimal places
    assert (df["MEAN_TEMP"].round(2) == df["MEAN_TEMP"]).all()
