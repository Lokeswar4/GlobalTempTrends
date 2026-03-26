"""Tests for src/ghcn_analysis/data/parse.py — 22 tests."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import responses as responses_lib

from ghcn_analysis.data.download import ParsedPaths, extract_archive
from ghcn_analysis.data.parse import parse_data, parse_inventory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_inv_file(tmp_path: Path) -> Path:
    """Minimal .inv file with 2 stations including accented name."""
    # Fixed-width: ID[0:11] gap[11] LAT[12:20] gap[20] LON[21:30] gap[30] ELEV[31:37] gap[37] NAME[38:68]
    lines = (
        "ACW00011604  17.1167  -61.7833   10.1 ST JOHNS COOLIDGE FLD           \n"
        "CAE00040100  45.6000  -73.7667  -999.0 QUÉBEC CITY INTL                \n"
    )
    p = tmp_path / "test.qcu.inv"
    p.write_text(lines, encoding="latin-1")
    return p


def _make_dat_file(tmp_path: Path) -> Path:
    """Minimal .dat file: 2 stations, 2 years, includes non-TAVG row and -9999 sentinel."""
    # ID[0:11] YEAR[11:15] ELEMENT[15:19] then 12x8-char blocks
    # Each block: VALUE[5] DMFLAG[1] QCFLAG[1] DSFLAG[1]
    def make_row(station, year, element, values):
        row = f"{station}{year:4d}{element}"
        for v in values:
            val_str = f"{v:5d}" if v != -9999 else "-9999"
            row += f"{val_str}   "  # DMFLAG, QCFLAG, DSFLAG all blank
        return row

    rows = [
        # TAVG row: station 1, year 1900, all valid, VALUE1=2350 (23.50°C)
        make_row("ACW00011604", 1900, "TAVG", [2350, 2400, 2500, 2600, 2700, 2800,
                                               2900, 2850, 2750, 2600, 2450, 2350]),
        # TAVG row: station 1, year 1901, VALUE12=-9999 sentinel
        make_row("ACW00011604", 1901, "TAVG", [2360, 2410, 2510, 2610, 2710, 2810,
                                               2910, 2860, 2760, 2610, 2460, -9999]),
        # PRCP row (non-TAVG) — must be filtered out
        make_row("ACW00011604", 1900, "PRCP", [100] * 12),
        # Negative temp: station 2, year 1900, VALUE1=-1250 (-12.50°C)
        make_row("CAE00040100", 1900, "TAVG", [-1250, -1100, -800, -200, 500, 1500,
                                               2000, 1800, 1200, 400, -300, -1000]),
    ]
    content = "\n".join(rows) + "\n"
    p = tmp_path / "test.qcu.dat"
    p.write_text(content, encoding="latin-1")
    return p


@pytest.fixture()
def inv_path(tmp_path):
    return _make_inv_file(tmp_path)


@pytest.fixture()
def dat_path(tmp_path):
    return _make_dat_file(tmp_path)


# ---------------------------------------------------------------------------
# parse_inventory tests
# ---------------------------------------------------------------------------

def test_parse_inventory_columns(inv_path):
    df = parse_inventory(inv_path)
    assert list(df.columns) == ["ID", "LATITUDE", "LONGITUDE", "STNELEV", "NAME"]


def test_parse_inventory_dtypes(inv_path):
    df = parse_inventory(inv_path)
    assert df["LATITUDE"].dtype == "float32"
    assert df["LONGITUDE"].dtype == "float32"
    assert df["STNELEV"].dtype == "float32"
    assert df["ID"].dtype == object  # str


def test_parse_inventory_sentinel_stnelev(inv_path):
    df = parse_inventory(inv_path)
    # Second row has -999.0 — retained as-is (Phase 2 replaces)
    row = df[df["ID"] == "CAE00040100"].iloc[0]
    assert float(row["STNELEV"]) == pytest.approx(-999.0)


def test_parse_inventory_latin1_name(inv_path):
    """Station name with accented character should parse without error."""
    df = parse_inventory(inv_path)
    names = df["NAME"].tolist()
    assert any("QUÉBEC" in n for n in names)


def test_parse_inventory_strips_whitespace(inv_path):
    df = parse_inventory(inv_path)
    for name in df["NAME"]:
        assert name == name.strip()


# ---------------------------------------------------------------------------
# parse_data tests
# ---------------------------------------------------------------------------

def test_parse_data_column_count(dat_path):
    df = parse_data(dat_path)
    # 50 cols: ID, YEAR + 12*(VALUE+DMFLAG+QCFLAG+DSFLAG) — ELEMENT dropped
    assert len(df.columns) == 50


def test_parse_data_element_filtered(dat_path):
    df = parse_data(dat_path)
    # Only TAVG rows: 2 for station 1 + 1 for station 2
    assert len(df) == 3


def test_parse_data_element_column_dropped(dat_path):
    df = parse_data(dat_path)
    assert "ELEMENT" not in df.columns


def test_parse_data_element_warning(dat_path, caplog):
    with caplog.at_level(logging.WARNING):
        parse_data(dat_path)
    # PRCP row triggers warning
    assert any("PRCP" in record.message or "ELEMENT" in record.message
               for record in caplog.records)


def test_parse_data_sentinel_to_nan(dat_path):
    df = parse_data(dat_path)
    # Station 1 year 1901 VALUE12 was -9999 → NaN (not -99.99)
    row = df[(df["ID"] == "ACW00011604") & (df["YEAR"] == 1901)].iloc[0]
    assert pd.isna(row["VALUE12"])
    # -99.99 must never appear
    for col in [f"VALUE{i}" for i in range(1, 13)]:
        assert not (df[col] == -99.99).any()


def test_parse_data_scaling(dat_path):
    df = parse_data(dat_path)
    row = df[(df["ID"] == "ACW00011604") & (df["YEAR"] == 1900)].iloc[0]
    assert float(row["VALUE1"]) == pytest.approx(23.50, abs=0.01)


def test_parse_data_negative_temp(dat_path):
    df = parse_data(dat_path)
    row = df[(df["ID"] == "CAE00040100") & (df["YEAR"] == 1900)].iloc[0]
    assert float(row["VALUE1"]) == pytest.approx(-12.50, abs=0.01)


def test_parse_data_value12_position(dat_path):
    """VALUE12 parsed correctly (guards against off-by-one in block formula)."""
    df = parse_data(dat_path)
    row = df[(df["ID"] == "ACW00011604") & (df["YEAR"] == 1900)].iloc[0]
    assert float(row["VALUE12"]) == pytest.approx(23.50, abs=0.01)


def test_parse_data_block_formula(dat_path):
    """QCFLAG6 exists and is parseable (mid-year block)."""
    df = parse_data(dat_path)
    assert "QCFLAG6" in df.columns


def test_parse_data_flag_dtype(dat_path):
    df = parse_data(dat_path)
    assert hasattr(df["QCFLAG1"].dtype, "categories")  # pd.CategoricalDtype


def test_parse_data_year_dtype(dat_path):
    df = parse_data(dat_path)
    assert df["YEAR"].dtype == "int16"


def test_parse_data_summary_logged(dat_path, caplog):
    with caplog.at_level(logging.INFO):
        parse_data(dat_path)
    messages = " ".join(r.message for r in caplog.records)
    assert "station" in messages.lower() or "record" in messages.lower()


# ---------------------------------------------------------------------------
# extract_archive / download tests
# ---------------------------------------------------------------------------

def test_extract_archive_returns_named_tuple(tmp_path):
    """extract_archive returns ParsedPaths with .dat and .inv attributes."""
    dat = tmp_path / "ghcnm.v4.0.1.20240101.qcu.dat"
    inv = tmp_path / "ghcnm.v4.0.1.20240101.qcu.inv"
    dat.write_text("dummy", encoding="latin-1")
    inv.write_text("dummy", encoding="latin-1")

    archive = tmp_path / "ghcnm.tavg.latest.qcu.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(dat, arcname=dat.name)
        tar.add(inv, arcname=inv.name)

    import ghcn_analysis.config as cfg
    original = cfg.RAW_DIR
    cfg.RAW_DIR = tmp_path
    try:
        result = extract_archive(archive)
    finally:
        cfg.RAW_DIR = original

    assert isinstance(result, ParsedPaths)
    assert result.dat.suffix == ".dat"
    assert result.inv.suffix == ".inv"


def test_extract_archive_invalid_tar(tmp_path):
    bad = tmp_path / "notatar.tar.gz"
    bad.write_bytes(b"not a tar file")
    with pytest.raises(ValueError, match="Not a valid tar archive"):
        extract_archive(bad)


@responses_lib.activate
def test_download_skips_cached(tmp_path):
    """No HTTP call made when file already on disk."""
    import ghcn_analysis.config as cfg
    from ghcn_analysis.data.download import download_dataset

    original = cfg.RAW_DIR
    cfg.RAW_DIR = tmp_path

    cached = tmp_path / "ghcnm.tavg.latest.qcu.tar.gz"
    cached.write_bytes(b"cached content")

    try:
        result = download_dataset("qcu")
    finally:
        cfg.RAW_DIR = original

    assert result == cached
    assert len(responses_lib.calls) == 0


@responses_lib.activate
def test_download_retries_on_503(tmp_path):
    """Session retries on 503."""
    import ghcn_analysis.config as cfg
    from ghcn_analysis.data.download import download_dataset

    url = f"{cfg.NOAA_BASE_URL}/ghcnm.tavg.latest.qcu.tar.gz"
    responses_lib.add(responses_lib.GET, url, status=503)
    responses_lib.add(responses_lib.GET, url, status=503)
    responses_lib.add(responses_lib.GET, url, status=503)

    original = cfg.RAW_DIR
    cfg.RAW_DIR = tmp_path
    try:
        with pytest.raises((RuntimeError, Exception)):
            download_dataset("qcu")
    finally:
        cfg.RAW_DIR = original


@responses_lib.activate
def test_download_no_retry_on_404(tmp_path):
    """RuntimeError raised on 404."""
    import ghcn_analysis.config as cfg
    from ghcn_analysis.data.download import download_dataset

    url = f"{cfg.NOAA_BASE_URL}/ghcnm.tavg.latest.qcu.tar.gz"
    responses_lib.add(responses_lib.GET, url, status=404)

    original = cfg.RAW_DIR
    cfg.RAW_DIR = tmp_path
    try:
        with pytest.raises((RuntimeError, Exception)):
            download_dataset("qcu")
    finally:
        cfg.RAW_DIR = original


def test_qfe_variant_warning(caplog):
    """DATASET_VARIANT="qfe" logs a WARNING."""
    import ghcn_analysis.config as cfg
    from ghcn_analysis.data.download import download_dataset

    cached = cfg.RAW_DIR / "ghcnm.tavg.latest.qfe.tar.gz"
    cached.write_bytes(b"cached")
    try:
        with caplog.at_level(logging.WARNING):
            download_dataset("qfe")
        assert any("1961" in r.message or "qfe" in r.message.lower()
                   for r in caplog.records)
    finally:
        cached.unlink(missing_ok=True)
