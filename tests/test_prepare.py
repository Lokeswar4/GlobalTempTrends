"""Tests for src/ghcn_analysis/data/prepare.py — 6 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ghcn_analysis.data.prepare import load_countries, prepare


COUNTRIES_TXT = """\
US United States
CA Canada
AU Australia
RU Russia
DE Germany
FR France
"""


def _make_metadata_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": ["US123456789", "CA987654321", "FR111111111"],
        "LATITUDE": [40.0, 45.0, 48.0],
        "LONGITUDE": [-75.0, -73.0, 2.0],
        "STNELEV": [100.0, -999.0, 50.0],
        "NAME": ["NEW YORK", "MONTREAL", "PARIS"],
    })


def _make_data_df() -> pd.DataFrame:
    rows = []
    for sid in ["US123456789", "CA987654321", "FR111111111"]:
        for year in range(1950, 1960):
            row = {"ID": sid, "YEAR": year}
            for i in range(1, 13):
                row[f"VALUE{i}"] = float(i * 2)
                row[f"QCFLAG{i}"] = ""
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def countries_file(tmp_path) -> Path:
    p = tmp_path / "ghcnm-countries.txt"
    p.write_text(COUNTRIES_TXT)
    return p


@pytest.fixture()
def meta_df():
    return _make_metadata_df()


@pytest.fixture()
def data_df():
    return _make_data_df()


def test_sentinel_replacement(meta_df, data_df, countries_file):
    # Insert -99.99 sentinel in VALUE1 for US station
    data_df.loc[data_df["ID"] == "US123456789", "VALUE1"] = -99.99
    meta_df.loc[meta_df["ID"] == "US123456789", "STNELEV"] = -999.0

    result = prepare(meta_df, data_df, countries_file, countries=["United States"])
    df = result["United States"]

    assert not (df["VALUE1"] == -99.99).any(), "VALUE sentinel not replaced"
    assert df["STNELEV"].isna().any(), "STNELEV -999.0 should become NaN"


def test_dmflag_dropped(meta_df, data_df, countries_file):
    for i in range(1, 13):
        data_df[f"DMFLAG{i}"] = ""
    result = prepare(meta_df, data_df, countries_file, countries=["United States"])
    df = result["United States"]
    assert not any(c.startswith("DMFLAG") for c in df.columns)


def test_dsflag_dropped(meta_df, data_df, countries_file):
    for i in range(1, 13):
        data_df[f"DSFLAG{i}"] = ""
    result = prepare(meta_df, data_df, countries_file, countries=["United States"])
    df = result["United States"]
    assert not any(c.startswith("DSFLAG") for c in df.columns)


def test_country_filter_fips(meta_df, data_df, countries_file):
    """Only rows matching target countries returned; uses FIPS (first 2 chars of ID)."""
    result = prepare(meta_df, data_df, countries_file, countries=["United States"])
    assert "United States" in result
    df = result["United States"]
    assert all(sid.startswith("US") for sid in df["ID"])
    assert not any(sid.startswith("FR") for sid in df["ID"])


def test_multi_country_dict_keys(meta_df, data_df, countries_file):
    result = prepare(
        meta_df, data_df, countries_file, countries=["United States", "Canada"]
    )
    assert set(result.keys()) == {"United States", "Canada"}


def test_merge_completeness(meta_df, data_df, countries_file):
    """Output has both metadata cols and data cols."""
    result = prepare(meta_df, data_df, countries_file, countries=["United States"])
    df = result["United States"]
    assert "NAME" in df.columns
    assert "LATITUDE" in df.columns
    assert "VALUE1" in df.columns
    assert "VALUE12" in df.columns
    assert "Country Name" in df.columns
