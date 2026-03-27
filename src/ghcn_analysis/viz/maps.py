"""Folium interactive map generators for GHCN station data."""

from __future__ import annotations

import logging

import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import MarkerCluster

from ghcn_analysis import config

__all__ = ["make_cluster_map", "make_temperature_map"]

logger = logging.getLogger(__name__)


def make_cluster_map(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> folium.Map:
    """Folium map with MarkerCluster. Popup shows NAME, ID, MEAN_TEMP."""
    station_df = (
        df.groupby("ID")
        .agg(
            LATITUDE=("LATITUDE", "first"),
            LONGITUDE=("LONGITUDE", "first"),
            NAME=("NAME", "first"),
            MEAN_TEMP=("MEAN_TEMP", "mean"),
        )
        .reset_index()
    )
    center_lat = float(station_df["LATITUDE"].mean())
    center_lon = float(station_df["LONGITUDE"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    cluster = MarkerCluster().add_to(m)

    for _, row in station_df.iterrows():
        popup = (
            f"<b>{row['NAME']}</b><br>"
            f"ID: {row['ID']}<br>"
            f"MEAN_TEMP: {row['MEAN_TEMP']:.2f}°C"
        )
        folium.Marker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            popup=folium.Popup(popup, max_width=200),
        ).add_to(cluster)

    if save:
        out = config.MAPS_DIR / f"{country}_cluster_map.html"
        m.save(str(out))
        logger.info("Saved %s", out)
    return m


def make_temperature_map(
    df: pd.DataFrame,
    country: str = "",
    save: bool = True,
) -> folium.Map:
    """Folium map with CircleMarkers coloured by MEAN_TEMP (viridis gradient)."""
    station_df = (
        df.groupby("ID")
        .agg(
            LATITUDE=("LATITUDE", "first"),
            LONGITUDE=("LONGITUDE", "first"),
            NAME=("NAME", "first"),
            MEAN_TEMP=("MEAN_TEMP", "mean"),
        )
        .reset_index()
    )
    center_lat = float(station_df["LATITUDE"].mean())
    center_lon = float(station_df["LONGITUDE"].mean())

    temps = station_df["MEAN_TEMP"].dropna()
    norm = mcolors.Normalize(vmin=float(temps.min()), vmax=float(temps.max()))
    cmap = plt.cm.viridis

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    for _, row in station_df.iterrows():
        if pd.isna(row["MEAN_TEMP"]):
            continue
        rgba = cmap(norm(row["MEAN_TEMP"]))
        color = mcolors.to_hex(rgba)
        popup = (
            f"<b>{row['NAME']}</b><br>"
            f"MEAN_TEMP: {row['MEAN_TEMP']:.2f}°C<br>"
            f"LAT: {row['LATITUDE']:.4f}, LON: {row['LONGITUDE']:.4f}"
        )
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=200),
        ).add_to(m)

    if save:
        out = config.MAPS_DIR / f"{country}_temperature_map.html"
        m.save(str(out))
        logger.info("Saved %s", out)
    return m
