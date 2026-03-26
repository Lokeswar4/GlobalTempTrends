"""GHCN v4 dataset downloader.

All network operations use HTTPS only (no FTP).
Retries: 3 attempts, exponential backoff (1s, 2s, 4s) on 5xx / ConnectionError / Timeout.
Progress: tqdm streaming progress bar during archive download.
Caching: each file is skipped if already present and non-zero bytes on disk.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import NamedTuple

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from ghcn_analysis import config

__all__ = ["download_dataset", "download_support_files", "extract_archive", "ParsedPaths"]

logger = logging.getLogger(__name__)


class ParsedPaths(NamedTuple):
    dat: Path  # path to .dat temperature records file
    inv: Path  # path to .inv station inventory file


def _make_session(retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Build a requests.Session with retry + backoff for transient failures."""
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def download_dataset(variant: str = config.DATASET_VARIANT) -> Path:
    """Download the main GHCN .tar.gz archive via HTTPS.

    Args:
        variant: Dataset variant — "qcu" (default), "qcf", or "qfe".
                 Logs a WARNING if variant is "qfe" (covers 1961–2010 only).

    Returns:
        Path to the downloaded .tar.gz file in RAW_DIR.

    Raises:
        RuntimeError: If download fails after all retries.
    """
    if variant == "qfe":
        logger.warning(
            "QFE variant covers 1961–2010 only — earlier/later years will be absent"
        )

    filename = f"ghcnm.tavg.latest.{variant}.tar.gz"
    dest = config.RAW_DIR / filename
    url = f"{config.NOAA_BASE_URL}/{filename}"

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Using cached: %s", dest)
        return dest

    logger.info("Downloading %s ...", url)
    session = _make_session()
    try:
        with session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=filename
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
    except requests.HTTPError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed [{exc.response.status_code}]: {url}") from exc
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {url} — {exc}") from exc

    logger.info("Saved to %s", dest)
    return dest


def download_support_files() -> None:
    """Download ghcnm-countries.txt, ghcnm-flags.txt, readme.txt to RAW_DIR."""
    session = _make_session()
    for filename in config.SUPPORT_FILES:
        dest = config.RAW_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            logger.info("Using cached: %s", dest)
            continue
        url = f"{config.NOAA_BASE_URL}/{filename}"
        logger.info("Downloading %s ...", url)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.info("Saved to %s", dest)


def extract_archive(archive_path: Path) -> ParsedPaths:
    """Extract .tar.gz archive and locate .dat and .inv files.

    Args:
        archive_path: Path to the .tar.gz file.

    Returns:
        ParsedPaths(dat=..., inv=...) NamedTuple.

    Raises:
        ValueError: If archive_path is not a valid tar archive.
        FileNotFoundError: If no .dat or .inv found after extraction.
    """
    if not tarfile.is_tarfile(archive_path):
        raise ValueError(f"Not a valid tar archive: {archive_path}")

    # Infer variant from filename
    stem = archive_path.stem.replace(".tar", "")
    variant = stem.split(".")[-1]  # e.g. "qcu"

    extracted_dir = config.RAW_DIR / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s ...", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extracted_dir)

    dat_files = sorted(extracted_dir.glob(f"**/*.{variant}.dat"))
    inv_files = sorted(extracted_dir.glob(f"**/*.{variant}.inv"))

    if not dat_files:
        raise FileNotFoundError(
            f"No .{variant}.dat found in {extracted_dir}. "
            f"Contents: {list(extracted_dir.rglob('*'))}"
        )
    if not inv_files:
        raise FileNotFoundError(
            f"No .{variant}.inv found in {extracted_dir}."
        )

    if len(dat_files) > 1:
        logger.warning("Multiple .dat files found; using first: %s", dat_files[0])
    if len(inv_files) > 1:
        logger.warning("Multiple .inv files found; using first: %s", inv_files[0])

    return ParsedPaths(dat=dat_files[0], inv=inv_files[0])
