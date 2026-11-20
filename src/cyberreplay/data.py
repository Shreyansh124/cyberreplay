"""Dataset acquisition and loading for CIC-IDS2017 (pipeline stage 1).

Responsibilities:
    * download the official ``GeneratedLabelledFlows.zip`` (via a byte-identical
      Hugging Face mirror, see DECISIONS.md D1) and verify its SHA-256;
    * locate the eight day CSVs in any folder layout (local ``data/raw``, a
      Google Drive folder in Colab, the ``TrafficLabelling`` or the
      ``MachineLearningCSV`` variant);
    * read them robustly (cp1252 bytes, empty trailing rows, duplicated
      column, whitespace in headers) and add day / row-order metadata;
    * parse the 12-hour, day-first timestamps into real datetimes so that the
      replay step can stream flows chronologically;
    * cache the combined frame as parquet so later runs load in seconds.

Nothing here alters feature values - cleaning lives in ``clean.py``.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def download_dataset(dest_dir: Path | str = config.DATA_DIR, url: str = config.DATASET_MIRROR_URL,
                     verify: bool = True) -> Path:
    """Download and extract the CIC-IDS2017 TrafficLabelling CSVs.

    Args:
        dest_dir: Folder that will contain ``GeneratedLabelledFlows.zip`` and the
            extracted ``TrafficLabelling/`` folder.
        url: Download URL (defaults to the Hugging Face mirror of the official zip).
        verify: If True, check the zip against the known SHA-256.

    Returns:
        Path to the folder holding the extracted day CSVs.

    Side effects:
        Writes ~285 MB (zip) + ~1.2 GB (CSVs) under ``dest_dir``. Skips the
        download if the CSVs are already present.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if find_csv_files(dest_dir, missing_ok=True):
        return dest_dir

    zip_path = dest_dir / "GeneratedLabelledFlows.zip"
    if not zip_path.exists():
        print(f"Downloading CIC-IDS2017 from {url} ...")
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:  # noqa: S310 - fixed https URL
            shutil.copyfileobj(resp, fh, length=1 << 20)
        tmp.rename(zip_path)

    if verify:
        digest = _sha256(zip_path)
        if digest != config.DATASET_ZIP_SHA256:
            raise ValueError(f"SHA-256 mismatch for {zip_path}: {digest}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # The official archive names its folder "TrafficLabelling " (trailing space).
    for child in dest_dir.iterdir():
        if child.is_dir() and child.name != child.name.strip():
            target = dest_dir / child.name.strip()
            if not target.exists():
                child.rename(target)
    return dest_dir


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file (streamed in 1 MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_csv_files(data_dir: Path | str, missing_ok: bool = False) -> list[Path]:
    """Find the eight CIC-IDS2017 day CSVs anywhere below ``data_dir``.

    Args:
        data_dir: Root folder to search recursively (local or Google Drive).
        missing_ok: If True return an empty list instead of raising when no
            files are found.

    Returns:
        Day CSV paths sorted in capture order (Monday ... Friday DDoS). When both
        the TrafficLabelling and MachineLearningCSV variants exist, the
        TrafficLabelling files win because they carry timestamps.
    """
    data_dir = Path(data_dir)
    found: dict[str, Path] = {}
    if data_dir.exists():
        for path in sorted(data_dir.rglob("*.csv")):
            stem = path.name.split(".pcap")[0]
            if stem not in config.DAY_FILES:
                continue
            prefer = "TrafficLabelling" in str(path) or stem not in found
            if prefer:
                found[stem] = path
    if not found and not missing_ok:
        raise FileNotFoundError(
            f"No CIC-IDS2017 day CSVs found under {data_dir}. Run with --download or set DATA_DIR."
        )
    return sorted(found.values(), key=lambda p: config.DAY_FILES[p.name.split(".pcap")[0]][1])


def parse_timestamps(raw: pd.Series) -> pd.Series:
    """Convert CIC-IDS2017 timestamp strings to datetimes.

    The TrafficLabelling CSVs store ``d/m/Y H:M`` (Monday: ``dd/mm/YYYY HH:MM:SS``)
    in a 12-hour clock *without* AM/PM. Capture ran roughly 08:00-17:00, so any
    hour below 8 is an afternoon hour and is shifted by 12 hours.

    Args:
        raw: Series of timestamp strings.

    Returns:
        ``datetime64[ns]`` Series (NaT where parsing fails).
    """
    s = raw.astype("string").str.strip()
    parsed = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s[missing], format="%d/%m/%Y %H:%M", errors="coerce")
    afternoon = parsed.dt.hour < 8
    parsed.loc[afternoon] = parsed.loc[afternoon] + pd.Timedelta(hours=12)
    return parsed


def load_day_file(path: Path | str) -> pd.DataFrame:
    """Read one CIC-IDS2017 day CSV into a tidy DataFrame.

    Args:
        path: Path to a ``*.pcap_ISCX.csv`` file.

    Returns:
        DataFrame with stripped column names, empty rows and the duplicated
        ``Fwd Header Length`` column removed, a parsed ``Timestamp`` (if the file
        has one) and the metadata columns ``day``, ``day_order``, ``row_order``.
    """
    path = Path(path)
    stem = path.name.split(".pcap")[0]
    day, order = config.DAY_FILES[stem]
    # latin-1 decodes every byte; the cp1252 en dash (0x96) is fixed in clean.py.
    df = pd.read_csv(path, encoding="latin-1", low_memory=False, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")], errors="ignore")
    # Rows with no label are the empty ",,,,," lines at the end of the Thursday-AM file.
    df = df[df[config.RAW_LABEL_COL].notna()].reset_index(drop=True)
    if config.TIMESTAMP_COL in df.columns:
        df[config.TIMESTAMP_COL] = parse_timestamps(df[config.TIMESTAMP_COL])
    df[config.DAY_COL] = day
    df[config.DAY_ORDER_COL] = np.int8(order)
    df[config.ROW_ORDER_COL] = np.arange(len(df), dtype=np.int32)
    return df


def load_raw(data_dir: Path | str | None = None, use_cache: bool = True,
             cache_dir: Path | str = config.PROCESSED_DIR, download: bool = False) -> pd.DataFrame:
    """Load all eight day files into one DataFrame (with parquet caching).

    Args:
        data_dir: Folder containing the CSVs (searched recursively). Defaults to
            ``config.DATA_DIR``.
        use_cache: Read / write ``raw_flows.parquet`` in ``cache_dir``.
        cache_dir: Folder for the parquet cache.
        download: Download the dataset into ``data_dir`` if no CSVs are found.

    Returns:
        Combined DataFrame of all flows in capture order (~2.83M rows).

    Side effects:
        May download the dataset and writes ``raw_flows.parquet`` (gitignored).
    """
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    cache_path = Path(cache_dir) / "raw_flows.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    if download and not find_csv_files(data_dir, missing_ok=True):
        download_dataset(data_dir)
    files = find_csv_files(data_dir)
    frames = [load_day_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if config.TIMESTAMP_COL not in df.columns:
        # MachineLearningCSV variant: no timestamps. Chronology is rebuilt from
        # the day order + row order inside each file (documented in the README).
        df[config.TIMESTAMP_COL] = pd.NaT
    # Store measurement columns compactly; float32 keeps ~7 significant digits,
    # ample for flow statistics, and halves memory.
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(
        [config.DAY_ORDER_COL, config.ROW_ORDER_COL]
    )
    df[num_cols] = df[num_cols].astype(np.float32)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def describe_files(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Summarise the day files (size on disk) for the dataset description.

    Args:
        data_dir: Folder containing the CSVs.

    Returns:
        DataFrame with one row per day file: file name, day label, size in MB.
    """
    files = find_csv_files(data_dir if data_dir is not None else config.DATA_DIR)
    rows = []
    for f in files:
        stem = f.name.split(".pcap")[0]
        rows.append({"file": f.name, "day": config.DAY_FILES[stem][0],
                     "size_mb": round(f.stat().st_size / 1e6, 1)})
    return pd.DataFrame(rows)
