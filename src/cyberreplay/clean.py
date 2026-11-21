"""Row-level cleaning of CIC-IDS2017 flows (pipeline stage 3).

Every operation here is *deterministic and statistic-free* (no means,
medians or quantiles), so it is safe to run on the full dataset before the
train/test split without leaking test information. Anything that needs a
fitted statistic - median imputation of NaN, outlier capping, scaling - is
done later inside sklearn Pipelines fitted on the training split only
(see ``features.py``).

Steps:
    1. normalise raw labels (cp1252 en dash, whitespace) and map them to the
       8 attack families + binary ``is_malicious``;
    2. replace +/-inf with NaN (inf appears when Flow Duration is 0);
    3. clip physically impossible negative values to 0 (keeping the -1
       sentinel of the TCP window columns);
    4. drop exact duplicate flows (same features and same family) so the same
       vector can never sit in both train and test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class CleaningReport:
    """Counts of everything the cleaning step changed (shown in notebook/report).

    Attributes:
        rows_in: Rows before cleaning.
        rows_out: Rows after cleaning.
        inf_counts: Per-column number of +/-inf values replaced by NaN.
        nan_counts: Per-column NaN counts after inf replacement (imputed later).
        negative_counts: Per-column number of invalid negatives clipped to 0.
        duplicates_removed: Duplicate rows removed, per family.
        conflicting_rows: Rows whose feature vector appears with >1 family.
    """

    rows_in: int = 0
    rows_out: int = 0
    inf_counts: dict[str, int] = field(default_factory=dict)
    nan_counts: dict[str, int] = field(default_factory=dict)
    negative_counts: dict[str, int] = field(default_factory=dict)
    duplicates_removed: dict[str, int] = field(default_factory=dict)
    conflicting_rows: int = 0

    def summary(self) -> pd.DataFrame:
        """Return a one-column table of the headline cleaning numbers."""
        return pd.DataFrame(
            {
                "value": {
                    "rows before cleaning": self.rows_in,
                    "+/-inf values -> NaN": sum(self.inf_counts.values()),
                    "NaN values (median-imputed in pipeline)": sum(self.nan_counts.values()),
                    "invalid negatives clipped to 0": sum(self.negative_counts.values()),
                    "duplicate rows removed": sum(self.duplicates_removed.values()),
                    "rows with conflicting labels (kept)": self.conflicting_rows,
                    "rows after cleaning": self.rows_out,
                }
            }
        )


def normalise_label(label: str) -> str:
    """Normalise one raw CIC-IDS2017 label string.

    Args:
        label: Raw label, e.g. ``"Web Attack \\x96 XSS"``.

    Returns:
        Clean label with an ASCII hyphen and single spaces, e.g. ``"Web Attack - XSS"``.
    """
    for bad in ("\x96", "\u2013", "\u2014", "\ufffd"):
        label = label.replace(bad, "-")
    return " ".join(label.split())


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw labels and add the family and binary target columns.

    Args:
        df: Raw flows with a ``Label`` column.

    Returns:
        The same frame with ``Label`` normalised, plus ``family`` (8 classes) and
        ``is_malicious`` (0/1).

    Raises:
        ValueError: If a label is not in ``config.LABEL_TO_FAMILY``.
    """
    df = df.copy()
    labels = df[config.RAW_LABEL_COL].astype(str)
    mapping = {raw: normalise_label(raw) for raw in labels.unique()}
    df[config.RAW_LABEL_COL] = labels.map(mapping)
    unknown = set(df[config.RAW_LABEL_COL].unique()) - set(config.LABEL_TO_FAMILY)
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    df[config.TARGET_COL] = df[config.RAW_LABEL_COL].map(config.LABEL_TO_FAMILY)
    df[config.BINARY_COL] = (df[config.TARGET_COL] != config.BENIGN).astype(np.int8)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """List candidate model features: numeric columns that are not identifiers/meta.

    Args:
        df: Flow DataFrame.

    Returns:
        Column names in file order (Protocol + 76 flow statistics before selection).
    """
    excluded = set(config.IDENTIFIER_COLS) | set(config.META_COLS)
    return [c for c in df.select_dtypes(include="number").columns if c not in excluded]


def replace_infinite(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Replace +/-inf with NaN in place.

    Args:
        df: Flow DataFrame (modified in place).
        cols: Columns to check.

    Returns:
        Mapping column -> number of infinite values replaced (non-zero only).
    """
    counts = {}
    for c in cols:
        mask = np.isinf(df[c].to_numpy())
        if mask.any():
            counts[c] = int(mask.sum())
            df.loc[mask, c] = np.nan
    return counts


def clip_negatives(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Clip invalid negative values to 0 in place.

    Durations, counts, byte lengths, rates and inter-arrival times cannot be
    negative; the few negatives (mostly BENIGN, from clock skew in
    CICFlowMeter) are measurement errors. Clipping keeps the rows - including
    the handful of DoS/DDoS flows affected - instead of deleting them.
    ``Init_Win_bytes_*`` are skipped because -1 there is a meaningful sentinel.

    Args:
        df: Flow DataFrame (modified in place).
        cols: Columns to check.

    Returns:
        Mapping column -> number of values clipped (non-zero only).
    """
    counts = {}
    for c in cols:
        if c in config.SENTINEL_NEGATIVE_COLS:
            continue
        mask = df[c].to_numpy() < 0
        if mask.any():
            counts[c] = int(mask.sum())
            df.loc[mask, c] = 0
    return counts


def drop_duplicates(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Drop duplicate flows (identical features *and* family), keeping the first.

    Duplicates are judged on model features only (identifiers excluded), because
    two flows from different IPs with identical statistics are the same sample
    as far as the model is concerned and would leak across train/test.

    Args:
        df: Flow DataFrame.
        cols: Feature columns.

    Returns:
        ``(deduplicated frame, removed-per-family counts, conflicting-row count)``
        where conflicting rows share a feature vector with a different family.
    """
    dup = df.duplicated(subset=cols + [config.TARGET_COL], keep="first")
    removed = df.loc[dup, config.TARGET_COL].value_counts().to_dict()
    out = df.loc[~dup].reset_index(drop=True)
    conflicting = int(out.duplicated(subset=cols, keep=False).sum())
    return out, {k: int(v) for k, v in removed.items()}, conflicting


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full row-level cleaning sequence.

    Args:
        df: Raw flows from ``data.load_raw``.

    Returns:
        ``(clean frame, CleaningReport)``. The clean frame keeps identifiers,
        timestamp and raw label for analysis/replay; NaN remain in
        ``Flow Bytes/s`` / ``Flow Packets/s`` and are imputed inside the model
        pipeline (fit on train only).
    """
    report = CleaningReport(rows_in=len(df))
    df = add_labels(df)
    cols = feature_columns(df)
    report.inf_counts = replace_infinite(df, cols)
    nan = df[cols].isna().sum()
    report.nan_counts = {k: int(v) for k, v in nan[nan > 0].items()}
    report.negative_counts = clip_negatives(df, cols)
    df, report.duplicates_removed, report.conflicting_rows = drop_duplicates(df, cols)
    report.rows_out = len(df)
    return df, report


def clean_cached(raw: pd.DataFrame | None = None, cache_dir: Path = config.PROCESSED_DIR,
                 use_cache: bool = True) -> tuple[pd.DataFrame, CleaningReport]:
    """Return the cleaned dataset, reusing a parquet cache when available.

    Args:
        raw: Raw flows; only needed when the cache is missing (loaded via
            ``data.load_raw`` if None).
        cache_dir: Folder for ``clean_flows.parquet`` and ``cleaning_report.json``.
        use_cache: Read/write the cache.

    Returns:
        ``(clean frame, CleaningReport)``.

    Side effects:
        Writes the two cache files (gitignored) on a cache miss.
    """
    data_path = Path(cache_dir) / "clean_flows.parquet"
    report_path = Path(cache_dir) / "cleaning_report.json"
    if use_cache and data_path.exists() and report_path.exists():
        return pd.read_parquet(data_path), CleaningReport(**json.loads(report_path.read_text()))
    if raw is None:
        from .data import load_raw
        raw = load_raw()
    df, report = clean(raw)
    if use_cache:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        report_path.write_text(json.dumps(asdict(report), indent=2))
    return df, report


def outlier_summary(df: pd.DataFrame, cols: list[str], quantile: float = config.OUTLIER_QUANTILE) -> pd.DataFrame:
    """Quantify extreme values per family (for EDA; nothing is modified).

    Args:
        df: Clean flows.
        cols: Feature columns.
        quantile: Global quantile used as the "extreme" threshold.

    Returns:
        DataFrame indexed by family with the share of rows (%) that have at
        least one feature above the threshold, and the row count.
    """
    thresholds = df[cols].quantile(quantile)
    extreme = (df[cols] > thresholds).any(axis=1)
    out = extreme.groupby(df[config.TARGET_COL]).agg(["mean", "size"])
    out.columns = ["share_rows_with_extreme_value_pct", "rows"]
    out["share_rows_with_extreme_value_pct"] = (out["share_rows_with_extreme_value_pct"] * 100).round(2)
    return out.reindex(config.CLASS_ORDER)
