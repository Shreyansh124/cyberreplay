"""Chronological traffic replay with live predictions (pipeline stage 12).

Flows are sorted by their capture ``Timestamp`` (ties broken by capture
file and original row order, which preserves CICFlowMeter's output order
within the same minute) and streamed in batches. The replay clock maps
*capture* time to *wall* time through a speed factor: at ``speed=60`` one
minute of recorded traffic plays in one second. Long idle gaps are capped so
the stream never stalls.

Used by the Streamlit dashboard and runnable headless (``simulate``) for tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config


def order_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Sort flows into capture order.

    Args:
        df: Flows with ``Timestamp`` (may be NaT for the MachineLearningCSV
            variant), ``day_order`` and ``row_order``.

    Returns:
        Sorted copy with a fresh index. If timestamps are missing, the order is
        reconstructed from day file order + row order.
    """
    df = df.copy()
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL], errors="coerce")
    keys = [k for k in (config.TIMESTAMP_COL, config.DAY_ORDER_COL, config.ROW_ORDER_COL) if k in df.columns]
    if df[config.TIMESTAMP_COL].isna().all():
        keys = keys[1:]
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


def predict_batch(pipe: Pipeline, batch: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Predict family, confidence and binary flag for a batch of raw flows.

    Args:
        pipe: Fitted best pipeline.
        batch: Raw flow rows.
        features: Selected features expected by the pipeline.

    Returns:
        DataFrame with ``predicted``, ``confidence`` and ``pred_is_malicious``
        (index aligned with ``batch``).
    """
    proba = pipe.predict_proba(batch[features])
    codes = np.asarray(pipe.classes_)[proba.argmax(axis=1)]
    predicted = np.asarray(config.CLASS_ORDER, dtype=object)[codes]
    return pd.DataFrame({"predicted": predicted, "confidence": proba.max(axis=1),
                         "pred_is_malicious": (predicted != config.BENIGN).astype(int)}, index=batch.index)


@dataclass
class ReplayStep:
    """One replay tick: the batch of flows and its predictions.

    Attributes:
        index: Batch number (0-based).
        batch: Raw flows with prediction columns appended.
        capture_time: Timestamp of the last flow in the batch (NaT if unknown).
        progress: Fraction of the stream replayed so far (0-1].
    """

    index: int
    batch: pd.DataFrame
    capture_time: pd.Timestamp
    progress: float


def replay(df: pd.DataFrame, pipe: Pipeline, features: list[str], batch_size: int = 200,
           speed: float = 60.0, max_sleep: float = 1.0, start: int = 0,
           sleep: Callable[[float], None] = time.sleep) -> Iterator[ReplayStep]:
    """Stream ordered flows batch by batch with real-time pacing and predictions.

    Args:
        df: Flows already in capture order (see ``order_stream``).
        pipe: Fitted pipeline used for live predictions.
        features: Selected features.
        batch_size: Flows per tick.
        speed: Capture seconds replayed per wall-clock second (<= 0 = no pacing).
        max_sleep: Cap on a single wait, so idle gaps in the capture do not freeze the stream.
        start: Row offset to resume from.
        sleep: Sleep function (injectable for tests).

    Yields:
        ``ReplayStep`` objects in order.
    """
    ts = pd.to_datetime(df[config.TIMESTAMP_COL], errors="coerce") if config.TIMESTAMP_COL in df else None
    n = len(df)
    prev_time = None
    for k, lo in enumerate(range(start, n, batch_size)):
        batch = df.iloc[lo:lo + batch_size]
        preds = predict_batch(pipe, batch, features)
        out = pd.concat([batch, preds], axis=1)
        cap_time = ts.iloc[min(lo + batch_size, n) - 1] if ts is not None else pd.NaT
        if speed > 0 and prev_time is not None and pd.notna(cap_time) and pd.notna(prev_time):
            gap = (cap_time - prev_time).total_seconds() / speed
            sleep(float(np.clip(gap, 0.0, max_sleep)))
        prev_time = cap_time
        yield ReplayStep(index=k, batch=out, capture_time=cap_time, progress=min(lo + batch_size, n) / n)


def simulate(df: pd.DataFrame, pipe: Pipeline, features: list[str], batch_size: int = 500,
             speed: float = 0.0) -> pd.DataFrame:
    """Run a whole replay without pacing and return every flow with its prediction.

    Args:
        df: Flows (ordered or not; they are ordered here).
        pipe: Fitted pipeline.
        features: Selected features.
        batch_size: Flows per tick.
        speed: Pacing factor (0 = as fast as possible).

    Returns:
        Ordered flows with ``predicted``, ``confidence``, ``pred_is_malicious``.
    """
    steps = replay(order_stream(df), pipe, features, batch_size=batch_size, speed=speed)
    return pd.concat([s.batch for s in steps])


def detection_timeline(replayed: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Count predicted flows per time bin and family (for the timeline chart).

    Args:
        replayed: Output of ``simulate`` or the concatenated live batches.
        freq: Pandas time-bin frequency.

    Returns:
        Long DataFrame with ``time``, ``predicted`` and ``flows``.
    """
    t = pd.to_datetime(replayed[config.TIMESTAMP_COL], errors="coerce")
    g = replayed.assign(time=t.dt.floor(freq)).groupby(["time", "predicted"]).size()
    return g.rename("flows").reset_index()
