"""Unit tests for the data-handling helpers (labels, timestamps, cleaning, replay order)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyberreplay import clean, config, data, features, replay  # noqa: E402


def test_normalise_label_handles_cp1252_dash():
    """The 0x96 byte read via latin-1 becomes an ASCII hyphen."""
    assert clean.normalise_label("Web Attack \x96 XSS") == "Web Attack - XSS"
    assert config.LABEL_TO_FAMILY[clean.normalise_label("Web Attack \x96 Sql Injection")] == "Web Attack"


def test_parse_timestamps_12_hour_clock():
    """Hours before 08:00 are afternoon hours; both CIC formats parse day-first."""
    ts = data.parse_timestamps(pd.Series(["7/7/2017 3:30", "4/7/2017 8:54", "03/07/2017 01:13:28"]))
    assert ts.iloc[0] == pd.Timestamp("2017-07-07 15:30")
    assert ts.iloc[1] == pd.Timestamp("2017-07-04 08:54")
    assert ts.iloc[2] == pd.Timestamp("2017-07-03 13:13:28")


def test_clean_drops_duplicates_and_fixes_values():
    """inf -> NaN, negatives clipped (sentinel kept), duplicates removed."""
    df = pd.DataFrame({
        "Flow Duration": [-5.0, 10.0, 10.0],
        "Flow Bytes/s": [np.inf, 1.0, 1.0],
        "Init_Win_bytes_forward": [-1.0, 5.0, 5.0],
        "Label": ["BENIGN", "DDoS", "DDoS"],
    })
    out, rep = clean.clean(df)
    assert len(out) == 2 and rep.duplicates_removed == {"DDoS": 1}
    assert out.loc[0, "Flow Duration"] == 0 and np.isnan(out.loc[0, "Flow Bytes/s"])
    assert out.loc[0, "Init_Win_bytes_forward"] == -1
    assert out[config.BINARY_COL].tolist() == [0, 1]


def test_identifiers_never_features():
    """Flow ID, IPs, ports and Timestamp are excluded from candidate features."""
    df = pd.DataFrame({c: [1.0] for c in ["Source Port", "Destination Port", "Protocol", "Flow Duration"]})
    df["Label"] = "BENIGN"
    assert clean.feature_columns(clean.add_labels(df)) == ["Protocol", "Flow Duration"]


def test_stratified_sample_keeps_minorities():
    """Small classes are kept whole in the tuning sample."""
    df = pd.DataFrame({config.TARGET_COL: ["BENIGN"] * 1000 + ["Rare"] * 5, "x": range(1005)})
    s = features.stratified_sample(df, 100, keep_all_below=50)
    assert (s[config.TARGET_COL] == "Rare").sum() == 5 and len(s) == 100


def test_replay_order_is_chronological():
    """order_stream sorts by timestamp, then day file and row order."""
    df = pd.DataFrame({config.TIMESTAMP_COL: pd.to_datetime(["2017-07-05 10:00", "2017-07-04 09:00",
                                                             "2017-07-04 09:00"]),
                       config.DAY_ORDER_COL: [2, 1, 1], config.ROW_ORDER_COL: [0, 7, 3]})
    out = replay.order_stream(df)
    assert out[config.ROW_ORDER_COL].tolist() == [3, 7, 0]
