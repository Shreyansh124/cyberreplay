"""Small shared helpers (sampling, timing, table export) used by several stages."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config


def cap_per_class(df: pd.DataFrame, col: str, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Sample at most ``n`` rows from each class of ``col`` (small classes kept whole).

    Args:
        df: Input frame.
        col: Class column.
        n: Maximum rows per class.
        seed: Random seed.

    Returns:
        Sampled frame (original index preserved).
    """
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby(col, sort=False)]
    return pd.concat(parts)


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager that records elapsed wall time in ``result['seconds']``.

    Yields:
        A dict that receives the ``seconds`` key when the block exits.
    """
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def save_table(df: pd.DataFrame, name: str, out_dir: Path = config.RESULTS_DIR, index: bool = True,
               floatfmt: str = ".4f") -> None:
    """Write a result table as both CSV and Markdown.

    Args:
        df: Table to save.
        name: File stem (``<name>.csv`` and ``<name>.md``).
        out_dir: Destination folder.
        index: Whether to write the index.
        floatfmt: Float format for the Markdown version.

    Side effects:
        Writes two files in ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=index)
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=floatfmt) + "\n")


def save_json(obj: dict, path: Path) -> None:
    """Write a dict as pretty JSON, converting NumPy scalars to Python types.

    Args:
        obj: JSON-serialisable dict (NumPy scalars allowed).
        path: Destination file.

    Side effects:
        Writes ``path``.
    """
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default))


def load_json(path: Path) -> dict:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text())
