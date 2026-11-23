"""Print-size style shared by every figure of the pipeline.

Each plotting function draws its figure at the size it is printed in the A4
report (the full text width or half of it), so a font size set here is the
size on paper. ``styled`` wraps a plotting function in a Matplotlib
``rc_context`` with these sizes; ``save`` writes the PNG at print resolution.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

from . import config

# A4 width (210 mm) minus two 12.5 mm margins, in inches.
FULL_WIDTH = 7.2
HALF_WIDTH = 3.55
DPI = 220


def report_rc(base: float = 8.5) -> dict:
    """Matplotlib rcParams for a figure printed at its drawn size.

    Args:
        base: Body font size in points (tick and legend labels are 0.5 pt smaller).

    Returns:
        Dict of rcParams for ``plt.rc_context``.
    """
    return {
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "legend.title_fontsize": base,
        "figure.titlesize": base + 1,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.5,
        "patch.linewidth": 0.5,
    }


def styled(base: float = 8.5) -> Callable:
    """Decorator that runs a plotting function inside ``report_rc(base)``.

    Args:
        base: Body font size in points.

    Returns:
        Decorator; the wrapped function keeps its signature and return value.
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def inner(*args, **kwargs):
            with plt.rc_context(report_rc(base)):
                return fn(*args, **kwargs)
        return inner
    return deco


def save(fig: plt.Figure, name: str, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Save a figure as ``<out_dir>/<name>.png`` at print resolution.

    Args:
        fig: Figure to save.
        name: File stem.
        out_dir: Destination folder (default ``reports/figures``).

    Returns:
        Path of the PNG.

    Side effects:
        Writes the PNG file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    return path
