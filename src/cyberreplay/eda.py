"""Exploratory data analysis plots (pipeline stage 2).

Six Matplotlib/Seaborn figures, each saved as PNG in ``reports/figures``:
    1. class distribution (raw labels and families)
    2. traffic timeline per capture day (benign vs attack families)
    3. attack family x capture day matrix (why GroupKFold by day is infeasible)
    4. data-quality overview (duplicates, inf/NaN, negatives per family)
    5. per-family distributions of key flow features (log scale)
    6. feature correlation heatmap (motivates the correlation filter)

The functions only read data; observations are written in the notebook/report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy

from . import config
from .utils import cap_per_class

FAMILY_COLORS: dict[str, str] = {
    "BENIGN": "#4c72b0",
    "DoS": "#dd8452",
    "DDoS": "#c44e52",
    "PortScan": "#8172b3",
    "Bot": "#937860",
    "Brute Force": "#da8bc3",
    "Web Attack": "#55a868",
    "Rare": "#000000",
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _save(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    """Save a figure as PNG (150 dpi) and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_class_distribution(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Plot raw-label counts (log scale) and family shares.

    Args:
        df: Labelled flows (needs ``Label`` and ``family``).
        out_dir: Folder for ``01_class_distribution.png``.

    Returns:
        The Matplotlib figure.
    """
    raw = df[config.RAW_LABEL_COL].value_counts()
    fam = df[config.TARGET_COL].value_counts().reindex(config.CLASS_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = [FAMILY_COLORS[config.LABEL_TO_FAMILY[l]] for l in raw.index]
    axes[0].barh(raw.index[::-1], raw.values[::-1], color=colors[::-1])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("flows (log scale)")
    axes[0].set_title("Raw CIC-IDS2017 labels (15)")
    for i, v in enumerate(raw.values[::-1]):
        axes[0].text(v, i, f" {v:,}", va="center", fontsize=8)
    pct = fam / fam.sum() * 100
    axes[1].bar(fam.index, fam.values, color=[FAMILY_COLORS[f] for f in fam.index])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("flows (log scale)")
    axes[1].set_title("Attack families (8 classes)")
    axes[1].tick_params(axis="x", rotation=35)
    for i, (v, p) in enumerate(zip(fam.values, pct.values)):
        axes[1].text(i, v, f"{p:.3f}%" if p < 1 else f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 1 - Class distribution", fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_class_distribution", out_dir)
    return fig


def plot_traffic_timeline(df: pd.DataFrame, freq: str = "5min", out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Plot flows per time bin for each capture day, stacked by family.

    Args:
        df: Labelled flows with parsed ``Timestamp``.
        freq: Pandas resampling frequency for the bins.
        out_dir: Folder for ``02_traffic_timeline.png``.

    Returns:
        The Matplotlib figure.
    """
    ts = df[config.TIMESTAMP_COL]
    fig, axes = plt.subplots(len(WEEKDAYS), 1, figsize=(14, 12), sharey=False)
    for ax, day in zip(axes, WEEKDAYS):
        mask = ts.dt.day_name() == day
        sub = df.loc[mask, [config.TIMESTAMP_COL, config.TARGET_COL]]
        counts = (sub.groupby([pd.Grouper(key=config.TIMESTAMP_COL, freq=freq), config.TARGET_COL])
                  .size().unstack(fill_value=0))
        counts = counts[[c for c in config.CLASS_ORDER if c in counts.columns]]
        hours = counts.index.hour + counts.index.minute / 60
        ax.stackplot(hours, counts.T.values, labels=counts.columns,
                     colors=[FAMILY_COLORS[c] for c in counts.columns], alpha=0.9)
        ax.set_xlim(8.5, 17.25)
        ax.set_ylabel(f"{day}\nflows / {freq}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("hour of day")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[c]) for c in config.CLASS_ORDER]
    fig.legend(handles, config.CLASS_ORDER, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Figure 2 - Traffic volume over the capture week", fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "02_traffic_timeline", out_dir)
    return fig


def plot_attack_day_matrix(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Heatmap of raw attack label x capture day (log-coloured counts).

    Shows that every attack type occurs on exactly one day, which is why a
    GroupKFold split by day would leave each attack absent from training in
    the fold where its day is held out.

    Args:
        df: Labelled flows.
        out_dir: Folder for ``03_attack_day_matrix.png``.

    Returns:
        The Matplotlib figure.
    """
    order = [d for d, _ in sorted(config.DAY_FILES.values(), key=lambda x: x[1])]
    tab = pd.crosstab(df[config.RAW_LABEL_COL], df[config.DAY_COL]).reindex(columns=order)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(np.log10(tab.replace(0, np.nan)), annot=tab.replace(0, np.nan), fmt=",.0f",
                cmap="rocket_r", cbar_kws={"label": "log10(flows)"}, ax=ax, annot_kws={"fontsize": 7},
                linewidths=0.5, linecolor="#eeeeee")
    ax.set_xlabel("capture file (day)")
    ax.set_ylabel("raw label")
    ax.set_title("Figure 3 - Each attack type appears on a single capture day", fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_attack_day_matrix", out_dir)
    return fig


def plot_data_quality(labelled: pd.DataFrame, report, out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Plot duplicate share per family and the counts of invalid values.

    Args:
        labelled: Flows *before* de-duplication (with ``family``).
        report: ``clean.CleaningReport`` returned by ``clean.clean``.
        out_dir: Folder for ``04_data_quality.png``.

    Returns:
        The Matplotlib figure.
    """
    total = labelled[config.TARGET_COL].value_counts().reindex(config.CLASS_ORDER)
    removed = pd.Series(report.duplicates_removed).reindex(config.CLASS_ORDER).fillna(0)
    share = removed / total * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(share.index, share.values, color=[FAMILY_COLORS[c] for c in share.index])
    axes[0].set_ylabel("% of family rows that are exact duplicates")
    axes[0].set_title("Duplicate flows removed before splitting")
    axes[0].tick_params(axis="x", rotation=35)
    for i, (p, n) in enumerate(zip(share.values, removed.values)):
        axes[0].text(i, p, f"{p:.1f}%\n({int(n):,})", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, max(share.max() * 1.25, 5))
    issues = pd.concat([
        pd.Series(report.inf_counts, name="+/-inf"),
        pd.Series(report.nan_counts, name="NaN (incl. former inf)"),
        pd.Series(report.negative_counts, name="invalid negative"),
    ], axis=1).fillna(0)
    issues.plot(kind="barh", ax=axes[1], logx=True, color=["#c44e52", "#8172b3", "#dd8452"])
    axes[1].set_xlabel("values affected (log scale)")
    axes[1].set_title("Invalid values per column")
    fig.suptitle("Figure 4 - Data quality issues", fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_data_quality", out_dir)
    return fig


KEY_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Flow Bytes/s",
    "Packet Length Mean",
    "Init_Win_bytes_forward",
]


def plot_feature_distributions(df: pd.DataFrame, features: list[str] | None = None,
                               per_class: int = 20_000, out_dir: Path = config.FIGURES_DIR) -> plt.Figure:
    """Box plots of key flow features per family on a symmetric-log scale.

    Args:
        df: Clean flows.
        features: Features to show (defaults to ``KEY_FEATURES``).
        per_class: Max rows sampled per family (keeps plotting fast).
        out_dir: Folder for ``05_feature_distributions.png``.

    Returns:
        The Matplotlib figure.
    """
    features = features or KEY_FEATURES
    sample = cap_per_class(df[features + [config.TARGET_COL]], config.TARGET_COL, per_class)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, feat in zip(axes.ravel(), features):
        sns.boxplot(data=sample, x=config.TARGET_COL, y=feat, order=config.CLASS_ORDER, hue=config.TARGET_COL,
                    palette=FAMILY_COLORS, ax=ax, showfliers=False, legend=False)
        ax.set_yscale("symlog")
        # Only Init_Win_bytes_* can be negative (-1 sentinel); trim the empty negative half.
        ax.set_ylim(bottom=-1.5 if feat in config.SENTINEL_NEGATIVE_COLS else -0.1)
        ax.set_xlabel("")
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Figure 5 - Key flow features by attack family (symlog scale, outliers hidden)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_feature_distributions", out_dir)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list[str], sample_rows: int = 200_000,
                             out_dir: Path = config.FIGURES_DIR) -> tuple[plt.Figure, pd.DataFrame]:
    """Clustered Pearson correlation heatmap of the candidate features.

    Constant columns are skipped (their correlation is undefined).

    Args:
        df: Clean flows.
        cols: Candidate feature columns.
        sample_rows: Rows sampled for the correlation estimate.
        out_dir: Folder for ``06_correlation_heatmap.png``.

    Returns:
        ``(figure, table of feature pairs with |r| > CORRELATION_THRESHOLD)``.
    """
    sample = df[cols].sample(min(sample_rows, len(df)), random_state=config.SEED)
    sample = sample.loc[:, sample.std() > 0]
    corr = sample.corr().fillna(0)
    link = hierarchy.linkage(corr.values, method="average")
    order = hierarchy.leaves_list(link)
    corr = corr.iloc[order, order]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Pearson r", "shrink": 0.6})
    ax.tick_params(labelsize=6)
    ax.set_title("Figure 6 - Clustered feature correlation (blocks = redundant feature groups)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_correlation_heatmap", out_dir)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("r").reset_index()
    pairs.columns = ["feature_a", "feature_b", "r"]
    pairs = pairs[pairs["r"].abs() > config.CORRELATION_THRESHOLD].sort_values("r", ascending=False)
    return fig, pairs.reset_index(drop=True)
