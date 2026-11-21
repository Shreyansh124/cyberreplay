"""Attack-pattern analysis and per-detection explanations (pipeline stages 10 and 14).

* model-native feature importance (tree impurity / XGBoost gain, |coef| for LogReg);
* SHAP ``TreeExplainer`` values on a stratified test subsample, summarised as
  "which features push a flow towards each attack family";
* per-attack feature distributions and a table of how each family differs
  from BENIGN traffic;
* ``explain_flow``: predicted class + top SHAP contributions + a plain-language
  reason, used by the Streamlit dashboard.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline

from . import config
from .eda import FAMILY_COLORS
from .features import preprocessing_only
from .models import WeightedXGBClassifier
from .utils import cap_per_class, save_table

TIME_HINTS = ("Duration", "IAT", "Active", "Idle")


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------
def native_importance(pipe: Pipeline, features: list[str]) -> pd.Series:
    """Model-native importance of each feature, normalised to sum to 1.

    Args:
        pipe: Fitted pipeline.
        features: Feature names in pipeline order.

    Returns:
        Series indexed by feature, sorted descending. Logistic Regression uses
        the mean absolute coefficient over classes (features are standardised).
    """
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        imp = np.abs(model.coef_).mean(axis=0)
    imp = imp / imp.sum()
    return pd.Series(imp, index=features).sort_values(ascending=False)


def plot_importances(pipes: dict[str, Pipeline], features: list[str], top: int = 15) -> plt.Figure:
    """Side-by-side top-``top`` importances for every model.

    Args:
        pipes: ``{model name: fitted pipeline}``.
        features: Selected features.
        top: Features shown per model.

    Returns:
        Figure (saved as ``12_feature_importance.png``).

    Side effects:
        Writes ``reports/results/feature_importance.{csv,md}`` (all models).
    """
    imps = {n: native_importance(p, features) for n, p in pipes.items()}
    save_table(pd.DataFrame(imps).sort_values(list(imps)[-1], ascending=False), "feature_importance")
    fig, axes = plt.subplots(1, len(imps), figsize=(5 * len(imps), 6))
    for ax, (name, imp) in zip(np.atleast_1d(axes), imps.items()):
        s = imp.head(top)[::-1]
        ax.barh(s.index, s.values, color="#4c72b0")
        ax.set_title(name + (" (|coef|)" if name == "LogisticRegression" else ""))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("normalised importance")
    fig.suptitle("Figure 12 - Model-native feature importance (top 15)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "12_feature_importance.png", dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def make_explainer(pipe: Pipeline) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for the tree model inside a pipeline.

    Args:
        pipe: Fitted pipeline whose ``model`` step is a tree / tree ensemble.

    Returns:
        ``shap.TreeExplainer`` (raw margin output for XGBoost, probabilities for sklearn trees).
    """
    model = pipe.named_steps["model"]
    if isinstance(model, WeightedXGBClassifier):
        model = model.model_
    return shap.TreeExplainer(model)


def transform_features(pipe: Pipeline, X_raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Apply the fitted preprocessing steps and keep feature names.

    Args:
        pipe: Fitted pipeline.
        X_raw: Raw rows (any superset of the feature columns).
        features: Selected features.

    Returns:
        Preprocessed DataFrame with the feature names as columns.
    """
    Xp = preprocessing_only(pipe).transform(X_raw)
    return pd.DataFrame(np.asarray(Xp), columns=features, index=X_raw.index)


def shap_values(explainer: shap.TreeExplainer, Xp: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values as a ``(rows, features, classes)`` array.

    Handles both the legacy list-of-arrays and the newer 3-D return formats.

    Args:
        explainer: TreeExplainer.
        Xp: Preprocessed features.

    Returns:
        3-D NumPy array of SHAP values.
    """
    sv = explainer.shap_values(Xp, check_additivity=False)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    return np.asarray(sv)


def shap_sample(test: pd.DataFrame, n: int = config.SHAP_SAMPLE_ROWS) -> pd.DataFrame:
    """Stratified test subsample for SHAP (equal cap per family; rare families kept whole).

    Args:
        test: Test split.
        n: Approximate total rows.

    Returns:
        Subsample of the test split.
    """
    return cap_per_class(test, config.TARGET_COL, max(1, n // len(config.CLASS_ORDER)))


def shap_by_family(sv: np.ndarray, y_family: pd.Series, features: list[str]) -> pd.DataFrame:
    """Mean SHAP value of each feature towards class *c*, over flows truly of class *c*.

    Positive values mean the feature pushes those flows towards their correct
    family - i.e. what makes that attack recognisable.

    Args:
        sv: SHAP array ``(rows, features, classes)``.
        y_family: True family of each row (aligned with ``sv``).
        features: Feature names.

    Returns:
        DataFrame (features x families) of mean |SHAP|.
    """
    out = {}
    fam = y_family.to_numpy()
    for code, name in enumerate(config.CLASS_ORDER):
        rows = fam == name
        if rows.any():
            out[name] = np.abs(sv[rows, :, code]).mean(axis=0)
    return pd.DataFrame(out, index=features)


def plot_shap_heatmap(by_family: pd.DataFrame, top: int = 15) -> plt.Figure:
    """Heatmap of the top SHAP features per family (each column scaled to its max).

    Args:
        by_family: Output of ``shap_by_family``.
        top: Number of features shown (ranked by their best per-family share).

    Returns:
        Figure (saved as ``13_shap_by_family.png``).

    Side effects:
        Writes ``reports/results/shap_by_family.{csv,md}``.
    """
    save_table(by_family, "shap_by_family")
    share = by_family / by_family.sum(axis=0)
    order = share.max(axis=1).sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(share.loc[order], annot=True, fmt=".2f", cmap="magma_r", ax=ax,
                cbar_kws={"label": "share of the family's total mean |SHAP|"})
    ax.set_title("Figure 13 - What identifies each family: mean |SHAP| towards the true class",
                 fontweight="bold")
    ax.set_xlabel("true family (SHAP for that class's output)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "13_shap_by_family.png", dpi=150, bbox_inches="tight")
    return fig


def plot_shap_beeswarms(sv: np.ndarray, Xp: pd.DataFrame, families: list[str] | None = None,
                        top: int = 8) -> plt.Figure:
    """SHAP beeswarm-style plots for selected attack families.

    Each dot is a flow; x = SHAP value towards the family, colour = the
    feature's value percentile (red = high, blue = low).

    Args:
        sv: SHAP array ``(rows, features, classes)``.
        Xp: Preprocessed features aligned with ``sv``.
        families: Families to show (default: DoS, DDoS, PortScan, Web Attack).
        top: Features per panel.

    Returns:
        Figure (saved as ``14_shap_beeswarm.png``).
    """
    families = families or ["DoS", "DDoS", "PortScan", "Web Attack"]
    pct = Xp.rank(pct=True).to_numpy()
    rng = np.random.default_rng(config.SEED)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"wspace": 0.55, "hspace": 0.3})
    for ax, fam in zip(axes.ravel(), families):
        code = config.CLASS_TO_CODE[fam]
        vals = sv[:, :, code]
        order = np.argsort(np.abs(vals).mean(axis=0))[::-1][:top]
        for rank, j in enumerate(order[::-1]):
            y = rank + rng.uniform(-0.3, 0.3, len(vals))
            ax.scatter(vals[:, j], y, c=pct[:, j], cmap="coolwarm", s=4, alpha=0.6, vmin=0, vmax=1)
        ax.set_yticks(range(len(order)), [Xp.columns[j] for j in order[::-1]], fontsize=8)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_title(f"SHAP towards '{fam}'")
        ax.set_xlabel("SHAP value (model output units)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="feature value percentile (low -> high)")
    fig.suptitle("Figure 14 - SHAP beeswarm for four attack families", fontweight="bold")
    fig.savefig(config.FIGURES_DIR / "14_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# Per-attack feature distributions
# --------------------------------------------------------------------------
def family_medians(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Median of every feature per family (used for explanations and the report).

    Args:
        df: Flows (training split).
        features: Feature names.

    Returns:
        DataFrame (families x features) of medians.
    """
    return df.groupby(config.TARGET_COL)[features].median().reindex(config.CLASS_ORDER)


def plot_attack_distributions(df: pd.DataFrame, features: list[str], per_class: int = 5_000) -> plt.Figure:
    """Violin plots (symlog) of the most explanatory features for every family.

    Args:
        df: Flows to plot (training split).
        features: Up to 8 features (e.g. top SHAP features).
        per_class: Max rows per family.

    Returns:
        Figure (saved as ``15_attack_feature_distributions.png``).
    """
    sample = cap_per_class(df[features + [config.TARGET_COL]], config.TARGET_COL, per_class)
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.2 * rows))
    for ax, feat in zip(axes.ravel(), features):
        vals = sample[[feat, config.TARGET_COL]].copy()
        vals[feat] = np.sign(vals[feat]) * np.log10(1 + np.abs(vals[feat]))
        sns.violinplot(data=vals, x=config.TARGET_COL, y=feat, order=config.CLASS_ORDER, hue=config.TARGET_COL,
                       palette=FAMILY_COLORS, ax=ax, cut=0, density_norm="width", inner="quartile",
                       legend=False, linewidth=0.6)
        ax.set_ylabel("sign(x)*log10(1+|x|)")
        ax.set_xlabel("")
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Figure 15 - Distribution of the most explanatory features per attack family",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "15_attack_feature_distributions.png", dpi=150, bbox_inches="tight")
    return fig


def pretty_value(feature: str, value: float) -> str:
    """Format a feature value with sensible units (CICFlowMeter times are microseconds).

    Args:
        feature: Feature name.
        value: Raw value.

    Returns:
        Human-readable string, e.g. ``"98.2 s"`` or ``"1,460"``.
    """
    if value is None or not np.isfinite(value):
        return "n/a"
    if any(h in feature for h in TIME_HINTS):
        a = abs(value)
        if a >= 1e6:
            return f"{value / 1e6:.2f} s"
        if a >= 1e3:
            return f"{value / 1e3:.1f} ms"
        return f"{value:.0f} us"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def compare_phrase(value: float, reference: float) -> str:
    """Describe a value relative to a benign reference ("much higher than", ...)."""
    if not np.isfinite(value) or not np.isfinite(reference):
        return "differs from"
    if np.isclose(value, reference):
        return "is close to"
    if reference <= 0 or value < 0:
        # Ratios are meaningless around zero / the -1 sentinel; compare directly.
        return "is higher than" if value > reference else "is lower than"
    ratio = value / reference
    if ratio >= 3:
        return "is much higher than"
    if ratio >= 1.2:
        return "is higher than"
    if ratio <= 1 / 3:
        return "is much lower than"
    if ratio <= 1 / 1.2:
        return "is lower than"
    return "is close to"


def distinguishing_table(by_family: pd.DataFrame, medians: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Summarise, for each attack family, its top SHAP features versus BENIGN.

    Args:
        by_family: Output of ``shap_by_family``.
        medians: Output of ``family_medians``.
        top: Features per family.

    Returns:
        Table with one row per (family, feature): family median, BENIGN
        median and a short description.

    Side effects:
        Writes ``reports/results/attack_signatures.{csv,md}``.
    """
    rows = []
    for fam in by_family.columns:
        if fam == config.BENIGN:
            continue
        for feat in by_family[fam].sort_values(ascending=False).index[:top]:
            v, b = medians.loc[fam, feat], medians.loc[config.BENIGN, feat]
            rows.append({"family": fam, "feature": feat, "family_median": pretty_value(feat, v),
                         "benign_median": pretty_value(feat, b),
                         "pattern": f"median {compare_phrase(v, b)} benign"})
    out = pd.DataFrame(rows)
    save_table(out, "attack_signatures", index=False)
    return out


# --------------------------------------------------------------------------
# Per-detection explanation (dashboard, step 14)
# --------------------------------------------------------------------------
def explain_flow(pipe: Pipeline, explainer: shap.TreeExplainer, row: pd.DataFrame, features: list[str],
                 benign_medians: pd.Series, top_k: int = 5) -> dict:
    """Explain one flow: predicted class, confidence, top SHAP features and a reason.

    Args:
        pipe: Fitted best pipeline.
        explainer: TreeExplainer for the pipeline's model.
        row: One-row DataFrame of raw flow columns.
        features: Selected features.
        benign_medians: BENIGN median per feature (training split).
        top_k: Number of contributing features returned.

    Returns:
        Dict with ``predicted`` (family), ``confidence``, ``probabilities``
        (Series), ``contributions`` (DataFrame: feature, value, benign_median,
        shap) and ``reason`` (plain-language text).
    """
    proba = pipe.predict_proba(row[features])[0]
    code = int(np.argmax(proba))
    predicted = config.CLASS_ORDER[code]
    Xp = transform_features(pipe, row, features)
    sv = shap_values(explainer, Xp)[0, :, code]
    raw_vals = row[features].iloc[0].astype(float)
    contrib = pd.DataFrame({"feature": features, "value": raw_vals.to_numpy(),
                            "benign_median": benign_medians.reindex(features).to_numpy(), "shap": sv})
    contrib = contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index).head(top_k)
    pushes = contrib[contrib["shap"] > 0].head(3)
    if predicted == config.BENIGN:
        lead = f"Classified as BENIGN with {proba[code]:.0%} confidence: the flow looks like normal traffic."
    else:
        lead = f"Flagged as **{predicted}** with {proba[code]:.0%} confidence."
    parts = []
    for _, r in pushes.iterrows():
        parts.append(f"{r.feature} = {pretty_value(r.feature, r.value)} "
                     f"{compare_phrase(r.value, r.benign_median)} typical benign traffic "
                     f"({pretty_value(r.feature, r.benign_median)})")
    reason = lead
    if parts:
        reason += " Main evidence: " + "; ".join(parts) + "."
    return {"predicted": predicted, "confidence": float(proba[code]),
            "probabilities": pd.Series(proba, index=[config.CLASS_ORDER[c] for c in pipe.classes_]),
            "contributions": contrib.reset_index(drop=True), "reason": reason}
