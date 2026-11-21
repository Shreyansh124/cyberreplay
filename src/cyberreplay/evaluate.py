"""Final training, test-set evaluation and model comparison (pipeline stages 8 and 9).

* ``fit_final`` refits each tuned pipeline on the *full* training split and
  records the wall-clock training time.
* ``evaluate_model`` scores it once on the untouched test split: accuracy,
  macro precision/recall/F1, per-class report, confusion matrix, binary
  ``is_malicious`` metrics (derived as prediction != BENIGN), false-positive
  rate on BENIGN, minority-class recall and inference speed.
* plotting helpers produce the comparison figures and ``choose_best`` applies
  the documented selection rule.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)
from sklearn.pipeline import Pipeline

from . import config
from .eda import FAMILY_COLORS
from .features import encode_target
from .models import build_pipeline
from .utils import save_json, save_table

# Families that make up < 1% of the training data; their mean recall is the
# "minority recall" criterion used when choosing the final model.
MINORITY_CLASSES = ["PortScan", "Bot", "Brute Force", "Web Attack", "Rare"]
MODEL_COLORS = {"LogisticRegression": "#8c8c8c", "DecisionTree": "#55a868",
                "RandomForest": "#4c72b0", "XGBoost": "#c44e52"}


def fit_final(name: str, features: list[str], params: dict, train: pd.DataFrame,
              cache: bool = True) -> tuple[Pipeline, float]:
    """Fit a tuned pipeline on the full training split.

    Args:
        name: Model name.
        features: Selected features.
        params: Tuned hyperparameters (no ``model__`` prefix).
        train: Full training split.
        cache: Save the fitted pipeline to ``models/cache/<name>.joblib``.

    Returns:
        ``(fitted pipeline, training seconds)``.

    Side effects:
        Writes the cached pipeline (gitignored).
    """
    pipe = build_pipeline(name, features, **params)
    y = encode_target(train[config.TARGET_COL])
    start = time.perf_counter()
    pipe.fit(train[features], y)
    seconds = time.perf_counter() - start
    if cache:
        config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipe, "train_seconds": seconds}, config.MODEL_CACHE_DIR / f"{name}.joblib")
    print(f"[fit] {name}: {seconds:.1f}s on {len(train):,} rows", flush=True)
    return pipe, seconds


def load_cached(name: str) -> tuple[Pipeline, float] | None:
    """Load a cached final pipeline and its training time, or None if absent."""
    path = config.MODEL_CACHE_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    obj = joblib.load(path)
    return obj["pipeline"], obj["train_seconds"]


def benign_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of truly BENIGN flows predicted as any attack (false alarms)."""
    benign = y_true == config.CLASS_TO_CODE[config.BENIGN]
    return float((y_pred[benign] != config.CLASS_TO_CODE[config.BENIGN]).mean())


def evaluate_model(name: str, pipe: Pipeline, test: pd.DataFrame, features: list[str],
                   train_seconds: float) -> dict:
    """Score one fitted pipeline on the test split.

    Args:
        name: Model name.
        pipe: Fitted pipeline.
        test: Test split (never used before this point).
        features: Selected features.
        train_seconds: Training time from ``fit_final``.

    Returns:
        Dict with ``metrics`` (flat dict for the comparison table), ``report``
        (per-class DataFrame), ``confusion`` (8x8 array), ``y_pred`` (codes).
    """
    X, y = test[features], encode_target(test[config.TARGET_COL])
    start = time.perf_counter()
    y_pred = pipe.predict(X)
    infer = time.perf_counter() - start
    labels = list(range(len(config.CLASS_ORDER)))
    p, r, f, s = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=config.CLASS_ORDER)
    yb, pb = (y != 0).astype(int), (y_pred != 0).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        # Rare has only 9 test rows, so one flow moves its F1 by ~0.1; report macro F1 without it too.
        "f1_macro_excl_rare": float(report.drop(index="Rare")["f1"].mean()),
        "minority_recall": float(report.loc[MINORITY_CLASSES, "recall"].mean()),
        "benign_fpr": benign_false_positive_rate(y, y_pred),
        "binary_accuracy": accuracy_score(yb, pb),
        "binary_precision": precision_score(yb, pb, zero_division=0),
        "binary_recall": recall_score(yb, pb, zero_division=0),
        "binary_f1": f1_score(yb, pb, zero_division=0),
        "train_time_s": train_seconds,
        "inference_time_s": infer,
        "inference_us_per_flow": infer / len(X) * 1e6,
    }
    return {"metrics": metrics, "report": report, "y_pred": y_pred,
            "confusion": confusion_matrix(y, y_pred, labels=labels)}


def save_evaluation(results: dict[str, dict]) -> pd.DataFrame:
    """Persist the comparison table and per-class reports.

    Args:
        results: ``{model: evaluate_model output}``.

    Returns:
        Comparison DataFrame indexed by model.

    Side effects:
        Writes ``test_comparison``, ``test_binary_comparison``,
        ``per_class_<model>``, ``per_class_f1`` and ``per_class_recall``
        tables (CSV + Markdown) and ``test_metrics.json`` in ``reports/results``.
    """
    comp = pd.DataFrame([r["metrics"] for r in results.values()]).set_index("model")
    main_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_macro_excl_rare",
                 "minority_recall", "benign_fpr", "train_time_s", "inference_time_s", "inference_us_per_flow"]
    save_table(comp[main_cols], "test_comparison")
    save_table(comp[["binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "benign_fpr"]],
               "test_binary_comparison")
    for name, r in results.items():
        save_table(r["report"], f"per_class_{name}")
        save_table(pd.DataFrame(r["confusion"], index=config.CLASS_ORDER, columns=config.CLASS_ORDER),
                   f"confusion_{name}", floatfmt=".0f")
    save_table(pd.DataFrame({n: r["report"]["f1"] for n, r in results.items()}), "per_class_f1")
    save_table(pd.DataFrame({n: r["report"]["recall"] for n, r in results.items()}), "per_class_recall")
    save_json({n: r["metrics"] for n, r in results.items()}, config.RESULTS_DIR / "test_metrics.json")
    return comp


def _macro_f1_from_codes(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Fast macro F1 from integer labels via a bincount confusion matrix."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def bootstrap_f1(results: dict[str, dict], test: pd.DataFrame, n_boot: int = 300,
                 seed: int = config.SEED) -> pd.DataFrame:
    """Bootstrap 95% confidence intervals of test macro F1 (paired resamples for all models).

    Args:
        results: ``{model: evaluate_model output}``.
        test: Test split (true labels).
        n_boot: Number of bootstrap resamples of the test rows.
        seed: Random seed.

    Returns:
        Table per model: macro F1, CI bounds, mean paired difference to the best
        model and the share of resamples in which the best model scores higher.

    Side effects:
        Writes ``reports/results/test_bootstrap_ci.{csv,md}``.
    """
    y = encode_target(test[config.TARGET_COL])
    k = len(config.CLASS_ORDER)
    rng = np.random.default_rng(seed)
    names = list(results)
    scores = np.zeros((n_boot, len(names)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for j, n in enumerate(names):
            scores[b, j] = _macro_f1_from_codes(y[idx], results[n]["y_pred"][idx], k)
    point = np.array([results[n]["metrics"]["f1_macro"] for n in names])
    best = int(np.argmax(point))
    out = pd.DataFrame({
        "f1_macro": point,
        "ci95_low": np.percentile(scores, 2.5, axis=0),
        "ci95_high": np.percentile(scores, 97.5, axis=0),
        "diff_to_best_mean": (scores - scores[:, [best]]).mean(axis=0),
        "share_best_higher": (scores[:, [best]] > scores).mean(axis=0),
    }, index=pd.Index(names, name="model"))
    save_table(out, "test_bootstrap_ci")
    return out


def choose_best(comp: pd.DataFrame, tolerance: float = 0.005) -> tuple[str, str]:
    """Pick the final model.

    Rule: highest test macro F1; models within ``tolerance`` of the best are
    treated as tied and ranked by minority recall, then BENIGN false-positive
    rate, then inference time.

    Args:
        comp: Comparison table from ``save_evaluation``.
        tolerance: Macro-F1 gap regarded as a tie.

    Returns:
        ``(best model name, plain-language justification)``.
    """
    top = comp["f1_macro"].max()
    tied = comp[comp["f1_macro"] >= top - tolerance]
    ranked = tied.sort_values(["minority_recall", "benign_fpr", "inference_time_s"],
                              ascending=[False, True, True])
    best = ranked.index[0]
    b = comp.loc[best]
    others = comp.drop(index=best)
    why = (f"{best} has macro F1 {b.f1_macro:.4f} (next best {others.f1_macro.max():.4f}), "
           f"minority-class recall {b.minority_recall:.4f}, BENIGN false-positive rate {b.benign_fpr:.4%}, "
           f"trains in {b.train_time_s:.0f}s and predicts at {b.inference_us_per_flow:.2f} us/flow.")
    if len(tied) > 1:
        why += f" Tied within {tolerance} macro F1: {', '.join(tied.index)}; tie broken by minority recall."
    return best, why


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG in ``reports/figures``."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_cv_folds(folds: pd.DataFrame) -> plt.Figure:
    """Per-fold macro F1 of the default-parameter 5-fold CV for each model.

    Args:
        folds: Output of ``tuning.cross_validate_models``.

    Returns:
        Figure (saved as ``07_cv_folds.png``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, g in folds.groupby("model", sort=False):
        axes[0].plot(g["fold"], g["f1_macro"], marker="o", label=name, color=MODEL_COLORS[name])
    axes[0].set_xlabel("fold")
    axes[0].set_ylabel("macro F1")
    axes[0].set_xticks(sorted(folds["fold"].unique()))
    axes[0].set_title("Macro F1 per StratifiedKFold fold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    long = folds.melt(id_vars="model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                      var_name="metric", value_name="score")
    sns.barplot(data=long, x="metric", y="score", hue="model", palette=MODEL_COLORS, ax=axes[1],
                errorbar="sd", capsize=0.1)
    axes[1].set_ylim(max(0, long["score"].min() - 0.05), 1.0)
    axes[1].set_title("Mean +/- std over folds")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Figure 7 - 5-fold stratified CV with default hyperparameters (tuning sample)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_cv_folds")
    return fig


def plot_tuning(summary: pd.DataFrame, tuning_results: dict[str, pd.DataFrame]) -> plt.Figure:
    """Show the score of every RandomizedSearchCV candidate and the gain over defaults.

    Args:
        summary: Output of ``tuning.tuning_summary``.
        tuning_results: ``{model: cv_results DataFrame}``.

    Returns:
        Figure (saved as ``08_tuning.png``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, (name, res) in enumerate(tuning_results.items()):
        axes[0].scatter(np.full(len(res), i) + np.random.default_rng(config.SEED).uniform(-0.15, 0.15, len(res)),
                        res["mean_test_score"], color=MODEL_COLORS[name], alpha=0.7)
    axes[0].set_xticks(range(len(tuning_results)), list(tuning_results), rotation=10)
    axes[0].set_ylabel("mean 3-fold macro F1")
    axes[0].set_title("All RandomizedSearchCV candidates")
    axes[0].grid(alpha=0.3)
    x = np.arange(len(summary))
    axes[1].bar(x - 0.2, summary["cv5_default_f1_macro"], 0.4, label="default (5-fold)", color="#bbbbbb")
    axes[1].bar(x + 0.2, summary["cv3_tuned_f1_macro"], 0.4, label="tuned (3-fold)",
                color=[MODEL_COLORS[m] for m in summary.index])
    for i, (a, b) in enumerate(zip(summary["cv5_default_f1_macro"], summary["cv3_tuned_f1_macro"])):
        axes[1].text(i + 0.2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(i - 0.2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x, summary.index, rotation=10)
    axes[1].set_ylim(max(0, summary[["cv5_default_f1_macro", "cv3_tuned_f1_macro"]].min().min() - 0.1), 1.02)
    axes[1].set_title("Best tuned vs default CV macro F1")
    axes[1].legend(loc="lower right")
    fig.suptitle("Figure 8 - Hyperparameter tuning (scoring = f1_macro)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "08_tuning")
    return fig


def plot_confusion_matrices(results: dict[str, dict]) -> plt.Figure:
    """2x2 grid of row-normalised confusion matrices (recall per true class).

    Args:
        results: ``{model: evaluate_model output}``.

    Returns:
        Figure (saved as ``09_confusion_matrices.png``).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        cm = r["confusion"]
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        annot = np.array([[f"{v:.2f}\n({c:,})" if c else "" for v, c in zip(rn, rc)] for rn, rc in zip(norm, cm)])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER, annot_kws={"fontsize": 7},
                    linewidths=0.5, linecolor="#dddddd")
        ax.set_title(f"{name} (macro F1 {r['metrics']['f1_macro']:.4f})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Figure 9 - Test-set confusion matrices (row-normalised, counts in brackets)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "09_confusion_matrices")
    return fig


def plot_comparison(comp: pd.DataFrame) -> plt.Figure:
    """Bar charts comparing the headline test metrics, binary metrics and timings.

    Args:
        comp: Comparison table from ``save_evaluation``.

    Returns:
        Figure (saved as ``10_model_comparison.png``).
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    m = comp[["accuracy", "precision_macro", "recall_macro", "f1_macro", "minority_recall"]]
    m.T.plot(kind="bar", ax=axes[0], color=[MODEL_COLORS[c] for c in m.index], width=0.8)
    axes[0].set_ylim(max(0, m.values.min() - 0.05), 1.0)
    axes[0].set_title("Multi-class test metrics")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8, loc="lower left")
    b = comp[["binary_precision", "binary_recall", "binary_f1"]].copy()
    b["1 - benign_fpr"] = 1 - comp["benign_fpr"]
    b.T.plot(kind="bar", ax=axes[1], color=[MODEL_COLORS[c] for c in b.index], width=0.8, legend=False)
    axes[1].set_ylim(max(0, b.values.min() - 0.05), 1.0)
    axes[1].set_title("Binary is_malicious (derived) metrics")
    axes[1].tick_params(axis="x", rotation=20)
    t = comp[["train_time_s", "inference_time_s"]]
    t.plot(kind="bar", ax=axes[2], logy=True, color=["#4c72b0", "#dd8452"])
    axes[2].set_title("Training (full train split) and inference (full test split) time")
    axes[2].set_ylabel("seconds (log)")
    axes[2].tick_params(axis="x", rotation=20)
    for i, (tr, inf) in enumerate(t.values):
        axes[2].text(i - 0.12, tr, f"{tr:.0f}s", ha="center", va="bottom", fontsize=8)
        axes[2].text(i + 0.12, inf, f"{inf:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 10 - Model comparison on the held-out test split", fontweight="bold")
    fig.tight_layout()
    _save(fig, "10_model_comparison")
    return fig


def plot_per_class_f1(results: dict[str, dict]) -> plt.Figure:
    """Heatmap of per-class F1 (rows = families, columns = models).

    Args:
        results: ``{model: evaluate_model output}``.

    Returns:
        Figure (saved as ``11_per_class_f1.png``).
    """
    f1 = pd.DataFrame({n: r["report"]["f1"] for n, r in results.items()})
    support = next(iter(results.values()))["report"]["support"]
    f1.index = [f"{c} (n={int(s):,})" for c, s in zip(f1.index, support)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(f1, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "F1"})
    ax.set_title("Figure 11 - Per-class test F1 (support = test rows)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_per_class_f1")
    return fig


def classification_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """sklearn classification report with family names (for printing)."""
    return classification_report(y_true, y_pred, labels=list(range(len(config.CLASS_ORDER))),
                                 target_names=config.CLASS_ORDER, digits=4, zero_division=0)
