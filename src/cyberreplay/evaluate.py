"""Final training, test-set evaluation and model comparison (pipeline stages 8 and 9).

* ``fit_final`` refits each tuned pipeline on the *full* training split and
  records the wall-clock training time.
* ``evaluate_model`` scores it once on the untouched test split: accuracy,
  macro precision/recall/F1, per-class report, confusion matrix, binary
  ``is_malicious`` metrics (derived as prediction != BENIGN), false-positive
  rate on BENIGN, minority-class recall and inference speed.
* ``per_raw_label_recall`` breaks test recall down by the 15 original labels
  (error analysis by attack sub-type).
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline

from . import config
from .eda import FAMILY_COLORS
from .features import encode_target
from .models import build_pipeline
from .plotstyle import FULL_WIDTH, HALF_WIDTH, save, styled
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


def full_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities with one column per family in ``config.CLASS_ORDER``.

    A column of zeros is used for a class the fitted model never saw.

    Args:
        pipe: Fitted pipeline.
        X: Rows to score.

    Returns:
        Array ``(rows, 8)``.
    """
    proba = pipe.predict_proba(X)
    out = np.zeros((len(X), len(config.CLASS_ORDER)))
    out[:, np.asarray(pipe.classes_, dtype=int)] = proba
    return out


def roc_auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-vs-rest ROC-AUC over the eight families.

    Args:
        y_true: True class codes.
        proba: Output of ``full_proba``.

    Returns:
        Mean of the per-family AUCs.
    """
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                               labels=list(range(len(config.CLASS_ORDER)))))


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
        (per-class DataFrame), ``confusion`` (8x8 array), ``y_pred`` (codes)
        and ``proba`` (class probabilities, for the ROC analysis).
    """
    X, y = test[features], encode_target(test[config.TARGET_COL])
    start = time.perf_counter()
    y_pred = pipe.predict(X)
    infer = time.perf_counter() - start
    # Outside the timer so the inference time stays a pure predict() measurement.
    proba = full_proba(pipe, X)
    labels = list(range(len(config.CLASS_ORDER)))
    p, r, f, s = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=config.CLASS_ORDER)
    yb, pb = (y != 0).astype(int), (y_pred != 0).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        # Mean per-class recall; for a multi-class problem this equals macro recall.
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr": roc_auc_ovr(y, proba),
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
    return {"metrics": metrics, "report": report, "y_pred": y_pred, "proba": proba,
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
    main_cols = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr",
                 "f1_macro_excl_rare",
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


def per_raw_label_recall(pipe: Pipeline, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Test recall of a fitted pipeline for each of the 15 original CIC-IDS2017 labels.

    The model predicts families, so a flow counts as detected when the predicted
    family equals the family its raw label maps to (e.g. a ``DoS slowloris`` flow
    predicted as ``DoS``). Splitting family recall this way shows which attack
    tools inside a family the model misses.

    Args:
        pipe: Fitted pipeline (normally the saved best model).
        test: Test split with the normalised raw ``Label`` and ``family`` columns.
        features: Selected features expected by the pipeline.

    Returns:
        Table indexed by raw label (ordered as ``config.LABEL_TO_FAMILY``) with
        ``family``, ``support``, ``detected``, ``recall`` and ``main_error``, the
        most frequent wrong prediction with its count ("" when none).

    Side effects:
        Writes ``reports/results/per_raw_label_recall.{csv,md}``.
    """
    predicted = pd.Series(np.asarray(config.CLASS_ORDER, dtype=object)[pipe.predict(test[features])],
                          index=test.index)
    rows = []
    for label, family in config.LABEL_TO_FAMILY.items():
        mask = test[config.RAW_LABEL_COL] == label
        if not mask.any():
            continue
        wrong = predicted[mask & (predicted != family)].value_counts()
        detected = int((predicted[mask] == family).sum())
        rows.append({"raw_label": label, "family": family, "support": int(mask.sum()), "detected": detected,
                     "recall": detected / int(mask.sum()),
                     "main_error": f"{wrong.index[0]} ({wrong.iloc[0]})" if len(wrong) else ""})
    out = pd.DataFrame(rows).set_index("raw_label")
    save_table(out, "per_raw_label_recall")
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
    """Save a figure as PNG in ``reports/figures`` at print resolution."""
    return save(fig, name)


@styled(11)
def plot_cv_folds(folds: pd.DataFrame) -> plt.Figure:
    """Per-fold macro F1 of the default-parameter 5-fold CV for each model (one panel).

    Logistic Regression scores about 0.53 and the trees 0.92-0.98, so the
    y-axis is broken into a tall band for the trees and a short band for the
    linear model; both bands share the fold axis. The legend gives the fold
    mean +/- standard deviation.

    Args:
        folds: Output of ``tuning.cross_validate_models``.

    Returns:
        Figure (saved as ``07_cv_folds.png``).
    """
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(FULL_WIDTH, 2.45),
                                      gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    trees = folds[folds["model"] != "LogisticRegression"]["f1_macro"]
    linear = folds[folds["model"] == "LogisticRegression"]["f1_macro"]
    for name, g in folds.groupby("model", sort=False):
        label = f"{name}  {g['f1_macro'].mean():.3f} ± {g['f1_macro'].std():.3f}"
        for ax in (top, bottom):
            ax.plot(g["fold"], g["f1_macro"], marker="o", ms=6, lw=2, label=label, color=MODEL_COLORS[name])
    pad = 0.006
    top.set_ylim(trees.min() - pad, trees.max() + pad)
    bottom.set_ylim(linear.min() - 0.004, linear.max() + 0.004)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(axis="x", bottom=False)
    # Diagonal break marks between the two y bands.
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=9, linestyle="none", color="k", mec="k", mew=1,
              clip_on=False)
    top.plot([0, 1], [0, 0], transform=top.transAxes, **kw)
    bottom.plot([0, 1], [1, 1], transform=bottom.transAxes, **kw)
    for ax in (top, bottom):
        ax.grid(alpha=0.3)
        ax.spines["right"].set_visible(False)
    top.spines["top"].set_visible(False)
    bottom.set_xticks(sorted(folds["fold"].unique()))
    bottom.set_xlabel("StratifiedKFold fold")
    fig.supylabel("macro F1", x=0.01)
    handles, labels = top.get_legend_handles_labels()
    top.legend(handles, labels, title="model: mean ± std", loc="upper left", bbox_to_anchor=(1.02, 1.0),
               frameon=False, fontsize=10.5, title_fontsize=11, alignment="left")
    fig.subplots_adjust(left=0.115, right=0.555, top=0.97, bottom=0.19)
    _save(fig, "07_cv_folds")
    return fig


@styled(7.5)
def plot_tuning(summary: pd.DataFrame, tuning_results: dict[str, pd.DataFrame]) -> plt.Figure:
    """Show the score of every RandomizedSearchCV candidate and the gain over defaults.

    Args:
        summary: Output of ``tuning.tuning_summary``.
        tuning_results: ``{model: cv_results DataFrame}``.

    Returns:
        Figure (saved as ``08_tuning.png``).
    """
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6))
    for i, (name, res) in enumerate(tuning_results.items()):
        axes[0].scatter(np.full(len(res), i) + np.random.default_rng(config.SEED).uniform(-0.15, 0.15, len(res)),
                        res["mean_test_score"], color=MODEL_COLORS[name], alpha=0.75, s=14)
    axes[0].set_xticks(range(len(tuning_results)), list(tuning_results))
    axes[0].set_ylabel("mean 3-fold macro F1")
    axes[0].set_title("(a) All RandomizedSearchCV candidates")
    axes[0].grid(alpha=0.3)
    x = np.arange(len(summary))
    axes[1].bar(x - 0.2, summary["cv5_default_f1_macro"], 0.4, label="default (5-fold)", color="#bbbbbb")
    axes[1].bar(x + 0.2, summary["cv3_tuned_f1_macro"], 0.4, label="tuned (3-fold)",
                color=[MODEL_COLORS[m] for m in summary.index])
    for i, (a, b) in enumerate(zip(summary["cv5_default_f1_macro"], summary["cv3_tuned_f1_macro"])):
        axes[1].text(i + 0.2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=6.5)
        axes[1].text(i - 0.2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=6.5)
    axes[1].set_xticks(x, summary.index)
    axes[1].set_ylim(max(0, summary[["cv5_default_f1_macro", "cv3_tuned_f1_macro"]].min().min() - 0.1), 1.06)
    axes[1].set_title("(b) Best tuned vs default CV macro F1")
    axes[1].legend(loc="lower right", framealpha=0.9)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.5)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "08_tuning")
    return fig


@styled(7)
def plot_confusion_matrices(results: dict[str, dict]) -> plt.Figure:
    """2x2 grid of row-normalised confusion matrices (recall per true class, counts below).

    Args:
        results: ``{model: evaluate_model output}``.

    Returns:
        Figure (saved as ``09_confusion_matrices.png``).
    """
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 7.0))
    for i, (ax, (name, r)) in enumerate(zip(axes.ravel(), results.items())):
        cm = r["confusion"]
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        annot = np.array([[f"{v:.2f}\n{c:,}" if c else "" for v, c in zip(rn, rc)] for rn, rc in zip(norm, cm)])
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER if i % 2 == 0 else False,
                    annot_kws={"fontsize": 5.6, "linespacing": 1.0}, linewidths=0.4, linecolor="#dddddd")
        ax.set_title(f"{name} (macro F1 {r['metrics']['f1_macro']:.4f})")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if i % 2 == 0 else "")
        ax.tick_params(axis="x", rotation=40, labelsize=6.5)
        ax.tick_params(axis="y", rotation=0, labelsize=6.5)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    _save(fig, "09_confusion_matrices")
    return fig


@styled(7)
def plot_comparison(comp: pd.DataFrame) -> plt.Figure:
    """Bar charts comparing the headline test metrics, binary metrics and timings.

    Args:
        comp: Comparison table from ``save_evaluation`` (with the binary columns).

    Returns:
        Figure (saved as ``10_model_comparison.png``).
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.75))
    m = comp[["accuracy", "precision_macro", "recall_macro", "f1_macro", "minority_recall"]]
    m.T.plot(kind="bar", ax=axes[0], color=[MODEL_COLORS[c] for c in m.index], width=0.8, legend=False)
    axes[0].set_ylim(max(0, m.values.min() - 0.05), 1.0)
    axes[0].set_title("(a) Multi-class test metrics")
    axes[0].set_xticklabels(["accu-\nracy", "preci-\nsion", "recall", "F1", "minority\nrecall"], rotation=0)
    b = comp[["binary_precision", "binary_recall", "binary_f1"]].copy()
    b["1 - benign_fpr"] = 1 - comp["benign_fpr"]
    b.T.plot(kind="bar", ax=axes[1], color=[MODEL_COLORS[c] for c in b.index], width=0.8, legend=False)
    axes[1].set_ylim(max(0, b.values.min() - 0.05), 1.0)
    axes[1].set_title("(b) Derived binary is_malicious")
    axes[1].set_xticklabels(["preci-\nsion", "recall", "F1", "1 - BENIGN\nFPR"], rotation=0)
    t = comp[["train_time_s", "inference_time_s"]]
    t.plot(kind="bar", ax=axes[2], logy=True, color=["#4c72b0", "#dd8452"], width=0.7)
    axes[2].set_title("(c) Train / inference time")
    axes[2].set_ylabel("seconds (log)")
    axes[2].set_ylim(0.03, 2000)
    axes[2].set_xticklabels([n.replace("LogisticRegression", "LogReg").replace("RandomForest", "RF")
                             .replace("DecisionTree", "DT").replace("XGBoost", "XGB") for n in t.index], rotation=0)
    axes[2].legend(["train (1.76 M rows)", "predict (440k rows)"], loc="upper right", fontsize=6.3, frameon=False)
    for i, (tr, inf) in enumerate(t.values):
        axes[2].text(i - 0.18, tr, f"{tr:.0f}s", ha="center", va="bottom", fontsize=6)
        axes[2].text(i + 0.18, inf, f"{inf:.2f}s", ha="center", va="bottom", fontsize=6)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.3)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[c]) for c in m.index]
    fig.legend(handles, list(m.index), loc="upper center", ncol=len(m.index), frameon=False,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.8)
    _save(fig, "10_model_comparison")
    return fig


@styled(7.5)
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
    fig, ax = plt.subplots(figsize=(HALF_WIDTH + 0.6, 2.9))
    sns.heatmap(f1, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "F1"},
                annot_kws={"fontsize": 7}, linewidths=0.4, linecolor="#eeeeee")
    ax.tick_params(axis="x", rotation=20, labelsize=6.8)
    ax.tick_params(axis="y", labelsize=7)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fig.tight_layout()
    _save(fig, "11_per_class_f1")
    return fig


def classification_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """sklearn classification report with family names (for printing)."""
    return classification_report(y_true, y_pred, labels=list(range(len(config.CLASS_ORDER))),
                                 target_names=config.CLASS_ORDER, digits=4, zero_division=0)
