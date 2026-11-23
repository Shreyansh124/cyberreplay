"""Report extensions: split overview, ROC analysis, majority baseline and fit diagnostics.

This stage runs after the four final models exist (``pipeline.run_diagnostics``)
and never refits or replaces them:

* ``split_table`` / ``plot_split`` - flows per family in the stratified 80/20
  split (dataset section of the report);
* ``make_dummy`` - majority-class baseline (``DummyClassifier``, most frequent)
  scored like the real models in CV, on the training split and on the test split;
* ``check_reproduction`` - confirms that the refitted models reproduce the
  stored test metrics before anything new is written;
* ``roc_curves`` / ``macro_roc`` - one-vs-rest ROC curves and ROC-AUC on the
  test split, plotted for the final model and as a macro average for all models;
* ``fit_diagnosis`` - train vs validation macro F1 with a validation slice
  carved from the *training* split (the test split stays untouched);
* ``learning_curve_scores`` / ``validation_curve_scores`` /
  ``boosting_curve_scores`` - learning and validation curves on a
  minority-preserving training subsample with StratifiedKFold.

Leakage: apart from the test-split scoring every computation uses training rows only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .eda import FAMILY_COLORS
from .evaluate import MODEL_COLORS
from .features import ColumnSelector, encode_target, preprocessing_only
from .models import build_pipeline
from .plotstyle import FULL_WIDTH, HALF_WIDTH, save, styled
from .tuning import CV_SCORING, _outer_jobs
from .utils import save_table

DUMMY = "Dummy"
DUMMY_COLOR = "#bbbbbb"
# Test metrics that the refitted models must reproduce exactly (timings are re-measured and excluded).
REPRO_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_macro_excl_rare",
              "minority_recall", "benign_fpr"]
ROC_MIN_FPR = 1e-6


# --------------------------------------------------------------------------
# Split overview
# --------------------------------------------------------------------------
def split_table(train: pd.DataFrame, test: pd.DataFrame, raw_counts: pd.Series | None = None) -> pd.DataFrame:
    """Flows per family before de-duplication and in each part of the split.

    Args:
        train: Training split.
        test: Test split.
        raw_counts: Flows per family before de-duplication (optional).

    Returns:
        Table with one row per family plus a ``Total`` row: raw labels, meaning,
        raw and unique flow counts, train and test counts and the test share.

    Side effects:
        Writes ``reports/results/split_table.{csv,md}``.
    """
    rows = []
    for fam in config.CLASS_ORDER:
        n_train = int((train[config.TARGET_COL] == fam).sum())
        n_test = int((test[config.TARGET_COL] == fam).sum())
        row = {"family": fam,
               "raw_labels": ", ".join(l for l, f in config.LABEL_TO_FAMILY.items() if f == fam),
               "meaning": config.FAMILY_MEANING[fam]}
        if raw_counts is not None:
            row["raw_flows"] = int(raw_counts.get(fam, 0))
        row.update(unique_flows=n_train + n_test, train=n_train, test=n_test)
        rows.append(row)
    out = pd.DataFrame(rows)
    total = {"family": "Total", "raw_labels": "", "meaning": ""}
    total.update({c: int(out[c].sum()) for c in out.columns if c not in total})
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    out["test_share"] = out["test"] / out["unique_flows"]
    save_table(out, "split_table", index=False, floatfmt=".4f")
    return out


@styled(8)
def plot_split(table: pd.DataFrame) -> plt.Figure:
    """Train/test split: total flows per part and flows per family (log scale, counts annotated).

    Args:
        table: Output of ``split_table``.

    Returns:
        Figure (saved as ``17_train_test_split.png``).
    """
    fam = table[table["family"] != "Total"].set_index("family")
    tot = table[table["family"] == "Total"].iloc[0]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.55), gridspec_kw={"width_ratios": [1, 4.2]})
    parts = {"train": "#4c72b0", "test": "#dd8452"}
    for i, (part, color) in enumerate(parts.items()):
        n = int(tot[part])
        ax0.bar(i, n / 1e6, color=color, width=0.65)
        ax0.text(i, n / 1e6, f"{n:,}\n({n / tot['unique_flows']:.0%})", ha="center", va="bottom", fontsize=7)
    ax0.set_xticks([0, 1], list(parts))
    ax0.set_ylim(0, 2.3)
    ax0.set_ylabel("flows (millions)")
    ax0.set_title("All flows")
    x = np.arange(len(fam))
    for k, (part, color) in enumerate(parts.items()):
        pos = x + (k - 0.5) * 0.4
        ax1.bar(pos, fam[part], 0.4, color=color, label=part)
        for p, v in zip(pos, fam[part]):
            ax1.text(p, v * 1.15, f"{int(v):,}", ha="center", va="bottom", rotation=90, fontsize=6.5)
    ax1.set_yscale("log")
    ax1.set_ylim(1, 3e8)
    ax1.set_xticks(x, fam.index)
    ax1.set_ylabel("flows (log scale)")
    ax1.set_title("Flows per family (stratified 80/20, seed 42)")
    ax1.legend(loc="upper right", ncol=2, frameon=False)
    for ax in (ax0, ax1):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "17_train_test_split")
    return fig


# --------------------------------------------------------------------------
# Majority-class baseline
# --------------------------------------------------------------------------
def make_dummy(features: list[str]) -> Pipeline:
    """Majority-class baseline with the same input interface as the real pipelines.

    Args:
        features: Selected feature columns (ignored by the classifier, kept so the
            pipeline accepts the same raw rows).

    Returns:
        Unfitted ``Pipeline(select -> DummyClassifier(strategy="most_frequent"))``.
    """
    return Pipeline([("select", ColumnSelector(features)),
                     ("model", DummyClassifier(strategy="most_frequent"))])


def dummy_cv(tune_df: pd.DataFrame, features: list[str], folds: int = config.CV_FOLDS) -> pd.DataFrame:
    """Score the majority baseline with the same 5-fold StratifiedKFold as the default-parameter CV.

    Args:
        tune_df: Tuning sample (training rows only).
        features: Selected feature columns.
        folds: Number of folds.

    Returns:
        Long table with one row per fold (same columns as ``cv_baseline_folds``).
    """
    X, y = tune_df[features], encode_target(tune_df[config.TARGET_COL])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    res = cross_validate(make_dummy(features), X, y, cv=cv, scoring=CV_SCORING)
    return pd.DataFrame([{"model": DUMMY, "fold": k + 1, "fit_time_s": res["fit_time"][k],
                          **{m: res[f"test_{m}"][k] for m in CV_SCORING}} for k in range(folds)])


def training_f1(pipes: dict[str, Pipeline], train: pd.DataFrame, features: list[str]) -> pd.Series:
    """Macro F1 of each fitted pipeline re-scored on the rows it was trained on.

    Args:
        pipes: ``{model: fitted pipeline}`` (fitted on ``train``).
        train: Training split.
        features: Selected feature columns.

    Returns:
        Series of training macro F1 indexed by model.
    """
    y = encode_target(train[config.TARGET_COL])
    return pd.Series({n: f1_score(y, p.predict(train[features]), average="macro", zero_division=0)
                      for n, p in pipes.items()}, name="train_f1_macro")


def cv_overview(folds: pd.DataFrame, tuning_summary: pd.DataFrame, train_f1: pd.Series) -> pd.DataFrame:
    """Combine default-parameter CV, training macro F1 and tuned CV into one table.

    Balanced accuracy equals macro recall for a multi-class problem (the mean of
    the per-class recalls), so the CV balanced accuracy is the fold mean of
    ``recall_macro``.

    Args:
        folds: Per-fold CV scores of the four models plus the baseline.
        tuning_summary: Output of ``tuning.tuning_summary`` (tuned 3-fold CV).
        train_f1: Training macro F1 of the final models (``training_f1``).

    Returns:
        Table indexed by model.

    Side effects:
        Writes ``reports/results/cv_overview.{csv,md}``.
    """
    g = folds.groupby("model", sort=False)
    out = pd.DataFrame({"cv5_f1_macro_mean": g["f1_macro"].mean(), "cv5_f1_macro_std": g["f1_macro"].std(),
                        "cv5_balanced_accuracy": g["recall_macro"].mean()})
    out["train_f1_macro"] = train_f1.reindex(out.index)
    out["cv3_tuned_f1_macro"] = tuning_summary["cv3_tuned_f1_macro"].reindex(out.index)
    out.index.name = "model"
    save_table(out, "cv_overview")
    return out


# --------------------------------------------------------------------------
# Reproduction check and extended test table
# --------------------------------------------------------------------------
def check_reproduction(metrics: dict[str, dict], stored: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Compare freshly computed test metrics with the stored comparison table.

    Args:
        metrics: ``{model: metrics dict}`` from ``evaluate.evaluate_model``.
        stored: Stored ``test_comparison`` table indexed by model.
        tol: Largest allowed absolute difference.

    Returns:
        Table of absolute differences (models x ``REPRO_COLS``).

    Raises:
        RuntimeError: If a model is missing or any metric differs by more than ``tol``.
    """
    missing = [m for m in stored.index if m not in metrics]
    if missing:
        raise RuntimeError(f"no refitted model for {missing}")
    new = pd.DataFrame(metrics).T.loc[stored.index, REPRO_COLS].astype(float)
    diff = (new - stored[REPRO_COLS].astype(float)).abs()
    if (diff > tol).any().any():
        raise RuntimeError("refitted models do not reproduce the stored test metrics:\n" + diff.to_string())
    return diff


def extend_comparison(stored: pd.DataFrame, metrics: dict[str, dict],
                      new_cols: tuple[str, ...] = ("balanced_accuracy", "roc_auc_ovr")) -> pd.DataFrame:
    """Add new metric columns to the stored comparison table without changing its values.

    Args:
        stored: Stored ``test_comparison`` table indexed by model.
        metrics: ``{model: metrics dict}`` containing ``new_cols``.
        new_cols: Columns to add (inserted after ``accuracy`` and ``f1_macro``).

    Returns:
        Extended table (existing columns untouched).

    Side effects:
        Rewrites ``reports/results/test_comparison.{csv,md}``.
    """
    out = stored.drop(columns=[c for c in new_cols if c in stored.columns]).copy()
    for col in new_cols:
        out[col] = [metrics[m][col] for m in out.index]
    order = [c for c in out.columns if c not in new_cols]
    order.insert(order.index("accuracy") + 1, "balanced_accuracy")
    order.insert(order.index("f1_macro") + 1, "roc_auc_ovr")
    out = out[order]
    save_table(out, "test_comparison")
    return out


def baseline_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Headline test metrics of the four models and the majority baseline.

    Args:
        metrics: ``{model: metrics dict}`` including ``DUMMY``.

    Returns:
        Table indexed by model.

    Side effects:
        Writes ``reports/results/test_comparison_baseline.{csv,md}``.
    """
    cols = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr",
            "minority_recall", "benign_fpr"]
    out = pd.DataFrame(metrics).T[cols].astype(float)
    out.index.name = "model"
    save_table(out, "test_comparison_baseline")
    return out


# --------------------------------------------------------------------------
# ROC analysis
# --------------------------------------------------------------------------
def roc_curves(y: np.ndarray, proba: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """One-vs-rest ROC curve and AUC for every family.

    Args:
        y: True class codes.
        proba: Predicted probabilities ``(rows, 8)`` in ``config.CLASS_ORDER``.

    Returns:
        ``{family: (fpr, tpr, auc)}``.
    """
    out = {}
    for code, fam in enumerate(config.CLASS_ORDER):
        pos = (y == code).astype(int)
        fpr, tpr, _ = roc_curve(pos, proba[:, code])
        out[fam] = (fpr, tpr, roc_auc_score(pos, proba[:, code]))
    return out


def macro_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """Macro-average ROC curve: per-class TPR interpolated on a common FPR grid and averaged.

    Args:
        curves: Output of ``roc_curves``.

    Returns:
        ``(fpr grid, mean tpr, macro AUC)``; the AUC is the mean of the per-class
        AUCs, i.e. ``roc_auc_score(..., multi_class="ovr", average="macro")``.
    """
    grid = np.unique(np.concatenate([np.geomspace(ROC_MIN_FPR, 1, 400)] + [f for f, _, _ in curves.values()]))
    tpr = np.mean([np.interp(grid, f, t) for f, t, _ in curves.values()], axis=0)
    return grid, tpr, float(np.mean([a for _, _, a in curves.values()]))


def per_class_auc(curves_by_model: dict[str, dict]) -> pd.DataFrame:
    """Per-family one-vs-rest AUC of every model.

    Args:
        curves_by_model: ``{model: roc_curves output}``.

    Returns:
        Table (families x models).

    Side effects:
        Writes ``reports/results/roc_auc_per_class.{csv,md}``.
    """
    out = pd.DataFrame({m: {f: c[f][2] for f in config.CLASS_ORDER} for m, c in curves_by_model.items()})
    out.index.name = "family"
    save_table(out, "roc_auc_per_class")
    return out


def _roc_axis(ax: plt.Axes) -> None:
    """Shared look of the ROC panels: logarithmic FPR axis, chance line, grid."""
    ax.set_xscale("log")
    ax.set_xlim(ROC_MIN_FPR, 1)
    ax.set_ylim(-0.02, 1.02)
    grid = np.geomspace(ROC_MIN_FPR, 1, 100)
    ax.plot(grid, grid, ls=":", color="#888888", lw=0.9)
    ax.set_xlabel("false-positive rate (log scale)")
    ax.set_ylabel("true-positive rate")
    ax.grid(alpha=0.3)


@styled(8)
def plot_final_cm_roc(name: str, confusion: np.ndarray, curves: dict) -> plt.Figure:
    """Confusion matrix with raw counts next to the one-vs-rest ROC curves of the final model.

    Args:
        name: Model name.
        confusion: 8x8 test confusion matrix (rows = true class).
        curves: ``roc_curves`` output for the model.

    Returns:
        Figure (saved as ``18_final_model_cm_roc.png``).
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.25), gridspec_kw={"width_ratios": [1.08, 1]})
    annot = np.array([[f"{v:,}" if v else "" for v in row] for row in confusion])
    sns.heatmap(confusion, annot=annot, fmt="", cmap="Blues", norm=LogNorm(vmin=1, vmax=confusion.max()),
                cbar=False, ax=ax0, xticklabels=config.CLASS_ORDER, yticklabels=config.CLASS_ORDER,
                annot_kws={"fontsize": 6.3}, linewidths=0.4, linecolor="#dddddd", mask=confusion == 0)
    ax0.set_facecolor("white")
    ax0.set_xlabel("predicted family")
    ax0.set_ylabel("true family")
    ax0.tick_params(axis="x", rotation=40, labelsize=7)
    ax0.tick_params(axis="y", rotation=0, labelsize=7)
    plt.setp(ax0.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax0.set_title(f"(a) {name}: test confusion matrix (counts)")
    for fam, (fpr, tpr, auc) in curves.items():
        ax1.step(np.clip(fpr, ROC_MIN_FPR, 1), tpr, where="post", color=FAMILY_COLORS[fam],
                 label=f"{fam} ({auc:.4f})", lw=1.2)
    _roc_axis(ax1)
    ax1.legend(title="family (AUC)", loc="lower right", fontsize=6.8, title_fontsize=7.2, frameon=True,
               framealpha=0.9)
    ax1.set_title("(b) One-vs-rest ROC curves")
    fig.tight_layout(w_pad=1.2)
    save(fig, "18_final_model_cm_roc")
    return fig


@styled(8)
def plot_roc_comparison(curves_by_model: dict[str, dict]) -> plt.Figure:
    """Macro-average one-vs-rest ROC curve of every model (majority baseline = chance line).

    Args:
        curves_by_model: ``{model: roc_curves output}`` (may include ``DUMMY``).

    Returns:
        Figure (saved as ``19_roc_comparison.png``).
    """
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.75))
    for name, curves in curves_by_model.items():
        grid, tpr, auc = macro_roc(curves)
        if name == DUMMY:
            ax.plot(grid, tpr, color=DUMMY_COLOR, ls="--", lw=1.2, label=f"Dummy ({auc:.3f})")
        else:
            ax.plot(grid, tpr, color=MODEL_COLORS[name], label=f"{name} ({auc:.4f})")
    _roc_axis(ax)
    ax.legend(title="macro ROC-AUC (OvR)", loc="upper left", fontsize=6.8, title_fontsize=7.2, framealpha=0.9)
    fig.tight_layout()
    save(fig, "19_roc_comparison")
    return fig


# --------------------------------------------------------------------------
# Over/underfitting diagnosis
# --------------------------------------------------------------------------
def diagnose(train_f1: float, val_f1: float) -> str:
    """Label a model's fit from its training and validation macro F1.

    Rule (thresholds in ``config``): training F1 below ``UNDERFIT_TRAIN_F1`` means
    the model cannot even fit its training data (underfit, high bias); otherwise
    the gap ``train - validation`` decides between overfit (high variance),
    slight overfit and good fit.

    Args:
        train_f1: Macro F1 on the rows the model was fitted on.
        val_f1: Macro F1 on held-out validation rows.

    Returns:
        Diagnosis label.
    """
    gap = train_f1 - val_f1
    if train_f1 < config.UNDERFIT_TRAIN_F1:
        return "underfit (high bias)"
    if gap > config.OVERFIT_GAP:
        return "overfit (high variance)"
    if gap > config.SLIGHT_OVERFIT_GAP:
        return "slight overfit"
    return "good fit"


def fit_diagnosis(train: pd.DataFrame, features: list[str], params: dict[str, dict],
                  val_size: float = config.DIAG_VALIDATION_SIZE, seed: int = config.SEED) -> pd.DataFrame:
    """Train vs validation macro F1 of every tuned model and the majority baseline.

    A stratified validation slice is carved from the training split; each model
    is refitted with its tuned hyperparameters on the remaining training rows.

    Args:
        train: Training split (never the test split).
        features: Selected feature columns.
        params: ``{model: tuned hyperparameters}``.
        val_size: Share of the training split held out for validation.
        seed: Random seed of the carve.

    Returns:
        One row per model: rows used, train / validation macro F1, gap, diagnosis.
    """
    fit_part, val = train_test_split(train, test_size=val_size, stratify=train[config.TARGET_COL],
                                     random_state=seed)
    y_fit, y_val = encode_target(fit_part[config.TARGET_COL]), encode_target(val[config.TARGET_COL])
    pipes = {n: build_pipeline(n, features, **params[n]) for n in params}
    pipes[DUMMY] = make_dummy(features)
    rows = []
    for name, pipe in pipes.items():
        pipe.fit(fit_part[features], y_fit)
        tr = f1_score(y_fit, pipe.predict(fit_part[features]), average="macro", zero_division=0)
        va = f1_score(y_val, pipe.predict(val[features]), average="macro", zero_division=0)
        rows.append({"model": name, "fit_rows": len(fit_part), "val_rows": len(val), "train_f1_macro": tr,
                     "val_f1_macro": va, "gap": tr - va, "diagnosis": diagnose(tr, va)})
        print(f"[diag] {name}: train {tr:.4f} val {va:.4f}", flush=True)
    return pd.DataFrame(rows)


def _fit_and_score(name: str, params: dict, features: list[str], X: pd.DataFrame, y: np.ndarray,
                   fit_idx: np.ndarray, val_idx: np.ndarray) -> tuple[float, float]:
    """Fit one pipeline on ``fit_idx`` rows and return (train macro F1, validation macro F1)."""
    pipe = build_pipeline(name, features, **params)
    pipe.fit(X.iloc[fit_idx], y[fit_idx])
    tr = f1_score(y[fit_idx], pipe.predict(X.iloc[fit_idx]), average="macro", zero_division=0)
    va = f1_score(y[val_idx], pipe.predict(X.iloc[val_idx]), average="macro", zero_division=0)
    return tr, va


def _folds(y: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled StratifiedKFold (train, validation) index pairs."""
    return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def learning_curve_scores(name: str, params: dict, df: pd.DataFrame, features: list[str],
                          fractions: list[float] = config.LEARNING_CURVE_FRACTIONS,
                          folds: int = config.CURVE_FOLDS, seed: int = config.SEED) -> pd.DataFrame:
    """Train and cross-validated macro F1 against the number of training rows.

    In every StratifiedKFold split the model is fitted on a stratified fraction
    of the training fold and scored on that fraction (train) and on the whole
    validation fold (CV).

    Args:
        name: Model name.
        params: Tuned hyperparameters.
        df: Training subsample (training split only).
        features: Selected feature columns.
        fractions: Shares of the training fold to fit on.
        folds: Number of folds.
        seed: Random seed (folds and stratified fractions).

    Returns:
        Long table: model, fraction, train_rows, fold, train_f1, val_f1.
    """
    X, y = df[features], encode_target(df[config.TARGET_COL])
    tasks = []
    for k, (tr, va) in enumerate(_folds(y, folds, seed)):
        for frac in fractions:
            idx = tr if frac >= 1 else train_test_split(tr, train_size=frac, stratify=y[tr], random_state=seed)[0]
            tasks.append((k, frac, idx, va))
    scores = Parallel(n_jobs=_outer_jobs(name))(
        delayed(_fit_and_score)(name, params, features, X, y, idx, va) for _, _, idx, va in tasks)
    return pd.DataFrame([{"model": name, "fraction": frac, "train_rows": len(idx), "fold": k + 1,
                          "train_f1": s[0], "val_f1": s[1]} for (k, frac, idx, _), s in zip(tasks, scores)])


def validation_curve_scores(name: str, params: dict, df: pd.DataFrame, features: list[str], param: str,
                            values: list, folds: int = config.CURVE_FOLDS,
                            seed: int = config.SEED) -> pd.DataFrame:
    """Train and cross-validated macro F1 while one hyperparameter varies (others stay tuned).

    Args:
        name: Model name.
        params: Tuned hyperparameters.
        df: Training subsample (training split only).
        features: Selected feature columns.
        param: Hyperparameter to vary (e.g. ``max_depth``).
        values: Values to try (``None`` allowed).
        folds: Number of StratifiedKFold folds.
        seed: Random seed.

    Returns:
        Long table: model, param, value (as text), position, fold, train_f1, val_f1.
    """
    X, y = df[features], encode_target(df[config.TARGET_COL])
    tasks = [(i, v, k, tr, va) for i, v in enumerate(values) for k, (tr, va) in enumerate(_folds(y, folds, seed))]
    scores = Parallel(n_jobs=_outer_jobs(name))(
        delayed(_fit_and_score)(name, {**params, param: v}, features, X, y, tr, va) for _, v, _, tr, va in tasks)
    return pd.DataFrame([{"model": name, "param": param, "value": str(v), "position": i, "fold": k + 1,
                          "train_f1": s[0], "val_f1": s[1]} for (i, v, k, _, _), s in zip(tasks, scores)])


def boosting_curve_scores(params: dict, df: pd.DataFrame, features: list[str],
                          rounds: list[int] = config.XGB_ROUNDS, folds: int = config.CURVE_FOLDS,
                          seed: int = config.SEED) -> pd.DataFrame:
    """XGBoost train and cross-validated macro F1 against the number of boosting rounds.

    One booster with ``max(rounds)`` trees is fitted per fold; the score after
    ``r`` rounds uses only its first ``r`` trees (``iteration_range``).

    Args:
        params: Tuned XGBoost hyperparameters (``n_estimators`` is overridden).
        df: Training subsample (training split only).
        features: Selected feature columns.
        rounds: Round counts to score.
        folds: Number of StratifiedKFold folds.
        seed: Random seed.

    Returns:
        Long table: model, param, value, position, fold, train_f1, val_f1.
    """
    X, y = df[features], encode_target(df[config.TARGET_COL])
    rows = []
    for k, (tr, va) in enumerate(_folds(y, folds, seed)):
        pipe = build_pipeline("XGBoost", features, **{**params, "n_estimators": max(rounds)})
        pipe.fit(X.iloc[tr], y[tr])
        pre, model = preprocessing_only(pipe), pipe.named_steps["model"]
        Xtr, Xva = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
        for i, r in enumerate(rounds):
            p_tr = model.classes_[model.model_.predict_proba(Xtr, iteration_range=(0, r)).argmax(axis=1)]
            p_va = model.classes_[model.model_.predict_proba(Xva, iteration_range=(0, r)).argmax(axis=1)]
            rows.append({"model": "XGBoost", "param": "n_estimators", "value": str(r), "position": i, "fold": k + 1,
                         "train_f1": f1_score(y[tr], p_tr, average="macro", zero_division=0),
                         "val_f1": f1_score(y[va], p_va, average="macro", zero_division=0)})
        print(f"[curves] XGBoost fold {k + 1} done", flush=True)
    return pd.DataFrame(rows)


def _curve_panel(ax: plt.Axes, df: pd.DataFrame, x: str, color: str) -> None:
    """Draw mean +/- std of train and CV macro F1 over folds on one panel."""
    g = df.groupby(x)[["train_f1", "val_f1"]].agg(["mean", "std"])
    xs = g.index.to_numpy()
    for col, ls, label in (("train_f1", "--", "train"), ("val_f1", "-", "cross-validation")):
        m, s = g[(col, "mean")].to_numpy(), g[(col, "std")].fillna(0).to_numpy()
        ax.plot(xs, m, ls=ls, marker="o", color=color, label=label, ms=3.2)
        ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


@styled(7.5)
def plot_learning_curves(lc: pd.DataFrame) -> plt.Figure:
    """Learning curves (train vs CV macro F1 against training rows), one panel per model.

    Args:
        lc: Concatenated ``learning_curve_scores`` output.

    Returns:
        Figure (saved as ``20_learning_curves.png``).
    """
    models = list(dict.fromkeys(lc["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(HALF_WIDTH, 2.45))
    for ax, name in zip(np.atleast_1d(axes), models):
        sub = lc[lc["model"] == name].copy()
        sub["k_rows"] = sub.groupby("fraction")["train_rows"].transform("mean") / 1e3
        _curve_panel(ax, sub, "k_rows", MODEL_COLORS[name])
        ax.set_title(name.replace("LogisticRegression", "Logistic Regression")
                     .replace("RandomForest", "Random Forest"))
        ax.set_xlabel("training rows (thousands)")
        ax.set_xlim(0, sub["k_rows"].max() * 1.05)
    np.atleast_1d(axes)[0].set_ylabel("macro F1")
    np.atleast_1d(axes)[0].legend(loc="lower right", frameon=False, handlelength=1.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "20_learning_curves")
    return fig


@styled(7.5)
def plot_validation_curves(dt: pd.DataFrame, xgb: pd.DataFrame) -> plt.Figure:
    """Validation curves: Decision Tree vs ``max_depth`` and XGBoost vs boosting rounds.

    Args:
        dt: ``validation_curve_scores`` output for the Decision Tree.
        xgb: ``boosting_curve_scores`` output.

    Returns:
        Figure (saved as ``21_validation_curves.png``).
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(HALF_WIDTH, 2.45))
    _curve_panel(ax0, dt, "position", MODEL_COLORS["DecisionTree"])
    labels = dt.drop_duplicates("position").sort_values("position")["value"].replace({"None": "∞"})
    ax0.set_xticks(range(len(labels)), labels, fontsize=6.5)
    ax0.set_xlabel("max_depth (∞ = no limit)")
    ax0.set_ylabel("macro F1")
    ax0.set_title("Decision Tree")
    ax0.legend(loc="lower right", frameon=False, handlelength=1.8)
    x = xgb.copy()
    x["rounds"] = x["value"].astype(int)
    _curve_panel(ax1, x, "rounds", MODEL_COLORS["XGBoost"])
    ax1.set_xscale("log")
    ax1.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
    ax1.set_xlabel("boosting rounds (log)")
    ax1.set_title("XGBoost")
    fig.tight_layout(w_pad=0.6)
    save(fig, "21_validation_curves")
    return fig


def load_or_compute(name: str, compute, force: bool = False) -> pd.DataFrame:
    """Return ``reports/results/<name>.csv`` if present, otherwise compute and save it.

    Args:
        name: Result file stem.
        compute: Zero-argument callable returning a DataFrame.
        force: Recompute even if the file exists.

    Returns:
        The table.

    Side effects:
        Writes ``<name>.{csv,md}`` when computed.
    """
    path: Path = config.RESULTS_DIR / f"{name}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    out = compute()
    save_table(out, name, index=False)
    return out
