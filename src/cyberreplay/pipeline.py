"""End-to-end CyberReplay pipeline and command-line entry point (stage 15).

Runs every step of the project in order and writes all artefacts:

    data -> EDA figures -> cleaning -> split + feature selection -> dataset sample
    -> 5-fold CV (defaults) -> RandomizedSearchCV -> final fit on full train
    -> test evaluation + comparison -> best model (joblib) -> SHAP / attack patterns
    -> replay simulation

Usage::

    python -m cyberreplay.pipeline --download          # full run (downloads data if missing)
    python -m cyberreplay.pipeline --data-dir /path    # use an existing copy
    python -m cyberreplay.pipeline --force             # ignore cached tuning / fits
    python -m cyberreplay.pipeline --quick             # small smoke test (minutes)
    python -m cyberreplay.pipeline --diagnostics       # report extensions only (see below)
    python -m cyberreplay.pipeline --figures           # redraw all figures from caches and tables

Expensive stages (tuning, final fits) are cached under ``reports/results`` and
``models/cache`` and reused unless ``--force`` is given. The notebook calls
the same ``stage_*`` functions.

``--diagnostics`` runs the separate diagnostics stage on top of the cached
tuning results and final fits: it first checks that the refitted models
reproduce the stored test metrics, then adds balanced accuracy and ROC-AUC,
the majority-class baseline, ROC curves, the train/test split figure and the
over/underfitting analysis (``diagnostics.py``). With ``--force`` only the
cached diagnostic tables are recomputed.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

# Headless backend for the CLI only; inside Jupyter the inline backend must stay active.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402

from . import (clean, config, data, diagnostics, eda, evaluate, explain, features, models,  # noqa: E402
               plotstyle, replay, tuning)
from .utils import load_json, save_json, save_table  # noqa: E402


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_load(data_dir: Path | None = None, download: bool = False) -> pd.DataFrame:
    """Load the raw dataset (downloading it if requested and missing)."""
    df = data.load_raw(data_dir, download=download)
    print(f"[load] {len(df):,} flows x {df.shape[1]} columns")
    return df


def stage_eda(raw: pd.DataFrame, clean_df: pd.DataFrame, report: clean.CleaningReport) -> pd.DataFrame:
    """Produce the six EDA figures; returns the highly-correlated feature pairs."""
    labelled = clean.add_labels(raw)
    eda.plot_class_distribution(labelled)
    eda.plot_traffic_timeline(labelled)
    eda.plot_attack_day_matrix(labelled)
    eda.plot_data_quality(labelled, report)
    eda.plot_feature_distributions(clean_df)
    _, pairs = eda.plot_correlation_heatmap(clean_df, clean.feature_columns(clean_df))
    save_table(pairs, "eda_correlated_pairs", index=False)
    plt.close("all")
    return pairs


def stage_select(train: pd.DataFrame, candidates: list[str], force: bool = False) -> dict:
    """Run (or load) feature selection on the training split.

    Side effects:
        Writes ``feature_selection.json`` and the correlation / importance tables.
    """
    path = config.RESULTS_DIR / "feature_selection.json"
    if path.exists() and not force:
        cached = load_json(path)
        missing = [f for f in cached["selected"] if f not in candidates]
        if not missing:
            return cached
        # e.g. the MachineLearningCSV variant has no Protocol column.
        print(f"[select] cached selection uses features missing from this data {missing}; recomputing")
    sel = features.select_features(train, candidates)
    save_table(sel["correlated"], "selection_correlated_dropped", index=False)
    save_table(sel["importance"], "selection_importance", index=False)
    out = {"selected": sel["selected"], "constant": sel["constant"], "counts": sel["counts"],
           "correlated_dropped": sel["correlated"]["dropped"].tolist(),
           "low_importance_dropped": sel["importance"].loc[~sel["importance"]["selected"], "feature"].tolist()}
    save_json(out, path)
    return out


def stage_sample(test: pd.DataFrame, n: int = config.DATASET_SAMPLE_ROWS) -> pd.DataFrame:
    """Write the committed dataset sample (stratified, drawn from the *test* split).

    Drawing it from the test split means the dashboard demo shows the model on
    flows it never trained on. Every attack family is kept whole when small;
    BENIGN / DoS / DDoS are capped.

    Side effects:
        Writes ``data/sample/cicids2017_sample.csv.gz``.
    """
    counts = test[config.TARGET_COL].value_counts()
    small = counts[counts < 2_000].index
    keep = test[test[config.TARGET_COL].isin(small)]
    budget = n - len(keep)
    big = test[~test[config.TARGET_COL].isin(small)]
    # BENIGN gets 60% of the remaining budget so the replay looks like real traffic.
    shares = {config.BENIGN: 0.6}
    other = [c for c in big[config.TARGET_COL].unique() if c != config.BENIGN]
    for c in other:
        shares[c] = 0.4 / len(other)
    parts = [keep]
    for c, s in shares.items():
        g = big[big[config.TARGET_COL] == c]
        parts.append(g.sample(min(len(g), int(budget * s)), random_state=config.SEED))
    sample = replay.order_stream(pd.concat(parts))
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(config.SAMPLE_PATH, index=False, compression="gzip")
    print(f"[sample] {len(sample):,} rows -> {config.SAMPLE_PATH}")
    return sample


def stage_cv(tune_df: pd.DataFrame, feats: list[str], force: bool = False) -> pd.DataFrame:
    """Run (or load) the default-parameter 5-fold stratified CV."""
    path = config.RESULTS_DIR / "cv_baseline_folds.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    return tuning.cross_validate_models(tune_df, feats)


def stage_tune(tune_df: pd.DataFrame, feats: list[str], force: bool = False,
               n_iter: dict[str, int] | None = None) -> dict[str, dict]:
    """Run (or load) RandomizedSearchCV for every model."""
    out = {}
    for name in config.MODEL_NAMES:
        path = config.RESULTS_DIR / f"best_params_{name}.json"
        if path.exists() and not force:
            r = load_json(path)
            r["cv_results"] = pd.read_csv(config.RESULTS_DIR / f"tuning_{name}.csv")
            out[name] = r
        else:
            out[name] = tuning.tune_model(name, tune_df, feats, n_iter=(n_iter or {}).get(name))
    return out


def stage_fit(train: pd.DataFrame, feats: list[str], tuned: dict[str, dict], force: bool = False):
    """Fit (or load) every tuned model on the full training split.

    Returns:
        ``{model: (pipeline, train_seconds)}``.
    """
    fitted = {}
    for name in config.MODEL_NAMES:
        cached = None if force else evaluate.load_cached(name)
        if cached and list(cached[0].named_steps["select"].columns) != list(feats):
            cached = None  # fitted on a different feature set (e.g. another dataset variant)
        fitted[name] = cached or evaluate.fit_final(name, feats, tuned[name]["best_params"], train)
    return fitted


def stage_evaluate(fitted: dict, test: pd.DataFrame, feats: list[str]) -> tuple[dict, pd.DataFrame]:
    """Evaluate every fitted model on the test split and draw the comparison figures."""
    results = {n: evaluate.evaluate_model(n, p, test, feats, t) for n, (p, t) in fitted.items()}
    comp = evaluate.save_evaluation(results)
    ci = evaluate.bootstrap_f1(results, test)
    print(ci.round(4).to_string())
    evaluate.plot_confusion_matrices(results)
    evaluate.plot_comparison(comp)
    evaluate.plot_per_class_f1(results)
    plt.close("all")
    return results, comp


def stage_save_best(best: str, why: str, fitted: dict, comp: pd.DataFrame, train: pd.DataFrame,
                    feats: list[str], tuned: dict) -> Path:
    """Save the best pipeline (compressed joblib) and its metadata JSON.

    Side effects:
        Writes ``models/best_model.joblib`` and ``models/best_model_metadata.json``.
    """
    pipe, seconds = fitted[best]
    joblib.dump(pipe, config.BEST_MODEL_PATH, compress=("xz", 3))
    medians = explain.family_medians(train, feats)
    meta = {
        "model": best,
        "justification": why,
        "features": feats,
        "class_order": config.CLASS_ORDER,
        "best_params": tuned[best]["best_params"],
        "test_metrics": comp.loc[best].to_dict(),
        "train_rows": len(train),
        "train_seconds": seconds,
        "family_medians": {f: medians.loc[f].to_dict() for f in medians.index},
        "dataset": {"source": config.DATASET_MIRROR_URL, "official_page": config.OFFICIAL_DATASET_PAGE,
                    "citation": config.DATASET_CITATION},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "seed": config.SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_mb": round(config.BEST_MODEL_PATH.stat().st_size / 1e6, 2),
    }
    save_json(meta, config.MODEL_METADATA_PATH)
    print(f"[best] {best} saved ({meta['size_mb']} MB): {why}")
    return config.BEST_MODEL_PATH


def stage_explain(best: str, fitted: dict, train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> dict:
    """Feature importance, SHAP and per-attack distributions for the best model.

    Returns:
        Dict with ``by_family`` (SHAP table), ``signatures`` and ``top_features``.
    """
    pipes = {n: p for n, (p, _) in fitted.items()}
    explain.plot_importances(pipes, feats)
    pipe = pipes[best]
    sample = explain.shap_sample(test)
    Xp = explain.transform_features(pipe, sample, feats)
    explainer = explain.make_explainer(pipe)
    sv = explain.shap_values(explainer, Xp)
    by_family = explain.shap_by_family(sv, sample[config.TARGET_COL].reset_index(drop=True), feats)
    explain.plot_shap_heatmap(by_family)
    explain.plot_shap_beeswarms(sv, Xp.reset_index(drop=True))
    share = by_family / by_family.sum(axis=0)
    top = list(share.drop(columns=[config.BENIGN]).max(axis=1).sort_values(ascending=False).index[:8])
    explain.plot_attack_distributions(train, top)
    medians = explain.family_medians(train, feats)
    signatures = explain.distinguishing_table(by_family, medians)
    plt.close("all")
    return {"by_family": by_family, "signatures": signatures, "top_features": top,
            "shap_values": sv, "shap_X": Xp, "shap_sample": sample}


def stage_replay(pipe, feats: list[str], sample: pd.DataFrame) -> pd.DataFrame:
    """Replay the dataset sample through the best model and plot the detection timeline.

    Side effects:
        Writes ``16_replay_timeline.png`` and ``replay_summary.{csv,md}``.
    """
    replayed = replay.simulate(sample, pipe, feats)
    tl = replay.detection_timeline(replayed, freq="5min")
    pivot = tl.pivot_table(index="time", columns="predicted", values="flows", fill_value=0)
    pivot = pivot[[c for c in config.CLASS_ORDER if c in pivot.columns]]
    with plt.rc_context(plotstyle.report_rc(7)):
        fig = _plot_replay_timeline(pivot)
    plotstyle.save(fig, "16_replay_timeline")
    plt.close(fig)
    summary = pd.crosstab(replayed[config.TARGET_COL], replayed["predicted"]).reindex(
        index=config.CLASS_ORDER, columns=config.CLASS_ORDER, fill_value=0)
    save_table(summary, "replay_summary", floatfmt=".0f")
    return replayed


def _plot_replay_timeline(pivot: pd.DataFrame) -> plt.Figure:
    """Stacked bars of predicted families per 5-minute bin (active periods only)."""
    fig, ax = plt.subplots(figsize=(plotstyle.FULL_WIDTH, 2.4))
    days = pivot.index.normalize().unique()
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for c in pivot.columns:
        ax.bar(x, pivot[c].values, bottom=bottom, color=eda.FAMILY_COLORS[c], label=c, width=1.0)
        bottom += pivot[c].values
    ax.set_ylim(0, bottom.max() * 1.25)
    ticks = [np.argmax(pivot.index.normalize() == d) for d in days]
    ax.set_xticks(ticks, [d.strftime("%a %d %b") for d in days])
    ax.set_ylabel("flows / 5 min (predicted)")
    ax.legend(ncol=8, fontsize=6.3, loc="upper left", frameon=False, handlelength=1.2, columnspacing=0.9)
    fig.tight_layout()
    return fig


def stage_diagnostics(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], tuned: dict, fitted: dict,
                      tune_df: pd.DataFrame, cv_folds: pd.DataFrame, tune_summary: pd.DataFrame,
                      raw_counts: pd.Series | None = None, results: dict | None = None,
                      force: bool = False) -> dict:
    """Report extensions: reproduction check, ROC-AUC, baseline, split figure and fit diagnostics.

    Args:
        train: Training split.
        test: Test split.
        feats: Selected features.
        tuned: ``stage_tune`` output (best hyperparameters).
        fitted: ``stage_fit`` output (final pipelines and training seconds).
        tune_df: 500k-row tuning sample (training rows only).
        cv_folds: Default-parameter CV fold scores.
        tune_summary: ``tuning.tuning_summary`` output.
        raw_counts: Flows per family before de-duplication (for the split table).
        results: ``stage_evaluate`` results if already computed (the notebook passes them).
        force: Recompute the cached diagnostic tables (fit diagnosis, curves).

    Returns:
        Dict with the reproduction differences, whether the saved best model
        predicts exactly like the refit, and every new table.

    Raises:
        RuntimeError: If the refitted models do not reproduce the stored test metrics.

    Side effects:
        Adds ``balanced_accuracy`` and ``roc_auc_ovr`` to ``test_comparison`` (other
        columns unchanged) and writes ``test_comparison_baseline``, ``roc_auc_per_class``,
        ``split_table``, ``cv_overview``, ``fit_diagnosis``, ``learning_curves``,
        ``validation_curve_dt``, ``validation_curve_xgb`` and Figures 07 and 17-21.
    """
    stored = pd.read_csv(config.RESULTS_DIR / "test_comparison.csv", index_col=0)
    if results is None:
        results = {n: evaluate.evaluate_model(n, p, test, feats, t) for n, (p, t) in fitted.items()}
    results = dict(results)
    metrics = {n: r["metrics"] for n, r in results.items()}
    diff = diagnostics.check_reproduction(metrics, stored)
    print(f"[diag] stored test metrics reproduced (max abs difference {diff.to_numpy().max():.2e})")
    best, _ = evaluate.choose_best(stored)
    saved = joblib.load(config.BEST_MODEL_PATH)
    same_predictions = bool(np.array_equal(saved.predict(test[feats]), results[best]["y_pred"]))
    print(f"[diag] saved {best} predicts exactly like the refit: {same_predictions}")
    evaluate_table = diagnostics.extend_comparison(stored, metrics)

    y_train = features.encode_target(train[config.TARGET_COL])
    dummy = diagnostics.make_dummy(feats).fit(train[feats], y_train)
    results[diagnostics.DUMMY] = evaluate.evaluate_model(diagnostics.DUMMY, dummy, test, feats, 0.0)
    metrics[diagnostics.DUMMY] = results[diagnostics.DUMMY]["metrics"]
    baseline = diagnostics.baseline_table(metrics)
    y_test = features.encode_target(test[config.TARGET_COL])
    curves = {n: diagnostics.roc_curves(y_test, r["proba"]) for n, r in results.items()}
    auc_table = diagnostics.per_class_auc(curves)
    diagnostics.plot_final_cm_roc(best, results[best]["confusion"], curves[best])
    diagnostics.plot_roc_comparison(curves)

    split = diagnostics.split_table(train, test, raw_counts)
    diagnostics.plot_split(split)
    pipes = {n: p for n, (p, _) in fitted.items()} | {diagnostics.DUMMY: dummy}
    train_f1 = diagnostics.training_f1(pipes, train, feats)
    folds_all = pd.concat([cv_folds, diagnostics.dummy_cv(tune_df, feats)], ignore_index=True)
    overview = diagnostics.cv_overview(folds_all, tune_summary, train_f1)

    params = {n: tuned[n]["best_params"] for n in config.MODEL_NAMES}
    fit_diag = diagnostics.load_or_compute(
        "fit_diagnosis", lambda: diagnostics.fit_diagnosis(train, feats, params), force)
    curve_df = features.stratified_sample(train, config.CURVE_SAMPLE_ROWS)
    lc = diagnostics.load_or_compute("learning_curves", lambda: pd.concat(
        [diagnostics.learning_curve_scores(n, params[n], curve_df, feats)
         for n in ("LogisticRegression", "RandomForest")], ignore_index=True), force)
    vc_dt = diagnostics.load_or_compute("validation_curve_dt", lambda: diagnostics.validation_curve_scores(
        "DecisionTree", params["DecisionTree"], curve_df, feats, "max_depth", config.DT_DEPTHS), force)
    vc_xgb = diagnostics.load_or_compute("validation_curve_xgb", lambda: diagnostics.boosting_curve_scores(
        params["XGBoost"], curve_df, feats), force)
    diagnostics.plot_learning_curves(lc)
    diagnostics.plot_validation_curves(vc_dt, vc_xgb)
    stage_result_figures()
    return {"reproduction_diff": diff, "best": best, "best_model_same_predictions": same_predictions,
            "test_comparison": evaluate_table, "baseline": baseline, "roc_auc_per_class": auc_table,
            "split_table": split, "cv_overview": overview, "fit_diagnosis": fit_diag, "learning_curves": lc,
            "validation_curve_dt": vc_dt, "validation_curve_xgb": vc_xgb, "roc_curves": curves}


def stage_result_figures() -> None:
    """Redraw Figures 07, 08 and 10 from the stored result tables.

    The committed tables keep the timings of the reported run, so drawing these
    figures from the tables (rather than from a fresh evaluation, whose
    wall-clock times differ) keeps figures and report numbers identical.

    Side effects:
        Writes ``07_cv_folds.png``, ``08_tuning.png`` and ``10_model_comparison.png``.
    """
    r = config.RESULTS_DIR
    evaluate.plot_cv_folds(pd.read_csv(r / "cv_baseline_folds.csv"))
    evaluate.plot_tuning(pd.read_csv(r / "tuning_summary.csv", index_col=0),
                         {n: pd.read_csv(r / f"tuning_{n}.csv") for n in config.MODEL_NAMES})
    binary = pd.read_csv(r / "test_binary_comparison.csv", index_col=0).drop(columns="benign_fpr")
    evaluate.plot_comparison(pd.read_csv(r / "test_comparison.csv", index_col=0).join(binary))
    plt.close("all")


def _prepare_cached(data_dir: Path | None, download: bool) -> dict:
    """Rebuild the clean split, features, tuning results and final fits from the caches.

    Missing caches are recomputed with the same code as ``run``.

    Returns:
        Dict with ``train``, ``test``, ``feats``, ``tune_df``, ``folds``, ``tuned``,
        ``summary``, ``fitted``, ``report`` and ``raw_counts``.
    """
    config.ensure_dirs()
    np.random.seed(config.SEED)
    cache = config.PROCESSED_DIR / "clean_flows.parquet"
    raw = None if cache.exists() else stage_load(data_dir, download)
    clean_df, report = clean.clean_cached(raw)
    del raw
    raw_counts = clean_df[config.TARGET_COL].value_counts().add(pd.Series(report.duplicates_removed),
                                                                  fill_value=0)
    train, test = features.split_train_test(clean_df)
    del clean_df
    feats = stage_select(train, clean.feature_columns(train))["selected"]
    tune_df = features.stratified_sample(train, config.TUNING_SAMPLE_SIZE)
    folds = stage_cv(tune_df, feats)
    tuned = stage_tune(tune_df, feats)
    summary = tuning.tuning_summary(tuned, folds)
    fitted = stage_fit(train, feats, tuned)
    return {"train": train, "test": test, "feats": feats, "tune_df": tune_df, "folds": folds, "tuned": tuned,
            "summary": summary, "fitted": fitted, "report": report, "raw_counts": raw_counts}


def run_figures(data_dir: Path | None = None, download: bool = False) -> None:
    """Redraw every figure from cached data, fits and result tables (no evaluation table is rewritten).

    Args:
        data_dir: Folder with the CIC-IDS2017 CSVs (default ``config.DATA_DIR``).
        download: Download the dataset if it is missing.

    Side effects:
        Rewrites the PNGs in ``reports/figures`` and the deterministic EDA, SHAP and
        replay tables they are drawn from.
    """
    raw = stage_load(data_dir, download)
    clean_df, report = clean.clean_cached(raw)
    stage_eda(raw, clean_df, report)
    del raw, clean_df
    c = _prepare_cached(data_dir, download)
    results = {n: evaluate.evaluate_model(n, p, c["test"], c["feats"], t) for n, (p, t) in c["fitted"].items()}
    evaluate.plot_confusion_matrices(results)
    evaluate.plot_per_class_f1(results)
    best, _ = evaluate.choose_best(pd.read_csv(config.RESULTS_DIR / "test_comparison.csv", index_col=0))
    stage_explain(best, c["fitted"], c["train"], c["test"], c["feats"])
    sample = replay.order_stream(data.prepare_external(pd.read_csv(config.SAMPLE_PATH)))
    stage_replay(joblib.load(config.BEST_MODEL_PATH), c["feats"], sample)
    stage_result_figures()


def run_diagnostics(data_dir: Path | None = None, download: bool = False, force: bool = False) -> dict:
    """Run only the diagnostics stage, reusing cached cleaning, selection, tuning and final fits.

    Missing caches are rebuilt with the same code as ``run`` (final fits use the
    saved best hyperparameters). EDA, the test evaluation tables and the saved
    best model are not rewritten.

    Args:
        data_dir: Folder with the CIC-IDS2017 CSVs (default ``config.DATA_DIR``).
        download: Download the dataset if it is missing.
        force: Recompute the cached diagnostic tables.

    Returns:
        ``stage_diagnostics`` output.
    """
    c = _prepare_cached(data_dir, download)
    return stage_diagnostics(c["train"], c["test"], c["feats"], c["tuned"], c["fitted"], c["tune_df"], c["folds"],
                             c["summary"], c["raw_counts"], force=force)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run(data_dir: Path | None = None, download: bool = False, force: bool = False, quick: bool = False) -> dict:
    """Run the whole pipeline.

    Args:
        data_dir: Folder with the CIC-IDS2017 CSVs (default ``config.DATA_DIR``).
        download: Download the dataset if it is missing.
        force: Recompute tuning and final fits even if cached.
        quick: Smoke-test mode (50k tuning sample, 2 candidates per model,
            final fit on a 200k training subset). Never use for reported results.

    Returns:
        Dict with the comparison table and the best model name.
    """
    config.ensure_dirs()
    np.random.seed(config.SEED)
    # Caches written by a --quick smoke test must never be reused for real results.
    marker = config.RESULTS_DIR / ".quick_run"
    if quick:
        marker.touch()
    elif marker.exists():
        force = True
        marker.unlink()
    raw = stage_load(data_dir, download)
    clean_df, report = clean.clean_cached(raw)
    save_table(report.summary(), "cleaning_summary", floatfmt=".0f")
    stage_eda(raw, clean_df, report)
    del raw
    train, test = features.split_train_test(clean_df)
    del clean_df
    save_table(pd.DataFrame({"train": train[config.TARGET_COL].value_counts(),
                             "test": test[config.TARGET_COL].value_counts()}).reindex(config.CLASS_ORDER),
               "split_counts", floatfmt=".0f")
    sel = stage_select(train, clean.feature_columns(train), force=force)
    feats = sel["selected"]
    print(f"[select] {sel['counts']}")
    sample = stage_sample(test)

    tune_n = 50_000 if quick else config.TUNING_SAMPLE_SIZE
    tune_df = features.stratified_sample(train, tune_n)
    save_table(tune_df[config.TARGET_COL].value_counts().reindex(config.CLASS_ORDER).to_frame("rows"),
               "tuning_sample_counts", floatfmt=".0f")
    folds = stage_cv(tune_df, feats, force=force or quick)
    evaluate.plot_cv_folds(folds)
    tuned = stage_tune(tune_df, feats, force=force or quick,
                       n_iter={m: 2 for m in config.MODEL_NAMES} if quick else None)
    summary = tuning.tuning_summary(tuned, folds)
    evaluate.plot_tuning(summary, {n: r["cv_results"] for n, r in tuned.items()})

    fit_train = features.stratified_sample(train, 200_000) if quick else train
    fitted = stage_fit(fit_train, feats, tuned, force=force or quick)
    results, comp = stage_evaluate(fitted, test, feats)
    print(comp[["accuracy", "precision_macro", "recall_macro", "f1_macro", "train_time_s",
                "inference_time_s"]].round(4).to_string())
    best, why = evaluate.choose_best(comp)
    stage_save_best(best, why, fitted, comp, train, feats, tuned)
    stage_explain(best, fitted, train, test, feats)
    stage_replay(fitted[best][0], feats, sample)
    return {"comparison": comp, "best": best, "why": why}


def main(argv: list[str] | None = None) -> None:
    """Parse command-line arguments and run the pipeline."""
    ap = argparse.ArgumentParser(description="CyberReplay end-to-end pipeline")
    ap.add_argument("--data-dir", type=Path, default=None, help="folder with the CIC-IDS2017 CSVs")
    ap.add_argument("--download", action="store_true", help="download the dataset if missing")
    ap.add_argument("--force", action="store_true", help="recompute cached tuning and model fits")
    ap.add_argument("--quick", action="store_true", help="fast smoke test (not for reported results)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="run only the diagnostics stage (ROC-AUC, baseline, split figure, fit diagnosis)")
    ap.add_argument("--figures", action="store_true",
                    help="redraw every figure from cached data, fits and result tables")
    args = ap.parse_args(argv)
    if args.diagnostics:
        run_diagnostics(args.data_dir, args.download, args.force)
    elif args.figures:
        run_figures(args.data_dir, args.download)
    else:
        run(args.data_dir, args.download, args.force, args.quick)


if __name__ == "__main__":
    main()
