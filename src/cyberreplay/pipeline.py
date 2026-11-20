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

Expensive stages (tuning, final fits) are cached under ``reports/results`` and
``models/cache`` and reused unless ``--force`` is given. The notebook calls
the same ``stage_*`` functions.
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

from . import clean, config, data, eda, evaluate, explain, features, models, replay, tuning  # noqa: E402
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
        return load_json(path)
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
    fig, ax = plt.subplots(figsize=(14, 4.5))
    days = pivot.index.normalize().unique()
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for c in pivot.columns:
        ax.bar(x, pivot[c].values, bottom=bottom, color=eda.FAMILY_COLORS[c], label=c, width=1.0)
        bottom += pivot[c].values
    ax.set_ylim(0, bottom.max() * 1.08)
    ticks = [np.argmax(pivot.index.normalize() == d) for d in days]
    ax.set_xticks(ticks, [d.strftime("%a %d %b") for d in days])
    ax.set_ylabel("flows / 5 min (predicted)")
    ax.legend(ncol=8, fontsize=8, loc="upper left")
    ax.set_title("Figure 16 - Replay of the dataset sample: predicted families over capture time "
                 "(5-minute bins, active periods only)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "16_replay_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = pd.crosstab(replayed[config.TARGET_COL], replayed["predicted"]).reindex(
        index=config.CLASS_ORDER, columns=config.CLASS_ORDER, fill_value=0)
    save_table(summary, "replay_summary", floatfmt=".0f")
    return replayed


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
    args = ap.parse_args(argv)
    run(args.data_dir, args.download, args.force, args.quick)


if __name__ == "__main__":
    main()
