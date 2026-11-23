"""Cross-validation and hyperparameter tuning (pipeline stages 6 and 7).

Both stages run on the ~500k-row tuning sample drawn from the *training*
split only (all minority-class rows kept), so the test split is never seen.

* ``cross_validate_models`` - StratifiedKFold (5 folds) with default
  hyperparameters, recording per-fold macro F1 / accuracy / macro recall.
* ``tune_model`` - RandomizedSearchCV (3-fold StratifiedKFold,
  ``scoring="f1_macro"``) over the spaces in ``models.search_space``.

GroupKFold by capture day is not used: every attack type occurs on a single
day (Figure 3), so holding a day out would remove that attack from training
entirely and the fold could not learn it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from . import config
from .features import encode_target
from .models import N_ITER, build_pipeline, search_space
from .utils import save_json, save_table

CV_SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "recall_macro": "recall_macro",
    "precision_macro": "precision_macro",
}


def _outer_jobs(name: str) -> int:
    """Parallel folds/candidates for a model: tree ensembles parallelise internally."""
    return 1 if name in ("RandomForest", "XGBoost") else config.N_JOBS


def cross_validate_models(tune_df: pd.DataFrame, features: list[str], folds: int = config.CV_FOLDS,
                          names: list[str] | None = None) -> pd.DataFrame:
    """Run StratifiedKFold CV with default hyperparameters for each model.

    Args:
        tune_df: Tuning sample (training rows only).
        features: Selected feature columns.
        folds: Number of folds.
        names: Models to evaluate (default: all four).

    Returns:
        Long table with one row per (model, fold): fold scores and fit time.

    Side effects:
        Writes ``reports/results/cv_baseline_folds.{csv,md}`` and the per-model
        mean/std summary ``cv_baseline_summary.{csv,md}``.
    """
    names = names or config.MODEL_NAMES
    X, y = tune_df[features], encode_target(tune_df[config.TARGET_COL])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    rows = []
    for name in names:
        res = cross_validate(build_pipeline(name, features), X, y, cv=cv, scoring=CV_SCORING,
                             n_jobs=min(_outer_jobs(name), folds), error_score="raise")
        for k in range(folds):
            rows.append({"model": name, "fold": k + 1, "fit_time_s": res["fit_time"][k],
                         **{m: res[f"test_{m}"][k] for m in CV_SCORING}})
        print(f"[cv] {name}: f1_macro {np.mean(res['test_f1_macro']):.4f} +/- {np.std(res['test_f1_macro']):.4f}",
              flush=True)
    folds_df = pd.DataFrame(rows)
    save_table(folds_df, "cv_baseline_folds", index=False)
    summary = folds_df.groupby("model", sort=False)[list(CV_SCORING) + ["fit_time_s"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    save_table(summary, "cv_baseline_summary")
    return folds_df


def tune_model(name: str, tune_df: pd.DataFrame, features: list[str], n_iter: int | None = None,
               folds: int = config.TUNING_CV_FOLDS) -> dict:
    """Tune one model with RandomizedSearchCV (scoring = macro F1).

    Args:
        name: Model name.
        tune_df: Tuning sample (training rows only).
        features: Selected feature columns.
        n_iter: Number of sampled candidates (default ``models.N_ITER[name]``).
        folds: StratifiedKFold folds inside the search.

    Returns:
        Dict with ``best_params`` (without the ``model__`` prefix), ``best_score``,
        ``best_fold_scores``, ``search_seconds`` and ``cv_results`` (DataFrame).

    Side effects:
        Writes ``reports/results/tuning_<name>.csv`` (all candidates) and
        ``reports/results/best_params_<name>.json``.
    """
    X, y = tune_df[features], encode_target(tune_df[config.TARGET_COL])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    search = RandomizedSearchCV(
        build_pipeline(name, features), search_space(name), n_iter=n_iter or N_ITER[name],
        scoring="f1_macro", cv=cv, random_state=config.SEED, refit=False,
        n_jobs=_outer_jobs(name), error_score="raise", verbose=0,
    )
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    keep = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time"]
    keep += [f"split{k}_test_score" for k in range(folds)] + [c for c in res.columns if c.startswith("param_")]
    res = res[keep].sort_values("rank_test_score").reset_index(drop=True)
    res.columns = [c.replace("param_model__", "") for c in res.columns]
    res.to_csv(config.RESULTS_DIR / f"tuning_{name}.csv", index=False)
    best = res.iloc[0]
    out = {
        "model": name,
        "best_params": {k.replace("model__", ""): _py(v) for k, v in search.best_params_.items()},
        "best_score": float(search.best_score_),
        "best_fold_scores": [float(best[f"split{k}_test_score"]) for k in range(folds)],
        "search_seconds": float(res["mean_fit_time"].sum() * folds),
        "n_iter": int(n_iter or N_ITER[name]),
    }
    save_json(out, config.RESULTS_DIR / f"best_params_{name}.json")
    out["cv_results"] = res
    print(f"[tune] {name}: best f1_macro {out['best_score']:.4f} with {out['best_params']}", flush=True)
    return out


def _py(v):
    """Convert NumPy scalars from cv_results to plain Python values for JSON / estimators."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def tuning_summary(results: dict[str, dict], baseline: pd.DataFrame) -> pd.DataFrame:
    """Compare default-parameter CV with the tuned CV score for each model.

    Args:
        results: ``{model: tune_model output}``.
        baseline: Output of ``cross_validate_models``.

    Returns:
        Table indexed by model: baseline mean F1, tuned best F1, gain, best params.

    Side effects:
        Writes ``reports/results/tuning_summary.{csv,md}``.
    """
    base = baseline.groupby("model", sort=False)["f1_macro"].mean()
    rows = []
    for name, r in results.items():
        rows.append({"model": name, "cv5_default_f1_macro": base.get(name, np.nan),
                     "cv3_tuned_f1_macro": r["best_score"],
                     "tuned_fold_scores": ", ".join(f"{s:.4f}" for s in r["best_fold_scores"]),
                     "best_params": ", ".join(f"{k}={_fmt(v)}" for k, v in r["best_params"].items())})
    df = pd.DataFrame(rows).set_index("model")
    df["gain"] = df["cv3_tuned_f1_macro"] - df["cv5_default_f1_macro"]
    save_table(df, "tuning_summary")
    return df


def _fmt(v) -> str:
    """Short string for a hyperparameter value."""
    return f"{v:.4g}" if isinstance(v, float) else str(v)
