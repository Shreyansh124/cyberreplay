"""Generate ``notebooks/CyberReplay.ipynb`` from the cell definitions below.

The notebook is the main deliverable; keeping its source here makes it easy to
review and regenerate. After generation it is executed in place with::

    python scripts/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/CyberReplay.ipynb \
        --ExecutePreprocessor.timeout=-1

Observation cells are written from the results of the full pipeline run
(``reports/results``); re-check them if the results change.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "scripts" / "notebook_observations.md"


def load_observations() -> dict[str, str]:
    """Read observation blocks from ``notebook_observations.md`` (sections start with ``@@ key``)."""
    blocks: dict[str, str] = {}
    key = None
    for line in OBS.read_text().splitlines():
        if line.startswith("@@ "):
            key = line[3:].strip()
            blocks[key] = ""
        elif key:
            blocks[key] += line + "\n"
    return {k: v.strip() for k, v in blocks.items()}


def build() -> nbf.NotebookNode:
    """Assemble the notebook cells.

    Returns:
        Notebook object (not executed).
    """
    o = load_observations()
    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(text.strip()))

    def obs(key: str) -> None:
        md("**Observations**\n\n" + o[key])

    md("""
# CyberReplay: Machine Learning-Based Network Intrusion Detection and Attack Analysis

**[Student Name]** · **[Roll Number]** · **[Course]** · **[Institute]**

Multi-class classification of CIC-IDS2017 network flows into 8 attack families (Track A: scikit-learn / XGBoost),
plus a derived binary `is_malicious` output, a traffic-replay dashboard and SHAP explanations.

| Build step | Section |
|---|---|
| 1 Problem + dataset | §0-1 |
| 2 EDA (6 plots) | §2, §3.2 |
| 3 Cleaning | §3 |
| 4 Feature engineering | §4 |
| 5-7 Models, cross-validation, tuning | §5-7 |
| 8-9 Evaluation, best model | §8-9 |
| 10 Attack patterns (importance, SHAP) | §10 |
| 11-14 Dashboard, replay, interactive visuals, explanations | §11 |
| 15-16 Reproducibility, conclusions | §12-13 |

All code lives in the `src/cyberreplay` package; this notebook calls it step by step and shows every result.
Random seed **42** everywhere.
""")

    md("""
## 0. Setup (Google Colab or local)

This notebook runs both on **Google Colab** (the project's virtual lab) and locally.

In Colab the first cell clones the public repository
`https://github.com/Shreyansh124/cyberreplay.git` (branch set by `BRANCH`) into `/content/cyberreplay` and installs
the pinned `requirements.txt`; no setup or credentials are needed. Locally it simply locates the repository root.

If `pip` upgrades NumPy/pandas, the cell restarts the runtime automatically - simply run all cells again.
""")
    code("""
# Environment setup: clone the repo + install pinned requirements in Colab; locate the repo locally.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/Shreyansh124/cyberreplay.git"
BRANCH = "main"  # change if the project lives on another branch

if IN_COLAB:
    ROOT = Path("/content/cyberreplay")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, str(ROOT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")], check=True)
    import importlib.metadata as md
    stale = [p for p in ("numpy", "pandas", "scipy") if p in sys.modules
             and sys.modules[p].__version__ != md.version(p)]
    if stale:
        print(f"Upgraded {stale}; restarting the runtime - run all cells again.")
        os.kill(os.getpid(), 9)
else:
    ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()

os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
print("Project root:", ROOT, "| Colab:", IN_COLAB)
""")
    md("""
**Data location.** Set `DATA_SOURCE = "drive"` to use a copy of CIC-IDS2017 already in Google Drive
(point `DRIVE_DATA_DIR` at the folder holding the day CSVs - either the *TrafficLabelling/GeneratedLabelledFlows*
or the *MachineLearningCSV* variant; sub-folders are searched), or `"download"` to fetch the official
`GeneratedLabelledFlows.zip` (~285 MB) automatically. If the Drive folder has no CSVs the notebook falls back to the download.

`FORCE_RECOMPUTE = False` reuses the committed CV / tuning results (`reports/results`) and any cached fitted models;
set it to `True` to recompute everything (tuning takes about an hour on 8 CPUs).
`FINAL_FIT_ROWS` can limit the final fit to a stratified subset for a faster Colab run (reported results use `None` = full training split).
""")
    code("""
DATA_SOURCE = "drive"                               # "drive" or "download"
DRIVE_DATA_DIR = "/content/drive/MyDrive/CIC-IDS2017"
FORCE_RECOMPUTE = False
FINAL_FIT_ROWS = None                               # e.g. 500_000 for a quick Colab run

DATA_DIR = ROOT / "data" / "raw"
if IN_COLAB and DATA_SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    # The day files are named like "Monday-WorkingHours.pcap_ISCX.csv" in both official variants.
    if any(".pcap_ISCX" in p.name for p in Path(DRIVE_DATA_DIR).rglob("*.csv")):
        DATA_DIR = Path(DRIVE_DATA_DIR)
    else:
        print("No CIC-IDS2017 day CSVs found in", DRIVE_DATA_DIR, "- downloading to", DATA_DIR, "instead.")
os.environ["CYBERREPLAY_DATA_DIR"] = str(DATA_DIR)  # must be set before importing cyberreplay

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from cyberreplay import (clean, config, data, eda, evaluate, explain, features, models, pipeline,
                         replay, tuning, utils)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 40, "display.width", 200, "display.float_format", "{:.4f}".format)
plt.rcParams["figure.dpi"] = 90
np.random.seed(config.SEED)
config.ensure_dirs()
print("Data dir:", config.DATA_DIR, "| seed:", config.SEED, "| n_jobs:", config.N_JOBS)
""")

    # ------------------------------------------------------------------ 1
    md("""
## 1. Problem and dataset

**Problem.** Signature-based intrusion detection misses new variants and floods analysts with alerts. A flow-level
ML classifier can recognise *which kind* of attack a connection belongs to - DoS, DDoS, port scan, botnet, brute
force, web attack or rare intrusions - so that alerts can be triaged by family, while a binary
`is_malicious` flag (prediction ≠ BENIGN) supports simple allow/deny decisions. The task is a heavily imbalanced
multi-class problem, so the primary metric is **macro F1** (every family counts equally).

**Dataset.** CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick): five days
(3-7 July 2017) of realistic traffic from a 25-host testbed with scripted attacks, converted into
bidirectional flows by CICFlowMeter (80+ statistical features per flow, labelled with 15 classes).
It is a widely used, non-trivial IDS benchmark: 2.8 M flows, real protocols and a strong class imbalance.

*Source:* official page <https://www.unb.ca/cic/datasets/ids-2017.html>; the automated download uses a byte-identical
mirror of the official `GeneratedLabelledFlows.zip` (TrafficLabelling CSVs, which include timestamps) at
<https://huggingface.co/datasets/bencorn/CICIDS2017> (SHA-256 checked), because the official download now sits behind a
registration form.
*Citation:* I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and
Intrusion Traffic Characterization", ICISSP 2018, pp. 108-116.

**Attack families (target).**

| Family | Raw labels |
|---|---|
| BENIGN | BENIGN |
| DoS | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest |
| DDoS | DDoS (LOIC) |
| PortScan | PortScan |
| Bot | Bot (Ares) |
| Brute Force | FTP-Patator, SSH-Patator |
| Web Attack | Web Attack - Brute Force, XSS, Sql Injection |
| Rare | Infiltration, Heartbleed |
""")
    code("""
# Load the eight day files (downloads them first if missing). Cached as parquet after the first run.
raw = pipeline.stage_load(config.DATA_DIR, download=True)
display(data.describe_files(config.DATA_DIR))
print(raw.shape)
raw.iloc[:3, :12]  # first columns only; the full frame has 87
""")
    obs("load")

    # ------------------------------------------------------------------ 2
    md("""
## 2. Exploratory data analysis

Labels are normalised first (the Thursday file encodes the en dash in "Web Attack – XSS" as the cp1252 byte `0x96`)
and mapped to the 8 families; the raw label is kept for analysis.
""")
    code("""
labelled = clean.add_labels(raw)
fam_counts = pd.DataFrame({"flows": labelled[config.TARGET_COL].value_counts()}).reindex(config.CLASS_ORDER)
fam_counts["share_%"] = 100 * fam_counts["flows"] / fam_counts["flows"].sum()
display(fam_counts)
fig = eda.plot_class_distribution(labelled); plt.show()
""")
    obs("eda1")
    code("""
fig = eda.plot_traffic_timeline(labelled); plt.show()
""")
    obs("eda2")
    code("""
fig = eda.plot_attack_day_matrix(labelled); plt.show()
""")
    obs("eda3")

    # ------------------------------------------------------------------ 3
    md("""
## 3. Data cleaning

Row-level cleaning is deterministic (no statistics are fitted), so it is safe before the split:
1. `+/-inf` → NaN (rates are infinite when `Flow Duration` = 0); NaN are **median-imputed later inside the pipeline**
   (median fitted on the training split only).
2. Physically impossible negative durations / IATs / header lengths → clipped to 0; the `-1` of `Init_Win_bytes_*`
   is a CICFlowMeter sentinel ("no TCP window seen") and is kept.
3. Exact duplicate flows (same features and family, identifiers ignored) are dropped **before** splitting so no vector
   can appear in both train and test.
4. Outliers are *quantified* here but not deleted (see §3.2 and §4.3 for how they are handled).
""")
    code("""
clean_df, report = clean.clean_cached(raw)
display(report.summary())
print("inf per column:", report.inf_counts)
print("negatives clipped per column:", report.negative_counts)
print("duplicates removed per family:", report.duplicates_removed)
display(clean_df[config.TARGET_COL].value_counts().reindex(config.CLASS_ORDER).to_frame("rows after cleaning"))
fig = eda.plot_data_quality(labelled, report); plt.show()
""")
    obs("clean")
    md("### 3.2 EDA on the cleaned flows: feature behaviour, outliers and redundancy")
    code("""
fig = eda.plot_feature_distributions(clean_df); plt.show()
""")
    obs("eda5")
    code("""
candidates = clean.feature_columns(clean_df)
print(len(candidates), "candidate features (identifiers excluded)")
display(clean.outlier_summary(clean_df, candidates))
fig, corr_pairs = eda.plot_correlation_heatmap(clean_df, candidates); plt.show()
print(len(corr_pairs), "feature pairs with |r| >", config.CORRELATION_THRESHOLD)
display(corr_pairs.head(10))
""")
    obs("eda6")

    # ------------------------------------------------------------------ 4
    md("""
## 4. Feature engineering and leakage control

* **Target encoding:** families → integer codes 0-7 (`BENIGN` = 0); binary `is_malicious` derived from predictions.
* **Split:** stratified 80/20 train/test split of the de-duplicated data (seed 42). The test split is not touched
  until §8.
* **Identifiers excluded:** Flow ID, Source/Destination IP, Source/Destination Port, Timestamp.
* **Feature selection (train only):** variance filter → greedy correlation filter (|r| > 0.95) → class-weighted
  Random Forest importance (keep 99% cumulative importance).
* **Preprocessing inside sklearn Pipelines:** column selection + median imputation for all models; for Logistic
  Regression additionally 99.9th-percentile capping, signed `log1p` and `StandardScaler` - all fitted on training
  folds only.
""")
    code("""
train, test = features.split_train_test(clean_df)
split_counts = pd.DataFrame({"train": train[config.TARGET_COL].value_counts(),
                             "test": test[config.TARGET_COL].value_counts()}).reindex(config.CLASS_ORDER)
display(split_counts)
del raw, labelled  # free memory
sel = pipeline.stage_select(train, candidates, force=FORCE_RECOMPUTE)
FEATURES = sel["selected"]
print(sel["counts"])
print("Constant (dropped):", sel["constant"])
print("Correlated (dropped):", sel["correlated_dropped"])
print("Low importance (dropped):", sel["low_importance_dropped"])
print(f"\\n{len(FEATURES)} selected features:", FEATURES)
""")
    code("""
imp = pd.read_csv(config.RESULTS_DIR / "selection_importance.csv")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(imp["feature"][::-1], imp["importance"][::-1], color=np.where(imp["selected"][::-1], "#4c72b0", "#cccccc"))
ax.set_title("Selection step 3 - Random Forest importance (grey = dropped)")
ax.tick_params(axis="y", labelsize=7); plt.tight_layout(); plt.show()
""")
    obs("select")
    code("""
# The preprocessing pipelines (fitted later on training data only).
print(models.build_pipeline("LogisticRegression", FEATURES))
print(models.build_pipeline("XGBoost", FEATURES))
""")

    # ------------------------------------------------------------------ 5-7
    md("""
## 5. Models

Four Track A models, each wrapped with the preprocessing above:

| Model | Why it is included | Imbalance handling |
|---|---|---|
| Logistic Regression (multinomial, lbfgs) | linear, interpretable baseline | class-balancing sample weights |
| Decision Tree (CART) | non-linear, rule-like, very fast | class-balancing sample weights |
| Random Forest | bagged trees, robust to noise | class-balancing sample weights |
| XGBoost (hist) | gradient boosting, state of the art on tabular data | class-balancing sample weights |

The weights are computed inside each estimator's `fit` from the rows it trains on, so every CV fold is leak-free.
Their strength is a hyperparameter: `balanced` (n / (k·count)) or its square root `sqrt`.

**Tuning sample.** CV and tuning use a 500k-row stratified sample of the *training* split that keeps **all** rows of
every family with fewer than 50k training rows and samples BENIGN / DoS / DDoS proportionally.
""")
    code("""
tune_df = features.stratified_sample(train, config.TUNING_SAMPLE_SIZE)
display(pd.DataFrame({"tuning sample": tune_df[config.TARGET_COL].value_counts(),
                      "full train": train[config.TARGET_COL].value_counts()}).reindex(config.CLASS_ORDER))
""")
    md("""
## 6. Cross-validation (StratifiedKFold, training data only)

5-fold `StratifiedKFold(shuffle=True, random_state=42)` with default hyperparameters on the tuning sample.

*Why not GroupKFold by capture day?* Figure 3 shows that every attack type was recorded on exactly one day. Holding
out a day would remove that attack completely from the training folds (e.g. all DDoS is on Friday afternoon),
so the model could never learn it and the score would measure "unseen attack type" detection instead of
classification. Stratified folds keep every family in every fold.
""")
    code("""
cv_folds = pipeline.stage_cv(tune_df, FEATURES, force=FORCE_RECOMPUTE)
display(cv_folds.pivot(index="fold", columns="model", values="f1_macro"))
display(cv_folds.groupby("model", sort=False)[["f1_macro", "accuracy", "recall_macro", "precision_macro",
                                               "fit_time_s"]].agg(["mean", "std"]))
fig = evaluate.plot_cv_folds(cv_folds); plt.show()
""")
    obs("cv")
    md("""
## 7. Hyperparameter tuning (RandomizedSearchCV, scoring = f1_macro)

3-fold stratified RandomizedSearchCV on the tuning sample. Search spaces:
""")
    code("""
for name in config.MODEL_NAMES:
    space = {k.replace("model__", ""): (v if isinstance(v, list) else f"{v.dist.name}{v.args}")
             for k, v in models.search_space(name).items()}
    print(f"{name} (n_iter={models.N_ITER[name]}):", space)
""")
    code("""
tuned = pipeline.stage_tune(tune_df, FEATURES, force=FORCE_RECOMPUTE)
tune_summary = tuning.tuning_summary(tuned, cv_folds)
display(tune_summary)
fig = evaluate.plot_tuning(tune_summary, {n: r["cv_results"] for n, r in tuned.items()}); plt.show()
display(tuned["XGBoost"]["cv_results"].head(5))
""")
    obs("tune")

    # ------------------------------------------------------------------ 8-9
    md("""
## 8. Final training on the full training split and test evaluation

Each tuned pipeline is refitted on the **full** training split (1.76 M flows) and evaluated once on the untouched
test split (440k flows). Training time is the wall-clock fit time; inference time is `predict` on the whole test split.
""")
    code("""
fit_train = train if FINAL_FIT_ROWS is None else features.stratified_sample(train, FINAL_FIT_ROWS)
fitted = pipeline.stage_fit(fit_train, FEATURES, tuned, force=FORCE_RECOMPUTE)
results, comparison = pipeline.stage_evaluate(fitted, test, FEATURES)
display(comparison[["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_macro_excl_rare",
                    "minority_recall", "benign_fpr",
                    "train_time_s", "inference_time_s", "inference_us_per_flow"]])
""")
    code("""
y_test = features.encode_target(test[config.TARGET_COL])
for name, r in results.items():
    print(f"===== {name} =====")
    print(evaluate.classification_text(y_test, r["y_pred"]))
""")
    obs("eval_table")
    code("""
fig = evaluate.plot_confusion_matrices(results); plt.show()
""")
    obs("confusion")
    code("""
fig = evaluate.plot_comparison(comparison); plt.show()
fig = evaluate.plot_per_class_f1(results); plt.show()
display(comparison[["binary_accuracy", "binary_precision", "binary_recall", "binary_f1", "benign_fpr"]])
""")
    obs("comparison")
    md("## 9. Choosing the final model")
    code("""
display(pd.read_csv(config.RESULTS_DIR / "test_bootstrap_ci.csv", index_col=0))
best, why = evaluate.choose_best(comparison)
print(why)
pipeline.stage_save_best(best, why, fitted, comparison, train, FEATURES, tuned)
print(f"Saved {config.BEST_MODEL_PATH} ({config.BEST_MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
""")
    obs("best")

    # ------------------------------------------------------------------ 10
    md("""
## 10. Attack patterns: what distinguishes each attack?

Three complementary views: model-native importance (global), SHAP values of the best model on a stratified test
subsample (per family, direction of effect), and the raw feature distributions per family.
""")
    code("""
exp = pipeline.stage_explain(best, fitted, train, test, FEATURES)
from IPython.display import Image
display(Image(filename=str(config.FIGURES_DIR / "12_feature_importance.png")))
""")
    obs("importance")
    code("""
display(Image(filename=str(config.FIGURES_DIR / "13_shap_by_family.png")))
display(Image(filename=str(config.FIGURES_DIR / "14_shap_beeswarm.png")))
""")
    obs("shap")
    code("""
display(Image(filename=str(config.FIGURES_DIR / "15_attack_feature_distributions.png")))
display(exp["signatures"])
""")
    obs("patterns")

    # ------------------------------------------------------------------ 11-14
    md("""
## 11. Dashboard, traffic replay and per-detection explanations (steps 11-14)

`app/streamlit_app.py` loads `models/best_model.joblib` (preprocessing + model in one pipeline) and offers:
* **Overview** - predictions for every loaded flow, attack frequencies (predicted vs true), binary metrics, latest detections;
* **Live replay** - flows streamed in capture-time order with an adjustable speed (1x-3600x capture time) and batch size,
  live alert KPIs, a live detection timeline and an alert feed;
* **Attack patterns** - interactive Plotly box/scatter plots per family, detection timeline, confusion matrix,
  model comparison and per-class F1;
* **Explain a detection** - choose a detected attack → predicted class, confidence, class probabilities, SHAP top
  contributing features and a plain-language reason.

Run it with `streamlit run app/streamlit_app.py` (locally) - it uses the committed sample
`data/sample/cicids2017_sample.csv.gz` (29,999 flows drawn from the **test** split) or an uploaded CSV.
The cells below run the same replay and explanation code headless.
""")
    code("""
sample = pd.read_csv(config.SAMPLE_PATH)
sample = replay.order_stream(data.prepare_external(sample))
best_pipe = fitted[best][0]
replayed = pipeline.stage_replay(best_pipe, FEATURES, sample)
display(Image(filename=str(config.FIGURES_DIR / "16_replay_timeline.png")))
display(pd.read_csv(config.RESULTS_DIR / "replay_summary.csv", index_col=0))
""")
    obs("replay")
    code("""
# Step 12: paced replay generator - first 3 ticks at 300x capture speed.
for step in replay.replay(sample, best_pipe, FEATURES, batch_size=500, speed=300, max_sleep=0.5):
    print(step.index, step.capture_time, f"{step.progress:.1%}", step.batch["predicted"].value_counts().to_dict())
    if step.index == 2:
        break
""")
    code("""
# Step 14: explain one detection per attack family with SHAP + a plain-language reason.
explainer = explain.make_explainer(best_pipe)
benign_median = explain.family_medians(train, FEATURES).loc[config.BENIGN]
preds = replay.predict_batch(best_pipe, sample, FEATURES)
for fam in ["DoS", "DDoS", "PortScan", "Bot", "Brute Force", "Web Attack", "Rare"]:
    hits = sample.index[(preds["predicted"] == fam) & (sample[config.TARGET_COL] == fam)]
    if len(hits) == 0:
        continue
    res = explain.explain_flow(best_pipe, explainer, sample.loc[[hits[0]]], FEATURES, benign_median, top_k=5)
    display(Markdown(f"**{fam}** (true label: {sample.loc[hits[0], config.RAW_LABEL_COL]}) - {res['reason']}"))
    display(res["contributions"])
""")
    obs("explain")

    # ------------------------------------------------------------------ 15-16
    md("""
## 12. Reproducibility

* Everything above is also available as one command: `python -m cyberreplay.pipeline --download`
  (`--force` recomputes tuning and fits, `--quick` is a smoke test).
* Pinned dependencies in `requirements.txt`; seed 42 for every split, sample, model and search.
* The final pipeline (feature selection + imputation + model) is saved as `models/best_model.joblib`
  with metadata (`best_model_metadata.json`: features, parameters, test metrics, library versions).
* Unit tests and a headless dashboard test: `pytest tests`.
* All figures are in `reports/figures`, all tables in `reports/results`.
""")
    code("""
import platform, sklearn, xgboost, shap
print("python", platform.python_version(), "| sklearn", sklearn.__version__, "| xgboost", xgboost.__version__,
      "| shap", shap.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
print(utils.load_json(config.MODEL_METADATA_PATH)["justification"])
""")
    md("## 13. Conclusions, limitations and future work\n\n" + o["conclusions"])

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}, "colab": {"provenance": []}}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks" / "CyberReplay.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), out)
    print("wrote", out)
