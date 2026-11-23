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

**Shreyansh Krishna** (2026CS05055) · **Aamir Lanker** (2026CS05054) · **Harshith G. D** (2026CS05053) · **Anil Pandey** (2026CS05031) · Course: Machine Learning

Multi-class classification of CIC-IDS2017 network flows into 8 attack families (Track A: scikit-learn / XGBoost),
with a derived binary `is_malicious` output, a traffic-replay dashboard and SHAP explanations.

| Build step | Section |
|---|---|
| 1 Problem + dataset | §0-1 |
| 2 EDA (6 plots) | §2, §3.2 |
| 3 Cleaning | §3 |
| 4 Feature engineering | §4 |
| 5-7 Models, cross-validation, tuning | §5-7 |
| 8-9 Evaluation, best model | §8-9 |
| Report extensions: split overview, ROC-AUC, baseline, over/underfitting | §4.1, §9.1-9.2 |
| 10 Attack patterns (importance, SHAP) | §10 |
| 11-14 Dashboard, replay, interactive visuals, explanations | §11 |
| 15-16 Reproducibility, conclusions | §12-13 |

We keep the implementation in `src/cyberreplay` and use this notebook to run each stage and present its results.
Every stage uses random seed **42**. Figure numbers in this notebook refer to the file numbers in `reports/figures`
(for example, Figure 8 is `08_tuning.png`); the report numbers its figures in reading order instead.
""")

    md("""
## 0. Setup (Google Colab or local)

We designed this notebook to run in **Google Colab**, our virtual lab, or on a local machine.

On Colab, the first cell clones our public repository
`https://github.com/Shreyansh124/cyberreplay.git` (branch set by `BRANCH`) into `/content/cyberreplay` and installs
the pinned `requirements.txt`, without requiring credentials or manual setup. For local use, it finds the repository root.

When `pip` upgrades NumPy/pandas, the cell restarts the runtime automatically; we then run all cells once more.
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
**Data location.** We set `DATA_SOURCE = "drive"` when CIC-IDS2017 is already in Google Drive. `DRIVE_DATA_DIR`
must point to the folder containing the day CSVs, either the *TrafficLabelling/GeneratedLabelledFlows* or
*MachineLearningCSV* version; the notebook also searches sub-folders. Setting it to `"download"` retrieves the official
`GeneratedLabelledFlows.zip` (~285 MB) automatically. If no CSVs are present in Drive, we fall back to that download.

With `FORCE_RECOMPUTE = False`, we reuse committed CV / tuning results in `reports/results` and any cached fitted models.
Changing it to `True` recomputes every stage; tuning takes about an hour on 8 CPUs. For a quicker Colab run,
`FINAL_FIT_ROWS` can restrict the final fit to a stratified subset, while the reported results use `None` = full training split.
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

from cyberreplay import (clean, config, data, diagnostics, eda, evaluate, explain, features, models, pipeline,
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

**Problem.** Signature-based intrusion detectors can miss new variants and generate too many alerts for analysts.
We train a flow-level ML classifier to identify *which kind* of attack a connection represents: DoS, DDoS, port scan,
botnet, brute force, web attack or a rare intrusion. This lets us organise alerts by family, while the binary
`is_malicious` flag (prediction ≠ BENIGN) supports a simpler allow/deny decision. Because this multi-class task is
heavily imbalanced, our primary metric is **macro F1**, which gives each family equal weight.

**Dataset.** CIC-IDS2017 from the Canadian Institute for Cybersecurity, University of New Brunswick contains five days
(3-7 July 2017) of realistic network traffic from a 25-host testbed with scripted attacks. CICFlowMeter converts the
traffic into bidirectional flows with 80+ statistical features per flow and 15 labels. Its 2.8 M flows, real protocols
and strong imbalance make it a widely used, non-trivial IDS benchmark.

*Source:* official page <https://www.unb.ca/cic/datasets/ids-2017.html>. Since the official download now requires a
registration form, our automated download uses a byte-identical mirror of the official `GeneratedLabelledFlows.zip`
(TrafficLabelling CSVs, which include timestamps) at <https://huggingface.co/datasets/bencorn/CICIDS2017>, with
SHA-256 verification.
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

We first normalise labels because the Thursday file stores the en dash in "Web Attack – XSS" as the cp1252 byte
`0x96`. We then map labels into the 8 families while retaining the raw label for analysis.
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

Our row-level cleaning uses no fitted statistics, so we can safely perform it before splitting:
1. We replace `+/-inf` with NaN because rates become infinite when `Flow Duration` = 0. The pipeline
   **median-imputes NaN later**, using a median fitted only on the training split.
2. We clip physically impossible negative durations, IATs and header lengths to 0, but retain the `-1` value in
   `Init_Win_bytes_*` because CICFlowMeter uses it as a sentinel for "no TCP window seen".
3. We remove exact duplicate flows with the same features and family, ignoring identifiers, **before** splitting.
   This prevents one vector from appearing in both train and test.
4. We measure outliers here without deleting them; §3.2 and §4.3 describe their treatment.
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

* **Target encoding:** We encode families as integer codes 0-7 (`BENIGN` = 0) and derive binary `is_malicious`
  from the predictions.
* **Split:** We make a stratified 80/20 train/test split of de-duplicated data using seed 42, then leave the test
  split untouched until §8.
* **Identifiers excluded:** We remove Flow ID, Source/Destination IP, Source/Destination Port and Timestamp.
* **Feature selection (train only):** We apply a variance filter, then a greedy correlation filter (|r| > 0.95),
  followed by class-weighted Random Forest importance that retains 99% cumulative importance.
* **Preprocessing inside sklearn Pipelines:** Every model performs column selection and median imputation. Logistic
  Regression also applies 99.9th-percentile capping, signed `log1p` and `StandardScaler`. We fit each operation only
  on training folds.
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
    md("""
### 4.1 The train/test split per family

We show how many flows of each family land in each part of the split. The figure uses a log scale because BENIGN has
almost 40,000 times more unique flows than Rare.
""")
    code("""
raw_counts = clean_df[config.TARGET_COL].value_counts().add(pd.Series(report.duplicates_removed), fill_value=0)
split_table = diagnostics.split_table(train, test, raw_counts)
display(split_table)
fig = diagnostics.plot_split(split_table); plt.show()
""")
    obs("split")
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

We compare four Track A models, each paired with the preprocessing described above:

| Model | Why it is included | Imbalance handling |
|---|---|---|
| Logistic Regression (multinomial, lbfgs) | interpretable linear reference | class-balancing sample weights |
| Decision Tree (CART) | fast non-linear decision rules | class-balancing sample weights |
| Random Forest | averaged trees that resist noise | class-balancing sample weights |
| XGBoost (hist) | boosted trees suited to tabular data | class-balancing sample weights |

Each estimator calculates weights inside `fit` from its own training rows, keeping every CV fold free of leakage.
We tune weight strength between `balanced` (n / (k·count)) and its square root `sqrt`.

**Tuning sample.** For CV and tuning, we take a stratified 500k-row sample from the *training* split. It includes
**all** rows from each family with fewer than 50k training rows and proportionally samples BENIGN / DoS / DDoS.
""")
    code("""
tune_df = features.stratified_sample(train, config.TUNING_SAMPLE_SIZE)
display(pd.DataFrame({"tuning sample": tune_df[config.TARGET_COL].value_counts(),
                      "full train": train[config.TARGET_COL].value_counts()}).reindex(config.CLASS_ORDER))
""")
    md("""
## 6. Cross-validation (StratifiedKFold, training data only)

We first evaluate default hyperparameters on the tuning sample with
5-fold `StratifiedKFold(shuffle=True, random_state=42)`.

*Why not GroupKFold by capture day?* Figure 3 shows that each attack type was recorded on exactly one day. If we held
out a day, its attack would disappear from the training folds; for example, all DDoS occurs on Friday afternoon.
The model could not learn that class, and the score would test "unseen attack type" detection rather than
classification. Stratification keeps every family represented in every fold.
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

We run 3-fold stratified RandomizedSearchCV on the tuning sample with these search spaces:
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

We refit every tuned pipeline on the **full** training split of 1.76 M flows, then evaluate it once on the untouched
test split of 440k flows. We record training time as wall-clock fit time and inference time as `predict` over the
complete test split.
Wall-clock times depend on the machine and its load, so a re-run shows its own measurements; the observations below
quote the reported run.
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
    md("""
## 9.1 ROC curves, balanced accuracy and a majority-class baseline

This stage first checks that the fitted models reproduce the stored test metrics. It then adds three things:
* **balanced accuracy**, the mean of the per-class recalls, which for a multi-class problem equals macro recall;
* **one-vs-rest ROC-AUC (macro)**: for each family, the ROC curve of "this family vs the rest" from the predicted
  probabilities, averaged over the eight families;
* a **majority-class baseline** (`DummyClassifier(strategy="most_frequent")`), which always predicts BENIGN.

The ROC plots use a logarithmic false-positive axis, because every model is near the top-left corner on a linear axis.
""")
    code("""
from IPython.display import Image
diag = pipeline.stage_diagnostics(train, test, FEATURES, tuned, fitted, tune_df, cv_folds, tune_summary,
                                  raw_counts, results=results, force=FORCE_RECOMPUTE)
print("saved best model predicts like the refit:", diag["best_model_same_predictions"])
display(diag["baseline"])
display(diag["roc_auc_per_class"])
display(Image(filename=str(config.FIGURES_DIR / "18_final_model_cm_roc.png")))
display(Image(filename=str(config.FIGURES_DIR / "19_roc_comparison.png")))
""")
    obs("roc")
    md("""
## 9.2 Over- and underfitting

We carve a stratified 20% validation slice out of the **training** split (seed 42; the test split is not used), fit
every tuned model on the other 80% and compare training and validation macro F1. Our rule: training macro F1 below
0.90 means **underfitting** (high bias); otherwise a train-validation gap above 0.05 means **overfitting** (high
variance), a gap between 0.02 and 0.05 a slight overfit, and a smaller gap a good fit.

Learning curves (Logistic Regression and Random Forest) and validation curves (Decision Tree depth, XGBoost boosting
rounds) use a 200k-row stratified training subsample that keeps every minority row, with 3-fold StratifiedKFold.
""")
    code("""
display(diag["cv_overview"])
display(diag["fit_diagnosis"])
display(Image(filename=str(config.FIGURES_DIR / "20_learning_curves.png")))
display(Image(filename=str(config.FIGURES_DIR / "21_validation_curves.png")))
""")
    obs("fit")

    # ------------------------------------------------------------------ 10
    md("""
## 10. Attack patterns: what distinguishes each attack?

We study attack differences from three viewpoints: global model-native importance, per-family SHAP values and effect
directions from the best model on a stratified test subsample, and raw feature distributions for each family.
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

`app/streamlit_app.py` loads `models/best_model.joblib`, which combines preprocessing and the model in one pipeline.
The application provides:
* **Overview** - We show predictions for all loaded flows, predicted and true attack frequencies, binary metrics and
  the latest detections.
* **Live replay** - We stream flows in capture-time order with adjustable speed (1x-3600x capture time) and batch size,
  alongside live alert KPIs, a detection timeline and an alert feed.
* **Attack patterns** - We provide interactive Plotly box/scatter plots by family, a detection timeline, confusion
  matrix, model comparison and per-class F1.
* **Explain a detection** - A selected attack displays its predicted class, confidence, class probabilities, leading
  SHAP features and a plain-language reason.

For local use, we run `streamlit run app/streamlit_app.py`. It accepts the committed
`data/sample/cicids2017_sample.csv.gz`, containing 29,999 flows drawn from the **test** split, or an uploaded CSV.
The following cells run the same replay and explanation logic without the interface.
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

* We can reproduce the full workflow with `python -m cyberreplay.pipeline --download`; `--force` recomputes tuning
  and fitting, while `--quick` runs a smoke test.
* `requirements.txt` pins dependencies, and every split, sample, model and search uses seed 42.
* We save the final feature selection + imputation + model pipeline as `models/best_model.joblib`, together with
  `best_model_metadata.json` containing features, parameters, test metrics and library versions.
* We run unit tests and a headless dashboard test with `pytest tests`.
* We store all figures in `reports/figures` and all tables in `reports/results`.
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
