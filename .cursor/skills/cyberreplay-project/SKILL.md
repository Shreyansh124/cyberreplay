---
name: cyberreplay-project
description: Marking scheme, requirements, agreed design decisions and the 16-step build plan for the CyberReplay ML project (multi-class network intrusion detection on CIC-IDS2017, Track A). Use for any work on this repository - EDA, preprocessing, models, evaluation, dashboard, notebook or report.
---

# CyberReplay: ML-Based Network Intrusion Detection and Attack Analysis

## Objective
Classify network flows from CIC-IDS2017 into specific attack categories
(BENIGN / DoS / DDoS / PortScan / Brute Force / Web Attack / Bot / Rare), and
also expose a binary `is_malicious` column (BENIGN = 0, anything else = 1).
Track A (libraries): compare Logistic Regression, Decision Tree, Random Forest, XGBoost.

## Agreed design decisions (do not change without the user)
- **Track A** only. Four models: LogReg, DecisionTree, RandomForest, XGBoost.
- **Labels**: group the 15 raw labels into families; keep the raw label column for analysis.
  - DoS = DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest
  - DDoS, PortScan, Bot stay as is
  - Brute Force = FTP-Patator, SSH-Patator
  - Web Attack = Brute Force, XSS, Sql Injection (normalise the odd unicode dash in raw labels)
  - Rare = Infiltration, Heartbleed
- **Binary output**: derived from the multi-class prediction (`is_malicious = pred != BENIGN`); also report binary metrics.
- **Data size**: tune on a stratified sample (~500k rows, keep all minority rows); fit final models on the full training split.
- **Cross-validation**: StratifiedKFold on the training split only. GroupKFold by capture day is NOT feasible because each attack lives on a single day; explain this in the report.
- **Imbalance**: class weights / sample weights (no SMOTE by default). Primary metric is macro F1.
- **Leakage**: drop duplicates before splitting; fit scaler/encoders inside sklearn Pipelines on train only; drop identifier columns (Flow ID, IPs, ports used as IDs, Timestamp) from features.
- **Seed**: 42 everywhere.
- **Optional steps 11-14 are IN scope** (Streamlit dashboard, traffic replay, interactive visuals, SHAP explanations).
- **Delivery**: modular `src/` package + notebook that runs in Google Colab by cloning the repo (Colab = the "virtual lab").
- **Report**: Markdown converted to PDF, 3-5 pages.

## Rubric (20 marks)
| Common (12) | Marks | Track A (8) | Marks |
|---|---|---|---|
| Problem clarity + dataset justification | 2 | >= 4 algorithms trained | 2 |
| EDA (insightful plots + observations) | 2 | CV + hyperparameter tuning | 2 |
| Preprocessing + feature engineering | 3 | Metrics, comparison tables and plots for all models | 3 |
| Use of a virtual lab | 1 | Justification of final model | 1 |
| Code quality, docs, reproducibility | 2 | | |
| Report: conclusions, limitations, future work | 2 | | |

Hard rules: non-trivial dataset (CIC-IDS2017 qualifies), no plagiarism, cite all data sources and references.

## Deliverables
1. Well-commented notebook / scripts exported as PDF
2. Report (3-5 pages)
3. Dataset link or a sample of the dataset
Required components: problem + motivation, dataset description + source, EDA, cleaning/preprocessing/FE, model building, evaluation, discussion of results/limitations/future work.

## 16-step build plan
1. **Problem + dataset** - multi-class IDS motivation, CIC-IDS2017 justification, list attack categories.
2. **EDA** - attack distribution, traffic patterns, feature correlations; 5-6 meaningful Matplotlib/Seaborn plots with written observations.
3. **Cleaning** - NaN, +/-inf, duplicates, invalid (negative) values, outliers (pandas; prefer clipping/capping over deleting attack rows).
4. **Feature engineering** - encode labels, select features (variance/correlation filter + importance), scale where needed (LogReg), train/test split, prevent leakage.
5. **Train 4 models** - LogReg, DT, RF, XGBoost (multi-class).
6. **Cross-validation** - StratifiedKFold on train only; compare per-fold scores.
7. **Tuning** - search spaces + RandomizedSearchCV, scoring = f1_macro; retrain best.
8. **Evaluation** - accuracy, macro precision/recall/F1, per-class report, confusion matrices, model comparison table + plots.
9. **Best model** - macro F1, minority-class recall, false positives on BENIGN, training time, inference speed.
10. **Attack patterns** - feature importance, SHAP, per-attack feature distributions, what distinguishes each attack.
11. **Streamlit dashboard** - load best model + pipeline, show predictions and attack frequencies.
12. **Traffic replay** - sort by timestamp, simulated live stream, adjustable speed, live predictions.
13. **Interactive visuals** - per-attack features, detection timeline, frequencies, comparisons.
14. **Attack explanation** - pick a detection, show predicted class + SHAP top contributing features + plain-language reason.
15. **Reproducibility** - modular scripts, requirements.txt, README, fixed seeds, saved model + preprocessing pipeline.
16. **Final report** - problem, dataset, EDA findings, preprocessing, model comparison, results, attack patterns, limitations, future scope.
