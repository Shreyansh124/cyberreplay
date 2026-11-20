# Decisions log

Decisions taken autonomously during the build (the user was not available). Binding
decisions from the project skill are listed first; everything after is an implementation
choice made by the build agent, with the reason.

## Binding (from `.cursor/skills/cyberreplay-project`)
- Track A, four models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- 8 label families (BENIGN, DoS, DDoS, PortScan, Bot, Brute Force, Web Attack, Rare); raw label kept.
- Binary `is_malicious` derived from the multi-class prediction (pred != BENIGN).
- Tune on a ~500k stratified sample (all minority rows kept); final fit on the full training split.
- StratifiedKFold on train only; GroupKFold by day infeasible (each attack on a single day).
- Class / sample weights for imbalance, no SMOTE. Primary metric: macro F1. Seed 42.
- Duplicates removed before splitting; scalers fit inside Pipelines on train only; identifiers never used.

## Implementation decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Dataset downloaded from the Hugging Face mirror `bencorn/CICIDS2017` (`csvs/GeneratedLabelledFlows.zip`, SHA-256 `7bdbef28...a27a`). | The official CIC download (`cicresearch.ca/CICDataset/CIC-IDS-2017/`) now redirects to a registration form (name, email, organisation), which an unattended agent should not fill in on the user's behalf. The mirror holds the original `GeneratedLabelledFlows.zip` (TrafficLabelling CSVs with Timestamp, file dates 2017-08 / 2018-06), 2,830,743 labelled flows - the published row count. |
| D2 | CSVs read with `latin-1` encoding; the `0x96` byte in Web Attack labels normalised to `-`. | The Thursday-morning file is cp1252-encoded (en dash), which breaks UTF-8 decoding. |
| D3 | 288,602 fully empty trailing rows of the Thursday-morning file are dropped at load time. | They contain only commas (CSV export artefact), not flows. |
| D4 | Timestamps parsed as day-first; hours 1-7 are shifted +12 h. | TrafficLabelling timestamps are `d/m/Y H:M` in a 12-hour clock without AM/PM (e.g. Friday DDoS starts at "3:30" = 15:30). Capture ran ~08:00-17:00, so any hour < 8 must be afternoon. Monday uses `dd/mm/YYYY HH:MM:SS` with the same 12-hour convention. |
| D5 | The duplicated `Fwd Header Length` column (appears twice in the CSV header) is dropped. | Exact duplicate of the first column. |
| D6 | `Source Port` **and** `Destination Port` are excluded from features alongside Flow ID, IPs and Timestamp. | Ports identify services/hosts of this specific testbed (e.g. the victim web server); excluding them forces the models to learn flow behaviour and is the conservative reading of "ports used as IDs". |
| D7 | Duplicates are judged on the 77 candidate feature columns + family (identifiers ignored); 630,963 rows removed, 2,199,780 remain. | Two flows with identical statistics are the same training sample regardless of IP/port; leaving them would put identical vectors in train and test. Side effect: PortScan shrinks from 158,930 to 1,958 rows (98.8% of its flows are identical one/two-packet probes). |
| D8 | 622 rows whose feature vector appears with two different families are kept. | They are genuine label ambiguity (e.g. identical tiny flows labelled BENIGN and DoS); removing them would hide an irreducible error source. Documented as a limitation. |
| D9 | Invalid negatives (3,545 values; Flow Duration, IAT, header lengths, rates) are clipped to 0; the `-1` in `Init_Win_bytes_*` is kept. | Clipping is deterministic (no leakage) and keeps the handful of DoS/DDoS rows affected; `-1` is CICFlowMeter's "no window seen" sentinel. |
| D10 | `+/-inf` -> NaN, then median imputation **inside the pipeline** (fit on train). Outlier capping at the training 99.9th percentile + signed log1p + StandardScaler is applied **only in the Logistic Regression pipeline**. | Tree models are invariant to monotone transforms and robust to outliers; 48.9% of Rare-family rows (Infiltration/Heartbleed) carry a value above the global 99.9th percentile, so capping for trees would erase their signal. No attack rows are ever deleted. |
| D11 | Feature selection on the training split only: variance filter (drops 8 constant columns) -> greedy Pearson filter, abs(r) > 0.95, in CSV order (drops 25) -> class-weighted RF importance, keep 99% cumulative importance (drops 7). 77 -> 37 features. Correlation/importance use a 300k minority-preserving training sample. | Removes exact/near duplicates (e.g. `Subflow Fwd Bytes` = `Total Length of Fwd Packets`) and noise features while keeping the more basic statistic of each pair. |
| D12 | Class weighting strength is a tuned hyperparameter for all four models: `balanced` (n / (k * count_c)) or its square root `sqrt`. Weights are applied as per-row sample weights inside each estimator's `fit`. | With fully balanced weights the linear model reaches recall ~0.97 on every attack but precision 0.03-0.2 on the smallest classes and flags ~12% of BENIGN flows (diagnosed on the tuning sample). Both options are still class weighting (no resampling); computing weights inside `fit` keeps every CV fold leak-free. |
| D13 | Baseline CV: 5-fold StratifiedKFold with default hyperparameters. Tuning: RandomizedSearchCV, 3-fold StratifiedKFold, `scoring="f1_macro"`, n_iter = 10 (LogReg), 30 (DT), 20 (RF), 20 (XGB), all on the 500k tuning sample. | 3 folds keep the Rare class (38 training rows in the sample) at >= 12 rows per validation fold while allowing a reasonable number of candidates. |
| D14 | Final-model rule: highest test macro F1; models within 0.005 are a tie broken by minority-class recall (PortScan, Bot, Brute Force, Web Attack, Rare), then BENIGN false-positive rate, then inference time. | Matches the agreed criteria (macro F1, minority recall, BENIGN false positives, speed). |
| D15 | The committed dataset sample (`data/sample/cicids2017_sample.csv.gz`, 29,999 rows) is drawn from the **test split**: all rows of families with < 2,000 test rows, the rest 60% BENIGN / 40% split across DoS, DDoS, Brute Force. | The dashboard demo then shows predictions on flows the model never saw. |
| D16 | Inference time = wall-clock `predict` on the full test split (439,956 flows) in one process; training time = wall-clock fit on the full training split (1,759,824 flows). | Comparable, reproducible timing for all four models on the same 8-CPU VM. |
| D17 | SHAP: `TreeExplainer` on a stratified test subsample (max 250 rows per family, all rare rows), explaining the best model. | Exact tree SHAP on ~2k rows is fast and covers every family. |
| D18 | PDFs are produced with Markdown -> HTML -> headless Chrome (`scripts/build_pdfs.py`). | No LaTeX on the VM; Chrome gives consistent A4 output. |
