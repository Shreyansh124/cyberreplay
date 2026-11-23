Observation blocks inserted into the notebook by build_notebook.py (one block per "@@ key").
They are written from the results of the full run (reports/results).

@@ load
- We loaded 2,830,743 labelled flows from 8 day files (Monday-Friday, 3-7 July 2017), matching the published size of CIC-IDS2017.
- Loading retained 84 original columns after removing the duplicate `Fwd Header Length`, then added 3 metadata columns:
  `day`, `day_order` and `row_order`. We stored measurement columns as float32, which used 1.3 GB in memory.
- We removed 288,602 fully empty CSV lines from the Thursday-morning file while loading. Parsing timestamps day-first
  and correcting the 12-hour clock (e.g. Friday "3:30" = 15:30) produced a consistent 08:42-17:10 capture window.

@@ eda1
- The class distribution is highly uneven. BENIGN accounts for 80.3% of flows; only DoS 8.9%, PortScan 5.6% and DDoS
  4.5% are also sizeable. Brute Force (0.49%), Web Attack (0.077%), Bot (0.069%) and Rare (0.002%, 47 flows:
  36 Infiltration + 11 Heartbleed) are much smaller, giving a BENIGN:Rare ratio of ~48,000:1.
- A predictor that always returns BENIGN already reaches 80% accuracy, so accuracy mainly reflects the majority class.
  We therefore use macro F1 as the main metric and apply class weights.
- Combining related labels gives the model more examples: Web Attack - Sql Injection has just 21 flows, whereas its
  grouped Web Attack family has 2,180.

@@ eda2
- Monday contains no attacks, while the remaining attacks occur in short bursts: FTP-Patator (Tue ~09:20-10:20),
  SSH-Patator (Tue ~14:00-15:00), DoS Hulk at a peak of ~70k flows per 5 min on Wednesday ~10:45, web attacks on
  Thursday morning, Bot on Friday morning, and PortScan (~95k flows / 5 min at ~14:50) followed by DDoS
  (~15:56-16:16) on Friday afternoon.
- Attack traffic dominates during these bursts, but benign traffic also changes sharply, as seen on Thursday
  15:00-15:40. For that reason, volume by itself is not dependable and we need per-flow features.
- We use these timestamps for the replay in §11. They also reveal a limitation discussed in §13: a random split places
  flows from the same attack burst in both training and test sets.

@@ eda3
- Each attack type occurs in exactly **one** capture file: Heartbleed and the DoS family appear on Wednesday,
  Infiltration on Thursday afternoon, and PortScan and DDoS in separate Friday files.
- This prevents meaningful GroupKFold validation by day because a held-out day would remove *all* examples of its attacks
  from training. We instead use StratifiedKFold on the training split (§6).

@@ clean
- A `Flow Duration` of 0 produces 4,376 infinite rates: 1,509 in `Flow Bytes/s` and 2,867 in `Flow Packets/s`.
  Together with 1,358 existing NaN entries, this leaves 5,734 missing values, which the pipeline fills using a
  training-set median.
- We clipped 3,545 impossible negative values to 0. Most are in `Flow IAT Min` (2,891, nearly all BENIGN), while the
  -1 sentinel in `Init_Win_bytes_*` (1.0 M and 1.4 M rows) remains because it means "no TCP window observed".
- Before splitting, we removed 630,963 exact duplicate rows (22.3%). PortScan changes most: 98.8% of its 158,930 flows
  are repeated one/two-packet probes, leaving 1,958 unique flows. The corresponding losses are 33.8% for Brute Force,
  26.7% for Bot, 23.3% for DoS and 18.0% for BENIGN. This avoids identical vectors appearing in both train and test and
  overstating performance.
- We retained 622 rows whose feature vector is shared with another family because they represent label noise or
  irreducible ambiguity.
- After cleaning, 2,199,780 flows remain.

@@ eda5
- We found distinct flow profiles for each family. DoS flows stay open for a long time (median `Flow Duration` near
  100 s because of slow-rate attacks) and send large forward payloads. DDoS uses small, uniform forward packets;
  PortScan lasts microseconds, carries no payload and uses 1-2 packets; Web Attack also has almost no forward payload.
  Rare (Infiltration/Heartbleed) flows are both large and long because Heartbleed exfiltrates sizeable memory chunks.
- For Brute Force and Web Attack, `Init_Win_bytes_forward` is nearly constant and acts as a fingerprint of the attack
  tools' TCP stack. This may also mean the models learn tool-specific artefacts instead of general behaviour.
- Feature values span 6-8 orders of magnitude with heavy tails, which supports using a log transform in the linear model.

@@ eda6
- Values above the global 99.9th percentile occur in 1.5% of BENIGN rows but **48.9% of Rare rows**. Extreme values
  therefore carry useful information for the rarest attacks, so we do not delete these rows. Only Logistic Regression
  caps values at the training 99.9th percentile, preventing extreme magnitudes from distorting its scaled linear fit;
  tree models receive the raw values after imputation.
- We identified 50 feature pairs with |r| > 0.95, including 9 exact copies with r = 1.0, such as
  `Subflow Fwd Bytes` = `Total Length of Fwd Packets` and `SYN Flag Count` = `Fwd PSH Flags`. The clustered heatmap
  groups redundant packet counts / header lengths, packet-size statistics and IAT / idle times, which supports applying
  the correlation filter.

@@ select
- We began with 77 candidates and retained 69 after removing 8 constant columns; all bulk-rate features and the Bwd
  PSH/URG flags are always 0. The correlation filter then retained 44 by dropping 25 redundant twins while keeping each
  pair's basic statistic. Finally, retaining 99% of cumulative Random Forest importance produced **37** features after
  7 near-useless flags/statistics were removed.
- Forward payload size (`Fwd Packet Length Max`, `Total Length of Fwd Packets`), TCP handshake information
  (`Init_Win_bytes_backward/forward`, `min_seg_size_forward`) and timing (`Fwd IAT Std`, `Flow Duration`) contribute the
  strongest features.
- We fitted all three filters only on the training split, leaving the test split with no role in feature selection.

@@ split
- The stratified 80/20 split of the 2,199,780 de-duplicated flows gives 1,759,824 training and 439,956 test flows. Every
  family keeps the same share in both parts (test share 0.19-0.20), so even Rare has 38 training and 9 test flows.
- De-duplication changes the family sizes a lot: PortScan falls from 158,930 raw flows to 1,958 unique ones, while DDoS
  and Rare lose almost nothing. The test split therefore holds only 392 PortScan, 288 Bot and 9 Rare flows, which is why
  the per-class scores of these families move in large steps.

@@ cv
- With default parameters, 5-fold CV gives macro F1 scores of XGBoost 0.960 ± 0.007, Random Forest 0.954 ± 0.012,
  Decision Tree 0.945 ± 0.018 and Logistic Regression 0.534 ± 0.002.
- Tree-model results remain consistent between folds, with accuracy ≥ 0.998 in every fold. Their macro F1 variation
  (± 0.007-0.018) comes mainly from the smallest classes because each validation fold contains only ~8 Rare and
  ~230 Bot rows.
- Fully balanced Logistic Regression reaches macro recall 0.95 but only macro precision 0.47. Large weights for tiny
  classes move its linear boundaries far into BENIGN traffic and create many false alarms, so we tune the weight strength.

@@ tune
- Tuning improves all four models, with the largest change for Logistic Regression: 0.534 to **0.790** using C ≈ 10
  and square-root class weights. Figure 8 separates its candidates into fully balanced (0.51-0.55) and square-root
  weighting (0.70-0.79) groups, showing that weight strength matters much more than C.
- The tree models retain fully balanced weights and improve by smaller amounts. Decision Tree moves from 0.945 to 0.949
  (entropy, unlimited depth, `min_samples_leaf=2`, `max_features=0.8`); Random Forest moves from 0.954 to 0.963
  (150 trees, `max_depth=20`, `max_features=0.5`, entropy); and XGBoost moves from 0.960 to 0.966
  (300 rounds, depth 6, learning rate 0.24, row/column subsampling 0.77/0.68).
- Since the best XGBoost and Random Forest candidates differ by only 0.003 in CV, we use the test split and the extra
  criteria in §9 for the final decision.

@@ eval_table
- On the untouched test split of 439,956 flows, Random Forest records the highest macro F1 (**0.921**), ahead of XGBoost
  (0.912), Decision Tree (0.895) and Logistic Regression (0.715). All tree models have accuracy ≥ 0.998, which provides
  little information for this imbalanced problem.
- Test macro F1 falls below CV macro F1 (~0.96) for two main reasons. First, (1) the test split preserves the true BENIGN
  share (85%), while the tuning sample thins BENIGN, reducing small-class precision (Bot precision 0.57-0.68). Second,
  (2) Rare has only 9 test flows, so one miss changes that class by ~0.1 F1. Excluding Rare leaves the tree models within
  0.004 of one another at macro F1 0.947-0.950.
- The larger BENIGN, DoS, DDoS and Brute Force families reach F1 ≥ 0.995. Bot (F1 0.70-0.74), Rare (0.53-0.71), and
  PortScan and Web Attack for the linear model remain the difficult cases.
- Fitting 1.76 M flows takes 32 s for DT, 62 s for XGB, 92 s for RF and 231 s for LogReg. Every model predicts the
  complete 440k-flow test set in well under 2 s (≈ 0.2-3 µs per flow), comfortably within real-time requirements.

@@ confusion
- Most mistakes involve BENIGN. Random Forest labels 157 BENIGN flows as DoS and 202 as Bot, while it misses 4 of the
  9 Rare flows (all Infiltration) and 12 of 288 Bot flows as BENIGN. Confusion directly between attack families is rare.
- Of the 202 BENIGN flows labelled as Bot, 158 contact 205.174.165.73:8080, the Ares C&C server and port used by every
  labelled Bot flow, but outside the labelled Bot period. These are likely unlabelled bot beacons, consistent with the
  CIC-IDS2017 labelling issues documented by Engelen et al., 2021, although we still count them as errors.
- Every tree model detects both Heartbleed flows. Infiltration is hardest because traffic after the initial compromise
  resembles ordinary activity from an internal host.
- Logistic Regression produces a clearly poorer matrix, with 1.6% of BENIGN flows becoming false alarms distributed
  across every attack family.

@@ comparison
- For the binary `is_malicious` output derived from multi-class predictions, the three tree models perform almost the
  same: F1 0.996-0.997, recall 0.998-0.999 and a BENIGN false-positive rate of 0.10-0.11%, or ≈ 400 false alarms among
  373k benign flows. Logistic Regression gives F1 0.947 and a 1.64% false-positive rate, producing 15x more false alarms.
- Figure 11's per-class F1 separates the models more clearly. Random Forest leads on PortScan, Web Attack and Rare;
  Decision Tree leads on Bot; and XGBoost leads on DoS, DDoS and Brute Force. Except for Rare, the tree models stay
  within 0.04 on each class.
- Inference is fastest for Decision Tree at ~0.2 µs/flow, followed by Random Forest at ~1.1 µs/flow and XGBoost at
  ~3 µs/flow.

@@ best
- **We select Random Forest as the final model.** It achieves the highest test macro F1 (0.921, 95% bootstrap CI
  0.864-0.947), the highest macro F1 without Rare (0.950), and the best minority-class recall (0.893 across PortScan,
  Bot, Brute Force, Web Attack, Rare). Its BENIGN false-positive rate is 0.11%, equal to XGBoost; training takes
  ~1.5 min, and prediction is ~2.8x faster than XGBoost.
- In the paired bootstrap (`reports/results/test_bootstrap_ci.csv`), Random Forest beats XGBoost in 99% of resamples,
  although its average macro F1 advantage is only 0.009. XGBoost is therefore a close second and remains a reasonable
  alternative, while Decision Tree suits cases where inference speed and interpretability matter most.
- The dashboard uses our saved pipeline, which combines feature selection, imputation and the model in about 5.5 MB
  compressed.

@@ roc
- The refitted models reproduce the stored test metrics exactly, and the saved best model predicts the same class for
  every test flow as the refit.
- Balanced accuracy equals macro recall for every model (Random Forest 0.9329), as it should for a multi-class problem.
  The majority-class baseline reaches 0.847 accuracy but only 0.125 balanced accuracy, 0.115 macro F1 and ROC-AUC 0.5,
  which shows how little accuracy says on this data.
- On one-vs-rest ROC-AUC (macro) XGBoost leads with 0.992, then Logistic Regression 0.979, Random Forest 0.972 and
  Decision Tree 0.957. For Random Forest every family except Rare has AUC of at least 0.997; four of the nine Rare test
  flows get a Rare probability of zero, which caps its Rare AUC at 0.778. With only nine positives the Rare curve is
  noisy and moves in steps of 1/9.
- The ranking by AUC differs from the ranking by macro F1 because AUC judges how well the probabilities order the flows
  at every threshold, while macro F1 judges the actual predicted classes.

@@ fit
- On the validation slice carved from the training split, Logistic Regression scores 0.720 on its own rows and 0.716 on
  new rows: almost no gap, but too low, so it underfits (high bias). The Decision Tree drops from 0.973 to 0.918 (gap
  0.055, overfit), XGBoost from 0.971 to 0.944 (gap 0.026, slight overfit) and Random Forest from 0.965 to 0.948 (gap
  0.017, good fit). The validation ranking (Random Forest, XGBoost, Decision Tree, Logistic Regression) matches the test
  ranking.
- The learning curves show the same picture. Logistic Regression's training and CV scores meet at about 0.84 and stop
  improving, so more data would not help. Random Forest's CV score keeps rising, from 0.873 at 13k rows to 0.960 at 133k
  rows, while its gap to the training score shrinks from 0.089 to 0.028.
- The Decision Tree validation curve moves from underfitting (0.086 at depth 2) to overfitting: beyond depth 15 the
  training score reaches 0.98-0.99 while CV stays at 0.94-0.95. XGBoost's CV score peaks at 0.969 around 200-300 rounds
  and slips to 0.967 at 600 rounds while the training score keeps rising, so the tuned 300 rounds sit near the best point.

@@ importance
- Across tree models, **payload size** features rank first (`Total Length of Fwd Packets`, `Fwd Packet Length Max`,
  `Packet Length Mean`), followed by **TCP handshake** features (`min_seg_size_forward`,
  `Init_Win_bytes_backward/forward`) and forward inter-arrival variation (`Fwd IAT Std`).
- XGBoost gain gives more weight to `Bwd Packet Length Min`, `Fwd PSH Flags` and `PSH Flag Count`, which are correlated
  views of packet-size and flag behaviour. Alongside payload size, Logistic Regression depends more on timing
  (`Flow IAT Max`, `Bwd IAT Total`, `Flow Duration`, `Flow IAT Mean`) because log-scaled times provide its clearest
  linear separation.

@@ shap
- SHAP results for Random Forest use 1,759 stratified test flows, up to 250 per family, and show a separate signature
  for each family in Figure 13. DDoS is associated with small forward packets (`Total Length of Fwd Packets`,
  `Fwd Packet Length Max`) and `min_seg_size_forward`; PortScan with zero payload plus low `Packet Length Mean` and
  `Fwd IAT Std`; Web Attack with `Total Length of Fwd Packets` and high `Fwd IAT Std`; Brute Force with
  `min_seg_size_forward` from the tool's TCP options; DoS with large forward packets and `Init_Win_bytes_backward`;
  and Rare with very large payloads and long `Flow Duration`.
- Figure 14's beeswarms show effect direction. High forward payload and long duration move predictions toward DoS;
  *low* forward packet sizes toward DDoS; zero payload with regular timing toward PortScan; and little payload with
  irregular forward timing, represented by high `Fwd IAT Std`, toward Web Attack.

@@ patterns
- Raw distributions agree with SHAP. A median DoS flow sends 354 forward bytes in packets up to 343 B, compared with
  74 B / 44 B for benign traffic; the median DDoS flow sends only 26 B. PortScan and Web Attack generally send **no**
  forward payload, while Rare sends ~12 kB over ~100 s against a benign median flow of 59 ms.
- Constant `min_seg_size_forward` / `Init_Win_bytes_forward` values make the attack tools' TCP stack a fingerprint for
  Brute Force and Web Attack. This helps within this dataset but may not generalise to different tools (§13).
- Zero-payload Web Attack and PortScan flows correspond to the "TCP teardown" artefacts reported for CIC-IDS2017
  (Engelen et al., 2021), indicating that the models partly learn how CICFlowMeter divided attack connections.

@@ replay
- Replaying the 29,999-flow test sample in capture order reconstructs Figure 2's schedule: Brute Force on Tuesday,
  the Wednesday-morning DoS burst, Thursday-morning web attacks, Friday-morning Bot, then PortScan and DDoS on
  Friday afternoon.
- In this sample, the model passes 16,207 of 16,231 BENIGN flows and detects every DoS and DDoS flow. Its misses match
  the full test split: the same 4 Infiltration and 12 Bot flows become BENIGN, while 13 PortScan flows are mostly
  classified as DoS.
- The paced generator emits 500-flow ticks at 300x capture speed, and the dashboard runs the same code through a
  speed slider.

@@ explain
- For each flow, our explanation reports the predicted family, probability and the features moving the prediction away
  from the typical median benign value. The DoS example has forward packets ~5x larger than normal and a server TCP
  window (`Init_Win_bytes_backward` = 235) where most benign flows have none. The PortScan example carries only
  2 bytes of payload, while the Rare Heartbleed example transfers 13.7 kB over 119 s.
- The dashboard's "Explain a detection" tab presents these same per-detection explanations with a SHAP bar chart and
  class probabilities.

@@ conclusions
**Conclusions.** After removing 631k duplicate CIC-IDS2017 flows and every identifier, we trained a class-weighted Random
Forest on 37 selected flow features. It classifies 8 attack families at test macro F1 **0.921** (0.950 without the 9-flow
Rare class), detects 99.9% of malicious flows with a 0.11% false-alarm rate, and predicts each flow in about a microsecond.
XGBoost follows closely in the statistical comparison, while Decision Tree is nearly as accurate and has the fastest
inference. Logistic Regression is not sufficient for these non-linear families (macro F1 0.715, 15x more false alarms).
We found that payload size, TCP handshake parameters and forward timing separate the attacks; SHAP explains individual
alerts, and the replay dashboard demonstrates the model as an interactive monitoring prototype.

**Limitations.**
- *Optimistic split:* Our random stratified split places flows from one attack burst in both train and test. GroupKFold by
  day cannot solve this because each attack appears on one day, so the scores show recognition of known attacks within
  the same network rather than generalisation to new networks or attack tools.
- *Label quality:* CIC-IDS2017 contains documented labelling and flow-construction errors (Engelen et al., 2021). Most Bot
  "false positives" contact the Ares C&C server and are probably unlabelled bot traffic, while zero-payload
  Web Attack/PortScan flows partly reflect TCP-teardown artefacts.
- *Tiny classes:* Rare contains 47 flows (9 in test), giving its F1 a very wide confidence interval, and Infiltration
  recall remains low.
- *Tool fingerprints:* Features such as `Init_Win_bytes_*` and `min_seg_size_forward` identify the attack tools' TCP
  stacks and may not transfer to other tools.
- *Offline features:* CICFlowMeter calculates features only when a flow ends. The replay is therefore near-real-time
  rather than packet-level real-time, and timestamps have only minute resolution except Monday.

**Future work.** We would test per-class probability thresholds or cost-sensitive calibration to reduce the remaining
Bot/DoS false alarms; evaluate corrected CIC-IDS2017 labels and run cross-dataset tests such as CSE-CIC-IDS2018 to measure
generalisation; use time-aware validation within each attack day; compute streaming flow features for true online
detection; and add anomaly detection for unseen (zero-day) attacks that a supervised classifier cannot recognise.

**References.** Sharafaldin, Lashkari & Ghorbani, ICISSP 2018 (dataset); Engelen, Rimmer & Joosen, "Troubleshooting an
Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE S&P Workshops 2021; Pedregosa et al., JMLR 2011
(scikit-learn); Chen & Guestrin, KDD 2016 (XGBoost); Lundberg & Lee, NeurIPS 2017 and Lundberg et al., Nature Machine
Intelligence 2020 (SHAP).
