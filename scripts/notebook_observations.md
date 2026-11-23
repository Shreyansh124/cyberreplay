Observation blocks inserted into the notebook by build_notebook.py (one block per "@@ key").
They are written from the results of the full run (reports/results).

@@ load
- 2,830,743 labelled flows in 8 day files (Monday-Friday, 3-7 July 2017), the published size of CIC-IDS2017.
- 84 original columns survive loading (the duplicated `Fwd Header Length` is dropped) plus 3 metadata columns
  (`day`, `day_order`, `row_order`); measurement columns are stored as float32 (1.3 GB in memory).
- The Thursday-morning file contained 288,602 completely empty CSV lines, removed at load time; timestamps were parsed
  as day-first with the 12-hour clock fixed (e.g. Friday "3:30" = 15:30), giving a consistent 08:42-17:10 capture window.

@@ eda1
- Extreme imbalance: BENIGN is 80.3% of all flows; DoS 8.9%, PortScan 5.6% and DDoS 4.5% are the only other large
  families, while Brute Force (0.49%), Web Attack (0.077%), Bot (0.069%) and Rare (0.002%, 47 flows: 36 Infiltration +
  11 Heartbleed) are tiny. The BENIGN:Rare ratio is ~48,000:1.
- Accuracy would therefore be dominated by BENIGN (a constant BENIGN predictor already scores 80%), which is why macro F1
  is the primary metric and class weights are needed.
- Grouping helps: e.g. Web Attack - Sql Injection (21 flows) alone could never be learned reliably, but the Web Attack family has 2,180.

@@ eda2
- Monday is attack-free; each attack is a short, intense burst: FTP-Patator (Tue ~09:20-10:20) and SSH-Patator
  (Tue ~14:00-15:00), DoS Hulk peaking at ~70k flows per 5 min on Wednesday ~10:45, the web attacks on Thursday morning, Bot on
  Friday morning, PortScan (~95k flows / 5 min at ~14:50) and DDoS (~15:56-16:16) on Friday afternoon.
- Attack volume dwarfs normal traffic during the bursts, but benign volume also fluctuates strongly (e.g. Thursday 15:00-15:40),
  so volume alone is not a reliable detector - per-flow features are needed.
- These timestamps drive the replay in §11; the chronological structure also means that a random split mixes flows of the
  same attack burst into train and test (a limitation discussed in §13).

@@ eda3
- Every attack type appears on exactly **one** capture file (Heartbleed and the DoS family on Wednesday, Infiltration on
  Thursday afternoon, PortScan and DDoS in separate Friday files).
- Consequence for validation: GroupKFold by day would hold out *all* examples of the attacks of that day, so those classes
  could not be learned in that fold. StratifiedKFold on the training split is used instead (§6).

@@ clean
- 4,376 infinite rates (1,509 `Flow Bytes/s`, 2,867 `Flow Packets/s`) occur when `Flow Duration` is 0; with the 1,358
  original NaN they give 5,734 missing values, imputed by the training median inside the pipeline.
- 3,545 invalid negatives (mostly `Flow IAT Min`, 2,891, almost all BENIGN) were clipped to 0; the -1 sentinel of
  `Init_Win_bytes_*` (1.0 M and 1.4 M rows) is kept because it means "no TCP window observed".
- 630,963 rows (22.3%) are exact duplicates and were removed before splitting. PortScan is hit hardest: 98.8% of its
  158,930 flows are identical one/two-packet probes, leaving 1,958 unique flows; Brute Force loses 33.8%, Bot 26.7%,
  DoS 23.3% and BENIGN 18.0%. Without this step identical vectors would sit in train and test and inflate the scores.
- 622 rows share their feature vector with a different family (label noise / irreducible ambiguity); they are kept.
- 2,199,780 clean flows remain.

@@ eda5
- The families have clearly different "shapes": DoS flows are long (median `Flow Duration` near 100 s - slow-rate attacks hold
  connections open) with large forward payloads; DDoS flows have small, uniform forward packets; PortScan flows last
  microseconds, carry no payload and use 1-2 packets; Web Attack flows also carry almost no forward payload; Rare
  (Infiltration/Heartbleed) flows are huge and long (Heartbleed exfiltrates large memory chunks).
- `Init_Win_bytes_forward` is almost constant for Brute Force and Web Attack (the attack tools' TCP stack), a strong
  fingerprint - but also a sign that the models may learn tool artefacts rather than generic behaviour.
- All features are heavy-tailed over 6-8 orders of magnitude, motivating the log transform for the linear model.

@@ eda6
- Outliers (any value above the global 99.9th percentile) concern 1.5% of BENIGN rows but **48.9% of Rare rows**:
  for the rarest attacks extreme values *are* the signal. Therefore no rows are deleted; capping at the training 99.9th
  percentile is applied only in the Logistic Regression pipeline, where extreme magnitudes would distort the scaled linear
  model, and tree models see the raw (imputed) values.
- 50 feature pairs have |r| > 0.95 and 9 pairs are exact copies (r = 1.0, e.g. `Subflow Fwd Bytes` =
  `Total Length of Fwd Packets`, `SYN Flag Count` = `Fwd PSH Flags`). The clustered heatmap shows big redundant blocks
  (packet counts / header lengths, packet-size statistics, IAT / idle times) - the motivation for the correlation filter.

@@ select
- 77 candidates → 69 after removing 8 constant columns (all bulk-rate features and the Bwd PSH/URG flags are always 0)
  → 44 after the correlation filter (25 redundant twins dropped, the basic statistic of each pair is kept) → **37**
  after keeping 99% of cumulative Random Forest importance (7 near-useless flags/statistics dropped).
- The most important features describe forward payload size (`Fwd Packet Length Max`, `Total Length of Fwd Packets`),
  the TCP handshake (`Init_Win_bytes_backward/forward`, `min_seg_size_forward`) and timing (`Fwd IAT Std`, `Flow Duration`).
- All three filters were fitted on the training split only, so the test split had no influence on the feature set.

@@ cv
- Default-parameter 5-fold CV (macro F1): XGBoost 0.960 ± 0.007, Random Forest 0.954 ± 0.012, Decision Tree
  0.945 ± 0.018, Logistic Regression 0.534 ± 0.002.
- The tree models are stable across folds (accuracy ≥ 0.998 in every fold); their fold-to-fold spread of macro F1
  (± 0.007-0.018) is what one expects from the smallest classes - each validation fold holds only ~8 Rare and ~230 Bot rows.
- Logistic Regression with fully balanced weights has macro recall 0.95 but macro precision 0.47: the huge weights of the
  tiny classes push its linear boundaries deep into BENIGN territory (many false alarms). This motivated tuning the
  weighting strength.

@@ tune
- Tuning helps every model, most of all Logistic Regression: 0.534 → **0.790** (C ≈ 10, square-root class weights). Its
  candidates form two clusters in Figure 8 - fully balanced (0.51-0.55) vs square-root weights (0.70-0.79) - so the weighting
  strength matters far more than C.
- Tree models keep fully balanced weights and gain less: Decision Tree 0.945 → 0.949 (entropy, unlimited depth,
  `min_samples_leaf=2`, `max_features=0.8`), Random Forest 0.954 → 0.963 (150 trees, `max_depth=20`, `max_features=0.5`,
  entropy), XGBoost 0.960 → 0.966 (300 rounds, depth 6, learning rate 0.24, row/column subsampling 0.77/0.68).
- The best XGBoost and Random Forest candidates are within 0.003 of each other in CV, so the final choice is made on the test
  split with the additional criteria of §9.

@@ eval_table
- On the untouched test split (439,956 flows) Random Forest has the best macro F1 (**0.921**), followed by XGBoost (0.912),
  Decision Tree (0.895) and Logistic Regression (0.715). Accuracy is ≥ 0.998 for all tree models and is not informative.
- Test macro F1 is lower than CV macro F1 (~0.96) for two reasons: (1) the test split has the real BENIGN proportion
  (85%) whereas the tuning sample is BENIGN-thinned, so small-class precision drops (Bot precision 0.57-0.68), and
  (2) Rare has only 9 test flows, so each miss costs ~0.1 F1 on that class. Without Rare the tree models are
  within 0.004 of each other (macro F1 0.947-0.950).
- The large families (BENIGN, DoS, DDoS, Brute Force) are solved (F1 ≥ 0.995); the hard cases are Bot (F1 0.70-0.74),
  Rare (0.53-0.71) and, for the linear model, PortScan and Web Attack.
- Training on 1.76 M flows takes 32 s (DT), 62 s (XGB), 92 s (RF) and 231 s (LogReg); predicting all 440k test flows takes
  well under 2 s for every model (≈ 0.2-3 µs per flow), far below real-time requirements.

@@ confusion
- Almost all errors involve BENIGN: for Random Forest 157 BENIGN flows are predicted as DoS and 202 as Bot, while
  4 of the 9 Rare flows (all Infiltration) and 12 of 288 Bot flows are missed as BENIGN. Attack families are almost never
  confused with each other.
- 158 of the 202 BENIGN → Bot "false alarms" go to 205.174.165.73:8080 - the Ares C&C server and port used by every
  labelled Bot flow - outside the labelled Bot window. They are very likely unlabelled bot beacons (CIC-IDS2017 labelling
  issues are documented by Engelen et al., 2021); we nevertheless count them as errors.
- Both Heartbleed flows are detected by every tree model, Infiltration is the hardest class: after the initial compromise it
  generates ordinary-looking traffic from an internal host.
- Logistic Regression has a visibly worse matrix: 1.6% of BENIGN flows raise false alarms spread over all attack families.

@@ comparison
- For the binary `is_malicious` decision (derived from the multi-class prediction) the three tree models are almost
  identical: F1 0.996-0.997, recall 0.998-0.999, BENIGN false-positive rate 0.10-0.11% (≈ 400 false alarms per 373k benign
  flows). Logistic Regression reaches F1 0.947 with a 1.64% false-positive rate - 15x more false alarms.
- Per-class F1 (Figure 11) shows where the models differ: Random Forest is best on PortScan, Web Attack and Rare, Decision
  Tree on Bot, XGBoost on DoS, DDoS and Brute Force; the three tree models differ by less than 0.04 on every class except Rare.
- Speed: Decision Tree is the fastest at inference (~0.2 µs/flow), Random Forest ~1.1 µs/flow and XGBoost ~3 µs/flow.

@@ best
- **Random Forest is the final model.** It has the highest test macro F1 (0.921, 95% bootstrap CI 0.864-0.947) and the highest
  macro F1 without Rare (0.950), the best minority-class recall (0.893 over PortScan, Bot, Brute Force, Web Attack, Rare),
  a BENIGN false-positive rate of 0.11% (equal to XGBoost), trains in ~1.5 min and predicts ~2.8x faster than XGBoost.
- The paired bootstrap (`reports/results/test_bootstrap_ci.csv`) shows Random Forest ahead of XGBoost in 99% of resamples,
  but by only 0.009 macro F1 on average - XGBoost is a very close second and would be an equally reasonable choice;
  Decision Tree is the choice if inference speed and interpretability matter most.
- The saved pipeline (feature selection + imputation + model, about 5.5 MB compressed) is used by the dashboard.

@@ importance
- All tree models rank **payload size** features first (`Total Length of Fwd Packets`, `Fwd Packet Length Max`,
  `Packet Length Mean`) followed by **TCP handshake** features (`min_seg_size_forward`, `Init_Win_bytes_backward/forward`) and
  forward inter-arrival variability (`Fwd IAT Std`).
- XGBoost's gain importance favours `Bwd Packet Length Min`, `Fwd PSH Flags` and `PSH Flag Count`, i.e. different but
  correlated views of the same packet-size / flag behaviour. Logistic Regression leans more on timing (`Flow IAT Max`,
  `Bwd IAT Total`, `Flow Duration`, `Flow IAT Mean`) next to payload size, because log-scaled times are the most linearly
  separating directions it can use.

@@ shap
- SHAP (Random Forest, 1,759 stratified test flows: up to 250 per family) shows each family has its own signature (Figure 13):
  DDoS - small forward packets (`Total Length of Fwd Packets`, `Fwd Packet Length Max`) and `min_seg_size_forward`;
  PortScan - zero payload plus low `Packet Length Mean` and `Fwd IAT Std`; Web Attack - `Total Length of Fwd Packets` and a
  high `Fwd IAT Std`; Brute Force - `min_seg_size_forward` (the tool's TCP options); DoS - large forward packets and
  `Init_Win_bytes_backward`; Rare - very large payloads and long `Flow Duration`.
- The beeswarms (Figure 14) give the direction: high forward payload and long durations push towards DoS; *low* forward
  packet sizes push towards DDoS; zero payload and regular timing push towards PortScan; irregular forward timing (high
  `Fwd IAT Std`) with little payload pushes towards Web Attack.

@@ patterns
- The distributions confirm SHAP on the raw scale: the median DoS flow sends 354 forward bytes in packets of up to 343 B (benign:
  74 B / 44 B); the median DDoS flow only 26 B; PortScan and Web Attack flows typically send **no** forward payload at all;
  Rare flows send ~12 kB over ~100 s (benign median flow: 59 ms).
- Brute Force and Web Attack flows have a constant `min_seg_size_forward` / `Init_Win_bytes_forward`, i.e. the attack tools'
  TCP stack is a fingerprint - useful in this dataset, but a reason to doubt generalisation to other tools (§13).
- The zero-payload Web Attack and PortScan flows match the "TCP teardown" artefacts reported for CIC-IDS2017 (Engelen et al., 2021):
  part of what the models learn is how CICFlowMeter split the attack connections.

@@ replay
- Replaying the 29,999-flow sample (drawn from the test split) in capture order reproduces the attack schedule of Figure 2:
  Brute Force on Tuesday, the DoS burst on Wednesday morning, web attacks on Thursday morning, Bot on Friday morning,
  PortScan and DDoS on Friday afternoon.
- On the sample, 16,207 of 16,231 BENIGN flows are passed and every DoS and DDoS flow is detected; the misses are the same as
  on the full test split (the same 4 Infiltration and 12 Bot flows missed as BENIGN, 13 PortScan flows mostly predicted as DoS).
- The paced generator streams 500-flow ticks at 300x capture speed; the dashboard uses the same code with a speed slider.

@@ explain
- Each explanation names the predicted family, its probability and the features that pushed the flow towards it, compared
  with the typical (median) benign value - e.g. the DoS example is flagged because its forward packets are ~5x larger than
  normal and the server answers with a TCP window (`Init_Win_bytes_backward` = 235) where most benign flows have none, the
  PortScan example because it carries only 2 bytes of payload, and the Rare (Heartbleed) example because it moves 13.7 kB
  over 119 s.
- These per-detection explanations are exactly what the dashboard's "Explain a detection" tab shows, together with the
  SHAP bar chart and the class probabilities.

@@ conclusions
**Conclusions.** On CIC-IDS2017, after removing 631k duplicate flows and all identifiers, a class-weighted Random Forest on 37
selected flow features classifies 8 attack families with test macro F1 **0.921** (0.950 without the 9-flow Rare class),
detects 99.9% of malicious flows with a 0.11% false-alarm rate, and predicts in about a microsecond per flow. XGBoost is
statistically a close second, Decision Tree is nearly as good and the fastest, and Logistic Regression is clearly
insufficient (macro F1 0.715, 15x more false alarms) because the families are not linearly separable. Payload size, TCP
handshake parameters and forward timing are what distinguish the attacks; SHAP makes every individual alert explainable,
and the replay dashboard turns the model into an interactive monitoring prototype.

**Limitations.**
- *Optimistic split:* a random stratified split puts flows of the same attack burst into train and test; GroupKFold by day
  is impossible (one day per attack), so the scores measure recognition of known attacks in the same network, not
  generalisation to new networks or new attack tools.
- *Label quality:* CIC-IDS2017 has documented labelling and flow-construction errors (Engelen et al., 2021). Most of our Bot
  "false positives" go to the Ares C&C server, i.e. are probably unlabelled bot traffic; zero-payload Web Attack/PortScan flows
  are partly TCP-teardown artefacts.
- *Tiny classes:* Rare has 47 flows (9 in test), so its F1 has a very wide confidence interval; Infiltration recall is low.
- *Tool fingerprints:* features such as `Init_Win_bytes_*` and `min_seg_size_forward` identify the attack tools' TCP stacks,
  which may not transfer to other tools.
- *Offline features:* CICFlowMeter features are computed when a flow ends, so the replay is near-real-time, not packet-level
  real-time; timestamps have only minute resolution (except Monday).

**Future work.** Per-class probability thresholds or cost-sensitive calibration to cut the remaining Bot/DoS false alarms;
evaluation on the corrected CIC-IDS2017 labels and cross-dataset tests (e.g. CSE-CIC-IDS2018) to measure generalisation;
time-aware validation within each attack day; streaming flow features for true online detection; and an anomaly-detection
component for unseen (zero-day) attack types that a supervised classifier cannot recognise.

**References.** Sharafaldin, Lashkari & Ghorbani, ICISSP 2018 (dataset); Engelen, Rimmer & Joosen, "Troubleshooting an
Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE S&P Workshops 2021; Pedregosa et al., JMLR 2011
(scikit-learn); Chen & Guestrin, KDD 2016 (XGBoost); Lundberg & Lee, NeurIPS 2017 and Lundberg et al., Nature Machine
Intelligence 2020 (SHAP).
