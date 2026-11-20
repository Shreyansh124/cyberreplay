# Build progress

Resume from the first unticked sub-phase. Each completed phase is committed as
`phase X.Y: ...` and pushed to `cursor/cyberreplay-pipeline-ddea`.

## Phase 0 - Setup
- [x] 0.1 Read project + workflow skills
- [x] 0.2 Download CIC-IDS2017 GeneratedLabelledFlows into `data/raw/` (see DECISIONS D1)
- [x] 0.3 `.venv`, pinned `requirements.txt`, `pyproject.toml`, `.gitignore`
- [x] 0.4 `config.py` (paths, seed, label map, constants)

## Phase 1 - Problem + dataset (`data.py`)
- [x] 1.1 Download helper (mirror URL, SHA-256 check, unzip, folder-name fix)
- [x] 1.2 Loader: latin-1, strip columns, drop empty rows / duplicate column, day + row order
- [x] 1.3 Timestamp parsing (day-first, 12-hour fix) + parquet cache

## Phase 2 - EDA (`eda.py`)
- [x] 2.1 Class distribution, traffic timeline, attack x day matrix
- [x] 2.2 Data quality, per-family feature distributions, correlation heatmap

## Phase 3 - Cleaning (`clean.py`)
- [x] 3.1 Label normalisation (cp1252 dash) + families + `is_malicious`
- [x] 3.2 inf -> NaN, invalid negatives clipped, duplicates dropped, conflicts counted
- [x] 3.3 Outlier quantification; capping only inside the LogReg pipeline (fit on train)

## Phase 4 - Feature engineering (`features.py`)
- [x] 4.1 Stratified 80/20 split after de-duplication
- [x] 4.2 Variance -> correlation -> RF-importance selection on train only
- [x] 4.3 Pipeline transformers (selector, imputer, clipper, signed log, scaler)

## Phase 5-7 - Models, CV, tuning (`models.py`, `tuning.py`)
- [x] 5.1 Four pipelines with class weights (XGBoost weight wrapper)
- [x] 6.1 5-fold StratifiedKFold per-fold scores on the tuning sample (code)
- [x] 7.1 RandomizedSearchCV (f1_macro) search spaces (code)
- [ ] 7.2 Full CV + tuning run on the 500k sample (`python -m cyberreplay.pipeline --force`, log in `logs/`)

## Phase 8-9 - Evaluation and best model (`evaluate.py`)
- [x] 8.1 Final fit, test metrics, per-class reports, confusion matrices, comparison plots (code)
- [ ] 8.2 Full run on the full training split
- [ ] 9.1 Best-model choice + saved pipeline + metadata

## Phase 10 - Attack patterns (`explain.py`)
- [x] 10.1 Native importance, SHAP heatmap/beeswarm, per-attack distributions (code)
- [ ] 10.2 Full-run outputs reviewed

## Phase 11-14 - Dashboard, replay, interactive visuals, explanations
- [x] 12.1 `replay.py` (ordering, pacing, live predictions)
- [ ] 11.1 Streamlit dashboard (`app/streamlit_app.py`)
- [ ] 11.2 Headless verification (AppTest + `streamlit run --server.headless`)

## Phase 15 - Reproducibility
- [x] 15.1 `pipeline.py` CLI (smoke-tested with `--quick`)
- [ ] 15.2 Dataset sample committed
- [ ] 15.3 Notebook (Colab cells) built and executed
- [ ] 15.4 README

## Phase 16 - Report
- [ ] 16.1 `report.md` -> `report.pdf` (3-5 pages)
- [ ] 16.2 `notebook.pdf`
- [ ] 16.3 Pull request
