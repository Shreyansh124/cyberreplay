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

## Phase 2 - EDA
## Phase 3 - Cleaning
## Phase 4 - Feature engineering
## Phase 5 - Models
## Phase 6 - Cross-validation
## Phase 7 - Tuning
## Phase 8 - Evaluation
## Phase 9 - Best model
## Phase 10 - Attack patterns
## Phase 11-14 - Dashboard, replay, interactive visuals, explanations
## Phase 15 - Reproducibility (CLI, notebook, sample, README)
## Phase 16 - Report
