---
name: cyberreplay-workflow
description: Working rules for building the CyberReplay project - phase/sub-phase execution, verify-before-code, and documentation standards. Use whenever writing, editing or reviewing code in this repository.
---

# CyberReplay working rules

## 1. Work in phases and sub-phases
- Split each of the 16 build steps (see the `cyberreplay-project` skill) into small sub-phases.
- Finish one sub-phase completely (code, run, check output) before starting the next.
- Keep a progress checklist in `PROGRESS.md` and tick sub-phases off as they finish.
- Commit after each completed phase with a message like `phase 4.2: scale features inside pipeline`.

## 2. Verify before writing code
Before writing or changing code, check that the plan:
- Matches the agreed design decisions and rubric in `cyberreplay-project`.
- Is consistent with existing modules (same names, column names, paths, seed = 42).
- Does not leak test data (fit only on train; no test rows in CV or tuning).
- Uses only allowed tooling for Track A (pandas, NumPy, scikit-learn, XGBoost, Matplotlib/Seaborn, SHAP, Streamlit).
After writing code, run it and confirm the output before moving on.

## 3. Documentation standards
- Every module starts with a docstring explaining its purpose and how it fits the pipeline.
- Every function has a docstring at the top: what it does, Args, Returns, and any side effects (files written).
- Comments explain *why* a choice was made (e.g. "clip instead of drop so rare attacks keep their rows"), not what the next line does.
- Notebook: a markdown cell before each code section explaining the step, and an "Observations" markdown cell after every plot or result.
- Keep code readable: type hints, small functions, constants in `src/config.py`, no hard-coded absolute paths.
