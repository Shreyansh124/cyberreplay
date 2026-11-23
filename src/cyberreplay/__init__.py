"""CyberReplay: machine learning-based network intrusion detection on CIC-IDS2017.

Package layout (one module per pipeline stage):
    config    - paths, seed, label mappings, constants
    data      - dataset download, loading and timestamp parsing
    clean     - label normalisation and row-level cleaning
    features  - leakage-safe split, feature selection, preprocessing transformers
    models    - the four Track A model pipelines
    tuning    - tuning sample, stratified CV, RandomizedSearchCV
    evaluate  - test metrics, confusion matrices, comparison plots
    explain   - feature importance, SHAP, per-attack feature distributions
    replay    - chronological traffic replay with live predictions
    eda       - exploratory plots
    pipeline  - command-line entry point that runs everything end to end
"""

__version__ = "1.0.0"
