"""Central configuration for the CyberReplay pipeline.

Every other module imports paths, the random seed, label mappings and
experiment constants from here so that the whole project has a single
source of truth (and no hard-coded absolute paths). Paths can be
overridden with environment variables, which is how the Colab notebook
points the pipeline at a Google Drive copy of the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Raw CSVs (gitignored). Override with CYBERREPLAY_DATA_DIR, e.g. a Drive folder.
DATA_DIR: Path = Path(os.environ.get("CYBERREPLAY_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR: Path = Path(
    os.environ.get("CYBERREPLAY_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed")
)
SAMPLE_DIR: Path = PROJECT_ROOT / "data" / "sample"
SAMPLE_PATH: Path = SAMPLE_DIR / "cicids2017_sample.csv.gz"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_DIR: Path = REPORTS_DIR / "results"

MODELS_DIR: Path = PROJECT_ROOT / "models"
# Fitted intermediate models (all four, uncompressed) - gitignored, can be large.
MODEL_CACHE_DIR: Path = MODELS_DIR / "cache"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "best_model_metadata.json"

# --------------------------------------------------------------------------
# Dataset source
# --------------------------------------------------------------------------
OFFICIAL_DATASET_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
# The official download (cicresearch.ca) sits behind a registration form, so the
# automated download uses a byte-identical Hugging Face mirror of the official
# GeneratedLabelledFlows.zip (TrafficLabelling CSVs, which include Timestamp).
DATASET_MIRROR_URL = (
    "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/GeneratedLabelledFlows.zip"
)
DATASET_ZIP_SHA256 = "7bdbef286f8893f31c6db12105fa097fa5c2dcc6733179037a08129d150ea27a"
DATASET_CITATION = (
    "I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani, \"Toward Generating a New Intrusion "
    "Detection Dataset and Intrusion Traffic Characterization\", Proc. 4th International "
    "Conference on Information Systems Security and Privacy (ICISSP), Portugal, 2018, pp. 108-116."
)

# Day files of CIC-IDS2017 in capture order: file stem prefix -> (day label, order).
# The order is used to rebuild a chronological stream for the replay step.
DAY_FILES: dict[str, tuple[str, int]] = {
    "Monday-WorkingHours": ("Monday", 0),
    "Tuesday-WorkingHours": ("Tuesday", 1),
    "Wednesday-workingHours": ("Wednesday", 2),
    "Thursday-WorkingHours-Morning-WebAttacks": ("Thursday-AM", 3),
    "Thursday-WorkingHours-Afternoon-Infilteration": ("Thursday-PM", 4),
    "Friday-WorkingHours-Morning": ("Friday-AM", 5),
    "Friday-WorkingHours-Afternoon-PortScan": ("Friday-PM-PortScan", 6),
    "Friday-WorkingHours-Afternoon-DDos": ("Friday-PM-DDoS", 7),
}

# Capture window (date, start, end) of each day file. Only used to rebuild
# approximate timestamps for the MachineLearningCSV variant, which has none.
DAY_WINDOWS: dict[str, tuple[str, str, str]] = {
    "Monday": ("2017-07-03", "08:55", "17:02"),
    "Tuesday": ("2017-07-04", "08:53", "17:00"),
    "Wednesday": ("2017-07-05", "08:42", "17:10"),
    "Thursday-AM": ("2017-07-06", "08:59", "12:59"),
    "Thursday-PM": ("2017-07-06", "13:00", "17:04"),
    "Friday-AM": ("2017-07-07", "08:59", "12:59"),
    "Friday-PM-PortScan": ("2017-07-07", "13:00", "15:29"),
    "Friday-PM-DDoS": ("2017-07-07", "15:30", "17:02"),
}

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
RAW_LABEL_COL = "Label"          # original 15-value label (kept for analysis)
TARGET_COL = "family"            # 8 attack families (multi-class target)
BINARY_COL = "is_malicious"      # 0 = BENIGN, 1 = any attack
DAY_COL = "day"
DAY_ORDER_COL = "day_order"
ROW_ORDER_COL = "row_order"      # original row position inside its day file
TIMESTAMP_COL = "Timestamp"

# Identifiers / meta columns that must never be used as model features:
# they identify hosts or capture time rather than describing flow behaviour,
# so a model using them would memorise the testbed instead of learning attacks.
IDENTIFIER_COLS: list[str] = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    TIMESTAMP_COL,
]
META_COLS: list[str] = [RAW_LABEL_COL, TARGET_COL, BINARY_COL, DAY_COL, DAY_ORDER_COL, ROW_ORDER_COL]

# Columns where -1 is a CICFlowMeter sentinel ("no TCP window observed"),
# not an invalid negative value, so cleaning keeps it.
SENTINEL_NEGATIVE_COLS: list[str] = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
BENIGN = "BENIGN"
LABEL_TO_FAMILY: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Infiltration": "Rare",
    "Heartbleed": "Rare",
}
# Fixed class order -> integer codes 0..7 (BENIGN first so code 0 = benign).
CLASS_ORDER: list[str] = [
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "Bot",
    "Brute Force",
    "Web Attack",
    "Rare",
]
CLASS_TO_CODE: dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
# One-line meaning of each family (dataset table in the report and notebook).
FAMILY_MEANING: dict[str, str] = {
    "BENIGN": "normal traffic of the lab users (web, e-mail, FTP, SSH)",
    "DoS": "one host floods or starves the web server",
    "DDoS": "many hosts flood the web server (LOIC)",
    "PortScan": "Nmap probes that look for open ports",
    "Bot": "Ares botnet hosts talking to their C&C server",
    "Brute Force": "password guessing on FTP and SSH (Patator)",
    "Web Attack": "login guessing, XSS and SQL injection on a web app",
    "Rare": "internal infiltration and the Heartbleed memory leak",
}

# --------------------------------------------------------------------------
# Experiment constants
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5                 # StratifiedKFold folds for the baseline CV (step 6)
TUNING_CV_FOLDS = 3          # folds inside RandomizedSearchCV (step 7)
TUNING_SAMPLE_SIZE = 500_000
# Classes with fewer training rows than this are kept in full in the tuning sample.
MINORITY_KEEP_ALL_THRESHOLD = 50_000
SELECTION_SAMPLE_SIZE = 300_000   # train rows used for correlation / importance filters
VARIANCE_THRESHOLD = 0.0
CORRELATION_THRESHOLD = 0.95
IMPORTANCE_CUMULATIVE = 0.99      # keep features covering 99% of RF importance
OUTLIER_QUANTILE = 0.999          # upper cap for the scale-sensitive (LogReg) pipeline
DATASET_SAMPLE_ROWS = 30_000      # size of the committed dataset sample
SHAP_SAMPLE_ROWS = 2_000

# Fit diagnostics (report section 6). The validation slice is carved from the
# training split only; the test split is never touched.
DIAG_VALIDATION_SIZE = 0.20
CURVE_SAMPLE_ROWS = 200_000       # minority-preserving training subsample for learning/validation curves
CURVE_FOLDS = 3
LEARNING_CURVE_FRACTIONS: list[float] = [0.1, 0.25, 0.5, 0.75, 1.0]
DT_DEPTHS: list[int | None] = [2, 4, 6, 8, 10, 12, 15, 20, 30, None]
XGB_ROUNDS: list[int] = [10, 25, 50, 100, 150, 200, 300, 400, 600]
# Diagnosis rule: train macro F1 below UNDERFIT_TRAIN_F1 = underfit (high bias);
# otherwise a train - validation gap above OVERFIT_GAP = overfit (high variance),
# above SLIGHT_OVERFIT_GAP = slight overfit, else good fit.
UNDERFIT_TRAIN_F1 = 0.90
OVERFIT_GAP = 0.05
SLIGHT_OVERFIT_GAP = 0.02

# Parallelism: 8 CPUs on the build VM; leave headroom to avoid OOM.
N_JOBS: int = max(1, min(8, os.cpu_count() or 1))

MODEL_NAMES: list[str] = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline.

    Side effects:
        Creates the processed-data, sample, figures, results, models and model
        cache directories if they do not exist yet.
    """
    for d in (PROCESSED_DIR, SAMPLE_DIR, FIGURES_DIR, RESULTS_DIR, MODELS_DIR, MODEL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
