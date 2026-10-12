"""Configuration constants and paths."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# Modeling defaults
RANDOM_SEED = 42
TARGET_COL = "TYPE"

# Label mapping: Original is 0 = Ovarian Cancer (OC), 1 = Benign (BOT)
# For ML standard positive class: 1 = Malignant/Cancer, 0 = Benign
LABEL_MAPPING = {0: 1, 1: 0}

# Cost-sensitive weights (Cost of False Negative vs False Positive in cancer screening)
# e.g., Missing cancer (FN) is penalized significantly more than a false alarm (FP)
COST_FN = 10.0
COST_FP = 1.0
