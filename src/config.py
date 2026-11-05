"""Configuration: every shared path and constant lives here.

Change a setting in this one file and every step (s1-s6) picks it up.
Imported by: all modules in src/ and main.py.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project folder, so the code runs on any computer)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent      # the project folder
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"                         # input Excel files (on GitHub)
PROCESSED_DATA_DIR = DATA_DIR / "processed"             # reserved; not used yet
OUTPUT_DIR = BASE_DIR / "output"                        # trained models (.joblib), not on GitHub
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"       # charts (.png), not on GitHub
RESULTS_DIR = BASE_DIR / "evaluation" / "results"       # scores (.json/.csv), not on GitHub

# ---------------------------------------------------------------------------
# Modeling defaults
# ---------------------------------------------------------------------------
RANDOM_SEED = 42      # fixed seed so every run gives identical results
TARGET_COL = "TYPE"   # name of the label column in the Excel file

# Label mapping: Original is 0 = Ovarian Cancer (OC), 1 = Benign (BOT)
# For ML standard positive class: 1 = Malignant/Cancer, 0 = Benign
# (so "recall" and "positive" always refer to cancer)
LABEL_MAPPING = {0: 1, 1: 0}

# Cost-sensitive weights (Cost of False Negative vs False Positive in cancer screening)
# e.g., Missing cancer (FN) is penalized significantly more than a false alarm (FP)
# Used only to report "Total Clinical Cost" in s4_evaluate.py; it does not change training.
COST_FN = 10.0
COST_FP = 1.0
