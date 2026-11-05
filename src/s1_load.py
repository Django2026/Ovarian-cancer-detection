"""Step 1 - Load and clean the raw data.

Input : an Excel file in data/raw/ (default: data_raw_1.xlsx, 349 patients).
Output: X, a table of 49 numeric features (missing values left as NaN),
        and y, the labels (1 = cancer, 0 = benign).
Called by: main.py (train), s5_shap.py and s6_nested_cv.py.

Missing values are NOT filled here; that happens in s2_preprocess.py, inside
each training fold, so that test patients never influence the filling.
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from src.config import RAW_DATA_DIR, TARGET_COL, LABEL_MAPPING


def load_raw_dataset(
    file_name: str = "data_raw_1.xlsx",
    raw_dir: Path = RAW_DATA_DIR,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw Excel file and returns feature DataFrame X and binary target Series y.

    Args:
        file_name: Excel file name in data/raw
        raw_dir: Directory containing raw data files
        target_col: Name of the target column

    Returns:
        X (pd.DataFrame): Biomarker features
        y (pd.Series): Binary labels (1 = Cancer, 0 = Benign)
    """
    # Stop early with a clear message if the data file is missing
    path = raw_dir / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please check data/raw/ or see data/README.md"
        )

    # Read the spreadsheet; strip stray spaces from column names (e.g. "CA125 ")
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {file_name}")

    # Separate the label from the features.
    # Map target: 0 (cancer) -> 1, 1 (benign) -> 0, so that cancer is the positive class
    y = df[target_col].map(LABEL_MAPPING)
    X = df.drop(columns=[target_col])

    # Convert all feature columns to numeric, coercing text/sensor artifacts (e.g., '<0.1') to NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are completely empty / all NaN
    X = X.dropna(axis=1, how="all")

    # Drop rows where target is missing or unrecognized
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)

    # Drop non-predictive identifier columns (here: SUBJECT_ID), leaving 49 features
    id_cols = [c for c in X.columns if "id" in c.lower() or "patient" in c.lower()]
    if id_cols:
        X = X.drop(columns=id_cols)

    return X, y
