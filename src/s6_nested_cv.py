"""Step 6: Nested cross-validation with recall-constrained threshold tuning.

This is the project's proposed method (Milestone 3).

For every model, the 349 patients are split into K_OUTER folds. In each outer round:
  1. INNER LOOP (training patients only):
     a. Grid-search the model's hyperparameters with K_INNER-fold CV (scored by ROC-AUC).
     b. Using out-of-fold probabilities from the best settings, pick the decision
        threshold tau = the HIGHEST cutoff that still catches >= RECALL_FLOOR of the
        cancers. A higher cutoff means fewer false alarms, so this is the threshold
        with the fewest false positives that still meets the recall target.
  2. OUTER TEST: refit the best settings on all training patients, predict the
     held-out fold, and label a patient "cancer" if P(cancer) >= tau.
Test patients are never used to choose settings or tau, so the scores are not
optimistically biased. Results are averaged over the K_OUTER folds.

Labels follow s1_load.py: 1 = ovarian cancer (positive class), 0 = benign.
"""
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, RESULTS_DIR
from src.s1_load import load_raw_dataset
from src.s2_preprocess import build_preprocessor
from src.s3_ensemble import get_base_models

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
K_OUTER = 5          # outer folds: estimate how well the method generalizes
K_INNER = 3          # inner folds: choose hyperparameters and threshold
RECALL_FLOOR = 0.90  # must catch at least 90% of cancers

# Small hyperparameter grids, one per model (keys match get_base_models()).
# "classifier__" refers to the model step inside the Pipeline below.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "rf": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [4, 6, None],
    },
    "hgb": {
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 4],
    },
    "lr": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
    },
    "svm": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__gamma": ["scale", 0.01],
    },
}


def build_pipeline(model_name: str) -> Pipeline:
    """Preprocessing (median imputation + scaling) followed by the chosen model.

    Putting preprocessing inside the pipeline means it is re-fitted on the
    training patients of every fold, so no test information leaks in.
    """
    clf = get_base_models(random_state=RANDOM_SEED)[model_name]
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(impute_strategy="median", scale=True)),
        ("classifier", clf),
    ])


def select_threshold(y_true: np.ndarray, p_cancer: np.ndarray,
                     recall_floor: float = RECALL_FLOOR) -> float:
    """Return the highest threshold tau whose recall is still >= recall_floor.

    Every predicted probability is a candidate cutoff. Recall can only go up as
    tau goes down, so we scan from the highest candidate downwards and stop at
    the first one that meets the recall floor.
    """
    n_cancer = int(np.sum(y_true == 1))
    for tau in np.sort(np.unique(p_cancer))[::-1]:
        recall = np.sum((p_cancer >= tau) & (y_true == 1)) / n_cancer
        if recall >= recall_floor:
            return float(tau)
    return 0.0  # flag everyone (recall = 1.0) if nothing else meets the floor


def fold_metrics(y_true: np.ndarray, p_cancer: np.ndarray, tau: float) -> Dict[str, float]:
    """Confusion-matrix metrics for one outer test fold at threshold tau."""
    y_pred = (p_cancer >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": tau,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "roc_auc": roc_auc_score(y_true, p_cancer),
    }


def run_nested_cv(model_name: str, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
    """Nested CV for one model; returns one result dict per outer fold."""
    outer = StratifiedKFold(n_splits=K_OUTER, shuffle=True, random_state=RANDOM_SEED)
    inner = StratifiedKFold(n_splits=K_INNER, shuffle=True, random_state=RANDOM_SEED)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx].to_numpy()
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx].to_numpy()

        # 1a. Inner loop: tune hyperparameters on training patients only
        search = GridSearchCV(build_pipeline(model_name), PARAM_GRIDS[model_name],
                              scoring="roc_auc", cv=inner, n_jobs=-1)
        search.fit(X_train, y_train)

        # 1b. Inner loop: choose tau from out-of-fold training probabilities
        p_inner = cross_val_predict(clone(search.best_estimator_), X_train, y_train,
                                    cv=inner, method="predict_proba")[:, 1]
        tau = select_threshold(y_train, p_inner)

        # 2. Outer test: best model (already refit on all training data) + tau
        p_test = search.best_estimator_.predict_proba(X_test)[:, 1]
        row = {"model": model_name, "fold": fold, **fold_metrics(y_test, p_test, tau),
               "best_params": json.dumps(search.best_params_)}
        rows.append(row)
        print(f"  fold {fold}: tau={tau:.3f}  recall={row['recall']:.3f}  "
              f"FPR={row['fpr']:.3f}  AUC={row['roc_auc']:.3f}")
    return rows


def run_all(model_names: List[str]) -> pd.DataFrame:
    """Run nested CV for each model, print a summary and save CSV files."""
    X, y = load_raw_dataset(file_name="data_raw_1.xlsx")
    print(f"Loaded {X.shape[0]} patients, {X.shape[1]} features; "
          f"{int(y.sum())} cancer / {int((y == 0).sum())} benign")
    print(f"Outer folds={K_OUTER}, inner folds={K_INNER}, recall floor={RECALL_FLOOR}")

    all_rows = []
    for name in model_names:
        print(f"\nNested CV: {name.upper()}")
        all_rows.extend(run_nested_cv(name, X, y))

    per_fold = pd.DataFrame(all_rows)
    cols = ["threshold", "recall", "specificity", "fpr", "precision", "roc_auc"]
    summary = per_fold.groupby("model", sort=False)[cols].agg(["mean", "std"]).round(3)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(RESULTS_DIR / "nested_cv_per_fold.csv", index=False)
    summary.to_csv(RESULTS_DIR / "nested_cv_summary.csv")

    print("\nSUMMARY (mean and std over outer folds):")
    print(summary.to_string())
    print(f"\nSaved: {RESULTS_DIR / 'nested_cv_per_fold.csv'}")
    print(f"       {RESULTS_DIR / 'nested_cv_summary.csv'}")
    return summary