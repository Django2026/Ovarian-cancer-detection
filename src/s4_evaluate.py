"""Step 4 - Evaluation: scores and charts for a trained model.

Input : true labels, predicted labels (0/1) and predicted cancer probabilities
        for the 70 held-out test patients (single 80/20 split).
Output: a dictionary of metrics (saved as JSON by main.py) and PNG charts in
        evaluation/reports/: confusion matrix, ROC curve, precision-recall
        curve and a 3D PCA view of the test patients.
Called by: main.py (--mode evaluate and --mode ensemble).

Note: these scores use the default 0.5 cutoff on one fixed split. The
project's proposed method (tuned cutoff, nested CV) is in s6_nested_cv.py.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.decomposition import PCA
from src.config import COST_FN, COST_FP, REPORTS_DIR


def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Computes total clinical misclassification cost: (FN * cost_fn) + (FP * cost_fp)."""
    # confusion_matrix(...).ravel() returns the four counts in this fixed order:
    # true negatives, false positives, false negatives, true positives
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn * cost_fn + fp * cost_fp)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Generates standard and cost-sensitive evaluation metrics.

    y_pred are 0/1 labels; y_prob (optional) are cancer probabilities, needed
    for the threshold-free scores ROC-AUC and PR-AUC.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Sensitivity (recall): share of cancer patients that were caught
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity: share of benign patients correctly cleared
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    results = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Sensitivity (Recall)": float(sensitivity),
        "Specificity": float(specificity),
        "Total Clinical Cost": compute_cost(y_true, y_pred),
    }

    if y_prob is not None:
        # ROC-AUC: chance a random cancer patient gets a higher score than a
        # random benign one (0.5 = guessing, 1.0 = perfect)
        results["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        # PR-AUC (average precision): summarises precision across all recall levels
        results["PR-AUC"] = float(average_precision_score(y_true, y_prob))

    return results

"""
This is based off of the training model function and how ground truth is set up
Calculates weights for each model by comparing their saved JSON predictions 
    against the ground truth y_test.csv.
    
    Args:
        results_dir: Directory where eval_results_{model_name}.json files are stored.
        ground_truth_dir: Directory where y_test.csv is saved.
        model_names: List of model names (e.g., ["rf", "xgb", "lr"]).
        prediction_key: Key inside the JSON metrics dict if predictions are nested.
        
    Returns:
        A list of weights (accuracy scores) corresponding to each model.
"""
def calculate_weights(results_dir: Path, ground_truth_dir: Path, model_names: list[str], prediction_key: str = "predictions") -> list[float]:
    # Used only by the exploratory weighted vote (--mode ensemble).
    # Note: these weights are computed on the TEST labels, so they should not be
    # used to report an unbiased ensemble score (see ensemble_optimization in s3).
    y_test_path = ground_truth_dir / "y_test.csv"
    y_test_df = pd.read_csv(y_test_path)
    true_labels = y_test_df.iloc[:, 0].tolist()
    weights = []
    #calculate raw accuracy for each model and use that as the weight
    for name in model_names:
        results_file = results_dir / f"eval_results_{name}.json"
        with open(results_file, 'r') as file:
            data = json.load(file)
        predictions = data.get(prediction_key, data) if isinstance(data, dict) else data
        if len(predictions) != len(true_labels):
            raise ValueError(f"Model '{name}' prediction count ({len(predictions)}) "f"does not match y_test count ({len(true_labels)}).")
        #accuracy = correct predictions / total samples
        correct_count = sum(p == t for p, t in zip(predictions, true_labels))
        accuracy = correct_count / len(true_labels)
        weights.append(max(0.0, accuracy))
    return weights


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Generates and saves a confusion matrix heatmap in evaluation/reports/.

    Rows = true class, columns = predicted class; the diagonal is correct.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign (0)", "Cancer (1)"],
        yticklabels=["Benign (0)", "Cancer (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({model_name.upper()})")
    plt.tight_layout()
    
    out_path = reports_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_roc_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Generates and saves a ROC curve plot in evaluation/reports/.

    Each point is one possible cutoff: x = false-alarm rate, y = recall.
    The dashed diagonal is a random classifier.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve ({model_name.upper()})")
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = reports_dir / f"roc_curve_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_3d_pca_plot(
    X_transformed: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Save a 3D PCA visualization of test samples, marking misclassifications.

    PCA squeezes the 49 features into 3 axes that keep as much variation as
    possible, so the patients can be drawn in 3D. For illustration only.
    """
    if X_transformed.shape[1] < 3:
        raise ValueError("3D PCA requires at least three input features.")

    components = PCA(n_components=3).fit_transform(X_transformed)

    # One colour per true class; circles = correct, X = misclassified

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label, color, class_name in [
        (0, "steelblue", "Benign"),
        (1, "darkorange", "Cancer"),
    ]:
        for correct, marker, status in [
            (True, "o", "Correct"),
            (False, "X", "Misclassified"),
        ]:
            mask = (y_true == label) & ((y_pred == y_true) == correct)
            ax.scatter(
                components[mask, 0],
                components[mask, 1],
                components[mask, 2],
                color=color,
                marker=marker,
                label=f"{class_name} - {status}",
                s=55,
            )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title(f"3D PCA of Test Samples ({model_name.upper()})")
    ax.legend()
    fig.tight_layout()

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"pca_3d_{model_name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_pr_curve_plot(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Generates and saves a precision-recall curve plot in evaluation/reports/.

    Each point is one possible cutoff: x = recall, y = precision (share of
    "cancer" predictions that were really cancer).
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)  # prevalence: precision of a random classifier

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR (AP = {ap:.3f})")
    plt.axhline(baseline, color="navy", lw=1.5, linestyle="--", label=f"Baseline ({baseline:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({model_name.upper()})")
    plt.legend(loc="lower left")
    plt.tight_layout()

    out_path = reports_dir / f"pr_curve_{model_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_model_comparison_plot(pipeline, X_test, y_true: np.ndarray, model_name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Saves ROC and PR curves comparing each ensemble member against the full ensemble.

    Only runs for "voting"/"stacking" models, which main.py currently skips.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    X_t = pipeline.named_steps["preprocessor"].transform(X_test)
    ensemble = pipeline.named_steps["classifier"]

    # Fitted members of the voting/stacking ensemble, plus the ensemble itself
    models = dict(ensemble.named_estimators_)
    models[f"{model_name} (ensemble)"] = ensemble

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, clf in models.items():
        prob = clf.predict_proba(X_t)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, prob)
        prec, rec, _ = precision_recall_curve(y_true, prob)
        is_ens = "ensemble" in name
        style = dict(lw=3 if is_ens else 1.5, linestyle="-" if is_ens else "--")
        ax_roc.plot(fpr, tpr, label=f"{name.upper()} (AUC = {roc_auc_score(y_true, prob):.3f})", **style)
        ax_pr.plot(rec, prec, label=f"{name.upper()} (AP = {average_precision_score(y_true, prob):.3f})", **style)

    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)", title="ROC: Members vs. Ensemble")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall: Members vs. Ensemble")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="lower left")
    fig.tight_layout()

    out_path = reports_dir / f"model_comparison_{model_name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path