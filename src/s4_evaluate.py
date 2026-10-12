"""Evaluation module for metrics, cost-sensitive scoring, and visualizations."""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from src.config import COST_FN, COST_FP, REPORTS_DIR


def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    """Computes total clinical misclassification cost: (FN * cost_fn) + (FP * cost_fp)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn * cost_fn + fp * cost_fp)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Generates standard and cost-sensitive evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
        results["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        results["PR-AUC"] = float(average_precision_score(y_true, y_prob))

    return results


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """Generates and saves a confusion matrix heatmap in evaluation/reports/."""
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
    """Generates and saves a ROC curve plot in evaluation/reports/."""
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
