"""Step 7 - Milestone 4 evaluation: development/test protocol, RQ1-RQ3.

Protocol (Assignment 4 manual, section 4.5):
  1. Split the 349 patients once into an 80% development set (279) and a
     20% final test set (70), stratified, seed 42. This is the same split
     main.py uses, so all results line up.
  2. On the development set only, with the SAME 5 stratified folds for every
     model:
       - grid-search each model's hyperparameters by mean ROC-AUC (RQ2),
       - collect out-of-fold cancer probabilities with the best settings,
       - choose the cutoff tau: the highest value whose out-of-fold recall is
         >= rho (rho = 0.90 for the main comparison; 0.80-0.95 for RQ2).
  3. Refit each best model on the whole development set and evaluate it ONCE
     on the final test set:
       - baseline : the tuned model with the default cutoff 0.5
       - proposed : the same tuned model with the recall-constrained cutoff tau
     The only difference between baseline and proposed is the cutoff (RQ1).
  4. RQ3: SHAP explanations of the model with the best development-set AUC,
     computed on the final-test patients (background: 50 development patients).

The final test set is never used to choose settings, cutoffs or the model
that SHAP explains.

Called by: main.py (--mode experiment).
Outputs (evaluation/results/ and evaluation/reports/, prefixed "a4_"):
  a4_final_test.csv            RQ1 table: baseline vs proposed per model
  a4_rho_sensitivity.csv/.png  RQ2: effect of the recall floor rho (dev CV)
  a4_hyperparams.csv/.png      RQ2: CV AUC for every hyperparameter setting
  a4_confusion_matrices.png    RQ1: confusion matrices on the final test set
  a4_shap_importance.csv, shap_*_a4_<model>.png   RQ3
"""
import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED, REPORTS_DIR, RESULTS_DIR
from src.s1_load import load_raw_dataset
from src.s2_preprocess import build_preprocessor
from src.s3_ensemble import get_base_models
import shap
from src.s5_shap import make_predict_fn, compute_shap_values, save_shap_plots
from src.s6_nested_cv import select_threshold

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20                          # 80% development / 20% final test
N_FOLDS = 5                               # folds on the development set
RHO_MAIN = 0.90                           # recall floor for the main comparison
RHO_GRID = [0.80, 0.85, 0.90, 0.95]       # recall floors studied in RQ2
DEFAULT_CUTOFF = 0.5                      # scikit-learn's default (baseline)
MODELS = ["lr", "svm", "rf", "hgb"]
MODEL_LABELS = {"lr": "Logistic Regression", "svm": "SVM (RBF)",
                "rf": "Random Forest", "hgb": "Gradient Boosting"}

# Hyperparameter grids (RQ2). The first key is the parameter plotted on the
# x-axis; the second (if any) gives one line per value.
PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "lr":  {"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "svm": {"classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__gamma": ["scale", 0.001, 0.01, 0.1]},
    "rf":  {"classifier__max_depth": [2, 4, 6, 8, None],
            "classifier__n_estimators": [100, 300]},
    "hgb": {"classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [2, 3, 4]},
}


def build_pipeline(model_name: str) -> Pipeline:
    """Preprocessing (median fill + scaling) followed by the model.

    Inside cross-validation the whole pipeline is refitted on each training
    part, so preprocessing never sees the patients it is scored on.
    """
    clf = get_base_models(random_state=RANDOM_SEED)[model_name]
    return Pipeline([("preprocessor", build_preprocessor("median", True)),
                     ("classifier", clf)])


def metrics_at(y_true: np.ndarray, p_cancer: np.ndarray, cutoff: float) -> Dict:
    """Confusion-matrix counts and metrics when 'cancer' means p >= cutoff."""
    y_pred = (p_cancer >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "cutoff": float(cutoff),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "recall": recall,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "roc_auc": roc_auc_score(y_true, p_cancer),
    }


def fold_mean(y: np.ndarray, p: np.ndarray, folds, cutoff: float) -> Dict:
    """Average the metrics over the development folds (each fold's own patients)."""
    rows = [metrics_at(y[val], p[val], cutoff) for _, val in folds]
    keys = ["recall", "specificity", "fpr", "precision", "f1"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out.update({f"{k}_std": float(np.std([r[k] for r in rows])) for k in keys})
    return out


def run_evaluation() -> pd.DataFrame:
    """Run the whole Milestone 4 protocol and save all tables and charts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. One fixed development / final-test split (same as main.py) ----
    X, y = load_raw_dataset(file_name="data_raw_1.xlsx")
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    y_dev, y_test = y_dev.to_numpy(), y_test.to_numpy()
    print(f"Development set: {len(y_dev)} patients ({y_dev.sum()} cancer); "
          f"final test set: {len(y_test)} patients ({y_test.sum()} cancer)")

    # The same 5 folds are used for every model and every setting
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(cv.split(X_dev, y_dev))

    final_rows, rho_rows, hp_rows = [], [], []
    fitted, test_probs, dev_auc = {}, {}, {}

    for name in MODELS:
        print(f"\n== {MODEL_LABELS[name]} ==")

        # ---- 2a. Hyperparameter search by mean CV ROC-AUC (dev set only) ----
        search = GridSearchCV(build_pipeline(name), PARAM_GRIDS[name],
                              scoring="roc_auc", cv=folds, n_jobs=-1)
        search.fit(X_dev, y_dev)
        dev_auc[name] = float(search.best_score_)
        print(f"  best settings {search.best_params_}  CV AUC={search.best_score_:.3f}")
        for params, mean, std in zip(search.cv_results_["params"],
                                     search.cv_results_["mean_test_score"],
                                     search.cv_results_["std_test_score"]):
            # settings stored as text so None (= no depth limit) stays "None"
            hp_rows.append({"model": name,
                            **{k.replace("classifier__", ""): str(v) for k, v in params.items()},
                            "cv_auc_mean": mean, "cv_auc_std": std})

        # ---- 2b. Out-of-fold probabilities with the best settings ----
        p_oof = cross_val_predict(clone(search.best_estimator_), X_dev, y_dev,
                                  cv=folds, method="predict_proba")[:, 1]

        # ---- 2c. RQ2: effect of the recall floor rho (dev CV only) ----
        tau_main = None
        for rho in RHO_GRID:
            tau = select_threshold(y_dev, p_oof, rho)
            if rho == RHO_MAIN:
                tau_main = tau
            rho_rows.append({"model": name, "rho": rho, "tau": tau,
                             **fold_mean(y_dev, p_oof, folds, tau)})
        rho_rows.append({"model": name, "rho": "default 0.5", "tau": DEFAULT_CUTOFF,
                         **fold_mean(y_dev, p_oof, folds, DEFAULT_CUTOFF)})

        # ---- 3. Final test set, used once: baseline vs proposed ----
        model = search.best_estimator_          # already refitted on all of dev
        p_test = model.predict_proba(X_test)[:, 1]
        fitted[name], test_probs[name] = model, p_test
        for method, cutoff in [("baseline (0.5)", DEFAULT_CUTOFF),
                               (f"proposed (rho={RHO_MAIN})", tau_main)]:
            row = {"model": name, "method": method, **metrics_at(y_test, p_test, cutoff),
                   "best_params": json.dumps({k.replace("classifier__", ""): v
                                              for k, v in search.best_params_.items()})}
            final_rows.append(row)
            print(f"  {method:22s} cutoff={cutoff:.3f}  recall={row['recall']:.3f}  "
                  f"FP={row['FP']}  F1={row['f1']:.3f}")

    # ---- Save tables ----
    final = pd.DataFrame(final_rows)
    final.to_csv(RESULTS_DIR / "a4_final_test.csv", index=False)
    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(RESULTS_DIR / "a4_rho_sensitivity.csv", index=False)
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(RESULTS_DIR / "a4_hyperparams.csv", index=False)

    # ---- Charts ----
    plot_confusion_matrices(final)
    plot_rho_sensitivity(rho_df)
    plot_hyperparams(hp_df)

    # ---- 4. RQ3: SHAP for the model with the best development-set AUC ----
    best = max(dev_auc, key=dev_auc.get)
    print(f"\nRQ3: explaining {MODEL_LABELS[best]} (best development CV AUC "
          f"{dev_auc[best]:.3f}) on the {len(y_test)} final-test patients")
    run_shap(best, fitted[best], X_dev, X_test, y_test, test_probs[best])

    summary_cols = ["model", "method", "cutoff", "TP", "FP", "TN", "FN",
                    "recall", "specificity", "precision", "f1", "roc_auc"]
    print("\nFINAL TEST SET (RQ1):")
    print(final[summary_cols].round(3).to_string(index=False))
    return final


def run_shap(name, model, X_dev, X_test, y_test, p_test) -> None:
    """SHAP values on the final-test patients; background = 50 dev patients.

    Same explained output as s5_shap.py (P(cancer)) and the same background
    sampling, but the explainer is given a fixed seed: SHAP's permutation
    algorithm shuffles features randomly, and without a seed the values (and
    sometimes the feature ranking) change slightly between runs.
    """
    predict_cancer = make_predict_fn(model, X_dev.columns)
    background = shap.sample(X_dev, 50, random_state=RANDOM_SEED)
    explainer = shap.Explainer(predict_cancer, background, seed=RANDOM_SEED)
    sv = compute_shap_values(explainer, X_test)

    importance = (pd.DataFrame({"feature": X_test.columns,
                                "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
                  .sort_values("mean_abs_shap", ascending=False))
    importance.to_csv(RESULTS_DIR / "a4_shap_importance.csv", index=False)
    print("  top features by mean |SHAP|:",
          ", ".join(importance["feature"].head(5)))

    # Waterfall: the first final-test patient who truly has cancer
    idx = int(np.flatnonzero(y_test == 1)[0])
    paths = save_shap_plots(sv, f"a4_{name}", patient_idx=idx)
    print(f"  waterfall explains final-test patient #{idx} "
          f"(true label: cancer, predicted P(cancer) = {p_test[idx]:.2f})")
    for p in paths:
        print(f"  saved: {p}")


def plot_confusion_matrices(final: pd.DataFrame) -> None:
    """2 x 4 grid: rows = baseline / proposed, columns = models."""
    methods = list(dict.fromkeys(final["method"]))
    fig, axes = plt.subplots(len(methods), len(MODELS), figsize=(10, 4.8))
    for i, method in enumerate(methods):
        for j, name in enumerate(MODELS):
            r = final[(final["model"] == name) & (final["method"] == method)].iloc[0]
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
            ax = axes[i, j]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
            for (a, b), v in np.ndenumerate(cm):
                ax.text(b, a, int(v), ha="center", va="center", fontsize=12,
                        color="white" if v > cm.sum() / 3 else "black")
            ax.set_xticks([0, 1], ["Benign", "Cancer"], fontsize=8)
            ax.set_yticks([0, 1], ["Benign", "Cancer"], fontsize=8)
            if i == 0:
                ax.set_title(MODEL_LABELS[name], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{method}\nTrue", fontsize=8)
            if i == len(methods) - 1:
                ax.set_xlabel("Predicted", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "a4_confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_rho_sensitivity(rho_df: pd.DataFrame) -> None:
    """RQ2: false-positive rate and F1 vs the recall floor rho (dev CV means)."""
    d = rho_df[rho_df["rho"] != "default 0.5"].copy()
    d["rho"] = d["rho"].astype(float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for name in MODELS:
        m = d[d["model"] == name]
        ax1.plot(m["rho"], m["fpr"], marker="o", label=MODEL_LABELS[name])
        ax2.plot(m["rho"], m["f1"], marker="o", label=MODEL_LABELS[name])
    ax1.set(xlabel="Recall floor ρ", ylabel="False-positive rate (CV mean)")
    ax2.set(xlabel="Recall floor ρ", ylabel="F1 (CV mean)")
    ax1.legend(fontsize=7)
    for ax in (ax1, ax2):
        ax.set_xticks(RHO_GRID)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "a4_rho_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_hyperparams(hp_df: pd.DataFrame) -> None:
    """RQ2: mean CV ROC-AUC for each hyperparameter value, one panel per model."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 3))
    for ax, name in zip(axes, MODELS):
        keys = [k.replace("classifier__", "") for k in PARAM_GRIDS[name]]
        d = hp_df[hp_df["model"] == name].copy()
        xkey = keys[0]
        d[xkey] = d[xkey].astype(str)
        order = [str(v) for v in PARAM_GRIDS[name][f"classifier__{xkey}"]]
        groups = [(None, d)] if len(keys) == 1 else list(d.groupby(d[keys[1]].astype(str)))
        for g, sub in groups:
            sub = sub.set_index(xkey).reindex(order)
            ax.errorbar(range(len(order)), sub["cv_auc_mean"], yerr=sub["cv_auc_std"],
                        marker="o", capsize=2,
                        label=None if g is None else f"{keys[1]}={g}")
        ax.set_xticks(range(len(order)), order, fontsize=7)
        ax.set(title=MODEL_LABELS[name], xlabel=xkey, ylabel="CV ROC-AUC")
        ax.grid(alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "a4_hyperparams.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_evaluation()
