"""Single entrypoint script to train, evaluate, or run both."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    RANDOM_SEED,
    RESULTS_DIR,
    REPORTS_DIR,
    OUTPUT_DIR,
    BASE_DIR,
)
from src.s1_load import load_raw_dataset
from src.s2_preprocess import build_preprocessor
from src.s3_ensemble import build_voting_ensemble, build_stacking_ensemble, get_base_models
from src.s4_evaluate import (
    evaluate_predictions,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)

GROUND_TRUTH_DIR = BASE_DIR / "evaluation" / "ground_truth"


def get_model(ensemble_type: str):
    """Instantiates the specified model or ensemble."""
    if ensemble_type == "voting":
        return build_voting_ensemble(random_state=RANDOM_SEED)
    elif ensemble_type == "stacking":
        return build_stacking_ensemble(random_state=RANDOM_SEED)
    else:
        models = get_base_models(random_state=RANDOM_SEED)
        clf = models.get(ensemble_type)
        if clf is None:
            raise ValueError(f"Unknown model type: {ensemble_type}")
        return clf


def train_model(model_name: str = "rf", test_size: float = 0.2) -> Path:
    """Trains model pipeline, stores test partition in ground_truth, and saves model to output/."""
    print("=" * 60)
    print(f"TRAINING PHASE: {model_name.upper()}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading data from data/raw/data_raw_1.xlsx...")
    X, y = load_raw_dataset(file_name="data_raw_1.xlsx")
    print(f"      Total records: {X.shape[0]}, Features: {X.shape[1]}")

    # Hold out a test split and preserve ground truth
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    # Save test partition & labels for standalone evaluation
    X_test.to_csv(GROUND_TRUTH_DIR / "X_test.csv", index=False)
    y_test.to_csv(GROUND_TRUTH_DIR / "y_test.csv", index=False)
    print(f"      Saved test split ({len(X_test)} samples) to evaluation/ground_truth/")

    print(f"\n[2/3] Constructing pipeline and fitting on {len(X_train)} samples...")
    preprocessor = build_preprocessor(impute_strategy="median", scale=True)
    clf = get_model(model_name)
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    full_pipeline.fit(X_train, y_train)

    model_path = OUTPUT_DIR / f"model_{model_name}.joblib"
    joblib.dump(full_pipeline, model_path)
    print(f"\n[3/3] Trained model artifact saved to: {model_path}")
    return model_path


def evaluate_model(model_name: str = "rf"):
    """Loads saved model artifact and runs evaluation on ground_truth test data."""
    print("=" * 60)
    print(f"EVALUATION PHASE: {model_name.upper()}")
    print("=" * 60)

    model_path = OUTPUT_DIR / f"model_{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run with '--mode train' first.")

    X_test_path = GROUND_TRUTH_DIR / "X_test.csv"
    y_test_path = GROUND_TRUTH_DIR / "y_test.csv"
    if not (X_test_path.exists() and y_test_path.exists()):
        raise FileNotFoundError("Ground truth test data missing. Run with '--mode train' first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading model artifact and ground truth test set...")
    pipeline = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze().to_numpy()

    print("\n[2/3] Running inference...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = evaluate_predictions(y_test, y_pred, y_prob)

    print("\n" + "-" * 40)
    print(f"TEST METRICS ({model_name.upper()}):")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    # Save metrics JSON
    results_file = RESULTS_DIR / f"eval_results_{model_name}.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate figures in evaluation/reports/
    print("\n[3/3] Generating visual reports...")
    cm_plot = save_confusion_matrix_plot(y_test, y_pred, model_name=model_name)
    print(f"      Confusion matrix: {cm_plot}")

    if y_prob is not None:
        roc_plot = save_roc_curve_plot(y_test, y_prob, model_name=model_name)
        print(f"      ROC Curve:        {roc_plot}")

    print(f"\nArtifacts generated in:\n  - {results_file}\n  - {REPORTS_DIR}")

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
    y_test_path = ground_truth_dir / "y_test.csv"
    y_test_df = pd.read_csv(y_test_path)
    true_labels = y_test_df.iloc[:, 0].tolist()
    weights = []
    for name in model_names:
        results_file = results_dir / f"eval_results_{name}.json"
        with open(results_file, 'r') as file:
            data = json.load(file)
        predictions = data.get(prediction_key, data) if isinstance(data, dict) else data
        if len(predictions) != len(true_labels):
            raise ValueError(f"Model '{name}' prediction count ({len(predictions)}) "f"does not match y_test count ({len(true_labels)}).")
        correct_count = sum(p == t for p, t in zip(predictions, true_labels))
        accuracy = correct_count / len(true_labels)
        weights.append(max(0.0, accuracy))
        return weights

"""
Loads individual model result JSON files, extracts their predictions, 
and calculates the weighted ensemble decision.
    Args:
    results_dir: The directory (Path object) where results are stored.
    model_names: List of model names corresponding to the files.
    weights: List of numerical weights based on each model's past accuracy.
    prediction_key: The dictionary key inside metrics where the boolean prediction is stored, change string to key used to store bool pred inside metrics dict     
    threshold: The decision threshold (default is 0.5 for weighted majority).
        
    Returns:
        The final boolean classification (True or False).
"""
def ensemble_optimization(results_dir: Path, model_names: list[str], weights: list[float], prediction_key: str = "prediction", threshold: float = .5 ) -> bool:
    if len(model_names) != len(weights):
        raise ValueError("Model names and weights must be equal")
    
    predictions = []
    #Load result files
    for name in model_names:
        results_file = results_dir / f"eval_results_{name}.json"
        with open(results_file, 'r') as file:
            metrics = json.load(file)
        #extract predictions fron metrics dict
        pred = metrics.get(prediction_key)
        if pred is None:
            raise KeyError(f"Could not find key '{prediction_key}' in {results_file.name}")
        predictions.append(pred)

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total weight must be greater then 0")
    #normalize weights
    normalized_weights = [w / total_weight for w in weight]
    #sum weights of classifiers that pred true
    weighted_score = sum(w for pred, w in zip(predictions, normalized_weights) if pred)
    return weighted_score >= threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ovarian Cancer Detection ML Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "evaluate", "all"],
        help="Action to perform: 'train', 'evaluate', or 'all' (default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["voting", "stacking", "rf"],
        help="Model or ensemble architecture.",
    )
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train_model(model_name=args.model)
    if args.mode in ("evaluate", "all"):
        evaluate_model(model_name=args.model)
