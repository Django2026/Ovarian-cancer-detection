"""Model definitions and ensemble construction."""
import json
from pathlib import Path
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from src.config import RANDOM_SEED

def get_base_models(random_state: int = RANDOM_SEED) -> Dict[str, BaseEstimator]:
    """Returns base classifiers. Reduced to Random Forest for testing."""
    return {
        "rf": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

def build_voting_ensemble(
    voting: str = "soft", random_state: int = RANDOM_SEED
) -> BaseEstimator:
    """
    Constructs a voting ensemble.
    For testing with RF only, returns the standalone RF classifier
    to avoid single-estimator VotingClassifier overhead.
    """
    base_models = get_base_models(random_state=random_state)
    estimators = [(name, model) for name, model in base_models.items()]

    if len(estimators) == 1:
        return estimators[0][1]

    return VotingClassifier(estimators=estimators, voting=voting)


def build_stacking_ensemble(random_state: int = RANDOM_SEED) -> BaseEstimator:
    """
    Constructs a stacking ensemble.
    For testing with RF only, returns the standalone RF classifier.
    """
    base_models = get_base_models(random_state=random_state)
    estimators = [(name, model) for name, model in base_models.items()]

    if len(estimators) == 1:
        return estimators[0][1]

    meta_learner = LogisticRegression(class_weight="balanced", random_state=random_state)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
    )

def ensemble_optimization(
    results_dir: Path,
    model_names: list[str],
    weights: list[float],
    prediction_key: str = "prediction",
    threshold: float = 0.5,
) -> bool:
    """Return the weighted majority decision from model result JSON files.

    Args:
        results_dir: Directory containing the model result files.
        model_names: Model names corresponding to the result files.
        weights: Weights assigned to each model.
        prediction_key: Key containing each model's boolean prediction.
        threshold: Decision threshold for the weighted score.
    """
    if len(model_names) != len(weights):
        raise ValueError("Model names and weights must be equal")

    predictions = []
    for name in model_names:
        results_file = results_dir / f"eval_results_{name}.json"
        with open(results_file, "r") as file:
            metrics = json.load(file)

        pred = metrics.get(prediction_key)
        if pred is None:
            raise KeyError(f"Could not find key '{prediction_key}' in {results_file.name}")
        predictions.append(pred)

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total weight must be greater than 0")

    normalized_weights = [weight / total_weight for weight in weights]
    weighted_score = sum(
        weight for pred, weight in zip(predictions, normalized_weights) if pred
    )
    return weighted_score >= threshold
