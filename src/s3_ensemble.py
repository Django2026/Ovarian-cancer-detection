"""Step 3 - Model definitions (and ensemble builders).

Defines the four classifiers the project compares, each with
class_weight="balanced" so cancer and benign patients count equally:
    "rf"  - Random Forest: averages many decision trees.
    "hgb" - Histogram Gradient Boosting: trees built one after another,
            each correcting the previous ones.
    "lr"  - Logistic Regression: a simple, interpretable linear baseline.
    "svm" - Support Vector Machine with an RBF kernel: a curved boundary.
Called by: main.py (train) and s6_nested_cv.py (which tunes these settings).

Also contains voting/stacking ensemble builders and a weighted-vote helper,
which are not part of the Milestone 3 method (see notes on each function).
"""
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
from sklearn.svm import SVC
from src.config import RANDOM_SEED

def get_base_models(random_state: int = RANDOM_SEED) -> Dict[str, BaseEstimator]:
    """Return the four base classifiers, keyed by their short names.

    The settings below are starting values. In s6_nested_cv.py some of them
    (e.g. tree depth, C, gamma) are tuned by grid search on training data.
    """
    return {
        # Random Forest: 100 trees, each at most 6 levels deep
        "rf": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,               # use all CPU cores
        ),
        # Gradient boosting: 200 small trees (depth 4), learning rate 0.05,
        # with L2 regularization to reduce overfitting
        "hgb": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
        # Logistic Regression: C controls regularization strength (smaller = simpler
        # model); max_iter is a safety limit so the solver has time to converge
        "lr": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        # SVM: RBF kernel allows a curved boundary; gamma controls how curved.
        # probability=True gives a cancer probability (Platt scaling), which the
        # threshold tuning in s6 needs.
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
    }

def build_voting_ensemble(
    voting: str = "soft", random_state: int = RANDOM_SEED
) -> BaseEstimator:
    """
    Constructs a voting ensemble.
    For testing with RF only, returns the standalone RF classifier
    to avoid single-estimator VotingClassifier overhead.

    "soft" voting averages the models' predicted probabilities.
    Note: main.py currently skips "voting" when training and evaluating.
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

    The four base models each predict; a logistic-regression "meta-learner"
    then learns how to combine their predictions (using 5-fold CV internally).
    Note: main.py currently skips "stacking" when training and evaluating.
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

def ensemble_optimization(results_dir: Path, model_names: list[str], weights: list[float], prediction_key: str = "prediction", threshold: float = 0.5,) -> bool:
    """Return the weighted majority decision from model result JSON files.

    Args:
        results_dir: Directory containing the model result files.
        model_names: Model names corresponding to the result files.
        weights: Weights assigned to each model.
        prediction_key: Key containing each model's boolean prediction.
        threshold: Decision threshold for the weighted score.

    Known limitations (exploratory code, not part of the Milestone 3 method):
      - Each "prediction" loaded below is a list of all 70 test predictions,
        so `if pred` is True for every model and the result is always True.
        A per-patient vote would need to loop over patients.
      - The weights come from accuracy on the test set (see
        calculate_weights in s4_evaluate.py), so test data influences the
        ensemble. A fair version would set weights on training data only.
    """
    if len(model_names) != len(weights):
        raise ValueError("Model names and weights must be equal")
    #load prediction list for all models
    predictions = []
    for name in model_names:
        results_file = results_dir / f"eval_results_{name}.json"
        with open(results_file, "r") as file:
            metrics = json.load(file)

        pred = metrics.get(prediction_key)
        if pred is None:
            raise KeyError(f"Could not find key '{prediction_key}' in {results_file.name}")
        predictions.append(pred)
    #normalize weights so they add up to 1
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total weight must be greater than 0")

    normalized_weights = [weight / total_weight for weight in weights]
    # Add up the weights of the models that "voted" cancer
    weighted_score = sum(
        weight for pred, weight in zip(predictions, normalized_weights) if pred
    )
    # Cancer if the weighted vote reaches the threshold (0.5 = weighted majority)
    return weighted_score >= threshold
