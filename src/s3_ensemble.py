"""Model definitions and ensemble construction."""
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestClassifier,
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
        )
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
