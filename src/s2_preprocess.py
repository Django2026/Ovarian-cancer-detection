"""Step 2 - Preprocessing: fill missing values and scale features.

Input : nothing directly; it returns a scikit-learn Pipeline that is later
        placed in front of a model.
Output: a Pipeline of (1) an imputer that fills missing values and
        (2) a scaler that puts every feature on the same scale.
Called by: main.py (train) and s6_nested_cv.py.

Why a Pipeline? It is fitted on training patients only. The medians, means
and standard deviations it learns are then re-used, unchanged, on test
patients, so no test information leaks into training.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_SEED


def build_preprocessor(impute_strategy: str = "median", scale: bool = True) -> Pipeline:
    """
    Constructs a leakage-free preprocessing pipeline.

    Args:
        impute_strategy: 'median', 'mean', or 'knn'
        scale: Whether to apply standard scaling
    """
    steps = []

    # 1. Fill missing values. The project uses "median": robust to the very
    #    large outliers common in blood-test values.
    if impute_strategy == "knn":
        steps.append(("imputer", KNNImputer(n_neighbors=5)))
    else:
        steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))

    # 2. Standardise each feature: z = (x - mean) / std. Needed by logistic
    #    regression and the SVM, whose results depend on feature scale.
    if scale:
        steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=steps)


def build_feature_selector(random_state: int = RANDOM_SEED) -> SelectFromModel:
    """Selects top informative biomarkers using a random forest selector.

    Note: not currently used anywhere in the pipeline; kept for possible
    future experiments.
    """
    selector_estimator = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    )
    return SelectFromModel(estimator=selector_estimator, threshold="median")
