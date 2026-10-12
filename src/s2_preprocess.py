"""Preprocessing pipelines: imputation, scaling, and feature selection."""
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
    
    if impute_strategy == "knn":
        steps.append(("imputer", KNNImputer(n_neighbors=5)))
    else:
        steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))
        
    if scale:
        steps.append(("scaler", StandardScaler()))
        
    return Pipeline(steps=steps)


def build_feature_selector(random_state: int = RANDOM_SEED) -> SelectFromModel:
    """Selects top informative biomarkers using a random forest selector."""
    selector_estimator = RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    )
    return SelectFromModel(estimator=selector_estimator, threshold="median")
