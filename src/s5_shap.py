"""Step 5 - Explainability with SHAP.

SHAP explains WHY a model gave a patient a certain cancer probability: it
splits the prediction into a push from each feature (e.g. "high CA125 pushed
the risk up by +0.20"), measured against an average (background) patient.

Input : a trained model saved by main.py (output/model_<name>.joblib), the 70
        test patients (evaluation/ground_truth/X_test.csv) and 50 training
        patients used as the background.
Output: three charts in evaluation/reports/:
        beeswarm  - every test patient's feature pushes, most important on top
        bar       - average importance of the top 15 features
        waterfall - step-by-step breakdown for one patient
Called by: main.py (--mode explain); runs on one model at a time.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

from src.config import OUTPUT_DIR, BASE_DIR, RANDOM_SEED, REPORTS_DIR
from src.s1_load import load_raw_dataset

GROUND_TRUTH_DIR = BASE_DIR / "evaluation" / "ground_truth"

def load_model_and_data(model_name: str = "rf"):
    """Load a saved model (preprocessing + classifier) and the test patients."""

    model_path = OUTPUT_DIR / f"model_{model_name}.joblib"         

    pipeline = joblib.load(model_path)             

    Xtest = pd.read_csv(GROUND_TRUTH_DIR / "X_test.csv")              

    return pipeline, Xtest

def make_predict_fn(pipeline, columns):
    """Wrap the model so SHAP gets one number per patient: P(cancer)."""

    # Find which output column holds the cancer probability (label 1 = cancer)
    cancer_col = list(pipeline.classes_).index(1)

    def predict_cancer(X):

        # SHAP passes plain arrays; turn them back into a table with the
        # original column names, which the saved pipeline expects
        X_df = pd.DataFrame(X, columns=columns)

        probs = pipeline.predict_proba(X_df)

        return probs[:, cancer_col]

    return predict_cancer

#loads the training data for the SHAP explainer
def load_train_data(test_size: float = 0.2):
    """Rebuild the same 80% training split used in main.py.

    Same seed and stratification, so these are exactly the patients the
    model was trained on (no test patients in the background).
    """

    X, y = load_raw_dataset(file_name="data_raw_1.xlsx")

    X_train, _, _, _ = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)
    return X_train

#creates a SHAP explainer
#Picking 50 random samples from the training data, more than 50 is slower but more accurate 
def build_explainer(pipeline, X_train, n_background: int = 50):
    """Create a SHAP explainer that compares each patient to a background
    of n_background training patients (the "average patient")."""

    predict_cancer = make_predict_fn(pipeline, X_train.columns)

    background = shap.sample(X_train, n_background, random_state=RANDOM_SEED)

    return shap.Explainer(predict_cancer, background)

def compute_shap_values(explainer, Xtest):
    """One SHAP value per patient per feature: that feature's push on P(cancer)."""

    shap_values = explainer(Xtest)

    return shap_values

def save_shap_plots(sv, model_name: str, patient_idx: int = 0, top_n: int = 15):
    """Save beeswarm, bar and waterfall charts; return their file paths.

    patient_idx picks which test patient the waterfall explains;
    top_n is how many features each chart shows.
    """
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = []

    
    shap.plots.beeswarm(sv, max_display=top_n, show=False)

    path = REPORTS_DIR / f"shap_beeswarm_{model_name}.png"

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    paths.append(path)

   
    shap.plots.bar(sv, max_display=top_n, show=False)

    path = REPORTS_DIR / f"shap_bar_{model_name}.png"

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    paths.append(path)

    
    shap.plots.waterfall(sv[patient_idx], max_display=top_n, show=False)

    path = REPORTS_DIR / f"shap_waterfall_patient{patient_idx}_{model_name}.png"

    plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.close()

    paths.append(path)

    return paths

# Lets the file be run on its own for testing: python -m src.s5_shap
if __name__ == "__main__":

    pipeline, Xtest = load_model_and_data("rf")

    X_train = load_train_data()

    explainer = build_explainer(pipeline, X_train)

    sv = compute_shap_values(explainer, Xtest)

    for p in save_shap_plots(sv, "rf"):

        print("saved:", p)

