# Ovarian Cancer Detection
## SENG 474 Milestone Project: Cost-Sensitive Ovarian Cancer Classification
### Adam, Shaelynn, Rem, and Arunesh

## Intro
We classify ovarian cancer vs. benign tumours from 49 routine blood-test values
(349 patients; Mi et al., 2020). Existing studies use a fixed 0.5 cutoff and may tune
settings on test data. Our method uses nested cross-validation and picks each model's
cutoff so that at least 90% of cancers are caught with the fewest false alarms.
We compare Logistic Regression, an RBF SVM, Random Forest and Gradient Boosting.

## Illustrative Sample
A patient whose model probability of cancer is 0.45 would be called benign with the
default 0.5 cutoff. Our tuned Random Forest cutoff (about 0.39) flags her as cancer.

## Prerequisites
- Python 3.11
- Conda / Miniconda
- Without Conda you can run: pip install -r requirements.txt

## Installation Guide (Windows)
### Install Python 3.11
https://www.python.org/downloads/
### Install Miniconda (via PowerShell)
winget install --id Anaconda.Miniconda3 -e
conda --version
- If conda not found: Start Menu -> Anaconda Prompt -> conda init powershell
- You may then need to run this in PowerShell to allow: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
- **Alternatively**, work directly within Anaconda Prompt from this point forward
### Set up environment (via PowerShell or Anaconda Prompt, from within repo)
conda env create -f environment.yml
conda activate ovarian-cancer-ml

## Quick Start (from within environment; see above)
### Run everything
python quickstart.py
### Train and evaluate
python main.py --mode all --model rf
### Train only
python main.py --mode train --model rf
### Evaluate only
python main.py --mode evaluate --model rf

### Shap testing 
python main.py --mode explain --model rf 

### Nested cross-validation (proposed method)
python main.py --mode nested --model all

## Results
Nested 5-fold cross-validation, recall floor 90% (mean over folds):

| Model | Cutoff | Recall | False-positive rate | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 0.39 | 0.92 | 0.20 | 0.95 |
| Gradient Boosting | 0.26 | 0.90 | 0.19 | 0.95 |
| SVM (RBF) | 0.24 | 0.89 | 0.41 | 0.88 |
| Logistic Regression | 0.20 | 0.88 | 0.42 | 0.87 |

## Citation
Mi, W., et al. (2020). Data for: Using Machine Learning to Predict Ovarian Cancer. Mendeley Data, V11, doi:10.17632/th7fztbrv9.11.
Kaggle Dataset: [Predict Ovarian Cancer](https://www.kaggle.com/datasets/saurabhshahane/predict-ovarian-cancer)

## License
Distributed under the MIT License. See LICENSE for details.
