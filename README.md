# Ovarian Cancer Detection
## SENG 474 Milestone Project: Cost-Sensitive Ovarian Cancer Classification
### Adam, Shaelynn, Rem, and Arunesh

## Intro
TODO

## Illustrative Sample
TODO

## Prerequisites
Python 3.11
Conda / Miniconda

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
### Train and evaluate
python main.py --mode all --model voting
### Train only
python main.py --mode train --model voting
### Evaluate only
python main.py --mode evaluate --model voting

## Results
TODO

## Citation
Mi, W., et al. (2020). Data for: Using Machine Learning to Predict Ovarian Cancer. Mendeley Data, V11, doi:10.17632/th7fztbrv9.11.
Kaggle Dataset: [Predict Ovarian Cancer](https://www.kaggle.com/datasets/saurabhshahane/predict-ovarian-cancer)

## License
Distributed under the MIT License. See LICENSE for details.
