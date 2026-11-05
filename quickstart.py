"""Quickstart: runs the full pipeline end to end with one command.

Usage (from the project folder, inside the conda environment):
    python quickstart.py

Steps run, in order (each one is a call to main.py):
    1. train    - train all four models on an 80% split, save to output/
    2. evaluate - test them on the 20% split, save scores and charts
    3. nested   - the proposed method: nested CV with a tuned cutoff
    4. explain  - SHAP charts for the Random Forest model
Stops with an error message if any step fails.
"""
import subprocess
import sys

steps = [
    ["--mode", "train", "--model", "all"],
    ["--mode", "evaluate", "--model", "all"],
    ["--mode", "nested", "--model", "all"],
    ["--mode", "explain", "--model", "rf"],
]

for args in steps:
    print("\n>>> python main.py " + " ".join(args))
    # sys.executable = the Python of the current environment, so the same
    # installed packages are used for every step
    result = subprocess.run([sys.executable, "main.py", *args])
    if result.returncode != 0:
        sys.exit(f"Step failed: {' '.join(args)}")

print("\nQuickstart finished. Results are in evaluation/results and evaluation/reports.")
