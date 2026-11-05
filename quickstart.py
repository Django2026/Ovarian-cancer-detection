"""Quickstart: runs the full pipeline end to end with one command.

Usage (from the project folder, inside the conda environment):
    python quickstart.py
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
    result = subprocess.run([sys.executable, "main.py", *args])
    if result.returncode != 0:
        sys.exit(f"Step failed: {' '.join(args)}")

print("\nQuickstart finished. Results are in evaluation/results and evaluation/reports.")