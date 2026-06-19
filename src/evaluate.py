"""
evaluate.py
Standalone evaluation script. Run this after training to:
  - Print a full metrics report
  - Check the fairness recall gap
  - Fail with exit code 1 if metrics fall below thresholds
    (used in GitHub Actions CI to gate model pushes)

Thresholds are intentionally conservative — they should be set based on
the actual model's real performance, not aspirational targets.
"""

import json
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report

from preprocess import run_preprocessing

DATA_PATH   = "data/heart999.csv"
MODEL_PATH  = "models/rf_model.joblib"
METRICS_PATH = "models/metrics.json"

# CI gate thresholds — set these after your first real training run
MIN_OVERALL_F1     = 0.80   # fail CI if overall F1 drops below this
MIN_FEMALE_RECALL  = 0.70   # fail CI if female recall drops below this
MAX_RECALL_GAP     = 0.25   # fail CI if |male_recall - female_recall| exceeds this


def evaluate():
    print("Loading model and preprocessor...")
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    print("\n=== Stored Metrics (from last training run) ===")
    print(json.dumps(metrics, indent=2))

    # Re-run on held-out test set to confirm stored metrics are reproducible
    print("\n--- Re-running preprocessing to verify metrics ---")
    _, X_test, _, y_test, _, _ = run_preprocessing(DATA_PATH)
    y_pred = model.predict(X_test)

    print("\n=== Classification Report (full test set) ===")
    print(classification_report(y_test, y_pred, target_names=["No CVD", "CVD"]))

    # CI gate checks
    failures = []

    overall_f1 = metrics["overall"]["f1"]
    if overall_f1 < MIN_OVERALL_F1:
        failures.append(
            f"Overall F1 {overall_f1} below threshold {MIN_OVERALL_F1}"
        )

    female_recall = metrics["by_gender"]["female"]["recall"]
    if female_recall < MIN_FEMALE_RECALL:
        failures.append(
            f"Female recall {female_recall} below threshold {MIN_FEMALE_RECALL}"
        )

    recall_gap = metrics["fairness"]["recall_gap_male_female"]
    if recall_gap > MAX_RECALL_GAP:
        failures.append(
            f"Recall gap {recall_gap} exceeds max {MAX_RECALL_GAP}"
        )

    if failures:
        print("\n❌ CI GATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✅ All CI gates passed.")
        print(f"  Overall F1     : {overall_f1} >= {MIN_OVERALL_F1}")
        print(f"  Female recall  : {female_recall} >= {MIN_FEMALE_RECALL}")
        print(f"  Recall gap     : {recall_gap} <= {MAX_RECALL_GAP}")


if __name__ == "__main__":
    evaluate()
