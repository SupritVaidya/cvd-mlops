"""
train.py
Trains the selected model: Random Forest with SMOTE female balancing.

Why this model and this bias strategy?
- Random Forest beat Logistic Regression on the baseline comparison in the notebook.
- SMOTE on the female subgroup (not the full dataset) was chosen because the
  dataset is 79% male. Applying SMOTE only to the minority gender group avoids
  inflating the majority group and keeps the synthetic data grounded in real
  female physiology.
- Full-feature set (all encoded columns) was retained because the notebook's
  reduced-feature experiments (Pushing-4, Top-4, Metabolic) traded accuracy
  for interpretability — not appropriate for a production system where we need
  the full signal.

MLflow:
  MLflow is an open-source platform for tracking ML experiments. Every training
  run logs its parameters (e.g. n_estimators), metrics (accuracy, F1 per gender),
  and the serialised model artifact. This means you can compare runs, reproduce
  any past result, and see exactly what changed between experiments — which is
  what a production ML team does instead of manually tracking results in a notebook.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocess import run_preprocessing, save_preprocessor

DATA_PATH   = "data/heart999.csv"
MODEL_PATH  = "models/rf_model.joblib"
METRICS_PATH = "models/metrics.json"
MLFLOW_URI  = "sqlite:///mlruns.db"   # SQLite locally; swap for remote URI in production

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}


def apply_smote_female(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE only to the female subgroup to balance HeartDisease classes
    within that group, then recombine with the male data.
    Matches the notebook's SMOTE cell exactly.
    """
    sex_col = "Sex_M"   # after one-hot encoding, Sex_M=0 means Female

    X_male   = X_train[X_train[sex_col] == 1]
    y_male   = y_train.loc[X_male.index]
    X_female = X_train[X_train[sex_col] == 0]
    y_female = y_train.loc[X_female.index]

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_female_smote, y_female_smote = smote.fit_resample(X_female, y_female)

    X_female_smote = pd.DataFrame(X_female_smote, columns=X_female.columns)
    y_female_smote = pd.Series(y_female_smote)

    X_balanced = pd.concat([X_male, X_female_smote], axis=0).reset_index(drop=True)
    y_balanced = pd.concat([y_male, y_female_smote], axis=0).reset_index(drop=True)

    # Shuffle
    idx = X_balanced.sample(frac=1, random_state=42).index
    X_balanced = X_balanced.loc[idx].reset_index(drop=True)
    y_balanced = y_balanced.loc[idx].reset_index(drop=True)

    print(f"After SMOTE — train size: {len(X_balanced)}")
    print(f"  Male: {(X_balanced[sex_col]==1).sum()}, Female: {(X_balanced[sex_col]==0).sum()}")
    return X_balanced, y_balanced


def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute overall + gender-stratified metrics. These go to MLflow and metrics.json."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    overall = {
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    gender_metrics = {}
    sex_col = "Sex_M"
    for val, label in [(1, "male"), (0, "female")]:
        mask   = X_test[sex_col] == val
        y_g    = y_test[mask]
        y_pred_g = model.predict(X_test[mask])
        gender_metrics[label] = {
            "n_samples": int(mask.sum()),
            "accuracy" : round(accuracy_score(y_g, y_pred_g), 4),
            "precision": round(precision_score(y_g, y_pred_g, zero_division=0), 4),
            "recall"   : round(recall_score(y_g, y_pred_g, zero_division=0), 4),
            "f1"       : round(f1_score(y_g, y_pred_g, zero_division=0), 4),
        }

    # Fairness gap: difference in recall between male and female
    recall_gap = round(
        abs(gender_metrics["male"]["recall"] - gender_metrics["female"]["recall"]), 4
    )

    return {
        "overall": overall,
        "by_gender": gender_metrics,
        "fairness": {"recall_gap_male_female": recall_gap},
    }


def train():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("cvd-prediction")

    print("--- Preprocessing ---")
    X_train, X_test, y_train, y_test, feature_columns, preprocessor_state = \
        run_preprocessing(DATA_PATH)

    print("--- SMOTE (female subgroup) ---")
    X_train_bal, y_train_bal = apply_smote_female(X_train, y_train)

    print("--- Training RandomForest ---")
    model = RandomForestClassifier(**RF_PARAMS)

    with mlflow.start_run():
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("smote_strategy", "female_subgroup_only")
        mlflow.log_param("train_size", len(X_train_bal))
        mlflow.log_param("test_size", len(X_test))

        model.fit(X_train_bal, y_train_bal)

        metrics = compute_metrics(model, X_test, y_test)

        # Log flat metrics to MLflow
        for k, v in metrics["overall"].items():
            mlflow.log_metric(k, v)
        for gender, gm in metrics["by_gender"].items():
            for k, v in gm.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"{gender}_{k}", v)
        mlflow.log_metric("recall_gap", metrics["fairness"]["recall_gap_male_female"])

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="rf_model")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")

    # Print results
    print("\n=== Overall Metrics ===")
    for k, v in metrics["overall"].items():
        print(f"  {k:12s}: {v}")

    print("\n=== Gender-Stratified Metrics ===")
    for gender, gm in metrics["by_gender"].items():
        print(f"  {gender.upper()} (n={gm['n_samples']})")
        for k, v in gm.items():
            if k != "n_samples":
                print(f"    {k:12s}: {v}")

    print(f"\n=== Fairness ===")
    print(f"  Recall gap (|male - female|): {metrics['fairness']['recall_gap_male_female']}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_preprocessor(preprocessor_state)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved   : {MODEL_PATH}")
    print(f"Metrics saved : {METRICS_PATH}")
    print(f"Preprocessor  : models/preprocessor.joblib")

    return model, metrics


if __name__ == "__main__":
    train()
