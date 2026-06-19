"""
preprocess.py
Reproduces the exact preprocessing pipeline from CVD_Analysis_Final.ipynb.

Steps (in notebook order):
  1. Replace zero RestingBP and Cholesterol with NaN
  2. Fill NaN with gender-stratified median
  3. Winsorize at 1st/99th percentile (global, then gender-stratified)
  4. Yeo-Johnson power transformation on numerical columns
  5. One-hot encode categorical columns
  6. Train/test split (80/20, stratified, random_state=42)

Outputs:
  X_train, X_test, y_train, y_test   — DataFrames/Series ready for training
  preprocessor_state                  — dict with fitted transformer + winsorize
                                        bounds, needed at inference time
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
import joblib
import os

NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL_COLS = ["RestingECG", "ChestPainType", "ExerciseAngina",
                    "FastingBS", "ST_Slope", "Sex"]
TARGET = "HeartDisease"


def load_raw(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


def _replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Zero in RestingBP and Cholesterol is physiologically impossible."""
    df = df.copy()
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    return df


def _fill_nan_gender_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with gender-stratified median (notebook cell 5)."""
    df = df.copy()
    df["RestingBP"] = df.groupby("Sex")["RestingBP"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Cholesterol"] = df.groupby("Sex")["Cholesterol"].transform(
        lambda x: x.fillna(x.median())
    )
    return df


def _fit_winsorize_bounds(df: pd.DataFrame) -> dict:
    """
    Compute winsorize bounds from training data.
    Returns dict: {col: (lower, upper)} for global bounds
                  and {col: {'M': (lo, hi), 'F': (lo, hi)}} for gender-stratified.
    """
    global_bounds = {}
    for col in NUMERICAL_COLS:
        global_bounds[col] = (
            df[col].quantile(0.01),
            df[col].quantile(0.99),
        )

    gender_bounds = {}
    for col in NUMERICAL_COLS:
        gender_bounds[col] = {}
        for gender in ["M", "F"]:
            mask = df["Sex"] == gender
            gender_bounds[col][gender] = (
                df.loc[mask, col].quantile(0.01),
                df.loc[mask, col].quantile(0.99),
            )

    return {"global": global_bounds, "gender": gender_bounds}


def _apply_winsorize(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Apply global then gender-stratified winsorization (notebook cells 12-13)."""
    df = df.copy()

    # Global winsorize
    for col, (lower, upper) in bounds["global"].items():
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Gender-stratified winsorize
    for col in NUMERICAL_COLS:
        for gender in ["M", "F"]:
            mask = df["Sex"] == gender
            lo, hi = bounds["gender"][col][gender]
            clipped = df.loc[mask, col].clip(lower=lo, upper=hi)
            df.loc[mask, col] = clipped.astype(df[col].dtype)

    return df


def _fit_power_transformer(df: pd.DataFrame) -> PowerTransformer:
    pt = PowerTransformer(method="yeo-johnson")
    pt.fit(df[NUMERICAL_COLS])
    return pt


def _apply_power_transformer(df: pd.DataFrame, pt: PowerTransformer) -> pd.DataFrame:
    df = df.copy()
    df[NUMERICAL_COLS] = pt.transform(df[NUMERICAL_COLS])
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode + drop IsolationForest_Outlier if present."""
    df = df.drop(columns=["IsolationForest_Outlier"], errors="ignore")
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def run_preprocessing(data_path: str):
    """
    Full pipeline. Returns:
        X_train, X_test, y_train, y_test, feature_columns, preprocessor_state
    """
    df = load_raw(data_path)
    df = _replace_zero_with_nan(df)
    df = _fill_nan_gender_median(df)

    # Fit bounds on full data (notebook doesn't split before winsorizing)
    winsorize_bounds = _fit_winsorize_bounds(df)
    df = _apply_winsorize(df, winsorize_bounds)

    # Fit Yeo-Johnson on full data before split (matches notebook order)
    pt = _fit_power_transformer(df)
    df = _apply_power_transformer(df, pt)

    # Encode
    df_encoded = _encode_categoricals(df)

    X = df_encoded.drop(columns=[TARGET])
    y = df_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_columns = X_train.columns.tolist()

    preprocessor_state = {
        "winsorize_bounds": winsorize_bounds,
        "power_transformer": pt,
        "feature_columns": feature_columns,
    }

    return X_train, X_test, y_train, y_test, feature_columns, preprocessor_state


def preprocess_single(raw_input: dict, preprocessor_state: dict) -> pd.DataFrame:
    """
    Transform a single patient dict (from the API) into a model-ready DataFrame.
    raw_input keys must match the original CSV columns (before encoding).
    """
    df = pd.DataFrame([raw_input])
    df = _replace_zero_with_nan(df)

    # Fill NaN with the value itself if already provided; otherwise use global median
    # (at inference we don't have the full dataset — fill with 0 to match transform space)
    for col in ["RestingBP", "Cholesterol"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)

    # Apply winsorization using fitted bounds
    bounds = preprocessor_state["winsorize_bounds"]
    sex_val = raw_input.get("Sex", "M")
    for col in NUMERICAL_COLS:
        # Global clip
        lo_g, hi_g = bounds["global"][col]
        df[col] = df[col].clip(lower=lo_g, upper=hi_g)
        # Gender clip
        if sex_val in bounds["gender"][col]:
            lo_s, hi_s = bounds["gender"][col][sex_val]
            df[col] = df[col].clip(lower=lo_s, upper=hi_s)

    # Yeo-Johnson
    pt = preprocessor_state["power_transformer"]
    df[NUMERICAL_COLS] = pt.transform(df[NUMERICAL_COLS])

    # Encode
    df = _encode_categoricals(df)

    # Align to training columns (fill missing dummies with 0)
    feature_columns = preprocessor_state["feature_columns"]
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def save_preprocessor(preprocessor_state: dict, path: str = "models/preprocessor.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor_state, path)
    print(f"Preprocessor saved to {path}")


def load_preprocessor(path: str = "models/preprocessor.joblib") -> dict:
    return joblib.load(path)


if __name__ == "__main__":
    DATA_PATH = "data/heart999.csv"
    X_train, X_test, y_train, y_test, feature_columns, preprocessor_state = run_preprocessing(DATA_PATH)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Class balance — train: {y_train.value_counts().to_dict()}")
    save_preprocessor(preprocessor_state)
