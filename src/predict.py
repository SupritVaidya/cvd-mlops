"""
predict.py
Inference wrapper. Loads the trained model + preprocessor and runs
predictions on a single patient input dict.

The fairness_flag logic:
  If the patient is female AND model confidence is between 0.40 and 0.60,
  we raise a flag. Why? Female patients are underrepresented in the training
  data (193 vs 725 males). In that confidence band, the model's uncertainty
  is high enough that clinical review is warranted rather than treating the
  score as definitive.

  This is a conservative, explainable heuristic. It is NOT the same as
  saying the model is wrong — it signals "treat this score with more
  scrutiny than usual."
"""

import joblib
from preprocess import preprocess_single, load_preprocessor

MODEL_PATH        = "models/rf_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

# Confidence band where female predictions should be flagged for review
FAIRNESS_LOW  = 0.40
FAIRNESS_HIGH = 0.60


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor_state = load_preprocessor(PREPROCESSOR_PATH)
    return model, preprocessor_state


def predict(raw_input: dict, model, preprocessor_state: dict) -> dict:
    """
    raw_input: dict with keys matching original CSV columns
    Returns:
        prediction    : 0 or 1
        probability   : float, probability of CVD (class 1)
        fairness_flag : bool, True if female + uncertain prediction
    """
    X = preprocess_single(raw_input, preprocessor_state)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    is_female = raw_input.get("Sex", "M") == "F"
    in_uncertain_band = FAIRNESS_LOW <= probability <= FAIRNESS_HIGH
    fairness_flag = is_female and in_uncertain_band

    return {
        "prediction"   : prediction,
        "probability"  : round(probability, 4),
        "fairness_flag": fairness_flag,
    }


if __name__ == "__main__":
    # Quick smoke test with a sample patient
    sample = {
        "Age"          : 52,
        "Sex"          : "F",
        "ChestPainType": "NAP",
        "RestingBP"    : 140,
        "Cholesterol"  : 220,
        "FastingBS"    : 0,
        "RestingECG"   : "Normal",
        "MaxHR"        : 130,
        "ExerciseAngina": "N",
        "Oldpeak"      : 1,
        "ST_Slope"     : "Flat",
    }

    model, preprocessor_state = load_artifacts()
    result = predict(sample, model, preprocessor_state)
    print("Prediction result:", result)
