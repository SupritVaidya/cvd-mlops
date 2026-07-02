import sys
sys.path.insert(0, 'src')

import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

SAMPLE_PATIENT = {
    "Age": 63,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 145,
    "Cholesterol": 233,
    "FastingBS": 1,
    "RestingECG": "LVH",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 2.3,
    "ST_Slope": "Down"
}


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_metrics_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_contains_overall(client):
    response = client.get("/metrics")
    data = response.json()
    assert "overall" in data
    assert "accuracy" in data["overall"]
    assert "f1" in data["overall"]


def test_metrics_contains_fairness(client):
    response = client.get("/metrics")
    data = response.json()
    assert "by_gender" in data
    assert "fairness" in data


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200


def test_predict_response_shape(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "fairness_flag" in data


def test_predict_prediction_is_binary(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.json()["prediction"] in [0, 1]


def test_predict_probability_in_range(client):
    response = client.post("/predict", json=SAMPLE_PATIENT)
    prob = response.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_fairness_flag_triggers_for_female_in_uncertain_band(client):
    patient = {**SAMPLE_PATIENT, "Sex": "F", "Oldpeak": 1.5, "ExerciseAngina": "Y", "ST_Slope": "Flat"}
    response = client.post("/predict", json=patient)
    data = response.json()
    prob = data["probability"]
    if 0.40 <= prob <= 0.60:
        assert data["fairness_flag"] is True


def test_fairness_flag_not_triggered_for_male(client):
    patient = {**SAMPLE_PATIENT, "Sex": "M", "Oldpeak": 1.5, "ExerciseAngina": "Y", "ST_Slope": "Flat"}
    response = client.post("/predict", json=patient)
    assert response.json()["fairness_flag"] is False


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in SAMPLE_PATIENT.items() if k != "Age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
