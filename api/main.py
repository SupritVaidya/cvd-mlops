from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
from pydantic import BaseModel
from preprocess import preprocess_single
import json

state = {}

FAIRNESS_LOW = 0.40
FAIRNESS_HIGH = 0.60

#Defining the lifespan of the API

@asynccontextmanager
async def lifespan(app: FastAPI):
    #what to execute at startup
    state["model"] = joblib.load("models/rf_model.joblib")
    state["preprocess"] = joblib.load("models/preprocessor.joblib")
    yield

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.get("/health", include_in_schema=True)
@app.head("/health")
def get_health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    with open("models/metrics.json") as f:
        data = json.load(f)
    return data

@app.get("/drift-report")
def get_drift_report():
    return{
        "status": "no_data",
        "message": "Drift monitoring activates once production traffic is logged"
    }

@app.post("/predict")
def post_predict(patient: PatientInput):
    raw = patient.model_dump()
    X = preprocess_single(raw, state["preprocess"])
    '''
    predict() returns an array — [0] gets the first (only) element. 
    predict_proba() returns probabilities for both classes — [0][1] gets the probability of class 1 (CVD). 
    We cast to int and float so they serialise cleanly to JSON.
    '''

    prediction = int(state["model"].predict(X)[0])
    probability = float(state["model"].predict_proba(X)[0][1])

    is_female = raw.get("Sex", "M") == "F"
    in_uncertain_band = FAIRNESS_LOW <= probability <= FAIRNESS_HIGH
    fairness_flag = is_female and in_uncertain_band

    return{
        "prediction": prediction,
        "probability": round(probability, 4),
        "fairness_flag": fairness_flag
    }

