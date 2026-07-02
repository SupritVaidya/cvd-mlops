# CVD Prediction System with Fairness Monitoring

A production ML system for cardiovascular disease prediction that tracks model fairness across gender groups.

## What this is

Binary classification (HeartDisease: 0/1) on 999 patients. The system goes beyond accuracy — it explicitly monitors whether the model performs equitably for male and female patients, because the training data is 79% male.

## Real Model Performance (Random Forest, SMOTE-balanced)

| Metric    | Overall | Male (n=146) | Female (n=38) |
|-----------|---------|--------------|---------------|
| Accuracy  | 88.0%   | 86.3%        | 94.7%         |
| Precision | 87.7%   | 88.0%        | 83.3%         |
| Recall    | 91.2%   | 91.7%        | 83.3%         |
| F1        | 89.4%   | 89.8%        | 83.3%         |

**Recall gap (|male − female|): 0.083** — below the 0.25 CI gate threshold.

## Stack

| Layer      | Tool                           |
|------------|--------------------------------|
| ML         | scikit-learn, imbalanced-learn |
| Tracking   | MLflow                         |
| API        | FastAPI                        |
| Monitoring | Evidently                      |
| Frontend   | Angular 20                     |
| Containers | Docker + Docker Compose        |

## Project Structure

```
cvd-mlops/
├── data/heart999.csv          # 999 patients, 12 features
├── src/
│   ├── preprocess.py          # Cleaning, Yeo-Johnson, encoding
│   ├── train.py               # RF + SMOTE + MLflow tracking
│   ├── evaluate.py            # CI gate: fails if F1 < 0.80 or female recall < 0.70
│   └── predict.py             # Inference wrapper with fairness flag
├── api/main.py                # FastAPI endpoints
├── frontend/                  # Angular app (Home, Predict, Model Info)
├── monitoring/                # Evidently drift reports
├── models/                    # Serialised artifacts (gitignored)
├── Dockerfile                 # Backend container
├── frontend/Dockerfile        # Frontend container
├── docker-compose.yml         # Runs full stack
└── MODEL_CARD.md
```

## Quick Start

### Option 1 — Docker (recommended)

Requires Docker Desktop running.

```bash
docker compose up --build
```

- Frontend: http://localhost:4200
- Backend API: http://localhost:8000

### Option 2 — Local

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
PYTHONPATH=src python src/train.py

# Evaluate and run CI gates
PYTHONPATH=src python src/evaluate.py

# Start the API
PYTHONPATH=src uvicorn api.main:app --reload --port 8000

# Start the frontend (separate terminal)
cd frontend
ng serve
```

## API Endpoints

| Method | Endpoint        | Description                              |
|--------|-----------------|------------------------------------------|
| GET    | `/health`       | Health check                             |
| POST   | `/predict`      | Predict CVD risk for a patient           |
| GET    | `/metrics`      | Model performance metrics by gender      |
| GET    | `/drift-report` | Data drift monitoring status             |

### Example `/predict` request

```json
{
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
```

### Example response

```json
{
  "prediction": 1,
  "probability": 0.67,
  "fairness_flag": false
}
```

## Key Design Decisions

**Why Random Forest over Logistic Regression?**
RF outperformed LR on the baseline comparison (F1: 0.89 vs ~0.85). Logistic Regression would be preferred in a regulatory context requiring full explainability, which is a documented trade-off.

**Why SMOTE on the female subgroup only?**
The dataset is 79% male. Applying SMOTE globally would synthesise majority-class data unnecessarily. Applying it only to female patients balances the gender groups without distorting the overall class distribution.

**Why Yeo-Johnson over standard scaling?**
Shapiro-Wilk tests confirmed that all 5 numerical features are non-normal. Yeo-Johnson handles both positive and negative values (unlike Box-Cox) and reduces skew without removing outliers.

**Why Winsorization instead of outlier removal?**
With 999 samples, removing even 50 outliers loses 5% of the data. Winsorizing at 1st/99th percentile caps extreme values while retaining every row.

## Fairness Flag

Female patients with model probability between 0.40 and 0.60 receive a `fairness_flag: true` in the API response. This signals clinical review is advisable — not that the prediction is wrong, but that model uncertainty is high for a historically underrepresented group.
