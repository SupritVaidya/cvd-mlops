# Model Card — CVD Prediction System

## Model Details

- **Architecture**: Random Forest Classifier (300 estimators)
- **Task**: Binary classification — HeartDisease (0 = No, 1 = Yes)
- **Framework**: scikit-learn 1.6.x
- **Training date**: 2026-06-18
- **Version**: 1.0.0

## Intended Use

**Primary use**: Research and portfolio demonstration of production ML engineering practices including fairness monitoring.

**Not intended for**: Clinical diagnosis. This model has not been validated in a clinical setting and must not be used to make or inform medical decisions.

## Training Data

- **Dataset**: heart999.csv — 999 patients
- **Source**: UCI Heart Disease Dataset variant
- **Features**: 12 (5 numerical, 6 categorical after encoding → ~20 encoded features)
- **Target**: HeartDisease (binary, 0/1)
- **Class balance**: 54.8% positive (CVD), 45.2% negative
- **Gender split**: 72.6% male (725), 19.3% female (193), imbalance acknowledged

## Preprocessing Pipeline

1. Replace zero RestingBP and Cholesterol with NaN (physiologically impossible values)
2. Fill NaN with gender-stratified median
3. Winsorize at 1st/99th percentile (global, then gender-stratified)
4. Yeo-Johnson power transformation (all 5 numerical features were non-normal per Shapiro-Wilk)
5. One-hot encoding of 6 categorical features
6. 80/20 stratified train/test split (random_state=42)

## Bias Mitigation

SMOTE applied to the female training subgroup only to balance HeartDisease classes within that group, before combining with male data. This was chosen over global SMOTE to avoid distorting the male majority group, and over Fairlearn's ExponentiatedGradient because the ExponentiatedGradient approach in the research notebook reduced overall recall substantially when optimising for Equalized Odds.

## Performance

| Metric        | Overall | Male (n=146) | Female (n=38) |
|---------------|---------|--------------|---------------|
| Accuracy      | 88.0%   | 86.3%        | 94.7%         |
| Precision     | 87.7%   | 88.0%        | 83.3%         |
| Recall        | 91.2%   | 91.7%        | 83.3%         |
| F1-Score      | 89.4%   | 89.8%        | 83.3%         |

**Recall gap (|male − female|)**: 0.083

All metrics are from the held-out test set (n=184). No test-set data was used during training or preprocessing fit steps.

## Fairness Analysis

The model performs better on the overall male group by recall (0.917 vs 0.833). This gap is partly a consequence of the small female test set (n=38) — a single misclassification changes female recall by ~0.083. The SMOTE intervention improved female recall compared to the no-intervention baseline.

The API exposes a `fairness_flag` field: true for female patients with model probability in [0.40, 0.60], indicating elevated uncertainty where clinical review is recommended.

## Limitations

- **Small female test set (n=38)**: Metric estimates for the female group have high variance. A single misclassified patient moves recall by ~8 percentage points.
- **Single dataset**: Trained and evaluated on one dataset. Generalisation to different clinical populations is unknown.
- **No temporal validation**: The dataset has no timestamps. Model drift over time cannot be assessed from this data alone.
- **Not clinically validated**: Results have not been reviewed by clinicians or validated against diagnostic gold standards.

## CI Gates (GitHub Actions)

| Check               | Threshold | Current Value |
|---------------------|-----------|---------------|
| Overall F1          | ≥ 0.80    | 0.894         |
| Female Recall       | ≥ 0.70    | 0.833         |
| Recall Gap          | ≤ 0.25    | 0.083         |
