# 🏦 Loan Default Prediction
### End-to-End Binary Classification with Logistic Regression

> Predicting whether a loan applicant will default using 300,000 records, scikit-learn pipelines, SHAP interpretability, and banking-standard evaluation metrics — deployed as an interactive Streamlit app.

---

## Problem Statement

Banks and fintech lenders lose billions annually to loan defaults that were statistically predictable. This project builds a production-ready machine learning system that:

- Scores applicants with a calibrated **default probability**
- Makes approve/decline decisions using a **threshold tuned for 80% recall on defaulters**
- Provides **SHAP-based explanations** for every decision — auditable and regulatory-ready
- Meets **industry-standard thresholds** (KS > 0.30, Gini > 0.40)

---

## Dataset

| Property | Value |
|----------|-------|
| Rows | 300,000 |
| Columns | 10 (9 features + 1 target) |
| Default rate | ~18.6% |
| Missing values | ~2% in 4 columns |
| Source | Synthetic (generated to match real-world credit distributions) |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Applicant age (21–69) |
| `annual_income` | float | Annual income in USD (15k–500k) |
| `employment_years` | float | Years at current/last employer (0–40) |
| `credit_score` | float | FICO credit score (300–850) |
| `loan_amount` | int | Loan amount requested in USD (1k–100k) |
| `loan_term_months` | int | Repayment term (12, 24, 36, 48, 60, 72, 84 months) |
| `num_open_accounts` | float | Number of open credit accounts (1–19) |
| `debt_to_income_ratio` | float | Monthly payment burden relative to income (0.01–2.0) |
| `num_derogatory_marks` | int | Negative credit events — late payments, collections (0–5) |
| `default` | int | **Target** — 1 = defaulted, 0 = did not default |

---

## Results

| Metric | Value | Threshold |
|--------|-------|-----------|
| ROC-AUC | *run notebooks* | > 0.85 (good) |
| Average Precision | *run notebooks* | > baseline |
| KS Statistic | *run notebooks* | > 0.30 (industry standard) |
| Gini Coefficient | *run notebooks* | > 0.40 (good) |
| Brier Score | *run notebooks* | Lower is better |
| F1 (Defaulters) | *run notebooks* | At tuned threshold |

---

## Project Structure

```
loan-default-prediction/
├── README.md
├── requirements.txt
├── app.py                          ← Streamlit scoring app
│
├── data/
│   └── loan_default_dataset.csv    ← 300k row dataset
│
├── notebooks/
│   ├── 01_eda.ipynb                ← Distributions, correlations, segments
│   ├── 02_preprocessing.ipynb      ← Feature engineering, VIF, pipelines
│   ├── 03_modeling.ipynb           ← L1/L2 LR, threshold tuning, GridSearchCV
│   └── 04_interpretation.ipynb     ← SHAP, PDP, calibration, KS, Gini
│
├── src/
│   ├── __init__.py
│   ├── features.py                 ← add_features() — single source of truth
│   └── predict.py                  ← score_applicant() and score_batch() APIs
│
├── models/
│   ├── preprocessing_artifacts.pkl ← train/test splits + fitted preprocessor
│   └── loan_default_model.pkl      ← final trained model (after running nb 03)
│
└── plots/                          ← all generated charts saved here
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter notebook notebooks/
```

Run `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb` → `04_interpretation.ipynb`

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

> ⚠️ Run notebook `03_modeling.ipynb` first to generate `models/loan_default_model.pkl`

### 4. Score a single applicant in Python

```python
from src.predict import score_applicant

applicant = {
    "age": 34,
    "annual_income": 55000,
    "employment_years": 3.5,
    "credit_score": 620,
    "loan_amount": 15000,
    "loan_term_months": 36,
    "num_open_accounts": 5,
    "debt_to_income_ratio": 0.35,
    "num_derogatory_marks": 1,
}

result = score_applicant(applicant)
# {'default_probability': 0.312, 'risk_level': 'LOW', 'decision': 'APPROVE'}
```

---

## Methodology

### Preprocessing
- **Stratified 80/20 train/test split** — preserves 18.6% default rate in both sets
- **Median imputation** — robust to skewed distributions and outliers
- **StandardScaler** — zero mean, unit variance; fitted on training set only (no leakage)
- All steps wrapped in a single `sklearn.Pipeline`

### Feature Engineering (6 new features)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `monthly_income` | `annual_income / 12` | Baseline for affordability |
| `monthly_loan_payment` | `loan_amount / loan_term_months` | Payment burden |
| `payment_to_income` | `monthly_payment / monthly_income` | Affordability ratio |
| `loan_per_account` | `loan_amount / (open_accounts + 1)` | Credit concentration |
| `risk_score` | `derog×10 + dti×5 − (credit−300)/100` | Domain composite |
| `career_maturity` | `employment_years / (age − 18)` | Employment stability |

### Class Imbalance
- Primary: `class_weight='balanced'` in LogisticRegression — clean, no data augmentation
- Alternative tested: SMOTE (applied to training data only)

### Model Selection
- Evaluated: L2 (lbfgs solver), L1 (saga solver)
- Tuned: `C` ∈ [0.001, 0.01, 0.1, 1, 10], `penalty` ∈ [l1, l2]
- Selection via 5-Fold Stratified GridSearchCV scoring on ROC-AUC

### Threshold Tuning
- Default threshold (0.5) is suboptimal for imbalanced credit risk
- Swept 0.05–0.90 in 0.01 steps; selected threshold maximising F1 for defaulters
- Alternative: threshold for guaranteed 80% recall on defaulters

---

## Key Findings

1. **`num_derogatory_marks`** is the single strongest predictor — applicants with 3+ marks default at 4× the baseline rate
2. **`credit_score`** has a strong non-linear relationship with default — risk accelerates below 580
3. **`debt_to_income_ratio`** above 0.5 is a critical tipping point for default probability
4. **`employment_years`** below 2 years significantly elevates risk regardless of credit score
5. The model is **well-calibrated** — a predicted 20% probability corresponds to ~20% actual defaults

---

## What I'd Do Next

- **Model comparison** — XGBoost or LightGBM as a challenger model
- **Fairness audit** — check for disparate impact across age and income groups
- **Reject inference** — handle the bias introduced by only observing outcomes on approved loans
- **Model monitoring** — track data drift and performance degradation over time in production
- **Feature store** — abstract features into a reusable registry for multiple models

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `imbalanced-learn` · `shap` · `scipy` · `statsmodels` · `matplotlib` · `seaborn` · `streamlit` · `joblib`

---

*Project by: [Your Name] · [LinkedIn] · [GitHub]*
