"""
app.py — Loan Default Risk Predictor
-------------------------------------
Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

# Allow imports from src/ when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from src.features import add_features

# ── Page config ────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
)

# ── Load model ─────────────────────────────────────────────────────────────── #
MODEL_PATH = Path(__file__).resolve().parent / "models" / "loan_default_model.pkl"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    
    # Train a quick model on the fly for demo purposes
    st.info("Training model for demo... (takes ~30 seconds)")
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(Path(__file__).parent / "data" / "loan_default_dataset.csv")
    X = add_features(df.drop(columns=["default"]))
    y = df["default"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    features = X_train.columns.tolist()
    pre = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), features)])
    model = Pipeline([
        ("preprocessor", pre),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model
# ── Header ─────────────────────────────────────────────────────────────────── #
st.title("🏦 Loan Default Risk Predictor")
st.markdown(
    "Enter applicant details below. The model returns a **default probability** "
    "and a **risk decision** based on a threshold tuned for 80 % recall on defaulters."
)

if model is None:
    st.warning(
        "⚠️ No trained model found at `models/loan_default_model.pkl`. "
        "Run **notebook 03_modeling.ipynb** first, then refresh this page."
    )

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────── #
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Profile")
    age            = st.slider("Age", 21, 70, 34)
    annual_income  = st.number_input("Annual Income ($)", 15_000, 500_000, 55_000, step=1_000)
    employment_yrs = st.slider("Employment Years", 0.0, 40.0, 3.5, step=0.5)

with col2:
    st.subheader("💳 Credit Profile")
    credit_score   = st.slider("Credit Score (FICO)", 300, 850, 650)
    num_accounts   = st.slider("Open Credit Accounts", 1, 19, 5)
    derog_marks    = st.slider("Derogatory Marks", 0, 5, 0)

with col3:
    st.subheader("📋 Loan Details")
    loan_amount    = st.number_input("Loan Amount ($)", 1_000, 100_000, 15_000, step=500)
    loan_term      = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=2)
    dti            = st.slider("Debt-to-Income Ratio", 0.01, 2.0, 0.35, step=0.01)

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────── #
THRESHOLD = 0.35   # tuned in notebook 03 for ~80 % recall

applicant = {
    "age":                   age,
    "annual_income":         annual_income,
    "employment_years":      employment_yrs,
    "credit_score":          credit_score,
    "loan_amount":           loan_amount,
    "loan_term_months":      loan_term,
    "num_open_accounts":     num_accounts,
    "debt_to_income_ratio":  dti,
    "num_derogatory_marks":  derog_marks,
}

if st.button("⚡ Assess Risk", type="primary", disabled=(model is None)):
    df_in  = add_features(pd.DataFrame([applicant]))
    proba  = float(model.predict_proba(df_in)[0][1])
    risk   = "HIGH" if proba >= THRESHOLD else "LOW"
    decision = "DECLINE" if risk == "HIGH" else "APPROVE"

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Default Probability", f"{proba:.1%}")

    with res_col2:
        if risk == "HIGH":
            st.error(f"🔴 Risk Level: **{risk}**")
        else:
            st.success(f"🟢 Risk Level: **{risk}**")

    with res_col3:
        if decision == "DECLINE":
            st.error(f"❌ Decision: **{decision}**")
        else:
            st.success(f"✅ Decision: **{decision}**")

    # Probability gauge bar
    st.markdown("#### Default Probability")
    st.progress(min(proba, 1.0))
    st.caption(f"Threshold: {THRESHOLD:.0%} · Scores above threshold → HIGH risk")

    # Key engineered features for transparency
    st.markdown("#### Key Risk Drivers (Engineered Features)")
    fe = add_features(pd.DataFrame([applicant])).iloc[0]
    driver_data = {
        "Feature":        ["Payment-to-Income", "Risk Score", "Career Maturity", "Loan per Account"],
        "Value":          [
            f"{fe['payment_to_income']:.3f}",
            f"{fe['risk_score']:.2f}",
            f"{fe['career_maturity']:.3f}",
            f"${fe['loan_per_account']:,.0f}",
        ],
        "Interpretation": [
            "Higher → more payment burden",
            "Higher → more risk indicators",
            "Lower → less stable employment history",
            "Higher → large loan vs few accounts",
        ],
    }
    st.dataframe(pd.DataFrame(driver_data), use_container_width=True, hide_index=True)

st.divider()

# ── About section ──────────────────────────────────────────────────────────── #
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** Logistic Regression (L2 regularisation, `class_weight='balanced'`)

    **Dataset:** 300,000 synthetic loan records · 18.6 % default rate

    **Features:** 9 raw inputs + 6 engineered features = 15 total

    **Decision Threshold:** 0.35 (tuned for ~80 % recall on defaulters)

    **Metrics (test set):**
    | Metric | Value |
    |--------|-------|
    | ROC-AUC | *run notebook 03 to fill* |
    | KS Statistic | *run notebook 03 to fill* |
    | Gini Coefficient | *run notebook 03 to fill* |
    | F1 (Defaulters) | *run notebook 03 to fill* |

    **Training pipeline:** `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`
    """)
