import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
)

def add_features(df):
    df = df.copy()
    df["monthly_income"] = df["annual_income"] / 12
    df["monthly_loan_payment"] = df["loan_amount"] / df["loan_term_months"]
    df["payment_to_income"] = df["monthly_loan_payment"] / (df["monthly_income"] + 1e-9)
    df["loan_per_account"] = df["loan_amount"] / (df["num_open_accounts"] + 1)
    df["risk_score"] = (
        df["num_derogatory_marks"] * 10
        + df["debt_to_income_ratio"] * 5
        - (df["credit_score"] - 300) / 100
    )
    working_age = (df["age"] - 18).clip(lower=0.01)
    df["career_maturity"] = df["employment_years"] / working_age
    return df

@st.cache_resource
def load_model():
    base = Path(__file__).resolve().parent
    data_path = base / "data" / "loan_default_dataset.csv"
    if not data_path.exists():
        data_path = base.parent / "data" / "loan_default_dataset.csv"

    with st.spinner("Training model — please wait ~60 seconds..."):
        df = pd.read_csv(data_path)
        X = add_features(df.drop(columns=["default"]))
        y = df["default"]
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
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
    return model, features

st.title("🏦 Loan Default Risk Predictor")
st.markdown(
    "Enter applicant details below. The model returns a **default probability** "
    "and a **risk decision** based on a threshold tuned for 80% recall on defaulters."
)
st.divider()

model, features = load_model()
THRESHOLD = 0.35

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Applicant Profile")
    age = st.slider("Age", 21, 70, 34)
    annual_income = st.number_input("Annual Income ($)", 15000, 500000, 55000, step=1000)
    employment_yrs = st.slider("Employment Years", 0.0, 40.0, 3.5, step=0.5)

with col2:
    st.subheader("💳 Credit Profile")
    credit_score = st.slider("Credit Score (FICO)", 300, 850, 650)
    num_accounts = st.slider("Open Credit Accounts", 1, 19, 5)
    derog_marks = st.slider("Derogatory Marks", 0, 5, 0)

with col3:
    st.subheader("📋 Loan Details")
    loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=500)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=2)
    dti = st.slider("Debt-to-Income Ratio", 0.01, 2.0, 0.35, step=0.01)

st.divider()

applicant = {
    "age": age,
    "annual_income": annual_income,
    "employment_years": employment_yrs,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term,
    "num_open_accounts": num_accounts,
    "debt_to_income_ratio": dti,
    "num_derogatory_marks": derog_marks,
}

if st.button("⚡ Assess Risk", type="primary"):
    df_in = add_features(pd.DataFrame([applicant]))
    proba = float(model.predict_proba(df_in)[0][1])
    risk = "HIGH" if proba >= THRESHOLD else "LOW"
    decision = "DECLINE" if risk == "HIGH" else "APPROVE"

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Default Probability", f"{proba:.1%}")
    with r2:
        if risk == "HIGH":
            st.error(f"🔴 Risk Level: **{risk}**")
        else:
            st.success(f"🟢 Risk Level: **{risk}**")
    with r3:
        if decision == "DECLINE":
            st.error(f"❌ Decision: **{decision}**")
        else:
            st.success(f"✅ Decision: **{decision}**")

    st.markdown("#### Default Probability Gauge")
    st.progress(min(proba, 1.0))
    st.caption(f"Threshold: {THRESHOLD:.0%} · Above threshold = HIGH risk")

    st.markdown("#### Key Risk Drivers")
    fe = add_features(pd.DataFrame([applicant])).iloc[0]
    st.dataframe(pd.DataFrame({
        "Feature": ["Payment-to-Income", "Risk Score", "Career Maturity", "Loan per Account"],
        "Value": [
            f"{fe['payment_to_income']:.3f}",
            f"{fe['risk_score']:.2f}",
            f"{fe['career_maturity']:.3f}",
            f"${fe['loan_per_account']:,.0f}",
        ],
        "Interpretation": [
            "Higher = more payment burden",
            "Higher = more risk indicators",
            "Lower = less stable employment",
            "Higher = concentrated loan",
        ],
    }), use_container_width=True, hide_index=True)

with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** Logistic Regression (L2 · `class_weight='balanced'`)  
    **Dataset:** 300,000 synthetic loan records · 18.6% default rate  
    **Features:** 9 raw inputs + 6 engineered = 15 total  
    **Decision Threshold:** 0.35 (tuned for ~80% recall on defaulters)  
    **Pipeline:** Median imputation → StandardScaler → LogisticRegression
    """)
