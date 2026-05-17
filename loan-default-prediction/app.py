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
