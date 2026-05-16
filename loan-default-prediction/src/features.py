"""
features.py
-----------
All feature engineering logic lives here.
Import add_features() in notebooks, predict.py, and app.py
so that every environment uses identical transformations.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 6 domain-driven features to the raw applicant dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw applicant data with columns:
        age, annual_income, employment_years, credit_score,
        loan_amount, loan_term_months, num_open_accounts,
        debt_to_income_ratio, num_derogatory_marks

    Returns
    -------
    pd.DataFrame
        Original columns + 6 new engineered features.
    """
    df = df.copy()

    # --- Monthly income ---
    df["monthly_income"] = df["annual_income"] / 12

    # --- Monthly loan payment estimate (simple division) ---
    df["monthly_loan_payment"] = df["loan_amount"] / df["loan_term_months"]

    # --- Payment-to-income burden ---
    df["payment_to_income"] = df["monthly_loan_payment"] / (df["monthly_income"] + 1e-9)

    # --- Loan concentration per open account ---
    df["loan_per_account"] = df["loan_amount"] / (df["num_open_accounts"] + 1)

    # --- Composite risk score (domain heuristic) ---
    # Higher derogatory marks + higher DTI + lower credit score → higher risk
    df["risk_score"] = (
        df["num_derogatory_marks"] * 10
        + df["debt_to_income_ratio"] * 5
        - (df["credit_score"] - 300) / 100
    )

    # --- Career maturity (employment stability relative to working age) ---
    working_age = (df["age"] - 18).clip(lower=0.01)
    df["career_maturity"] = df["employment_years"] / working_age

    return df


def get_feature_names() -> list:
    """Return the full ordered list of features after engineering."""
    return [
        # Original features
        "age",
        "annual_income",
        "employment_years",
        "credit_score",
        "loan_amount",
        "loan_term_months",
        "num_open_accounts",
        "debt_to_income_ratio",
        "num_derogatory_marks",
        # Engineered features
        "monthly_income",
        "monthly_loan_payment",
        "payment_to_income",
        "loan_per_account",
        "risk_score",
        "career_maturity",
    ]
