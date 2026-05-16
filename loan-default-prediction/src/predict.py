"""
predict.py
----------
Production-ready scoring function.
Loads the saved model and exposes score_applicant()
for use in app.py, APIs, or batch scoring scripts.
"""

import joblib
import pandas as pd
from pathlib import Path
from src.features import add_features

# --------------------------------------------------------------------------- #
# Resolve model path relative to this file so it works from any working dir   #
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "loan_default_model.pkl"

_model = None  # lazy-loaded singleton


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run notebook 03_modeling.ipynb first to train and save the model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def score_applicant(applicant: dict, threshold: float = 0.35) -> dict:
    """
    Score a single loan applicant.

    Parameters
    ----------
    applicant : dict
        Raw applicant features (9 original columns).
        Example:
            {
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
    threshold : float
        Decision threshold for HIGH/LOW risk label. Default 0.35.
        (Tuned during training for ~80 % recall on defaulters.)

    Returns
    -------
    dict
        {
            "default_probability": float,   # 0–1
            "risk_level": str,              # "HIGH" or "LOW"
            "decision": str,                # "DECLINE" or "APPROVE"
        }
    """
    model = _load_model()

    df = pd.DataFrame([applicant])
    df = add_features(df)

    proba = float(model.predict_proba(df)[0][1])
    risk  = "HIGH" if proba >= threshold else "LOW"

    return {
        "default_probability": round(proba, 4),
        "risk_level": risk,
        "decision": "DECLINE" if risk == "HIGH" else "APPROVE",
    }


def score_batch(df_raw: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """
    Score a batch of applicants.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame with the 9 original feature columns (no target).
    threshold : float
        Decision threshold.

    Returns
    -------
    pd.DataFrame
        Original dataframe with three new columns appended:
        default_probability, risk_level, decision.
    """
    model = _load_model()
    df    = add_features(df_raw.copy())
    probas = model.predict_proba(df)[:, 1]

    df_out = df_raw.copy()
    df_out["default_probability"] = probas.round(4)
    df_out["risk_level"]          = ["HIGH" if p >= threshold else "LOW" for p in probas]
    df_out["decision"]            = ["DECLINE" if p >= threshold else "APPROVE" for p in probas]
    return df_out


# --------------------------------------------------------------------------- #
# Quick smoke-test when run directly:  python -m src.predict                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sample = {
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
    result = score_applicant(sample)
    print(result)
