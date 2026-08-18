"""
Customer Lifetime Value (CLV) Prediction App
--------------------------------------------
Streamlit front-end for the tuned Random Forest pipeline.

The .sav file contains the FULL pipeline (ColumnTransformer + RandomForestRegressor),
so raw user input can be passed straight to .predict() with no manual encoding.

Run locally:  streamlit run app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model_clv.sav"

# Column names must match the training data EXACTLY (including spacing).
# 'EmploymentStatus' has no space; 'Marital Status' does. This is the single
# most common cause of prediction failures in deployment.
FEATURE_ORDER = [
    "Vehicle Class",
    "Coverage",
    "Renew Offer Type",
    "EmploymentStatus",
    "Marital Status",
    "Education",
    "Monthly Premium Auto",
    "Total Claim Amount",
    "Income",
    "Policy Bin",
]

VEHICLE_CLASSES = [
    "Four-Door Car", "Two-Door Car", "SUV",
    "Sports Car", "Luxury SUV", "Luxury Car",
]
COVERAGES = ["Basic", "Extended", "Premium"]
OFFER_TYPES = ["Offer1", "Offer2", "Offer3", "Offer4"]
EMPLOYMENT_STATUSES = ["Employed", "Unemployed", "Medical Leave", "Disabled", "Retired"]
MARITAL_STATUSES = ["Married", "Single", "Divorced"]
EDUCATION_LEVELS = ["High School or Below", "College", "Bachelor", "Master", "Doctor"]

# Reference points from the training data, used to contextualise a prediction.
CLV_REFERENCE = {
    "median": 5_838,
    "mean": 8_059,
    "p90": 15_641,
    "p95": 21_922,
    "p99": 36_261,
}


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the pickled pipeline once and cache it across reruns."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def bin_policies(n_policies: int) -> str:
    """
    Replicate the Policy Bin feature engineering from training.

    Training used: pd.cut(x, bins=[0, 1, 2, 99], labels=['1', '2', '3+'])
    """
    if n_policies <= 1:
        return "1"
    if n_policies == 2:
        return "2"
    return "3+"


def describe_segment(prediction: float) -> tuple[str, str]:
    """Return a (tier, explanation) pair for a predicted CLV."""
    if prediction >= CLV_REFERENCE["p99"]:
        return "Top 1%", "Exceptional value. Prioritise for retention."
    if prediction >= CLV_REFERENCE["p95"]:
        return "Top 5%", "High value. Strong candidate for retention investment."
    if prediction >= CLV_REFERENCE["p90"]:
        return "Top 10%", "Above average value. Worth targeted offers."
    if prediction >= CLV_REFERENCE["median"]:
        return "Upper half", "Moderate value. Standard servicing."
    return "Lower half", "Below median value. Low-cost retention only."


# --------------------------------------------------------------------------
# Page layout
# --------------------------------------------------------------------------

st.set_page_config(page_title="CLV Prediction", page_icon="📊", layout="wide")

st.title("Customer Lifetime Value Prediction")
st.caption(
    "Estimates the lifetime value of an auto-insurance customer using a "
    "tuned Random Forest model. Use the estimate to guide acquisition and "
    "retention budgets."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file not found at `{MODEL_PATH.name}`. "
        "Place `model_clv.sav` in the same folder as this script."
    )
    st.stop()

st.divider()

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------

with st.form("clv_form"):
    st.subheader("Customer details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Policy**")
        vehicle_class = st.selectbox("Vehicle Class", VEHICLE_CLASSES, index=0)
        coverage = st.selectbox("Coverage", COVERAGES, index=0)
        renew_offer = st.selectbox("Renew Offer Type", OFFER_TYPES, index=0)

    with col2:
        st.markdown("**Demographics**")
        employment = st.selectbox("Employment Status", EMPLOYMENT_STATUSES, index=0)
        marital = st.selectbox("Marital Status", MARITAL_STATUSES, index=0)
        education = st.selectbox("Education", EDUCATION_LEVELS, index=2)

    with col3:
        st.markdown("**Financials**")
        n_policies = st.number_input(
            "Number of Policies", min_value=1, max_value=9, value=1, step=1,
            help="Customers with exactly 2 policies show markedly higher CLV.",
        )
        monthly_premium = st.number_input(
            "Monthly Premium Auto", min_value=60.0, max_value=300.0,
            value=93.0, step=1.0,
        )
        total_claim = st.number_input(
            "Total Claim Amount", min_value=0.0, max_value=3000.0,
            value=431.0, step=10.0,
        )
        income = st.number_input(
            "Income", min_value=0, max_value=100_000, value=37_868, step=1_000,
            help="Enter 0 for unemployed customers.",
        )

    submitted = st.form_submit_button("Predict CLV", type="primary")

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

if submitted:
    input_data = pd.DataFrame([{
        "Vehicle Class": vehicle_class,
        "Coverage": coverage,
        "Renew Offer Type": renew_offer,
        "EmploymentStatus": employment,
        "Marital Status": marital,
        "Education": education,
        "Monthly Premium Auto": float(monthly_premium),
        "Total Claim Amount": float(total_claim),
        "Income": float(income),
        "Policy Bin": bin_policies(int(n_policies)),
    }])[FEATURE_ORDER]

    prediction = float(model.predict(input_data)[0])
    tier, advice = describe_segment(prediction)

    st.divider()
    st.subheader("Result")

    res1, res2, res3 = st.columns(3)
    res1.metric("Predicted CLV", f"{prediction:,.0f}")
    res2.metric("Segment", tier)
    res3.metric("vs. median customer", f"{prediction / CLV_REFERENCE['median']:.1f}x")

    st.info(advice)

    # Honest uncertainty reporting. Error grows sharply with predicted value,
    # so a single point estimate would overstate precision for high-value customers.
    if prediction >= 20_000:
        margin = 13_400
        note = "High-value predictions carry wide uncertainty (residual std ≈ 13,400)."
    elif prediction >= 10_000:
        margin = 5_000
        note = "Moderate uncertainty in this range (residual std ≈ 5,000)."
    else:
        margin = 450
        note = "The model is most reliable in this range (residual std ≈ 450)."

    low = max(0.0, prediction - margin)
    high = prediction + margin
    st.caption(f"Indicative range: **{low:,.0f} – {high:,.0f}**. {note}")

    with st.expander("Model input (as sent to the pipeline)"):
        st.dataframe(input_data, use_container_width=True)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("About this model")
    st.markdown(
        """
        **Algorithm** — Random Forest Regressor
        `n_estimators=300`, `max_depth=5`, `min_samples_leaf=10`

        **Test performance**
        - R² ≈ 0.68
        - RMSE ≈ 3,900
        - MAE ≈ 1,690

        **Main value drivers**
        1. Number of policies (≈ 64% of model importance)
        2. Monthly premium (≈ 32%)

        Demographic fields contribute very little once policy count
        and premium are known.
        """
    )
    st.warning(
        "The model cannot predict above ~36,500, because Random Forest "
        "predictions are averages of observed training values. Estimates for "
        "the highest-value customers are conservative by design."
    )