import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# === LOAD MODEL, SCALER, SHAP EXPLAINER ===
with open("best_fraud_model_top20.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_top20.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("shap_explainer_top20.pkl", "rb") as f:
    explainer = pickle.load(f)

# === DEFINE TOP 20 FEATURES USED ===
top_20_features = [
    'Total_claim_payout',
    'Total_Claim_Amount',
    'Repair Cost Estimate',
    'Number of Previous Claims',
    'claim_amount_premium_ratio',
    'vehicle_claim',
    'policy_annual_premium',
    'property_claim_deductable_ratio',
    'Premium Amount',
    'Deductible',
    'property_claim',
    'Coverage Amount',
    'Odometer Reading at Time of Claim',
    'Annual Income',
    'Credit Score',
    'Vehicle Value',
    'Claim Amount',
    'Policy Tenure',
    'months_as_customer',
    'Fraud Investigation Days'
]

# === APP UI ===
st.set_page_config(page_title="ClaimShield", layout="wide")
st.title("CS.png" "Auto Insurance Fraud Predictor")
st.markdown("Enter claim details to receive a **fraud prediction**.")

# === INPUT FORM IN 3 COLUMNS ===
with st.form("fraud_form"):
    col1, col2, col3 = st.columns(3)
    user_input = []

    for i, feature in enumerate(top_20_features):
        if i % 3 == 0:
            val = col1.number_input(f"{feature}", format="%.2f")
        elif i % 3 == 1:
            val = col2.number_input(f"{feature}", format="%.2f")
        else:
            val = col3.number_input(f"{feature}", format="%.2f")
        user_input.append(val)

    submitted = st.form_submit_button("Predict")

# === ON SUBMIT: PREDICT AND EXPLAIN ===
if submitted:
    # Scale user input
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # === MODEL PREDICTION ===
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    risk_score = int(prob * 100)

    # === RISK & RESULT DISPLAY ===
    if risk_score <= 40:
        st.success(f"Risk Score: {risk_score}/100 — ✅ Low Risk – Recommend Claim Payout")
    elif risk_score <= 70:
        st.success(f"Risk Score: {risk_score}/100 — ⚠️ Medium Risk – Request Supporting Docs")
    else:
        st.error(f"Risk Score: {risk_score}/100 — 🚨 High Risk – Recommend for Further Investigation")

    st.subheader("🔍 Prediction")
    if pred == 1:
        st.error(f"🚨 This claim is predicted to be **FRAUDULENT** with {prob*100:.2f}% confidence.")
    else:
        st.success(f"✅ This claim is predicted to be **LEGITIMATE** with {(1 - prob)*100:.2f}% confidence.")

    # === SHAP EXPLANATION ===
    st.markdown("---")
    st.subheader("🧠 SHAP Explanation – Why this prediction?")

    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    # === SHAP TEXTUAL HINTS ===
    st.subheader("📋 Key Contributors")
    shap_vals = shap_values.values[0]
    feature_impact = list(zip(top_20_features, shap_vals))
    sorted_impact = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)

    for i, (feat, val) in enumerate(sorted_impact[:5], 1):
        direction = "increases" if val > 0 else "decreases"
        st.write(f"**{i}.** `{feat}` {direction} the likelihood of fraud (impact: {val:+.4f})")
