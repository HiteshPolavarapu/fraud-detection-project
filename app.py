import os
import joblib
import pandas as pd
import streamlit as st

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ==============================
# CUSTOM STYLING
# ==============================
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #ffffff;
        }
        .sub-text {
            font-size: 18px;
            color: #cfcfcf;
        }
        .section-header {
            font-size: 28px;
            font-weight: 700;
            margin-top: 20px;
        }
        .info-box {
            background-color: #111827;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #374151;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# FILE PATHS
# ==============================
MODEL_PATH = "fraud_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_names.pkl"

# ==============================
# CHECK FILES EXIST
# ==============================
required_files = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Required file not found: {file}")
        st.stop()

# ==============================
# LOAD FILES
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# ==============================
# DEFAULT VALUES
# ==============================
default_values = {feature: 0.0 for feature in feature_names}

if "Amount" in default_values:
    default_values["Amount"] = 50.0

if "Time" in default_values:
    default_values["Time"] = 10000.0

# ==============================
# HEADER
# ==============================
st.markdown('<div class="main-title">💳 Credit Card Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">A Machine Learning-powered analytics application for identifying potentially fraudulent credit card transactions.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("📝 Transaction Inputs")
st.sidebar.markdown("Adjust the transaction feature values below to test the fraud prediction model.")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(
        label=feature,
        value=float(default_values.get(feature, 0.0)),
        format="%.6f"
    )

input_df = pd.DataFrame([input_data])

# ==============================
# MAIN LAYOUT
# ==============================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section-header">🔍 Transaction Feature Preview</div>', unsafe_allow_html=True)
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.markdown('<div class="section-header">📊 Prediction Engine</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Fraud Prediction"):
        model_name = type(model).__name__

        if model_name == "RandomForestClassifier":
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
        else:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            st.markdown(
                f'<div class="result-box" style="background-color:#3b0d0d; color:#ff6b6b;">⚠ Fraudulent Transaction Detected</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box" style="background-color:#0f2f1d; color:#4ade80;">✅ Legitimate Transaction</div>',
                unsafe_allow_html=True
            )

        st.metric("Fraud Probability", f"{probability * 100:.2f}%")

        if probability < 0.30:
            st.info("Risk Level: Low")
        elif probability < 0.70:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

# ==============================
# PROJECT OVERVIEW
# ==============================
st.markdown("---")
st.markdown('<div class="section-header">📌 Project Overview</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>About this Project</h3>
    <p>
        This project demonstrates a <b>Credit Card Fraud Detection System</b> built using
        <b>Machine Learning</b> and designed as a <b>cloud-based analytics application</b>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>Key Features</h3>
    <ul>
        <li>Fraud / Legitimate transaction prediction</li>
        <li>Fraud probability score</li>
        <li>Risk level classification</li>
        <li>Interactive cloud-hosted dashboard</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>Model Used</h3>
    <p><b>Random Forest Classifier</b> (Final Selected Model)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>Business Value</h3>
    <p>
        This system can help financial institutions identify suspicious transactions early,
        reduce financial losses, and improve fraud prevention workflows.
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Developed as part of an Azure Cloud Technologies Final Project")
