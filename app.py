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
# TITLE
# ==============================
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    """
    This application predicts whether a transaction is **Fraudulent** or **Legitimate**
    using a trained **Machine Learning model**.
    """
)

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
# LOAD MODEL FILES
# ==============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# ==============================
# DEFAULT VALUES (safe demo defaults)
# ==============================
default_values = {feature: 0.0 for feature in feature_names}

if "Amount" in default_values:
    default_values["Amount"] = 50.0

if "Time" in default_values:
    default_values["Time"] = 10000.0

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("📝 Enter Transaction Details")

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
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Input Transaction Data")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("📊 Prediction Result")

    if st.button("Predict Fraud"):
        model_name = type(model).__name__

        if model_name == "RandomForestClassifier":
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
        else:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            st.error("⚠ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{probability * 100:.2f}%")

        if probability < 0.30:
            st.info("Risk Level: Low")
        elif probability < 0.70:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

# ==============================
# PROJECT OVERVIEW SECTION
# ==============================
st.markdown("---")
st.subheader("📌 Project Overview")

st.markdown(
    """
    ### About this Project
    This project demonstrates a **Credit Card Fraud Detection System**
    built using **Machine Learning** and designed for deployment as a
    **cloud-based analytics application**.

    ### Key Features
    - Fraud / Legitimate transaction prediction
    - Fraud probability score
    - Risk level classification
    - Interactive dashboard interface

    ### Model Used
    - **Random Forest Classifier** (Final Selected Model)

    ### Business Value
    - Helps identify suspicious financial transactions
    - Supports fraud prevention and financial security
    """
)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Developed as part of an Azure Cloud Technologies Final Project")
