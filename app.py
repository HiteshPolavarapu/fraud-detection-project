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
st.markdown("This application predicts whether a transaction is **Fraudulent** or **Legitimate** using a trained Machine Learning model.")

# ==============================
# FILE PATHS
# ==============================
MODEL_PATH = "model/fraud_model.pkl"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/feature_names.pkl"
DATA_PATH = "data/creditcard.csv"

# ==============================
# CHECK FILES EXIST
# ==============================
required_files = [MODEL_PATH, SCALER_PATH, FEATURES_PATH, DATA_PATH]

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
df = pd.read_csv(DATA_PATH)

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("📝 Enter Transaction Details")

# Use median values as default input
default_values = df.drop("Class", axis=1).median().to_dict()

input_data = {}

for feature in feature_names:
    input_data[feature] = st.sidebar.number_input(
        label=feature,
        value=float(default_values.get(feature, 0.0)),
        format="%.6f"
    )

input_df = pd.DataFrame([input_data])

# ==============================
# MAIN SECTION
# ==============================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Input Transaction Data")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("📊 Prediction Result")

    if st.button("Predict Fraud"):
        # Check if model is Random Forest or Logistic Regression
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
# DASHBOARD OVERVIEW
# ==============================
st.markdown("---")
st.subheader("📈 Dataset Overview")

total_transactions = len(df)
fraud_transactions = df["Class"].sum()
legit_transactions = total_transactions - fraud_transactions
fraud_rate = (fraud_transactions / total_transactions) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total_transactions:,}")
c2.metric("Fraud Cases", f"{fraud_transactions:,}")
c3.metric("Legitimate Cases", f"{legit_transactions:,}")
c4.metric("Fraud Rate", f"{fraud_rate:.4f}%")

# ==============================
# CLASS DISTRIBUTION CHART
# ==============================
st.markdown("### Fraud vs Legitimate Transactions")
chart_df = pd.DataFrame({
    "Transaction Type": ["Legitimate", "Fraud"],
    "Count": [legit_transactions, fraud_transactions]
})
st.bar_chart(chart_df.set_index("Transaction Type"))

# ==============================
# SAMPLE DATA
# ==============================
st.markdown("### Sample Transaction Records")
st.dataframe(df.head(10), use_container_width=True)