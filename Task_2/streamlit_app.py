import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

# Load model once
model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("📄 Upload a CSV file of transactions to detect fraud.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Optional: Drop 'Class' column if exists
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Drop 'Time' column if it was not used during training
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        # Standardize 'Amount' to match training condition
        if "Amount" in df.columns:
            scaler = StandardScaler()
            df["Amount"] = scaler.fit_transform(df[["Amount"]])

        # Predict
        predictions = model.predict(df)

        # Attach results
        df["Prediction"] = predictions
        df["Fraud_Status"] = df["Prediction"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

        # Show top rows
        st.subheader("🔍 Prediction Results (Top 10 Rows):")
        st.write(df.head(10))

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Results as CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
