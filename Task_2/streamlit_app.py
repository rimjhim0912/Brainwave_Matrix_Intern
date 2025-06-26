import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model_and_scaler():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

st.title("💳 Credit Card Fraud Detection")
st.write("📄 Upload a CSV file of transaction data to detect fraudulent activities.")

# Load model and scaler
model, scaler = load_model_and_scaler()

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# When file is uploaded
if uploaded_file is not None:
    try:
        # Read uploaded CSV
        data = pd.read_csv(uploaded_file)

        # Drop label column if exists
        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        # Optional: drop Time if your model was trained without it
        if "Time" in data.columns:
            data = data.drop(columns=["Time"])

        # Scale data
        data_scaled = scaler.transform(data)

        # Predict
        predictions = model.predict(data_scaled)

        # Add results to DataFrame
        data["Prediction"] = predictions
        data["Fraud_Status"] = data["Prediction"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

        # Display
        st.subheader("🔍 Prediction Results (First 10 Rows):")
        st.write(data.head(10))

        # Download results
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
