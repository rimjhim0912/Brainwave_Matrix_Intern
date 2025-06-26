import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

# Load model
model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.write("📄 Upload a CSV file of transactions to detect fraud.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)

        # Optional: Drop 'Class' column if present (label)
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Normalize 'Amount' if model expected that
        if "Amount" in df.columns:
            scaler = StandardScaler()
            df["Amount"] = scaler.fit_transform(df[["Amount"]])

        # ❗ Keep 'Time' column because model was trained with it

        # Predict
        predictions = model.predict(df)

        # Results
        df["Prediction"] = predictions
        df["Fraud_Status"] = df["Prediction"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

        # Show top rows
        st.subheader("🔍 Prediction Results (Top 10 Rows):")
        st.write(df.head(10))

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
