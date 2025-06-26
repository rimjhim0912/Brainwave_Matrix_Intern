import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

st.title("Credit Card Fraud Detection App")
st.write("📄 Upload a CSV file of transaction data to detect fraudulent activities.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

model = load_model()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Drop label column if present
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Drop Time if your model was trained without it
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        # Predict
        preds = model.predict(df)
        df["Prediction"] = preds
        df["Fraud_Status"] = df["Prediction"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

        st.subheader("Prediction Results (First 10 Rows):")
        st.write(df.head(10))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Results", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
