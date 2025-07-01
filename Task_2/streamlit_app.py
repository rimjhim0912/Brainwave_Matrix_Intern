import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report

@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

# Load model
model = load_model()

# Title and instructions
st.title("💳 Credit Card Fraud Detection")
st.write("📄 Upload a CSV file of transactions to detect fraudulent activities.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read uploaded data
        df = pd.read_csv(uploaded_file)

        # Extract true labels if present
        if 'Class' in df.columns:
            true_labels = df['Class'].astype(str).str.replace("'", "").str.strip().astype(int)
            df = df.drop(columns=['Class'])
        else:
            true_labels = None

        # Check for required features
        required_features = model.feature_names_in_
        missing_features = [col for col in required_features if col not in df.columns]

        if missing_features:
            st.error(f"❌ Missing required columns: {missing_features}")
        else:
            # Make predictions
            preds = model.predict(df)
            preds = pd.Series(preds).astype(str).str.replace("'", "").str.strip().astype(int)
            df["Prediction"] = preds
            df["Fraud_Status"] = df["Prediction"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud")

            # Show top 10 results
            st.subheader("🔍 Prediction Results (First 10 rows):")
            st.write(df.head(10))

            # Optional: show evaluation metrics
            if true_labels is not None:
                st.subheader("📊 Evaluation Metrics:")
                st.text(classification_report(true_labels, preds))

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full Results", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
