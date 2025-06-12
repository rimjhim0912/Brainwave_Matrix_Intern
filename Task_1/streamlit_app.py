import os
import streamlit as st
import joblib
import nltk
from src.preprocess import clean_text

nltk.download('stopwords')
nltk.download('wordnet')

model_path = "models/logistic_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error("Model files not found. Make sure 'logistic_model.pkl' and 'vectorizer.pkl' exist in the 'models' folder.")
    st.stop()

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article snippet below:")

text_input = st.text_area("News Text", height=250)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        st.write("🧹 **Cleaned Text:**", cleaned)

        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "🟩 Real News" if prediction == 1 else "🟥 Fake News"
        st.success(f"Prediction: **{label}**")
