import streamlit as st
import joblib
import nltk
from src.preprocess import clean_text

nltk.download('stopwords')
nltk.download('wordnet')
# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article snippet below:")

text_input = st.text_area("News Text", height=250)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "🟥 Fake News" if prediction == 1 else "🟩 Real News"
        st.success(f"Prediction: **{label}**")
