import streamlit as st
import joblib
import nltk
from src.preprocess import clean_text

model_path = "models/logistic_model.pkl"
vectorizer_path = "models/vectorizer.pkl"


nltk.download('stopwords')
nltk.download('wordnet')
# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article snippet below:")
st.write("Cleaned text:", cleaned)


text_input = st.text_area("News Text", height=250)

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "🟩 Real News" if prediction == 1 else "🟥 Fake News"
        st.success(f"Prediction: **{label}**")
