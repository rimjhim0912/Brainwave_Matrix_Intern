from flask import Flask, request, render_template
import joblib
from src.preprocess import clean_text


app = Flask(__name__)

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        text = request.form["news"]
        cleaned_text = clean_text(text)  
        vector = vectorizer.transform([cleaned_text])
        pred = model.predict(vector)[0]
        prediction = "Fake News" if pred == 0 else "Real News"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

