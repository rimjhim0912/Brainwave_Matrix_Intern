import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import clean_text

# Create model directory if not exists
os.makedirs('models', exist_ok=True)

# Load data
fake = pd.read_csv('data/raw/Fake.csv')
real = pd.read_csv('data/raw/True.csv')

fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real])

# Combine title and text
df['full_text'] = (df['title'] + " " + df['text']).apply(clean_text)

# Check label distribution
print("Label distribution:\n", df['label'].value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization (unigrams + bigrams, more features)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model: Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save model and vectorizer
joblib.dump(model, 'models/logistic_model.pkl')  # File name stays same
joblib.dump(vectorizer, 'models/vectorizer.pkl')
