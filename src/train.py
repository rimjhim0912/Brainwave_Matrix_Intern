import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from preprocess import clean_text
os.makedirs('models', exist_ok=True)

# Load and label data
fake = pd.read_csv('data/raw/Fake.csv')
real = pd.read_csv('data/raw/True.csv')
fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real])

print("Fake news samples:", len(fake))
print("Real news samples:", len(real))


# Preprocess
df['text'] = df['text'].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

