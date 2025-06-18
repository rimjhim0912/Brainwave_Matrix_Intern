# fraud_detection.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("creditcard.csv")

# Preprocess
df["NormalizedAmount"] = StandardScaler().fit_transform(df[["Amount"]])
df = df.drop(["Time", "Amount"], axis=1)
X = df.drop("Class", axis=1)
y = df["Class"]

# Handle imbalance
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Save model
joblib.dump(model, "fraud_model.pkl")
