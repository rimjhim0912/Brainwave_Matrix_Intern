import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("fraud_model.pkl")

# Load the dataset (same one used for training)
df = pd.read_csv("creditcard.csv")

# Preprocess the new input in the same way
df["NormalizedAmount"] = StandardScaler().fit_transform(df[["Amount"]])
df = df.drop(["Time", "Amount"], axis=1)

# Select a few samples for testing (for example: first 5 transactions)
sample_input = df.drop("Class", axis=1).iloc[:5]

# Predict using the model
predictions = model.predict(sample_input)

# Show the prediction results
print("Predictions for first 5 transactions:")
for i, pred in enumerate(predictions):
    result = "Fraud" if pred == 1 else "Not Fraud"
    print(f"Transaction {i+1}: {result}")
