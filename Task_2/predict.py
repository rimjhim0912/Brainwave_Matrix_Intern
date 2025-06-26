import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model using pickle
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Preprocess: Normalize the 'Amount' column and drop 'Time'
scaler = StandardScaler()
df["NormalizedAmount"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(["Time", "Amount"], axis=1)

# Select 5 fraudulent samples to test the model
fraud_samples = df[df["Class"] == 1].drop("Class", axis=1).iloc[:5]

# Predict using the trained model
predictions = model.predict(fraud_samples)

# Display results
print("Predictions for 5 actual fraudulent transactions:")
for i, pred in enumerate(predictions):
    result = "Fraud" if pred == 1 else "Not Fraud"
    print(f"Transaction {i+1}: {result}")
