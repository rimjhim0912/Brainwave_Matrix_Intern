import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard.csv")

# Get only fraud transactions
fraud_df = df[df["Class"] == 1]

# Save first 5 fraud transactions to a new CSV
fraud_df.iloc[:5].to_csv("fraud_samples.csv", index=False)

print("Saved 5 fraud transactions to fraud_samples.csv")
