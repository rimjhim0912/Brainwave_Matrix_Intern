import pandas as pd

url = "https://www.openml.org/data/get_csv/1673544/creditcard.csv"
df = pd.read_csv(url)

# Verify it has fraud cases
print("Shape:", df.shape)
print("Fraud cases:", (df['Class'] == 1).sum())
print("Not Fraud cases:", (df['Class'] == 0).sum())

# Save
df.to_csv("creditcard.csv", index=False)
print("✅ Dataset downloaded and saved.")
