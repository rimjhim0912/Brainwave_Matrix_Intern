# download_data.py

import pandas as pd

def download_dataset(save_path="creditcard.csv"):
    url = "https://www.openml.org/data/get_csv/1673544/creditcard.csv"
    print("Downloading dataset...")
    df = pd.read_csv(url)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    download_dataset()
