import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split


def load_vendor_invoice_data(db_path):
    print("=" * 50)
    print("Database Path:", db_path)

    conn = sqlite3.connect(db_path)

    print("Connected Successfully!")

    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )

    print("\nTables:")
    print(tables)

    df = pd.read_sql_query(
        "SELECT * FROM vendor_invoice",
        conn
    )

    print("\nData Loaded Successfully")
    print(df.head())

    conn.close()

    return df


def prepare_features(df):
    X = df[["Dollars"]]
    y = df["Freight"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )