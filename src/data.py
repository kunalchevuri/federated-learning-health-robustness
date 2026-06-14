import pyreadstat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load the XPT file ──────────────────────────────────────────
def load_brfss(path="data/LLCP2023.XPT"):
    df, meta = pyreadstat.read_xport(path, encoding="latin1")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

# ── 2. Select and clean columns ───────────────────────────────────
def preprocess(df):
    cols = ["MENTHLTH", "EXERANY2", "GENHLTH", "_RFBMI5", "ADDEPEV3"]
    df = df[cols].copy()

    df["MENTHLTH"] = df["MENTHLTH"].replace({77: np.nan, 99: np.nan, 88: 0})
    df["EXERANY2"] = df["EXERANY2"].replace({7: np.nan, 9: np.nan})
    df["GENHLTH"]  = df["GENHLTH"].replace({7: np.nan, 9: np.nan})
    df["_RFBMI5"]  = df["_RFBMI5"].replace({9: np.nan})
    df["ADDEPEV3"] = df["ADDEPEV3"].replace({7: np.nan, 9: np.nan})

    df = df.dropna()
    print(f"After cleaning: {len(df)} rows")

    df["ADDEPEV3"] = df["ADDEPEV3"].apply(lambda x: 1 if x == 1 else 0)
    df["EXERANY2"] = df["EXERANY2"].apply(lambda x: 1 if x == 1 else 0)
    df["MENTHLTH"] = df["MENTHLTH"] / 30.0

    print(f"Depression prevalence: {df['ADDEPEV3'].mean():.2%}")
    return df

# ── 3. Train/test split ───────────────────────────────────────────
def split_data(df):
    X = df.drop("ADDEPEV3", axis=1).values.astype(np.float32)
    y = df["ADDEPEV3"].values.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

# ── 4. Dirichlet partitioning across clients ──────────────────────
def partition_data(X_train, y_train, num_clients=20, alpha=0.5, seed=42):
    np.random.seed(seed)
    num_classes = 2
    client_data = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(y_train == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        splits = np.split(class_indices, np.cumsum(proportions[:-1]))
        for i, split in enumerate(splits):
            client_data[i].extend(split.tolist())

    partitions = []
    for i in range(num_clients):
        idx = client_data[i]
        partitions.append((X_train[idx], y_train[idx]))
        print(f"Client {i:2d}: {len(idx)} samples | "
              f"depression rate: {y_train[idx].mean():.2%}")
    return partitions

# ── 5. Standard dataset: UCI Breast Cancer Wisconsin ─────────────
def load_breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = (1 - data.target).astype(np.int64)  # 1=malignant, 0=benign
    pos_rate = y.mean()
    print(f"Loaded Breast Cancer: {len(y)} samples | malignant rate: {pos_rate:.2%}")
    return X, y


def split_arrays(X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_brfss()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    partitions = partition_data(X_train, y_train, num_clients=20, alpha=0.5)
    print("\nDone! Data pipeline working correctly.")