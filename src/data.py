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

# ── 2b. Expanded feature set (feature-choice sensitivity ablation) ─
# Adds 8 literature-standard depression-risk covariates (demographics +
# behavioral/physical health) to the original 4-feature set, so the main
# heterogeneity/noise findings can be checked for robustness to feature
# choice rather than resting on 4 hand-picked variables. Missing-value
# codes follow the published BRFSS 2023 codebook conventions (7/77/99/9
# family = don't know/refused, 88 = "none" recoded to 0 for day counts).
EXPANDED_COLS = [
    "MENTHLTH", "EXERANY2", "GENHLTH", "_RFBMI5",   # original 4
    "PHYSHLTH",   # days physical health not good, past 30 days
    "_AGEG5YR",   # age group (13 categories)
    "SEXVAR",     # sex
    "EDUCA",      # education level
    "INCOME3",    # income category
    "EMPLOY1",    # employment status
    "_SMOKER3",   # smoking status
    "_RFDRHV8",   # heavy drinking flag
    "ADDEPEV3",   # target
]


def preprocess_expanded(df):
    df = df[EXPANDED_COLS].copy()

    df["MENTHLTH"] = df["MENTHLTH"].replace({77: np.nan, 99: np.nan, 88: 0})
    df["PHYSHLTH"] = df["PHYSHLTH"].replace({77: np.nan, 99: np.nan, 88: 0})
    df["EXERANY2"] = df["EXERANY2"].replace({7: np.nan, 9: np.nan})
    df["GENHLTH"]  = df["GENHLTH"].replace({7: np.nan, 9: np.nan})
    df["_RFBMI5"]  = df["_RFBMI5"].replace({9: np.nan})
    df["_AGEG5YR"] = df["_AGEG5YR"].replace({14: np.nan})
    df["EDUCA"]    = df["EDUCA"].replace({9: np.nan})
    df["INCOME3"]  = df["INCOME3"].replace({77: np.nan, 99: np.nan})
    df["EMPLOY1"]  = df["EMPLOY1"].replace({9: np.nan})
    df["_SMOKER3"] = df["_SMOKER3"].replace({9: np.nan})
    df["_RFDRHV8"] = df["_RFDRHV8"].replace({9: np.nan})
    df["ADDEPEV3"] = df["ADDEPEV3"].replace({7: np.nan, 9: np.nan})

    df = df.dropna()
    print(f"[expanded] After cleaning: {len(df)} rows")

    df["ADDEPEV3"] = df["ADDEPEV3"].apply(lambda x: 1 if x == 1 else 0)
    df["EXERANY2"] = df["EXERANY2"].apply(lambda x: 1 if x == 1 else 0)
    df["MENTHLTH"] = df["MENTHLTH"] / 30.0
    df["PHYSHLTH"] = df["PHYSHLTH"] / 30.0
    # Remaining categorical covariates (_AGEG5YR, SEXVAR, EDUCA, INCOME3,
    # EMPLOY1, _SMOKER3, _RFDRHV8, GENHLTH, _RFBMI5) are left as raw ordinal
    # integer codes. NOTE: the BRFSS pipeline does NOT standardize features --
    # split_data() feeds raw values straight to the model, matching the
    # original 4-feature pipeline exactly. Feature ranges therefore differ
    # (MENTHLTH/PHYSHLTH in [0,1], _AGEG5YR in [1,13], INCOME3 in [1,11]),
    # which the BinaryMLP must absorb in its first layer. Only the Breast
    # Cancer pipeline applies StandardScaler. This is a known limitation of
    # the expanded-feature ablation and is disclosed in the paper.

    print(f"[expanded] Depression prevalence: {df['ADDEPEV3'].mean():.2%}")
    return df


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