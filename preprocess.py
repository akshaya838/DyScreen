"""
preprocess.py - Module 1: loading, cleaning, encoding, splitting and scaling.
Run on its own to see the preprocessing summary:  python preprocess.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
from feature_engineering import engineer_dataset


def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        print("\n[ERROR] Dataset not found at:", path)
        print("Download the Rello et al. (2020) dataset from Kaggle "
              "(kaggle.com/datasets/luzrello/dyslexia), take the DESKTOP csv file,")
        print("rename it to dataset.csv and place it inside the data/ folder.\n")
        sys.exit(1)
    # sep=None lets pandas detect ';' or ',' automatically (the Kaggle file uses ';')
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def find_column(df, name):
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None


def clean_dataset(df, verbose=True):
    """Duplicates, label encoding and basic checks. Returns cleaned df + label column name."""
    label_col = find_column(df, "Dyslexia")
    if label_col is None:
        raise ValueError("No 'Dyslexia' label column found in the dataset.")

    if verbose:
        print("=" * 60, "\nMODULE 1 - DATA PREPROCESSING\n" + "=" * 60)
        print("Dataset shape:", df.shape)
        print("\nFirst 5 rows:\n", df.head())
        print("\nColumn names (first 20 of", len(df.columns), "):", list(df.columns[:20]))
        missing = df.isnull().sum()
        print("\nMissing values - total:", int(missing.sum()))
        if missing.sum() > 0:
            print(missing[missing > 0])
        print("Duplicate rows:", int(df.duplicated().sum()))

    df = df.drop_duplicates().reset_index(drop=True)

    # Encode label Yes/No -> 1/0
    y = df[label_col].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    df = df[y.notnull()].copy()
    df[TARGET_COLUMN] = y[y.notnull()].astype(int)
    if label_col != TARGET_COLUMN:
        df = df.drop(columns=[label_col])
    df = df.reset_index(drop=True)

    if verbose:
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        print("\nClass distribution (0 = no dyslexia, 1 = dyslexia):")
        for k, v in counts.items():
            print(f"  Class {k} = {v}  ({v / len(df) * 100:.1f}%)")
    return df


def encode_raw_features(df):
    """
    One-hot encode categorical columns of the RAW dataset (Gender, Nativelang, Otherlang).
    Used for analysis of the raw 196 features (feature_selection.py).
    """
    X = df.drop(columns=[TARGET_COLUMN])
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    return X_enc, cat_cols


def build_engineered_dataset(verbose=True):
    """Full Module 1 + Module 2 chain: returns X (engineered features), y, raw_df."""
    raw = load_dataset()
    df = clean_dataset(raw, verbose=verbose)
    age_col = find_column(df, "Age")

    X_raw_enc, cat_cols = encode_raw_features(df)
    X = engineer_dataset(df, age_col=age_col)
    y = df[TARGET_COLUMN].reset_index(drop=True)

    if verbose:
        print("\nCategorical columns one-hot encoded:", cat_cols)
        print("Number of features BEFORE preprocessing (raw):", df.shape[1] - 1)
        print("Number of features after one-hot encoding (raw):", X_raw_enc.shape[1])
        print("Number of ENGINEERED skill-level features used by the model:", X.shape[1])
        print("Engineered features:", list(X.columns))
    return X, y, df


def split_data(X, y):
    """Stratified split keeps the same dyslexia ratio in train and test sets."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def make_imputer_and_scaler():
    """
    Imputer + scaler are placed INSIDE the model pipeline in train_model.py,
    so they are fitted on training data only (no data leakage).
    """
    return SimpleImputer(strategy="median"), StandardScaler()


if __name__ == "__main__":
    X, y, df = build_engineered_dataset(verbose=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    imputer, scaler = make_imputer_and_scaler()
    X_train_s = scaler.fit_transform(imputer.fit_transform(X_train))   # fit on TRAIN only
    X_test_s = scaler.transform(imputer.transform(X_test))            # only transform TEST
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
    print("Train class counts:", y_train.value_counts().sort_index().to_dict())
    print("Test class counts: ", y_test.value_counts().sort_index().to_dict())
    print("Scaled train mean (should be ~0):", np.round(X_train_s.mean(axis=0), 3))
