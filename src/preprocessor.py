import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
TEST_SIZE = 0.30


def drop_identifier(df):
    return df.drop(columns=["loan_id"])


def encode_categoricals(df):
    df = pd.get_dummies(df, columns=["occupation", "employment_sector"], drop_first=True)
    return df


def split_data(df, target_col="default_flag"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def run_preprocessing_pipeline(df):
    df = drop_identifier(df)
    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_features(X_train, X_test)
    X_train, y_train = apply_smote(X_train, y_train)
    return X_train, X_test, y_train, y_test

def get_raw_splits(df, target_col="default_flag"):
    """
    Return raw unscaled train/test split — for RL training only.
    No scaling, no SMOTE. Just split and separate features from target.
    """
    df = drop_identifier(df)
    df = encode_categoricals(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(df)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:\n", pd.Series(y_train).value_counts(normalize=True).round(3))
    print("y_test distribution:\n", pd.Series(y_test).value_counts(normalize=True).round(3))