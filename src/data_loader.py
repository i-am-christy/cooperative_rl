import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 5000
OUTPUT_PATH = "data/raw/cooperative_loans.csv"

def simulate_dataset(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """
    Simulate a cooperative loan dataset with realistic distribution.
    Return a raw DataFrame. Does not save to disk.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 75, size=n_samples)

    sectors = ["public", "private", "self_employed", "informal"]
    sector_weights = [0.40, 0.25, 0.20, 0.15]
    employment_sector = rng.choice(sectors, size=n_samples, p=sector_weights)

    occupations = ["civil_servant", "trader", "teacher", "artisan", "farmer", "other"]
    occ_weights = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
    occupation = rng.choice(occupations, size=n_samples, p=occ_weights)

    monthly_income = rng.exponential(scale=80000, size=n_samples)
    monthly_income = np.clip(monthly_income, 50_000, 500_000)

    wealth_index = rng.choice(
        [1, 2, 3, 4, 5],
        size=n_samples,
        p=[0.35, 0.30, 0.20, 0.10, 0.05]
    )

    loan_amount = rng.exponential(scale=350000, size=n_samples)
    loan_amount = np.clip(loan_amount, 50_000, 2_000_000)

    tenure_months = rng.normal(loc=12, scale=6, size=n_samples)
    tenure_months = np.clip(tenure_months, 6, 36).astype(int)

    interest_rate = rng.normal(loc=12, scale=4, size=n_samples)
    interest_rate = np.clip(interest_rate, 5, 25).round(1)

    missed_payments = rng.choice(
        range(7),
        size=n_samples,
        p=[0.55, 0.20, 0.10, 0.07, 0.04, 0.02, 0.02]
    )

    good_share = 0.80
    n_good = int(n_samples * good_share)
    n_bad = n_samples - n_good
    good_borrowers = rng.uniform(0.7, 1.0, size=n_good)
    bad_borrowers = rng.uniform(0.0, 0.4, size=n_bad)
    repayment_ratio = np.concatenate([good_borrowers, bad_borrowers])
    rng.shuffle(repayment_ratio)

    outstanding_balance = (loan_amount * (1 - repayment_ratio)).round(2)

    default_flag = ((missed_payments >= 3) | (repayment_ratio < 0.28)).astype(int)

    df = pd.DataFrame({
        "loan_id": [f"LN{str(i).zfill(5)}" for i in range(n_samples)],
        "age": age,
        "occupation": occupation,
        "employment_sector": employment_sector,
        "monthly_income": monthly_income.round(2),
        "wealth_index": wealth_index,
        "loan_amount": loan_amount.round(2),
        "tenure_months": tenure_months,
        "interest_rate": interest_rate,
        "missed_payments": missed_payments,
        "repayment_ratio": repayment_ratio.round(4),
        "outstanding_balance": outstanding_balance,
        "default_flag": default_flag,
    })

    return df


def save_dataset(df, path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path} — {df.shape[0]} rows, {df.shape[1]} columns")


def load_dataset(path=OUTPUT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No dataset found at {path}. Run simulate_dataset() first.")
    return pd.read_csv(path)


def inspect_dataset(df):
    print("=== Shape ===")
    print(df.shape)
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== Null Counts ===")
    print(df.isnull().sum())
    print("\n=== Default Flag Distribution ===")
    print(df["default_flag"].value_counts(normalize=True).round(3))
    print("\n=== Numeric Summary ===")
    print(df.describe().round(2))


if __name__ == "__main__":
    df = simulate_dataset()
    inspect_dataset(df)
    save_dataset(df)