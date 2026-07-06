import numpy as np
from rl_environment import (
    discretize_state, state_to_index,
    build_balance_percentiles,
    ACTION_CONTINUE, ACTION_RESTRUCTURE, ACTION_FLAG
)

ACTION_LABELS = {
    ACTION_CONTINUE: "Continue current repayment schedule",
    ACTION_RESTRUCTURE: "Restructure loan — extend tenure and reduce installment",
    ACTION_FLAG: "Flag borrower for administrator follow-up",
}

RISK_LABELS = {
    ACTION_CONTINUE: "Low Risk",
    ACTION_RESTRUCTURE: "Medium Risk",
    ACTION_FLAG: "High Risk",
}


def recommend(borrower_row, q_table, balance_low, balance_high):
    """
    Given a single borrower (pandas Series or dict),
    return a recommendation dict with keys:
    - action_code: int (0, 1, or 2)
    - action_label: str
    - risk_level: str
    - explanation: str
    """
    repayment_ratio = borrower_row["repayment_ratio"]

    missed_payments = borrower_row["missed_payments"]
    wealth_index = borrower_row["wealth_index"]
    outstanding_balance = borrower_row["outstanding_balance"]

    state = discretize_state(
        repayment_ratio, missed_payments, wealth_index, outstanding_balance,
        balance_low, balance_high
    )
    state_idx = state_to_index(state)

    action = int(np.argmax(q_table[state_idx]))

    explanation = (
        f"Repayment ratio {repayment_ratio:.2f}, "
        f"{missed_payments} missed payment(s), "
        f"wealth index {wealth_index}. "
        f"Recommended action: {ACTION_LABELS[action]}."
    )

    return {
        "action_code": action,
        "action_label": ACTION_LABELS[action],
        "risk_level": RISK_LABELS[action],
        "explanation": explanation,
    }


def batch_recommend(X_df, y_true, q_table, balance_low, balance_high):
    """
    Run recommend() on every row in X_df.
    Returns a DataFrame with columns:
    action_code, action_label, risk_level, true_label
    """
    import pandas as pd

    records = []
    y_true_reset = y_true.reset_index(drop=True)

    for i, (_, row) in enumerate(X_df.iterrows()):
        result = recommend(row, q_table, balance_low, balance_high)
        records.append({
            "action_code": result["action_code"],
            "action_label": result["action_label"],
            "risk_level": result["risk_level"],
            "true_label": int(y_true_reset.iloc[i]),
        })

    return pd.DataFrame(records)


def summarise_recommendations(results_df):
    """
    Print a summary of action distribution across all borrowers.
    """
    print("=== Recommendation Summary ===")
    print(results_df["action_label"].value_counts())
    print("\n=== Risk Level Distribution ===")
    print(results_df["risk_level"].value_counts())


if __name__ == "__main__":
    import pandas as pd
    from data_loader import load_dataset
    from preprocessor import run_preprocessing_pipeline
    from rl_agent import train

    df = load_dataset()
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(df)

    X_train_df = pd.DataFrame(X_train,
        columns=X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]))
    X_test_df = pd.DataFrame(X_test,
        columns=X_test.columns if hasattr(X_test, 'columns') else range(X_test.shape[1]))

    print("Training agent...")
    q_table, _ = train(X_train_df, y_train)
    balance_low, balance_high = build_balance_percentiles(X_train_df)

    print("Unique state indices in test set:")
    test_states = []
    for _, row in X_test_df.iterrows():
        s = discretize_state(
            row["repayment_ratio"], row["missed_payments"],
            row["wealth_index"], row["outstanding_balance"],
            balance_low, balance_high
        )
        test_states.append(state_to_index(s))
    import collections
    print(collections.Counter(test_states).most_common(10))

    sample = X_test_df.iloc[0]
    result = recommend(sample, q_table, balance_low, balance_high)
    print("\n=== Sample Recommendation ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    results_df = batch_recommend(X_test_df, y_test, q_table, balance_low, balance_high)
    summarise_recommendations(results_df)

    for state_idx in [54, 55, 56, 2, 1, 29, 28]:
        print(f"State {state_idx}: {q_table[state_idx].round(3)}")