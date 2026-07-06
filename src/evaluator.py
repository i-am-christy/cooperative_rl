import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from rl_environment import ACTION_RESTRUCTURE


def evaluate_rl_model(results_df):
    """
    Evaluate the RL model's recommendations against true labels.
    Treats ACTION_FLAG as the positive class (predicted default).
    Returns a dict of metrics.
    """
    y_true = results_df["true_label"]
    y_pred = (results_df["action_code"] >= ACTION_RESTRUCTURE).astype(int)
    y_score = results_df["action_code"].map({0: 0.1, 1: 0.5, 2: 0.9})

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score),
    }
    return metrics


def evaluate_baseline(X_train, y_train, X_test, y_test):
    """
    Train a Logistic Regression baseline and evaluate it.
    Returns a dict of metrics in the same format as evaluate_rl_model.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_score),
    }
    return metrics


def print_comparison(rl_metrics, baseline_metrics):
    """
    Print a side-by-side comparison table.
    """
    print("\n=== Model Performance Comparison ===")
    print(f"{'Metric':<15} {'Q-Learning RL':>15} {'Logistic Reg.':>15}")
    print("-" * 47)
    for metric in rl_metrics:
        rl_val = rl_metrics[metric]
        bl_val = baseline_metrics[metric]
        print(f"{metric:<15} {rl_val:>15.3f} {bl_val:>15.3f}")


def plot_rewards(rewards_per_episode, save_path="data/rewards_curve.png"):
    """
    Plot cumulative reward per episode to show convergence.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    window = 50
    smoothed = pd.Series(rewards_per_episode).rolling(window).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode, color="lightgray", alpha=0.5, label="Raw reward")
    plt.plot(smoothed, color="teal", label="Avg reward (50-ep window)")
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Q-Learning Agent Reward Convergence")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessor import run_preprocessing_pipeline
    from rl_agent import train
    from decision_engine import batch_recommend, build_balance_percentiles

    df = load_dataset()
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(df)

    X_train_df = pd.DataFrame(X_train,
        columns=X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]))
    X_test_df = pd.DataFrame(X_test,
        columns=X_test.columns if hasattr(X_test, 'columns') else range(X_test.shape[1]))

    print("Training agent...")
    q_table, rewards = train(X_train_df, y_train)

    balance_low, balance_high = build_balance_percentiles(X_train_df)
    results_df = batch_recommend(X_test_df, y_test, q_table, balance_low, balance_high)

    rl_metrics = evaluate_rl_model(results_df)
    baseline_metrics = evaluate_baseline(X_train, y_train, X_test, y_test)

    print_comparison(rl_metrics, baseline_metrics)
    plot_rewards(rewards)
    print("\nRewards curve saved to data/rewards_curve.png")