import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_dataset
from preprocessor import run_preprocessing_pipeline
from rl_agent import train
from decision_engine import recommend, batch_recommend
from rl_environment import build_balance_percentiles, ACTION_CONTINUE, ACTION_RESTRUCTURE, ACTION_FLAG

RISK_COLORS = {
    "Low Risk": "#02C39A",
    "Medium Risk": "#028090",
    "High Risk": "#D62828",
}


@st.cache_resource
def load_model():
    """Train the model once and cache it."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(df)
    X_train_df = pd.DataFrame(X_train,
        columns=X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]))
    X_test_df = pd.DataFrame(X_test,
        columns=X_test.columns if hasattr(X_test, 'columns') else range(X_test.shape[1]))
    q_table, rewards = train(X_train_df, y_train)
    balance_low, balance_high = build_balance_percentiles(X_train_df)
    results_df = batch_recommend(X_test_df, y_test, q_table, balance_low, balance_high)
    return q_table, balance_low, balance_high, results_df, rewards


def plot_portfolio(results_df):
    """Bar chart of action distribution."""
    counts = results_df["action_label"].value_counts()
    risk_by_label = results_df.drop_duplicates("action_label").set_index("action_label")["risk_level"]
    colors = [RISK_COLORS[risk_by_label[label]] for label in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Borrowers")
    ax.set_title("Recommendation Distribution")

    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + max(counts.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 str(value), va="center")

    fig.tight_layout()
    return fig


def plot_rewards(rewards):
    """Smoothed reward convergence curve."""
    window = 50
    smoothed = pd.Series(rewards).rolling(window).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, color="lightgray", alpha=0.5, label="Raw reward")
    ax.plot(smoothed, color="teal", label="Avg reward (50-ep window)")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Agent Training Convergence")
    ax.legend()
    fig.tight_layout()
    return fig


st.set_page_config(page_title="CoopGuard — Loan Risk Dashboard", layout="wide")
st.title("CoopGuard")
st.caption("Q-Learning Loan Repayment Risk System — Cooperative Administrator Dashboard")

with st.spinner("Loading model..."):
    q_table, balance_low, balance_high, results_df, rewards = load_model()

st.success("Model ready.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single Borrower Assessment")

    repayment_ratio = st.slider("Repayment Ratio", 0.0, 1.0, 0.65, step=0.01)
    missed_payments = st.slider("Missed Payments", 0, 6, 0)
    wealth_index = st.slider("Wealth Index", 1, 5, 3)
    outstanding_balance = st.number_input("Outstanding Balance (₦)", 0, 2000000, 50000, step=10000)

    if st.button("Get Recommendation"):
        borrower = {
            "repayment_ratio": repayment_ratio,
            "missed_payments": missed_payments,
            "wealth_index": wealth_index,
            "outstanding_balance": outstanding_balance,
        }
        result = recommend(borrower, q_table, balance_low, balance_high)

        st.metric("Recommended Action", result["action_label"])

        if result["risk_level"] == "Low Risk":
            st.success(f"{result['risk_level']}: {result['action_label']}")
        elif result["risk_level"] == "Medium Risk":
            st.warning(f"{result['risk_level']}: {result['action_label']}")
        else:
            st.error(f"{result['risk_level']}: {result['action_label']}")

        st.write(result["explanation"])

with col2:
    st.subheader("Portfolio Summary")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Borrowers", len(results_df))
    m2.metric("Flagged High Risk", int((results_df["risk_level"] == "High Risk").sum()))
    m3.metric("Low Risk", int((results_df["risk_level"] == "Low Risk").sum()))

    fig_portfolio = plot_portfolio(results_df)
    st.pyplot(fig_portfolio)

st.divider()
st.subheader("Agent Training Convergence")
fig_rewards = plot_rewards(rewards)
st.pyplot(fig_rewards)