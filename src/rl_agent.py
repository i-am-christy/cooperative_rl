import numpy as np
from rl_environment import (
    discretize_state, state_to_index, get_reward,
    build_balance_percentiles, N_STATES, N_ACTIONS,
    ACTION_CONTINUE, ACTION_RESTRUCTURE, ACTION_FLAG
)
import os

ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.1
N_EPISODES = 1000
RANDOM_SEED = 42


def build_qtable():
    return np.zeros((N_STATES, N_ACTIONS))


def epsilon_greedy(q_table, state_index, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, N_ACTIONS)
    return int(np.argmax(q_table[state_index]))


def update_qtable(q_table, state_idx, action, reward, next_state_idx):
    best_next = np.max(q_table[next_state_idx])
    td_target = reward + GAMMA * best_next
    q_table[state_idx, action] += ALPHA * (td_target - q_table[state_idx, action])


def train(X_train_df, y_train, n_episodes=N_EPISODES):
    rng = np.random.default_rng(RANDOM_SEED)
    q_table = build_qtable()
    balance_low, balance_high = build_balance_percentiles(X_train_df)

    rewards_per_episode = []
    epsilon_decay = (EPSILON_START - EPSILON_END) / n_episodes
    epsilon = EPSILON_START

    for episode in range(n_episodes):
        idx = rng.integers(0, len(X_train_df))
        borrower = X_train_df.iloc[idx]
        is_defaulter = int(y_train.iloc[idx])

        # repayment_ratio is scaled 0-1, maps directly — no change needed
        repayment_ratio = borrower["repayment_ratio"]

        # missed_payments was scaled from 0-6 range, unscale it
        missed_payments = round(borrower["missed_payments"] * 6)

        # wealth_index was scaled from 1-5 range, unscale it
        wealth_index = round(borrower["wealth_index"] * 4 + 1)

        # outstanding_balance — use raw percentile values, already unscaled
        outstanding_balance = borrower["outstanding_balance"]

        episode_reward = 0

        for step in range(12):
            state = discretize_state(
                repayment_ratio, missed_payments, wealth_index, outstanding_balance,
                balance_low, balance_high
            )
            state_idx = state_to_index(state)

            action = epsilon_greedy(q_table, state_idx, epsilon, rng)
            reward = get_reward(action, repayment_ratio, missed_payments)
            episode_reward += reward

            if is_defaulter and step > 6:
                update_qtable(q_table, state_idx, action, reward, state_idx)
                break

            if is_defaulter:
                next_repayment = max(0.0, repayment_ratio - 0.05)
                next_missed = missed_payments if action == ACTION_RESTRUCTURE else missed_payments + 1
            else:
                next_repayment = min(1.0, repayment_ratio + 0.05)
                next_missed = max(0, missed_payments - 1) if action == ACTION_RESTRUCTURE else missed_payments

            next_state = discretize_state(
                next_repayment, next_missed, wealth_index, outstanding_balance,
                balance_low, balance_high
            )
            next_state_idx = state_to_index(next_state)

            update_qtable(q_table, state_idx, action, reward, next_state_idx)

            repayment_ratio = next_repayment
            missed_payments = next_missed

        rewards_per_episode.append(episode_reward)
        epsilon = max(EPSILON_END, epsilon - epsilon_decay)

    return q_table, rewards_per_episode

def save_qtable(q_table, path="data/qtable.npy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, q_table)
    print(f"Q-table saved to {path}")

def load_qtable(path="data/qtable.npy"):
    return np.load(path)

if __name__ == "__main__":
    import pandas as pd
    from data_loader import load_dataset
    from preprocessor import run_preprocessing_pipeline

    df = load_dataset()
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(df)

    X_train_df = pd.DataFrame(X_train, columns=X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]))

    print("Training Q-learning agent...")
    q_table, rewards = train(X_train_df, y_train)

    print("Training complete.")
    print("Q-table shape:", q_table.shape)
    print("Mean reward (first 100 episodes):", round(sum(rewards[:100]) / 100, 3))
    print("Mean reward (last 100 episodes):", round(sum(rewards[-100:]) / 100, 3))
    print("Q-table sample (state 57):", q_table[57].round(3))