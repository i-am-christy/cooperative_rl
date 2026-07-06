import numpy as np
import pandas as pd

ACTION_CONTINUE = 0
ACTION_RESTRUCTURE = 1
ACTION_FLAG = 2
N_ACTIONS = 3

N_BINS = 3
N_STATE_VARS = 4
N_STATES = N_BINS ** N_STATE_VARS


def discretize_state(repayment_ratio, missed_payments, wealth_index, outstanding_balance,
                      balance_low, balance_high):
    """
    Convert continuous borrower features into a discrete state tuple.
    Each variable maps to 0 (low), 1 (medium), or 2 (high).
    balance_low and balance_high are the 33rd and 66th percentiles
    of outstanding_balance from the training set — passed in so this
    function never touches raw data directly.
    """
    if repayment_ratio < 0.33:
        repayment_bin = 0
    elif repayment_ratio < 0.66:
        repayment_bin = 1
    else:
        repayment_bin = 2

    if missed_payments <= 1:
        missed_bin = 0
    elif missed_payments <= 3:
        missed_bin = 1
    else:
        missed_bin = 2

    if wealth_index <= 2:
        wealth_bin = 0
    elif wealth_index == 3:
        wealth_bin = 1
    else:
        wealth_bin = 2

    if outstanding_balance < balance_low:
        balance_bin = 0
    elif outstanding_balance < balance_high:
        balance_bin = 1
    else:
        balance_bin = 2

    return (repayment_bin, missed_bin, wealth_bin, balance_bin)


def state_to_index(state_tuple):
    """
    Convert a (r, m, w, b) tuple into a single integer index for the Q-table.
    Uses base-3 encoding.
    """
    r, m, w, b = state_tuple
    return r * (N_BINS ** 3) + m * (N_BINS ** 2) + w * (N_BINS ** 1) + b * (N_BINS ** 0)


def get_reward(action, repayment_ratio, missed_payments):
    """
    Return reward signal based on action taken and borrower outcome.
    +1 if repayment_ratio >= 0.66 and missed_payments <= 1  (good outcome)
    -1 if repayment_ratio < 0.33 or missed_payments >= 4   (bad outcome)
     0 otherwise                                             (neutral)
    """
    if repayment_ratio >= 0.66 and missed_payments <= 1:
        return 1
    elif repayment_ratio < 0.33 or missed_payments >= 4:
        return -1
    else:
        return 0


def build_balance_percentiles(X_train_df):
    """
    Compute 33rd and 66th percentiles of outstanding_balance from training data.
    Returns (low_cutpoint, high_cutpoint).
    Call this once before training — pass the results into discretize_state.
    """
    low = np.percentile(X_train_df["outstanding_balance"], 33)
    high = np.percentile(X_train_df["outstanding_balance"], 66)
    return low, high


if __name__ == "__main__":
    low, high = 50000, 150000
    s = discretize_state(0.8, 0, 3, 30000, low, high)
    print("State tuple:", s)
    print("State index:", state_to_index(s))
    print("Reward (good borrower):", get_reward(ACTION_CONTINUE, 0.8, 0))
    print("Reward (bad borrower):", get_reward(ACTION_FLAG, 0.1, 5))
    print("Reward (neutral):", get_reward(ACTION_RESTRUCTURE, 0.5, 2))