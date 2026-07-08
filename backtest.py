"""
backtest.py

Walk-forward backtest of the TennisModel.

For each test match, the model is initialized with only matches that
occurred strictly before that match's date — no lookahead bias.

Usage:
    python backtest.py

Requires:
    - lr_model.pkl and lr_scaler.pkl (produced by train.py)
    - ATP dataset via kagglehub
"""

import pandas as pd
import numpy as np
import pickle
import kagglehub
import os
from model import TennisModel

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TEST_START_DATE = "2023-01-01"
SAMPLE_EVERY_N  = 20
# ─────────────────────────────────────────────────────────────────────────────


def run_backtest(df, lr_model, lr_scaler):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    test_df      = df[df['Date'] >= TEST_START_DATE].reset_index(drop=True)
    sampled_test = test_df.iloc[::SAMPLE_EVERY_N].reset_index(drop=True)
    total        = len(sampled_test)

    print(f"Total matches : {len(df):,}")
    print(f"Test matches  : {len(test_df):,} (from {TEST_START_DATE})")
    print(f"Sampled       : {total:,} predictions\n")

    results, skipped = [], 0

    for i, row in sampled_test.iterrows():
        if i % 50 == 0:
            print(f"  Progress: {i}/{total} ({100*i//max(total,1)}%)")

        match_date = row['Date']
        p1         = row['Player_1']
        p2         = row['Player_2']
        surface    = row['Surface']
        winner     = row['Winner']

        history_df = df[df['Date'] < match_date]

        p1_seen = ((history_df['Player_1'] == p1) | (history_df['Player_2'] == p1)).any()
        p2_seen = ((history_df['Player_1'] == p2) | (history_df['Player_2'] == p2)).any()
        if not p1_seen or not p2_seen:
            skipped += 1
            continue

        tennis_model = TennisModel(history_df)
        probs        = tennis_model.predict(p1, p2, surface, lr_model, lr_scaler,
                                            match_date=match_date)
        p1_prob = probs[0]

        if p1_prob == 0.5 and probs[1] == 0.5:
            skipped += 1
            continue

        predicted_winner = p1 if p1_prob >= 0.5 else p2
        correct          = int(predicted_winner == winner)
        actual_p1_won    = int(winner == p1)

        p_clipped    = np.clip(p1_prob, 1e-7, 1 - 1e-7)
        log_loss_val = -(actual_p1_won * np.log(p_clipped) +
                         (1 - actual_p1_won) * np.log(1 - p_clipped))

        results.append({
            'date'         : match_date.date(),
            'p1'           : p1,
            'p2'           : p2,
            'surface'      : surface,
            'winner'       : winner,
            'p1_prob'      : round(p1_prob, 4),
            'predicted'    : predicted_winner,
            'correct'      : correct,
            'log_loss'     : round(log_loss_val, 4),
        })

    results_df = pd.DataFrame(results)
    n          = len(results_df)
    accuracy   = results_df['correct'].mean()
    avg_ll     = results_df['log_loss'].mean()
    se         = np.sqrt(accuracy * (1 - accuracy) / n)
    ci_low     = accuracy - 1.96 * se
    ci_high    = accuracy + 1.96 * se

    summary = {
        'matches_tested' : n,
        'matches_skipped': skipped,
        'accuracy'       : f"{accuracy*100:.1f}%",
        'accuracy_95ci'  : f"{ci_low*100:.1f}% – {ci_high*100:.1f}%",
        'avg_log_loss'   : round(avg_ll, 4),
    }

    return results_df, summary


if __name__ == "__main__":
    # load data
    path     = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
    csv_file = os.path.join(path, "atp_tennis.csv")
    df       = pd.read_csv(csv_file, low_memory=False)

    # load trained model
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('lr_scaler.pkl', 'rb') as f:
        lr_scaler = pickle.load(f)

    results_df, summary = run_backtest(df, lr_model, lr_scaler)

    print("\n" + "="*45)
    print("BACKTEST RESULTS")
    print("="*45)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")

    results_df.to_csv('backtest_results.csv', index=False)
    print("\nFull results saved to backtest_results.csv")
