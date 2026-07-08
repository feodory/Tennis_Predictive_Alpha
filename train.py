"""
train.py

Builds the feature table and trains the logistic regression model.

Run this once to produce:
    - feature_df.csv   : precomputed features for all matches from START_DATE onward
    - lr_model.pkl     : fitted LogisticRegression
    - lr_scaler.pkl    : fitted StandardScaler

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import pickle
import kagglehub
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
START_DATE      = "2015-01-01"   # build features from this date onward
TRAIN_END_DATE  = "2023-01-01"   # train on data before this date
SAMPLE_EVERY_N  = 20             # sample every Nth match (reduces runtime)
BASE_ELO        = 1500
K_FACTOR        = 32
FEATURES        = ['elo_diff', 'swr_diff', 'p1_h2h', 'p1_co']
# ─────────────────────────────────────────────────────────────────────────────


def load_data():
    path     = kagglehub.dataset_download("dissfya/atp-tennis-2000-2023daily-pull")
    csv_file = os.path.join(path, "atp_tennis.csv")
    df       = pd.read_csv(csv_file, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df):,} matches ({df['Date'].min().year}–{df['Date'].max().year})")
    return df


def compute_elo_history(df):
    ratings = {}
    history = []
    for _, row in df.iterrows():
        p1, p2, winner = row['Player_1'], row['Player_2'], row['Winner']
        r1 = ratings.get(p1, BASE_ELO)
        r2 = ratings.get(p2, BASE_ELO)
        history.append({'date': row['Date'], 'p1': p1, 'p2': p2,
                        'p1_elo': r1, 'p2_elo': r2})
        expected_p1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        actual_p1   = 1 if winner == p1 else 0
        ratings[p1] = r1 + K_FACTOR * (actual_p1 - expected_p1)
        ratings[p2] = r2 + K_FACTOR * ((1 - actual_p1) - (1 - expected_p1))
    return pd.DataFrame(history)


def get_surface_win_rate(history_df, player, surface):
    matches = history_df[(history_df['Player_1'] == player) |
                         (history_df['Player_2'] == player)]
    surface_matches = matches[matches['Surface'] == surface]
    if surface_matches.empty:
        return 0.5
    return (surface_matches['Winner'] == player).sum() / len(surface_matches)


def get_h2h_score(history_df, p1, p2, surface):
    h2h = history_df[
        ((history_df['Player_1'] == p1) & (history_df['Player_2'] == p2)) |
        ((history_df['Player_1'] == p2) & (history_df['Player_2'] == p1))
    ]
    if h2h.empty:
        p1_swr = get_surface_win_rate(history_df, p1, surface)
        p2_swr = get_surface_win_rate(history_df, p2, surface)
        total  = p1_swr + p2_swr
        return p1_swr / total if total > 0 else 0.5
    return (h2h['Winner'] == p1).sum() / len(h2h)


def get_common_opponent_score(history_df, p1, p2):
    def get_opponents(player):
        won  = set(history_df[history_df['Winner'] == player].apply(
            lambda x: x['Player_2'] if x['Player_1'] == player else x['Player_1'], axis=1))
        lost = set(history_df[
            (history_df['Winner'] != player) &
            ((history_df['Player_1'] == player) | (history_df['Player_2'] == player))
        ]['Winner'])
        return won | lost

    common = get_opponents(p1) & get_opponents(p2)
    if not common:
        return 0.5

    p1_wins, p2_wins = 0.0, 0.0
    for opp in common:
        p1_m = history_df[
            ((history_df['Player_1'] == p1) & (history_df['Player_2'] == opp)) |
            ((history_df['Player_1'] == opp) & (history_df['Player_2'] == p1))
        ]
        if not p1_m.empty:
            p1_wins += (p1_m['Winner'] == p1).mean()
        p2_m = history_df[
            ((history_df['Player_1'] == p2) & (history_df['Player_2'] == opp)) |
            ((history_df['Player_1'] == opp) & (history_df['Player_2'] == p2))
        ]
        if not p2_m.empty:
            p2_wins += (p2_m['Winner'] == p2).mean()

    total = p1_wins + p2_wins
    return p1_wins / total if total > 0 else 0.5


def build_feature_table(df):
    print("\nComputing Elo history...")
    elo_history = compute_elo_history(df)
    elo_history['date'] = pd.to_datetime(elo_history['date'])
    print(f"Elo history built: {len(elo_history):,} matches")

    sample_df = df[df['Date'] >= START_DATE].iloc[::SAMPLE_EVERY_N].reset_index(drop=True)
    print(f"\nBuilding features for {len(sample_df):,} matches...\n")

    rows, skipped = [], 0

    for i, row in sample_df.iterrows():
        if i % 100 == 0:
            print(f"  {i}/{len(sample_df)} ({100*i//max(len(sample_df),1)}%)")

        match_date = row['Date']
        p1, p2     = row['Player_1'], row['Player_2']
        surface    = row['Surface']
        winner     = row['Winner']

        history_df = df[df['Date'] < match_date]
        elo_before = elo_history[elo_history['date'] < match_date]

        if history_df.empty:
            skipped += 1
            continue

        p1_elo_rows = elo_before[(elo_before['p1'] == p1) | (elo_before['p2'] == p1)]
        p2_elo_rows = elo_before[(elo_before['p1'] == p2) | (elo_before['p2'] == p2)]

        if p1_elo_rows.empty or p2_elo_rows.empty:
            skipped += 1
            continue

        last_p1 = p1_elo_rows.iloc[-1]
        p1_elo  = last_p1['p1_elo'] if last_p1['p1'] == p1 else last_p1['p2_elo']
        last_p2 = p2_elo_rows.iloc[-1]
        p2_elo  = last_p2['p1_elo'] if last_p2['p1'] == p2 else last_p2['p2_elo']

        p1_swr = get_surface_win_rate(history_df, p1, surface)
        p2_swr = get_surface_win_rate(history_df, p2, surface)

        rows.append({
            'date'    : match_date.date(),
            'p1'      : p1, 'p2': p2, 'surface': surface,
            'elo_diff': round(p1_elo - p2_elo, 1),
            'swr_diff': round(p1_swr - p2_swr, 4),
            'p1_h2h'  : round(get_h2h_score(history_df, p1, p2, surface), 4),
            'p1_co'   : round(get_common_opponent_score(history_df, p1, p2), 4),
            'outcome' : 1 if winner == p1 else 0,
        })

    feature_df = pd.DataFrame(rows)
    print(f"\nFeature table: {len(feature_df):,} rows, {skipped} skipped")
    return feature_df


def train(feature_df):
    feature_df = feature_df.copy()
    feature_df['date'] = pd.to_datetime(feature_df['date'])

    train_df = feature_df[feature_df['date'] < TRAIN_END_DATE]
    test_df  = feature_df[feature_df['date'] >= TRAIN_END_DATE]

    print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")

    X_train = train_df[FEATURES].values
    y_train = train_df['outcome'].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df['outcome'].values

    scaler        = StandardScaler()
    X_train_sc    = scaler.fit_transform(X_train)
    X_test_sc     = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_sc))
    logloss  = log_loss(y_test, model.predict_proba(X_test_sc))
    n        = len(y_test)
    se       = np.sqrt(accuracy * (1 - accuracy) / n)

    print(f"\nTest accuracy : {accuracy*100:.1f}%")
    print(f"95% CI        : {(accuracy - 1.96*se)*100:.1f}% – {(accuracy + 1.96*se)*100:.1f}%")
    print(f"Log-loss      : {logloss:.4f}")

    print("\nLearned feature weights:")
    for feat, coef in zip(FEATURES, model.coef_[0]):
        print(f"  {feat:<12}: {coef:.4f}")

    return model, scaler


if __name__ == "__main__":
    df         = load_data()
    feature_df = build_feature_table(df)
    feature_df.to_csv('feature_df.csv', index=False)
    print("\nSaved feature_df.csv")

    model, scaler = train(feature_df)

    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('lr_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved lr_model.pkl and lr_scaler.pkl")
