import numpy as np
import pandas as pd

BASE_ELO = 1500
K_FACTOR = 32


def compute_elo_ratings(df):
    """
    Single chronological pass through all matches.
    Computes each player's Elo rating after every match.

    Returns:
        ratings     : dict {player_name: final_elo_rating}
        history_df  : DataFrame of each player's Elo BEFORE each match
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    ratings = {}
    history = []

    for _, row in df.iterrows():
        p1, p2, winner = row['Player_1'], row['Player_2'], row['Winner']

        r1 = ratings.get(p1, BASE_ELO)
        r2 = ratings.get(p2, BASE_ELO)

        history.append({
            'date': row['Date'], 'p1': p1, 'p2': p2,
            'p1_elo_before': r1, 'p2_elo_before': r2
        })

        expected_p1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        actual_p1   = 1 if winner == p1 else 0

        ratings[p1] = r1 + K_FACTOR * (actual_p1 - expected_p1)
        ratings[p2] = r2 + K_FACTOR * ((1 - actual_p1) - (1 - expected_p1))

    return ratings, pd.DataFrame(history)


class TennisModel:
    """
    ATP tennis match prediction model.

    Uses logistic regression trained on four features:
        - Elo rating difference
        - Surface win rate difference
        - Head-to-head win rate
        - Common opponent win rate

    Requires a fitted LogisticRegression model and StandardScaler
    (produced by train.py) to make predictions.
    """

    def __init__(self, dataframe):
        self.df = dataframe
        self.elo_ratings, self.elo_history = compute_elo_ratings(dataframe)

    # ── FEATURE METHODS ───────────────────────────────────────────────────────

    def get_surface_win_rate(self, history_df, player, surface):
        """Win rate for a player on a given surface."""
        matches = history_df[
            (history_df['Player_1'] == player) |
            (history_df['Player_2'] == player)
        ]
        surface_matches = matches[matches['Surface'] == surface]
        if surface_matches.empty:
            return 0.5
        return (surface_matches['Winner'] == player).sum() / len(surface_matches)

    def get_h2h_score(self, history_df, p1, p2, surface):
        """
        p1's head-to-head win rate against p2.
        Falls back to relative surface win rates if no H2H history exists.
        """
        h2h = history_df[
            ((history_df['Player_1'] == p1) & (history_df['Player_2'] == p2)) |
            ((history_df['Player_1'] == p2) & (history_df['Player_2'] == p1))
        ]
        if h2h.empty:
            p1_swr = self.get_surface_win_rate(history_df, p1, surface)
            p2_swr = self.get_surface_win_rate(history_df, p2, surface)
            total  = p1_swr + p2_swr
            return p1_swr / total if total > 0 else 0.5
        return (h2h['Winner'] == p1).sum() / len(h2h)

    def get_common_opponent_score(self, history_df, p1, p2):
        """
        p1's win rate against common opponents relative to p2's.
        Returns a value between 0 and 1 (0.5 = neutral, no common opponents).
        """
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

    # ── PREDICTION ────────────────────────────────────────────────────────────

    def predict(self, p1, p2, match_surface, lr_model, lr_scaler, match_date=None):
        """
        Predicts win probability for p1 against p2.

        Args:
            p1, p2          : player name strings (must match dataset format)
            match_surface   : 'Clay', 'Hard', 'Grass', or 'Carpet'
            lr_model        : fitted LogisticRegression from train.py
            lr_scaler       : fitted StandardScaler from train.py
            match_date      : if provided, only uses data before this date
                              (ensures no lookahead during backtesting)

        Returns:
            [p1_prob, p2_prob] — win probabilities summing to 1.0
        """
        if match_date is not None:
            elo_hist = self.elo_history[self.elo_history['date'] < match_date]
            hist_df  = self.df[self.df['Date'] < match_date]
        else:
            elo_hist = self.elo_history
            hist_df  = self.df

        p1_elo_rows = elo_hist[(elo_hist['p1'] == p1) | (elo_hist['p2'] == p1)]
        p2_elo_rows = elo_hist[(elo_hist['p1'] == p2) | (elo_hist['p2'] == p2)]

        if p1_elo_rows.empty or p2_elo_rows.empty:
            return [0.5, 0.5]

        last_p1 = p1_elo_rows.iloc[-1]
        p1_elo  = last_p1['p1_elo_before'] if last_p1['p1'] == p1 else last_p1['p2_elo_before']
        last_p2 = p2_elo_rows.iloc[-1]
        p2_elo  = last_p2['p1_elo_before'] if last_p2['p1'] == p2 else last_p2['p2_elo_before']

        elo_diff = p1_elo - p2_elo
        p1_swr   = self.get_surface_win_rate(hist_df, p1, match_surface)
        p2_swr   = self.get_surface_win_rate(hist_df, p2, match_surface)
        swr_diff = p1_swr - p2_swr
        p1_h2h   = self.get_h2h_score(hist_df, p1, p2, match_surface)
        p1_co    = self.get_common_opponent_score(hist_df, p1, p2)

        features        = np.array([[elo_diff, swr_diff, p1_h2h, p1_co]])
        features_scaled = lr_scaler.transform(features)
        p1_prob         = lr_model.predict_proba(features_scaled)[0][1]

        return [round(p1_prob, 4), round(1 - p1_prob, 4)]
