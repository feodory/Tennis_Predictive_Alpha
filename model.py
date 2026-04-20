class TennisModel:
    def __init__(self, dataframe):
        """Initializes the model with a historical ATP match dataset."""
        self.df = dataframe

    def surface_win_rate(self, name, surface):
        """
        Calculates a player's historical win percentage on a specific surface.
        
        Returns:
            float: Win rate between 0.0 and 1.0.
        """
        player_matches = self.df[(self.df['winner_name'] == name) | 
                                 (self.df['loser_name'] == name)]
        
        if player_matches.empty:
            return 0

        def calc_win_rate(group):
            wins = (group['winner_name'] == name).sum()
            total = len(group)
            return (wins / total)

        surface_stats = player_matches.groupby('surface').apply(calc_win_rate, include_groups=False)
        
        return surface_stats.get(surface, 0)

    def get_common_opponent_score(self, p1, p2):
        """
        Computes a relative score based on how both players performed 
        against same opponents.
        """
        p1_winners = set(self.df[self.df['winner_name'] == p1]['loser_name'])
        p1_losers = set(self.df[self.df['loser_name'] == p1]['winner_name'])
        p1_opponents = p1_winners | p1_losers
        
        p2_winners = set(self.df[self.df['winner_name'] == p2]['loser_name'])
        p2_losers = set(self.df[self.df['loser_name'] == p2]['winner_name'])
        p2_opponents = p2_winners | p2_losers
    
        common = p1_opponents & p2_opponents
    
        if not common:
            return 0, 0
    
        p1_wins, p2_wins = 0, 0
        for opponent in common:
            
            p1_m = self.df[((self.df['winner_name'] == p1) & (self.df['loser_name'] == opponent)) |
                           ((self.df['winner_name'] == opponent) & (self.df['loser_name'] == p1))]
            if not p1_m.empty:
                p1_wins += (p1_m['winner_name'] == p1).mean()
            
            
            p2_m = self.df[((self.df['winner_name'] == p2) & (self.df['loser_name'] == opponent)) |
                           ((self.df['winner_name'] == opponent) & (self.df['loser_name'] == p2))]
            if not p2_m.empty:
                p2_wins += (p2_m['winner_name'] == p2).mean()
            
        total = p1_wins + p2_wins
        if total == 0: return 0, 0
    
        return (p1_wins / total) * 5, (p2_wins / total) * 5

    def get_simple_h2h(self, p1, p2, match_surface):
        """
        Calculates surface-weighted Head-to-Head scores.
        Weights recent matches higher using a recency decay function.
        """
        h2h = self.df[((self.df['winner_name'] == p1) & (self.df['loser_name'] == p2)) | 
                      ((self.df['winner_name'] == p2) & (self.df['loser_name'] == p1))]

        p1_score, p2_score = 0, 0
        weight = 5
        
        if h2h.empty:
            rate_diff = self.surface_win_rate(p1, match_surface) - self.surface_win_rate(p2, match_surface)
            if rate_diff >= 0:
                p1_score = rate_diff * weight 
            else:
                p2_score = abs(rate_diff) * weight
        else:
            for _, row in h2h.iterrows():
                match_year = row['tourney_date'].year 
                
                points = 1.0 + (1/(2027 - match_year))
                if row['surface'] == match_surface:
                    points += 0.5
                
                if row['winner_name'] == p1:
                    p1_score += points 
                else:
                    p2_score += points

            p1_score = (p1_score * 10) / len(h2h)
            p2_score = (p2_score * 10) / len(h2h)
            
        return round(p1_score, 2), round(p2_score, 2)

    def get_strength_score(self, p1_points, p2_points):
        """Converts ATP ranking points into a relative strength score (0-7)."""
        weight = 7
        total_points = p1_points + p2_points
        if total_points == 0: return 0, 0
        
        return (p1_points / total_points) * weight, (p2_points / total_points) * weight

    def predict_odds(self, p1, p2, match_surface, p1_points, p2_points, opinion):
        """
        Combines all factors to predict the final match strength scores.
        """
        str1, str2 = self.get_strength_score(p1_points, p2_points)
        h2h1, h2h2 = self.get_simple_h2h(p1, p2, match_surface)
        co1, co2 = self.get_common_opponent_score(p1, p2)
    
        total_score1 = str1 + h2h1 + co1 - (opinion / 5)
        total_score2 = str2 + h2h2 + co2 + (opinion / 5)
    
        return [round(total_score1, 3), round(total_score2, 3)]

    def get_final_verdict(self, lst):
        """Converts internal scores to win percentages and implied decimal odds."""
        s1, s2 = lst[0], lst[1]
        prob1 = (s1 / (s1 + s2)) * 100
        return f"Prob: {round(prob1, 1)}% | Odds: {round(100/prob1, 2)} vs {round(100/(100-prob1), 2)}"

    def check_alpha(self, p1_score, p2_score, market_mult):
        """
        Compares model probability vs Kalshi multiplier to find Expected Value.
        
        Args:
            p1_score (float): Model's raw score for Player 1.
            p2_score (float): Model's raw score for Player 2.
            market_mult (float): Kalshi multiplier (e.g., 1.4 for 1.4x).
        """
        
        my_prob = (p1_score / (p1_score + p2_score)) * 100
        market_prob = 100 / market_mult
        
        
        p = my_prob / 100
        net_profit = market_mult - 1
        ev = (p * net_profit) - (1 - p)
        
    
        if ev > 0.01:
            verdict = "BUY YES"
        elif ev < -0.01:
            verdict = "BUY NO"
        else:
            verdict = "NO TRADE"

        return [round(my_prob, 1), round(market_prob, 1), round(ev, 4), verdict]
