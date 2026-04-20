import pandas as pd
from model import TennisModel  # <--- This "grabs" your class from model.py

# 1. Load the data
df = pd.read_csv('atp_matches_2024.csv')

# 2. Initialize the model
model = TennisModel(df)

# 3. Test Stuff (The "Driving")
scores = model.predict_odds("Fils", "Musetti", "Clay", 1440, 3625, 0)
results = model.check_alpha(scores[0], scores[1], 1.91)

print(f"Verdict: {results[3]}")
