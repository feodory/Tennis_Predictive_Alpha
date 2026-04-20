import pandas as pd
from model import TennisModel

# The remote URL to the 2024 (or 2025/2026) data
DATA_URL = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv'

# Load the data directly from the web
df = pd.read_csv(DATA_URL)

model = TennisModel(df)
