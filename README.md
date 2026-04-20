# Tennis Quant: ATP Alpha Discovery Model

A quantitative trading tool designed to identify mispriced binary contracts in tennis markets (e.g., Kalshi) by leveraging historical ATP data and multi-factor statistical modeling.

## 🚀 Overview
This project implements a `TennisModel` class that calculates the "True Probability" of match outcomes. By comparing model-derived probabilities against market prices (multipliers), the system identifies trades with positive **Expected Value (EV)**.

## 📊 Methodology
The model calculates win probabilities by synthesizing four primary data streams:

1.  **Elo-Based Strength:** Relative player strength derived from ATP ranking points.
2.  **Surface-Specific H2H:** Historical head-to-head performance with a **recency decay function** (newer matches carry more weight).
3.  **Common Opponent Network:** Analyzes performance against shared rivals to solve for "A vs C" scenarios.
4.  **Surface Variance:** Adjusts win rates based on historical performance on Clay, Grass, or Hard courts.

## 📉 Mathematical Framework
The core of the "Alpha" discovery is the Expected Value formula:

$$EV = (P_{win} \times \text{Net Profit}) - (P_{loss} \times \text{Stake})$$

Where:
* $P_{win}$ is the model's predicted probability.
* **Net Profit** is $(Multiplier - 1)$.
* **Stake** is the amount risked ($1.00$).

## 🛠️ Usage
```python
from model import TennisModel
import pandas as pd

# Initialize with historical data
df = pd.read_csv('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv')
model = TennisModel(df)

# Predict Match
scores = model.predict_odds("Sinner", "Alcaraz", "Hard", 13000, 9000, opinion=0)

# Check for Alpha (Market Multiplier of 1.8x)
my_prob, m_prob, ev, verdict = model.check_alpha(scores[0], scores[1], 1.8)

print(f"Verdict: {verdict} | Expected ROI: {ev*100}%")
'''
## 🌐 Data Sourcing
To ensure the model utilizes the most recent match data without local storage overhead, this project pulls historical records directly from remote repositories.
* **Source:** [Jeff Sackmann's ATP Database](https://github.com/JeffSackmann/tennis_atp)
* **Implementation:** Data is streamed via `pandas.read_csv()` from raw GitHub URLs, allowing for seamless updates as new 2026 match results are processed.
