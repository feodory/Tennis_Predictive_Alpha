# Tennis Match Prediction Model

A machine learning pipeline for predicting ATP tennis match outcomes using historical match data, self-computed Elo ratings, and logistic regression.

## Results

| Metric | Value |
|---|---|
| Out-of-sample accuracy | **62.2%** |
| 95% Confidence Interval | 58.6% – 65.8% |
| Log-loss | 0.635 |
| Test set | 693 matches (2023–2026) |
| Baseline (random) | 50% |

Validated using a strict **walk-forward backtest**: each prediction only uses data from matches that occurred before that match's date — no lookahead bias.

## Methodology

### Features

Four features are computed for each match, all using only pre-match historical data:

| Feature | Description |
|---|---|
| `elo_diff` | Difference in self-computed Elo ratings |
| `swr_diff` | Difference in surface-specific win rates |
| `p1_h2h` | Player 1's historical H2H win rate vs Player 2 |
| `p1_co` | Player 1's win rate vs common opponents, relative to Player 2 |

### Elo Ratings

Elo ratings are computed from scratch using the full match history (2000–present). Each match updates both players' ratings using the standard logistic update formula:

```
expected = 1 / (1 + 10^((opponent_rating - player_rating) / 400))
new_rating = old_rating + K * (actual_result - expected)
```

K-factor is set to 32 (standard). Ratings start at 1500 for all players.

### Model

Logistic regression is trained on pre-2023 matches. Feature correlation with match outcome:

| Feature | Correlation |
|---|---|
| elo_diff | 0.402 |
| p1_co | 0.257 |
| swr_diff | 0.262 |
| p1_h2h | 0.198 |

Learned weights confirm Elo is the strongest predictor (coefficient 0.878), consistent with published tennis prediction research.

## Project Structure

```
├── model.py        # TennisModel class with Elo computation and feature methods
├── train.py        # Feature table builder and logistic regression training
├── backtest.py     # Walk-forward backtest
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train the model** (builds feature table, fits logistic regression, saves model files):
```bash
python train.py
```

**Run backtest** (requires trained model from train.py):
```bash
python backtest.py
```

**Predict a specific matchup:**
```python
import pickle
import pandas as pd
from model import TennisModel

# load data and trained model
df = pd.read_csv('your_atp_data.csv')
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('lr_scaler.pkl', 'rb') as f:
    lr_scaler = pickle.load(f)

model = TennisModel(df)
probs = model.predict("Sinner J.", "Alcaraz C.", "Clay", lr_model, lr_scaler)
print(f"Sinner win probability: {probs[0]:.1%}")
print(f"Alcaraz win probability: {probs[1]:.1%}")
```

## Data

Uses the [ATP Tennis Dataset](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) via kagglehub (68,000+ matches, 2000–2026).

## Requirements

```
pandas
numpy
scikit-learn
kagglehub
```
