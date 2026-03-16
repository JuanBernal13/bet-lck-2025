<div align="center">

#LCK Esports Betting Model · 2025

### Algorithmic Prediction & Risk Management System for League of Legends



</div>

---

## What does this project solve?

Esports betting markets exhibit **informational inefficiencies** that a quantitative model can systematically exploit. This system:

1. **Estimates** the true win probability of each team before the match begins.
2. **Compares** that probability against the implied probability from the bookmaker's odds.
3. **Sizes** each bet using the **Fractional Kelly Criterion** to maximize long-term capital growth while controlling drawdown.

The goal is not to predict who wins, but to find matches where the model's probability systematically exceeds the market's implied probability (**+EV**).

---

## Project Architecture

```
bets-lck-2025/
├── main.py                  # Full pipeline orchestrator
├── lol_esports.ipynb        # Original exploration and idea iteration
└── src/
    ├── config.py            # Hyperparameters and global constants
    ├── data_manager.py      # Oracle's Elixir dataset ingestion and cleaning
    ├── features.py          # Temporal Feature Engineering (rolling, streaks)
    ├── models.py            # Model definition, training, and calibration
    ├── strategy.py          # Expected Value, Kelly Criterion, simulation
    └── visualization.py     # Premium visual reports
```

---

## 🔬 Technical Pipeline

### 1 · Data

- **Source:** [Oracle's Elixir](https://oracleselixir.com/) — official professional LoL statistics dataset.
- **League:** LCK (League of Legends Champions Korea) · 2025 Season.
- **Scope:** 555 unique matches · 10 teams · 3 splits (Cup, Rounds 1-2, Rounds 3-5).
- **Granularity:** One team row per match (12 total rows per match: 10 players + 2 teams). The model works exclusively with **team-level rows**.

### 2 · Feature Engineering (Leak-Free)

The fundamental rule: **no feature can contain information from the match being predicted.**

All performance metrics are computed with `shift(1)` over each team's time series, ensuring the model only sees history prior to the match in question.

| Type | Features | Windows |
|------|----------|---------|
| **Rolling stats** | Win rate, Gold diff@15, Gold diff@25, Dragons, Barons, Towers, Earned GPM, GSPD, Vision Score, CKPM, Void Grubs | W = {3, 5, 10} |
| **Momentum** | Win streak (signed cumulative streak) | — |
| **Fatigue/Rest** | Days since last match | — |
| **Trend** | Season cumulative win rate | — |
| **Meta-game** | Patch (ordinal), Side (Blue/Red), Playoffs flag, Split | — |

**Differentials (Blue − Red):** For each feature, the differential between both teams is computed. These are the most informative predictors as they normalize absolute performance against the opponent's baseline.

**Total features:** 120 columns in the final match-level dataset.

### 3 · Time-Aware Cross-Validation

Standard k-fold introduces **temporal data leakage** by mixing past and future matches. Instead, **expanding chronological splits** are used:

```
Fold 1 │ Train: Cup (109 matches)              → Test: Rounds 1-2 (240 matches)
Fold 2 │ Train: Cup + Rounds 1-2 (349 matches) → Test: Rounds 3-5 (206 matches) ← primary evaluation
```

### 4 · Machine Learning Algorithms

| Algorithm | Description | Regularization |
|-----------|-------------|----------------|
| **Logistic Regression** | Linear classifier with L₂ penalty. Interpretable baseline. | `C=0.1` (strong) |
| **Random Forest** | 200 trees with bootstrap. Robust to non-linearities. | `max_depth=4`, `min_samples_leaf=10` |
| **XGBoost** | Gradient Boosting with 2nd-order Taylor expansion. | `reg_alpha=1.0`, `reg_lambda=2.0` |
| **LightGBM** | GBM optimized with GOSS + EFB for categorical datasets. | `num_leaves=15`, `min_child_samples=15` |

### 5 · Probability Calibration

Ensemble models tend to produce **poorly calibrated** probabilities (typically compressed toward the center). For betting, this is critical: if the model says "60%" but it actually happens 70% of the time, the Kelly strategy will mis-size every bet.

**Platt Scaling** fits a sigmoid transformation on the model's raw output:

```
p_calibrated = 1 / (1 + exp(−(α·f(x) + β)))
```

Parameters `α` and `β` are estimated by minimizing log-loss via 5-fold internal CV (`CalibratedClassifierCV`).

---

## Results (Fold 2 · Rounds 3-5)

### Comparative Metrics

| Model | Accuracy | ROC-AUC | Log Loss | Brier Score |
|-------|----------|---------|----------|-------------|
| Random Forest | 0.558 | **0.640** | 0.675 | — |
| LightGBM | 0.578 | 0.634 | 0.758 | — |
| XGBoost | 0.544 | 0.620 | 0.752 | — |
| Logistic Regression | 0.515 | 0.608 | 0.738 | — |
| **Calibrated Random Forest** | — | **0.644** | 0.676 | **0.242** |


### Bankroll Simulation (Quarter-Kelly · EV > 3%)

| Metric | Value |
|--------|-------|
| Matches evaluated | 206 |
| Bets executed | 57 (27.7%) |
| Betting win rate | 43.9% |
| Initial bankroll | $1,000 |
| Final bankroll | $825.56 |
| ROI | −17.4% |

---

## 💡 Design Decisions

- **Why `shift(1)` instead of simply excluding the target column?**  
  Leakage in team time series also occurs in performance metrics: if you include the current match result in the rolling mean calculation, you contaminate all features for that same match.

- **Why Quarter-Kelly instead of full Kelly?**  
  Full Kelly assumes that $\hat{p}$ is exactly the true probability. In practice, any estimation error amplifies the risk of ruin (**Kelly Criterion's curse**). The 0.25 factor absorbs model uncertainty.

- **Why Platt Scaling instead of Isotonic Regression?**  
  Isotonic Regression requires a large amount of data for the calibration curve. With ~350 training matches, Platt Scaling (2 parameters) is more stable and less prone to overfitting.

---

## Usage

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

### Running

```bash
# Place the Oracle's Elixir CSV file in the project root
python main.py
```

The pipeline will automatically generate:
1. EDA plots (win rates, correlations)
2. Calibration reliability diagram
3. Comparative ROC curves
4. Bankroll simulation with EV distribution

### Using the Serialized Model

```python
import joblib, json

model = joblib.load('lck_betting_model.pkl')
with open('lck_feature_cols.json') as f:
    cols = json.load(f)

prob_blue_wins = model.predict_proba(X_new[cols])[:, 1]
```


## Data

The dataset is not included in the repository due to its size (~80 MB). Download it directly from [Oracle's Elixir — 2025 Match Data](https://oracleselixir.com/tools/downloads) and place it in the project root with the name:

```
2025_LoL_esports_match_data_from_OraclesElixir.csv
```

---

<div align="center">



</div>
