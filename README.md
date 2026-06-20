# FIFA World Cup 2026 — Prediction System

A end-to-end machine learning pipeline for predicting match outcomes, simulating the FIFA World Cup 2026 tournament, and generating scoreline probabilities. Built across five notebooks and deployed as an interactive Streamlit dashboard.

---

## Project Structure

```
├── data/
│   ├── raw/                        # Source data (results, fixtures, team metadata)
│   └── processed/                  # Feature matrices, ELO snapshots, H2H records
├── models/                         # Trained models and Poisson parameters
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_tournament_simulation.ipynb
│   ├── 04_score_prediction.ipynb
│   └── 05_dashboard_prep.ipynb
└── dashboard/
    ├── app.py                      # Landing page
    ├── utils.py                    # Shared loaders and prediction functions
    └── pages/
        ├── Championship_Odds.py
        ├── Group_Stage.py
        ├── Stage_Progression.py
        ├── Tournament_Bracket.py
        └── Head_to_Head.py
```

---

## Data Sources

- **International match results** (1872–2026): home/away teams, scores, tournament, neutral venue flag
- **WC 2026 fixtures**: all 104 matches including group stage and knockout placeholders
- **WC 2026 team metadata**: group assignments, confederation, FIFA rank
- **Historical FIFA rankings** are not used for training rows — ELO ratings computed at match time are used instead to avoid data leakage

---

## Feature Engineering

### ELO Ratings

ELO ratings are computed chronologically across all matches from 1990 onwards. Every team starts at 1500. The K-factor varies by tournament type — World Cup finals use K=60, major tournament qualifiers use K=20–30, and friendlies use K=20. Home advantage adds 65 points to the expected score calculation for non-neutral venues.

This produces a time-aware ELO value for every team at every match, stored as `home_elo_before` and `away_elo_before`. These are strictly historical — no future information bleeds into any training row.

### Rolling Form Features

For each match, the previous 3, 5, and 10 matches for both teams are used to compute:
- Win rate
- Average goals scored and conceded
- Average goal difference

These are computed with `.shift(1).rolling(window)` to ensure the current match is never included. Both team perspectives (home and away) are handled separately before being merged back into the match row. The three window sizes capture short-term momentum (3), medium form (5), and longer-term consistency (10).

### Head-to-Head Features

For every pair of WC 2026 teams, H2H records are compiled from all historical matches. Features include total games, wins per side, and a recency-weighted win rate (exponential decay with a 0.1 annual decay constant). Both directions are stored so lookup is always a direct match rather than requiring reversal logic.

Teams with no H2H history default to 0.5 win rate — reflecting genuine uncertainty rather than a biased assumption.

### Rank Features

ELO pseudo-ranks are derived per match date by ranking all teams by their ELO at that point in time. This gives a relative rank signal without using FIFA's official rankings (which aren't available historically). For 2026 fixtures, actual FIFA rankings from the official snapshot are attached separately and never mixed into historical training rows.

### Feature Matrix

The final training matrix has 44 features per match after dropping highly correlated H2H columns (goal counts correlate >0.97 with win counts; unweighted win rate correlates >0.90 with the weighted version). The matrix covers 31,770 matches from 1990 onwards. NaN values from early matches with no form history are dropped for core ELO and form features; H2H NaNs are filled with neutral defaults.

---

## Model Training

### Train/Test Split

A strict chronological split is used — no random splitting. Training data covers all matches before January 2015. A calibration slice covers 2015–2017. The test set covers 2018 onwards, with WC-only results (128 matches) evaluated separately as a directional check.

This mirrors real deployment: the model is always predicting future matches it has never seen.

### Models

Three classifiers are trained:

- **Logistic Regression** (with StandardScaler): a strong linear baseline that performs well when ELO differential is the dominant signal
- **Random Forest** (300 trees, balanced class weights): captures non-linear interactions between form and H2H features
- **XGBoost** (tuned via RandomizedSearchCV on 5-fold time-series CV): gradient boosting with column and row subsampling

Hyperparameters are tuned using `neg_log_loss` as the scoring metric rather than accuracy — this rewards well-calibrated probabilities, which matters more than hard predictions for tournament simulation.

### Calibration

Random Forest and XGBoost produce poorly calibrated probabilities by default. Isotonic regression calibration is applied using the 2015–2017 held-out slice. Logistic Regression is already well-calibrated and is left as-is.

### Ensemble

The three calibrated models are combined in a soft-voting ensemble. Weights are set by validation performance: Logistic Regression at 3.2, XGBoost at 2.3, Random Forest at 0.9. The ensemble reaches 61% accuracy and 0.864 log loss on the all-tournament test set, and 63.3% accuracy on WC-only matches.

### Target Variable

Three classes: 0 = Away win, 1 = Draw, 2 = Home win. Draw prediction remains the hardest class (recall ~5%) — a known limitation of football prediction. The ensemble is trained with balanced class weights to reduce this bias.

---

## Tournament Simulation

### Probability Cache

Before any simulation runs, match probabilities are pre-computed for all 48×47 = 2,256 possible team pairings in a single batch `predict_proba` call. This reduces 50,000 simulation iterations from ~30 minutes to under 3 minutes, since every match lookup is then an O(1) dict access.

### Group Stage

Each group stage match is simulated by sampling from the cached [home win, draw, away win] probabilities. Goal totals are drawn from a Poisson distribution based on each team's recent attacking and defensive form (5-match averages), then forced to be consistent with the sampled outcome (e.g. if home win is drawn, the score is adjusted so home goals > away goals).

Groups are ranked by points, then goal difference, then goals scored, with a random tiebreak for equal records. The 12 third-place finishers are ranked by the same criteria and the best 8 advance to the Round of 32 — matching the WC 2026 format.

### Knockout Stage

In knockout rounds, draws lead to extra time (draw probability reduced by 70%), and if still drawn, a 50/50 coin flip represents penalties. This keeps the simulation fast while capturing the reduced likelihood of draws in extra time.

The standard WC 2026 R32 bracket is used: Group A winner vs Group B runner-up, B winner vs A runner-up, and so on for all 12 groups, with the 8 best third-place teams filling the remaining 16 slots.

### Monte Carlo Results

50,000 full tournament simulations produce stable probability estimates for every team at every stage: R32 qualification, R16, QF, SF, Final, and Champion. The results are saved to `simulation_results.csv` and loaded by the dashboard.

---

## Scoreline Prediction

### Dixon-Coles Model

A Dixon-Coles Poisson model is fitted on post-2018 competitive matches (friendlies excluded) involving at least one WC 2026 team. The model estimates an attack parameter (α) and defense parameter (β) for every team, plus a home advantage term (γ) and a low-score correction parameter (ρ).

Expected goals for a match are computed as:

```
λ_home = exp(α_home − β_away + γ)   [0 if neutral venue]
λ_away = exp(α_away − β_home)
```

The Dixon-Coles correction adjusts the probabilities of 0-0, 1-0, 0-1, and 1-1 scorelines, which Poisson models systematically under or over-estimate.

Matches are weighted by recency (exponential decay over 1 year) and tournament importance (World Cup finals = 2x, qualifiers and Nations League = 1.5x). This means recent high-stakes results carry the most weight in the fitted parameters.

### Ensemble Blend

The Dixon-Coles model produces xG and a scoreline matrix, but its win/draw/loss probabilities alone can miss signals captured by the ML ensemble (form, H2H, ELO momentum). The final prediction blends DC at 60% weight and the ensemble at 40%, then rescales the scoreline matrix so its marginal win/draw/loss probabilities match the blended values. This keeps all derived markets (BTTS, Over/Under, clean sheets) consistent with the headline result probabilities.

### Predicted Score

The reported predicted scoreline uses `round(xg_home) - round(xg_away)` rather than the matrix mode. The most probable single scoreline in a Poisson model is often 1-1 or 1-0 even when one team dominates, because probability mass is spread across many scorelines. Rounding xG gives a more intuitive score that reflects the actual expected goal difference.

---

## Dashboard

The dashboard is built with Streamlit and deployed on Streamlit Cloud. All heavy computation (model inference, probability caching) happens once at startup using `@st.cache_resource`. Subsequent page interactions are purely data lookups.

### Landing Page — Match Day Predictions

Shows all group stage matches for the selected date. Date navigation uses the host nation timezone (US Central) so Asian users see today's matches relative to the US, not their local date. Each match card shows predicted score, xG, win/draw/loss bar, and an expandable full prediction with scoreline probabilities, Over 2.5, BTTS, and clean sheet percentages.

### Championship Odds

Ranks all 48 teams by win probability from the Monte Carlo simulations, coloured by confederation. Filterable by confederation and top-N. Includes a full probability table showing R32 through to Win% for every team.

### Group Stage Breakdown

Shows group-by-group win and advancement probabilities in a 3×4 grid. The single group tab shows a separate horizontal bar chart for each stage (R32, R16, QF, SF, Final, Win) so it is easy to see where each team's probability drops off.

### Stage Progression

Overlaid bar chart showing all six stage probabilities per team simultaneously, useful for identifying teams likely to go deep but unlikely to win (strong QF/SF probability, low champion probability). A heatmap view covers the top 24 teams.

### Tournament Bracket

Tab 1 shows the most probable bracket derived from simulation results — each round picks the higher-probability team at each fixture. Tab 2 simulates one full tournament live, using the pre-built probability cache. Each round is clearly labelled with where the winner advances (e.g. "R32 Match 1 → R16"), and rounds flow top-down from R32 to Final. Group stage results are shown in a 3-column chart grid after simulation completes.

### Head to Head

Allows selecting any two WC 2026 teams for a live prediction. Form is pulled from raw historical results (not the pre-computed feature matrix), so it reflects the most recent matches. Output includes ensemble win probabilities, DC+ensemble blended scoreline prediction with xG, top 5 scorelines, betting markets, and a historical H2H record breakdown.

---

## Why These Choices

**Chronological split over random split** — football has temporal structure. A random split would allow the model to see 2022 matches when predicting 2018, inflating test performance. Time-based splitting gives honest estimates of how the model performs on future matches.

**ELO over raw FIFA rankings for training** — FIFA rankings change on a fixed schedule and aren't available historically per match. ELO computed at match time gives a continuous, match-by-match quality signal with no leakage.

**Log loss over accuracy for model selection** — a model that predicts 60% home, 30% draw, 10% away for a match that ends in a home win is better than one that predicts 50/25/25 and picks the same winner. Log loss rewards calibration; accuracy ignores it. For tournament simulation, calibrated probabilities matter more than hard picks.

**Dixon-Coles + ensemble blend** — neither model alone is optimal. DC gives coherent scoreline distributions and proper Poisson structure for goal markets. The ensemble captures features DC ignores (form trends, H2H history, ELO momentum). Blending both at 60/40 consistently outperforms either alone on log loss.

**50,000 simulations** — below 10,000, champion probabilities for weaker teams fluctuate too much between runs. Above 50,000, runtime becomes impractical in a notebook environment. At 50,000, standard error on a 10% probability estimate is ~0.13 percentage points, which is well within acceptable tolerance.

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
xgboost
streamlit
plotly
pytz
pickle
```
