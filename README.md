# Timberwolves Basketball Analytics

A basketball analytics system built to demonstrate shot prediction modeling and player performance trajectory analysis using real NBA tracking data.

Built as a portfolio project showcasing the same methodology used in my sports analytics research at Northeastern University — player tracking pipelines, machine learning models, and an interactive dashboard — focused specifically on the Minnesota Timberwolves roster.


## The Questions This Answers

- Can we predict whether a shot goes in based on location, zone, and game situation?
- Which factors matter most for shot success: location, player, or situation?
- Is a player trending up or down right now, and what should we expect next game?

## Key Findings

**1. Shot location dominates shot prediction**
`IS_RESTRICTED_AREA` accounts for 52% of model importance. Whether a shot is a layup or dunk versus everything else is the single biggest predictor of success, more than who is shooting or when.

**2. Corner 3s are meaningfully more efficient**
40.3 – 42.3% vs 37.1% above the break across our dataset, validating the modern NBA emphasis on corner 3 generation in offensive scheme design.

**3. Recent trajectory predicts better than season average**
`CONSISTENCY` and `PERFORMANCE_TREND` rank as top predictive features across players. A player's recent form predicts next game output better than their season average alone.

**4. Clutch performance varies significantly by player**
Some players maintain or improve efficiency in 4th quarter under 5 minutes, others drop significantly, directly relevant to late-game lineup decisions.


## Data

Source: [NBA Stats API](https://www.nba.com/stats) via `nba_api`

| Dataset | Rows | Description |
|---|---|---|
| Shot Charts | 13,661 | Every shot attempt across 5 players, 3 seasons |
| Game Logs | 1,090 | Game-by-game stats per player per season |
| Player Info | 5 | Profile data — age, position, experience |
| Roster | 18 | Full 2024-25 Timberwolves roster |

**Players(5):** Anthony Edwards, Rudy Gobert, Julius Randle, Naz Reid, Mike Conley

**Seasons(3):** 2022-23, 2023-24, 2024-25


## Models

### 1.  Shot Prediction — XGBoost Classifier
Predicts whether a shot goes in(1) or misses(0).

| Metric | Value |
|---|---|
| Accuracy | 63.2% |
| F1 Score | 0.575 |
| CV Accuracy | 61.6% ± 1.0% |
| Training shots | 10,928 |
| Test shots | 2,733 |

Features: shot distance, court coordinates, zone, distance bucket, clutch flag, corner 3 flag, restricted area flag, shot difficulty score, time remaining, player identity.

### 2. Player Performance Prediction — XGBoost Regressor
Predicts next game point total per player using recent form features.

| Player | Avg PPG | MAE | Within 5 pts |
|---|---|---|---|
| Anthony Edwards | 26.1 | 7.3 pts | 42.6% |
| Julius Randle | 22.6 | 7.8 pts | 44.7% |
| Mike Conley | 10.5 | 5.0 pts | 51.2% |
| Naz Reid | 13.2 | 5.1 pts | 56.5% |
| Rudy Gobert | 13.1 | 4.7 pts | 65.1% |

Features: rolling averages (L5, L10), performance trend, consistency score, fatigue proxy, hot/cold streak flags, season progress, win rate.

## Dashboard

An interactive Plotly Dash application visualizing all model outputs.

**Player-controlled panels** (update when player or season changes):
- **Shot Chart** — every shot plotted on a court, colored by made/missed, filterable by season
- **Scoring Trajectory** — rolling averages over time with hot streak markers
- **Next Game Prediction** — model prediction with confidence range based on consistency score
- **Stat Cards** — PPG, RPG, APG, FG%, 3P%, Clutch% for selected player and season

**Static panels** (always show full dataset):
- **Player Comparison** — all 5 players across PTS, REB, AST, FG%, PLUS_MINUS
- **Feature Importance** — which factors drive shot prediction most

****Added sample dashboard screenshots in the ['screenshots/'] folder.**


## Project Structure

```text
basketball-analytics/
├── config.py
├── main.py
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── fetch_data.py
│   │   └── process_data.py
│   ├── features/
│   │   └── engineer_features.py
│   └── models/
│       ├── shot_prediction.py
│       └── player_performance.py
└── dashboard/
    └── app.py
```


---

## How To Run

**1. Clone the repo**
```bash
git clone https://github.com/desairajviii/basketball-analytics.git
cd basketball-analytics
```

**2. Set up environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. On Mac — install OpenMP (required for XGBoost)**
```bash
brew install libomp
```

**4. Run the full pipeline**
```bash
python3 main.py
```

**5. Launch the dashboard**
```bash
python3 dashboard/app.py
```

Then open `http://127.0.0.1:8050` in your browser.


## Connection To Research

This project mirrors the methodology of my ongoing research as a Data Science Research Assistant in the Sports Analytics Lab at Northeastern University, where I work with NBA player tracking data to build performance models and coach-facing dashboards.

Due to lab confidentiality policies I cannot share that research directly. This project was built from scratch using public NBA data to demonstrate the same core skills — tracking data pipelines, time series feature engineering, uncertainty quantification, and interactive visualization.

## Computer Vision Pipeline

In my research role at Northeastern's Sports Analytics Lab, I build computer vision pipelines to extract biomechanical metrics from basketball video, work that complements the tracking and modeling shown in this project.


**Metrics extracted:**

| Metric | Description |
|---|---|
| Release Angle | Angle between shoulder, elbow, and wrist at ball release |
| Elbow Position | Shooting elbow angle at release — indicator of form consistency |
| Follow-Through | Wrist position after release — completion of shooting motion |
| Balance Score | Center of mass over base — stability at time of shot |
| First-Step Quickness | Lateral acceleration from standing position |

**Why this matters:**

Tracking data tells you where players are. Computer vision tells you how they're moving their bodies. A shot from the corner at 40% efficiency looks the same in tracking data regardless of whether the shooter has perfect form or is off-balance. Biomechanical data explains the why behind the numbers, directly applicable to player development and shot quality evaluation.

**Connection to this project:**

The shot difficulty score and shot prediction model in this project use location and context features. In the full research pipeline, biomechanical features are added on top - release angle, balance, and elbow position improve shot outcome prediction by capturing how a shot was taken, not just where it came from.

This component is not included in this repository as it requires proprietary video footage. The methodology and implementation exist in my Northeastern research work.

## Tech Stack

| Category | Tools |
|---|---|
| Data | nba_api, pandas, numpy |
| Modeling | XGBoost, scikit-learn |
| Visualization | Plotly, Dash |
| Infrastructure | Python 3.13, joblib |