# 🏆 FIFA World Cup 2026 Match Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-FIFA%202026%20Predictor-gold?style=for-the-badge&logo=football)](https://fifa-2026-predictor-production.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Deployed-blue?style=for-the-badge&logo=docker)](https://docker.com)
[![Railway](https://img.shields.io/badge/Railway-Live-purple?style=for-the-badge&logo=railway)](https://railway.app)

> **Live Predictions for all 104 FIFA World Cup 2026 matches using Poisson Regression and Elo Ratings**

🔗 **Live App:** — [Click Here](https://fifa-2026-predictor-production.up.railway.app/)

---

## 📌 Project Overview

This is a **end-to-end machine learning project** that predicts outcomes of FIFA World Cup 2026 matches. Given any two of the 48 qualified teams, the model predicts:

- ✅ Win / Draw / Loss probabilities
- ✅ Expected goals for each team
- ✅ Most likely scoreline and its probability
- ✅ Full tournament simulation (Group Stage → Final)
- ✅ Visual knockout bracket



---

## 🌐 Live Application

| Page | Description |
|------|-------------|
| `/` | Home — Select any two teams and predict match outcome |
| `/predictor` | Results — Win/Draw/Loss probabilities, expected goals, scoreline |
| `/groups` | All 12 World Cup groups with team rankings |
| `/simulator` | Full tournament simulation with knockout bracket |

---

## 📊 Dataset

**Source:** [Kaggle — International Football Results 1872–2026] — [Click Here](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

| File | Rows | Description |
|------|------|-------------|
| `results.csv` | 49,287 | All international match results |
| `shootouts.csv` | 675 | Penalty shootout results |
| `goalscorers.csv` | 47,601 | Individual goal events |
| `former_names.csv` | 36 | Historical country name changes |

**Additional:** FIFA 2026 Player Stats CSV — 1,248 players, 503 matched with top-5 league statistics

---

## 🔧 Feature Engineering

Starting from raw match data, we engineered **55 features** across multiple categories:
<details>
<summary><strong>📋 View All 55 Features</strong></summary>


| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `date` | Raw | Match date |
| 2 | `home_team` | Raw | Home team name |
| 3 | `away_team` | Raw | Away team name |
| 4 | `home_score` | Raw | Home team goals scored |
| 5 | `away_score` | Raw | Away team goals scored |
| 6 | `tournament` | Raw | Tournament name |
| 7 | `city` | Raw | Match city |
| 8 | `country` | Raw | Match country |
| 9 | `neutral` | Raw | Neutral ground flag (0/1) |
| 10 | `year` | Derived | Year extracted from date |
| 11 | `month` | Derived | Month extracted from date |
| 12 | `result` | Target | Win / Draw / Loss (home team perspective) |
| 13 | `home_ranking` | Rankings | FIFA ranking of home team |
| 14 | `away_ranking` | Rankings | FIFA ranking of away team |
| 15 | `ranking_difference` | Rankings | away_ranking minus home_ranking |
| 16 | `tournament_importance` | Context | 1=Friendly, 2=Nations League, 3=Continental, 4=Qualification, 5=World Cup |
| 17 | `home_form_5` | Form | Home team win rate in last 5 matches (0.0 to 1.0) |
| 18 | `away_form_5` | Form | Away team win rate in last 5 matches |
| 19 | `home_form_10` | Form | Home team win rate in last 10 matches |
| 20 | `away_form_10` | Form | Away team win rate in last 10 matches |
| 21 | `home_goals_scored_avg` | Goals | Average goals scored by home team in last 5 matches |
| 22 | `away_goals_scored_avg` | Goals | Average goals scored by away team in last 5 matches |
| 23 | `home_goals_conceded_avg` | Goals | Average goals conceded by home team in last 5 |
| 24 | `away_goals_conceded_avg` | Goals | Average goals conceded by away team in last 5 |
| 25 | `h2h_home_win_rate` | H2H | How often home team wins in direct head-to-head history |
| 26 | `h2h_draw_rate` | H2H | Draw rate between these two specific teams |
| 27 | `h2h_total_matches` | H2H | Total historical meetings between the two teams |
| 28 | `home_goal_diff_avg` | Goals | Average goal difference for home team in last 5 |
| 29 | `away_goal_diff_avg` | Goals | Average goal difference for away team in last 5 |
| 30 | `home_avg_rating` | Squad | Average FIFA 25 player rating across home team squad |
| 31 | `home_star_rating` | Squad | Best player rating in home team squad |
| 32 | `home_squad_depth` | Squad | Home team squad depth score |
| 33 | `home_avg_age` | Squad | Average age of home team squad |
| 34 | `away_avg_rating` | Squad | Average FIFA 25 player rating across away team squad |
| 35 | `away_star_rating` | Squad | Best player rating in away team squad |
| 36 | `away_squad_depth` | Squad | Away team squad depth score |
| 37 | `away_avg_age` | Squad | Average age of away team squad |
| 38 | `squad_rating_diff` | Squad | home_avg_rating minus away_avg_rating |
| 39 | `star_rating_diff` | Squad | home_star_rating minus away_star_rating |
| 40 | `squad_depth_diff` | Squad | home_squad_depth minus away_squad_depth |
| 41 | `home_elo` | Elo | Elo rating of home team before match |
| 42 | `away_elo` | Elo | Elo rating of away team before match |
| 43 | `elo_difference` | Elo | home_elo minus away_elo |
| 44 | `match_weight` | Time | Recent match weight (2018+=1.0, 2010+=0.7, 2000+=0.4, older=0.1) |
| 45 | `home_confederation` | Confederation | Home team confederation (UEFA/CONMEBOL/CAF/AFC/CONCACAF/OFC) |
| 46 | `away_confederation` | Confederation | Away team confederation |
| 47 | `home_conf_strength` | Confederation | Numeric strength of home confederation (1-5) |
| 48 | `away_conf_strength` | Confederation | Numeric strength of away confederation |
| 49 | `conf_strength_diff` | Confederation | home_conf_strength minus away_conf_strength |
| 50 | `same_confederation` | Confederation | 1 if both teams from same confederation |
| 51 | `poisson_home_win` | Poisson | Poisson model home win probability |
| 52 | `poisson_draw` | Poisson | Poisson model draw probability |
| 53 | `poisson_away_win` | Poisson | Poisson model away win probability |
| 54 | `poisson_home_exp` | Poisson | Poisson expected goals for home team |
| 55 | `poisson_away_exp` | Poisson | Poisson expected goals for away team |
</details>
---
 🎯 Why These Features?

These features capture:

- 📈 Team quality (FIFA Rankings & Elo Ratings)
- 🔥 Recent form and momentum
- ⚽ Scoring and defensive performance
- 🏆 Tournament importance
- 👥 Squad strength and depth
- 🌍 Confederation strength
- 📊 Statistical match probabilities via Poisson models

This feature engineering pipeline significantly improved predictive performance compared to using raw match data alone.

## 🤖 Models Trained

### 1. Decision Tree
- Simple if/else rule-based model
- **Accuracy: 46.10%**
- Problem: Too simple, struggled with draws

### 2. XGBoost — 7 Versions
Gradient boosting ensemble — each tree fixes mistakes of previous trees.

| Version | What We Tried | Accuracy |
|---------|--------------|----------|
| V1 | 18 basic features | 49.57% |
| V2 | Added squad + Elo features (26 total) | 49.30% |
| V3 | Balanced class weights for draws | 45.37% |
| V4 | Time-weighted matches (recent = more weight) | 46.64% |
| V5 | Only post-1990 competitive matches | 44.85% |
| V6 | Hyperparameter tuning (250 combinations) | 50.43% |
| V7 | All 30 features including confederation | **50.63%** |

**Why XGBoost hit a ceiling:**
- Draw prediction F1-score was only 0.03 in best version
- Model learned to always predict Win (most common at 46%)
- Needed a fundamentally different approach

---

## ✅ Final Model — Poisson Regression

### Why Poisson?

Instead of predicting Win/Draw/Loss directly, Poisson Regression predicts **how many goals each team will score**. Then we calculate the probability of every possible scoreline (0-0, 1-0, 0-1, 1-1, 2-0 etc.) and sum them to get Win, Draw, Loss probabilities.

Goals in football are **rare, independent events in a fixed time period** — which perfectly fits the Poisson distribution.

### How It Works

**Step 1 — Calculate Attack Strength and Defence Weakness per team:**

```
attack_strength  = team_avg_goals_per_match / overall_avg_goals
defence_weakness = team_avg_conceded_per_match / overall_avg_goals
```

**Step 2 — Calculate Expected Goals:**

```
home_expected = avg_goals × home_attack × away_defence × squad_score × conf_multiplier
away_expected = avg_goals × away_attack × home_defence × squad_score × conf_multiplier
```

**Step 3 — Calculate all scoreline probabilities:**

```
P(home=X, away=Y) = Poisson(X, home_expected) × Poisson(Y, away_expected)
```

**Step 4 — Sum probabilities:**

```
Win  = sum of all P where X > Y
Draw = sum of all P where X = Y
Loss = sum of all P where X < Y
```

### Additional Improvements

| Improvement | Description |
|-------------|-------------|
| **Squad Strength** | Player stats from top 5 leagues (PL, Bundesliga, La Liga, Serie A, Ligue 1) — form rating, goals+assists, tackles |
| **Confederation Multiplier** | UEFA=1.15x, CONMEBOL=1.12x, CAF=0.95x, AFC=0.93x, CONCACAF=0.97x, OFC=0.88x |
| **Host Bonus** | Only USA, Mexico, Canada get home advantage (1.284x) |
| **Elo Ratings** | Calculated from scratch using all 7,501 matches — updates after every game |

### Final Accuracy

| Model | Accuracy | vs Random Baseline |
|-------|----------|--------------------|
| Random Guessing | 33.0% | baseline |
| Decision Tree | 46.10% | +13.1% |
| XGBoost (best) | 50.63% | +17.6% |
| **Poisson Regression** | **52.48%** | **+19.5%** |


---

## 🏗️ Architecture

```
User Browser
     ↓
FastAPI (main.py)
     ↓
Poisson Prediction Engine
     ↓
├── team_stats (attack/defence from historical data)
├── SQUAD_DATA (player stats from top 5 leagues)
├── CONF_MULTIPLIER (confederation strength)
└── poisson_model.pkl (saved model data)
     ↓
Jinja2 Templates (HTML)
     ↓
Response to Browser
```

---

## 📁 Project Structure

```
fifa_2026/
├── main.py                    # FastAPI app — all routes and prediction logic
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container configuration
├── .gitignore
├── model/
│   └── poisson_model.pkl      # Saved model (team_stats, avg_goals, rankings)
├── static/
│   ├── style.css              # Dark theme UI styles
│   ├── fifa_logo.webp         # Official FIFA 2026 logo
│   └── worldcup_homepage.jpg  # Hero background image
└── templates/
    ├── index.html             # Home page — team selector + countdown timer
    ├── predictor.html         # Prediction results page
    ├── groups.html            # All 12 World Cup groups
    └── simulator.html         # Full tournament simulator 
    ├── Matches.html           # All 104 Matches Prediction
    └── Knockout.html          # Knockout Matches Bracket Section
```

---

## 🚀 Tech Stack

| Layer | Technology |
|---------|-----------|
| Language | Python 3.11 |
| Machine Learning | Poisson Regression (`scipy.stats`) |
| Backend API | FastAPI |
| Frontend Templates | Jinja2 |
| UI Styling | Custom CSS |
| Static Assets | FlagCDN |
| Automated Testing | Pytest |
| Continuous Integration (CI) | GitHub Actions |
| Containerization | Docker |
| Continuous Deployment (CD) | Railway |
| Version Control | Git & GitHub |

---


## 🏟️ Tournament Structure

```
48 Teams → 12 Groups of 4
        ↓
Group Stage (72 matches)
Top 2 from each group = 24 teams
Best 8 third-placed = 8 teams
Total = 32 teams
        ↓
Round of 32 (16 matches)
        ↓
Round of 16 (8 matches)
        ↓
Quarter Finals (4 matches)
        ↓
Semi Finals (2 matches)
        ↓
Third Place Match + Final
        ↓
🏆 Champion
```

---

## 📈 Key Insights

- **France, Spain, Germany** have highest squad scores from top 5 league data
- **Brazil** historically wins most World Cups but current squad score slightly lower
- **UEFA and CONMEBOL** teams win ~70% of cross-confederation matches
- **Home advantage** only applies to USA, Mexico, Canada (host nations)
- **Draw prediction** is the hardest — football genuinely produces ~25% draws

---


## Author

**Shodhan Moily**
- GitHub: [@shodhanmoily](https://github.com/shodhanmoily)

## 📄 License

MIT License — feel free to use and modify.

---

