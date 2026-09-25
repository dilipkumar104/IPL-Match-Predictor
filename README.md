# 🏏 IPL Match Outcome Predictor

> **Predict who wins an IPL match** before it starts — using 17 seasons of historical data and a production-grade Scikit-Learn pipeline.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-IPL%202008--2024-brightgreen)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
[![License](https://img.shields.io/badge/License-KMIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Key Insight — Why Binary Classification?](#-key-insight--why-binary-classification)
- [Dataset](#-dataset)
- [Data Quality Findings](#-data-quality-findings)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation & Execution](#-installation--execution)
- [Sample Predictions](#-sample-predictions)
- [Technical Stack](#-technical-stack)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)

---

## 🎯 Problem Statement

Given two IPL teams and match conditions (venue, toss outcome, toss decision), **predict which team is more likely to win** the match.

This is framed as a **binary classification** problem: given that `team1` is listed first, does `team1` win? (Target = 1).

---

## 💡 Key Insight — Why Binary Classification?

| Formulation | Classes | Test Accuracy |
|---|---|---|
| Multi-class: "Predict exact winner" | 15–19 unique teams | ~43% |
| **Binary: "Will team1 win?"** | **2 (win/loss)** | **~80%** |

Framing the problem as binary classification:
1. Eliminates sparsity — many teams play only a few dozen matches
2. Results in near-perfect class balance (~51% team1 wins, due to random name ordering in the CSV)
3. Allows Logistic Regression to shine, since the decision boundary is nearly linear in feature space

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — IPL Complete Dataset (2008–2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) |
| **File** | `data/matches.csv` |
| **Raw rows** | 1,095 matches |
| **Clean rows** | 1,076 (after removing ties/no-results) |
| **Seasons** | 17 (2007/08 – 2024) |
| **Teams** | 15 unique (after name standardisation) |
| **Columns** | 20 original → 10 engineered features |

---

## 🔍 Data Quality Findings

Six data quality issues were identified and resolved:

| # | Issue | Impact | Fix Applied |
|---|---|---|---|
| 1 | **Team name inconsistencies** — "Delhi Daredevils", "Kings XI Punjab", "Royal Challengers Bengaluru", "Rising Pune Supergiants" appear alongside their renamed equivalents | Creates duplicate entities, inflates team count | Unified via `TEAM_NAME_MAP` dictionary |
| 2 | **No-result / Tied matches** — 5 NaN winners + 14 ties + 5 "no result" rows | No ground-truth label for these rows | Dropped (kept only `result == 'runs'` or `'wickets'`) |
| 3 | **51 missing `city` values** | Incomplete metadata | Filled with `"Unknown"` |
| 4 | **19 missing `result_margin` values** | Cannot use this as a feature | Filled with `0` |
| 5 | **Heterogeneous `season` format** — some seasons are `"2007/08"`, others are `"2009"` | Cannot sort or use ordinally | Extracted first year: `"2007/08"` → `2007` |
| 6 | **Non-predictive columns** — `id`, `method`, `umpire1/2`, `player_of_match`, `target_runs/overs`, `super_over` | Add noise, no signal | Dropped before modelling |

---

## ⚙️ Feature Engineering

Ten features are computed from the cleaned dataset:

| Feature | Type | Description | Importance |
|---|---|---|---|
| `team1_total_wins` | Float | Team 1's all-time IPL win count — proxy for franchise strength | Very High |
| `team2_total_wins` | Float | Team 2's all-time IPL win count | Very High |
| `win_diff` | Float | `team1_total_wins − team2_total_wins` — net strength gap | High |
| `team1_h2h_win_rate` | Float [0,1] | Team 1's historical win rate directly against Team 2 | High |
| `team1_venue_win_rate` | Float [0,1] | Team 1's win % at the match venue | Medium |
| `team2_venue_win_rate` | Float [0,1] | Team 2's win % at the match venue | Medium |
| `venue_advantage_diff` | Float [-1,1] | Net venue win-rate advantage for Team 1 | Medium |
| `team1_won_toss` | Binary {0,1} | Whether Team 1 won the toss | Low |
| `toss_decision_bat` | Binary {0,1} | Whether the toss winner chose to bat | Low |
| `season_number` | Integer | Ordinal season year — captures league maturity trends | Low |

---

## 🧠 Model Architecture

The entire pipeline is wrapped in a Scikit-Learn `Pipeline` to prevent data leakage:

```
Data → StandardScaler → Logistic Regression (C=1.0, lbfgs solver)
```

Three models were evaluated:

| Model | CV Accuracy | CV ROC-AUC | Test Accuracy | Test ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** ✅ | **78.84% ±2.1%** | **88.99% ±1.1%** | **79.63%** | **90.02%** |
| Random Forest (300 trees) | 75.23% ±2.0% | 86.57% ±1.4% | — | — |
| Gradient Boosting | 71.86% ±4.1% | 83.93% ±2.4% | — | — |

**Winner: Logistic Regression** — Selected by highest mean CV ROC-AUC.

> The dominance of Logistic Regression (a linear model) over tree ensembles indicates that the relationship between our engineered features and the outcome is largely **linear** in nature. Team strength differences, venue advantage, and head-to-head record combine additively — there is no strong interaction effect that trees would capture.

---

## 📈 Results

### Confusion Matrix — Test Set (216 samples)

```
                 Predicted
                T2 Wins   T1 Wins
Actual T2 Wins [ TN=88    FP=18  ]
Actual T1 Wins [ FN=26    TP=84  ]
```

### Classification Report

```
              precision  recall  f1-score  support
Team 2 Wins      0.77    0.83      0.80      106
Team 1 Wins      0.82    0.76      0.79      110

accuracy                           0.80      216
macro avg        0.80    0.80      0.80      216
```

---

## 📁 Project Structure

```
ipl-match-predictor/
├── data/
│   ├── matches.csv          # IPL match records (2008–2024), 1,095 rows × 20 cols
│   └── deliveries.csv       # Ball-by-ball data (reserved for future player features)
│
├── models/
│   └── best_ipl_pipeline.pkl  # Serialised fitted Scikit-Learn Pipeline (auto-generated)
│
├── notebooks/
│   ├── generate_eda_plots.py  # EDA visualisation script
│   ├── matches_per_season.png
│   ├── total_wins_by_team.png
│   ├── toss_decision_distribution.png
│   ├── toss_effect.png
│   ├── top_venues.png
│   └── win_margins.png
│
├── src/
│   ├── ipl_predictor.py       # Original modular predictor (legacy)
│   └── step7_upgrades.py      # Multi-class experiment (legacy)
│
├── train.py                   # ⭐ Main ML pipeline — run this
├── TUTORIAL.md                # Step-by-step tutorial with engineering rationale
├── README.md                  # This file
└── .gitignore
```

---

## 🚀 Installation & Execution

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/ipl-match-predictor.git
cd ipl-match-predictor
```

### Step 2 — Install dependencies

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### Step 3 — Run the full ML pipeline

```bash
python train.py
```

This single command will:
1. Load and clean `data/matches.csv`
2. Engineer all 10 features
3. Run 5-fold stratified cross-validation on 3 models
4. Evaluate the best model on the held-out test set
5. Save the fitted pipeline to `models/best_ipl_pipeline.pkl`
6. Print live match predictions across all toss scenarios

### Step 4 — (Optional) Generate EDA visualisations

```bash
python notebooks/generate_eda_plots.py
```

---

## 🔮 Sample Predictions

```
Mumbai Indians vs Chennai Super Kings at Wankhede Stadium
──────────────────────────────────────────────────────────
Toss Winner     Toss Decision   MI Win%   CSK Win%   Winner
Mumbai Indians  bat             ~61%      ~39%       Mumbai Indians
Mumbai Indians  field           ~60%      ~40%       Mumbai Indians
Chennai SC      bat             ~47%      ~53%       Chennai Super Kings
Chennai SC      field           ~46%      ~54%       Chennai Super Kings

Kolkata Knight Riders vs Royal Challengers Bangalore at Eden Gardens
─────────────────────────────────────────────────────────────────────
Toss Winner     Toss Decision   KKR Win%  RCB Win%   Winner
Kolkata KR      bat             ~70%      ~30%       Kolkata Knight Riders
Kolkata KR      field           ~69%      ~31%       Kolkata Knight Riders
```

---

## 🛠 Technical Stack

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Core language |
| **Pandas** | 2.x | Data loading, cleaning, feature engineering |
| **NumPy** | 1.26+ | Numerical operations |
| **Scikit-Learn** | 1.4+ | `Pipeline`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, cross-validation, metrics |
| **joblib** | bundled | Model serialisation (`.pkl`) |
| **Matplotlib** | 3.x | EDA plots |
| **Seaborn** | 0.13+ | Styled EDA plots |

---

## 🔑 Key Insights

1. **Team strength is king** — Historical win count (`team1_total_wins` / `team2_total_wins`) is the most predictive feature. Strong franchises (MI, CSK) win consistently.

2. **Toss matters less than you think** — Toss-related features have the lowest feature importance. The probability swing from winning vs. losing the toss is less than 3 percentage points.

3. **Head-to-head record is a meaningful signal** — Historical matchup data between two specific teams provides useful signal beyond overall team strength.

4. **Venue advantage compounds** — Teams playing on their home ground (e.g., MI at Wankhede) show measurably better venue win rates.

5. **The problem is linearly separable** — Logistic Regression outperforms both Random Forest and Gradient Boosting, confirming that feature interactions are weak and the signal is additive.

---

## 🔭 Future Work & Recent Improvements

### ✅ Completed (September 2026)

1. **✅ Time-aware split** — Implemented temporal train/test split (2008–2022 train, 2023–2024 test) with rolling aggregates to prevent look-ahead bias
2. **✅ Player-level features** — Extracted 732 player stats from `deliveries.csv`; engineered squad power, recent form, batting/bowling strength indices (8 new features)
3. **✅ XGBoost / LightGBM + tuning** — Bayesian hyperparameter optimization with Optuna for 5 models; best params saved to `models/tuning_results.json`
4. **✅ Calibration** — Applied isotonic calibration via `CalibratedClassifierCV` for better probability estimates
5. **✅ Ensemble** — Weighted ensemble combining Logistic Regression, Gradient Boosting, and LightGBM

### 📊 New Results (Temporal Holdout 2023–2024)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.28%** ⬆️ from 79.63% |
| Test ROC-AUC | 0.9737 |
| Test Precision | 91% (Team 2), 90% (Team 1) |
| Test Recall | 88% (Team 2), 92% (Team 1) |

**See:** `IMPLEMENTATION_SUMMARY.md` for full details on all 5 phases

### 🎯 Remaining Work Toward 95%

- [ ] Match context features (days between matches, home/away, fatigue)
- [ ] Time-series models (LSTM on match sequences for momentum capture)
- [ ] Feature interactions (polynomial, cross-team dynamics)
- [ ] Advanced threshold optimization (cost-sensitive learning per season)
- [ ] **Streamlit App** — Interactive web interface for real-time match predictions

---

## 👤 Author

**Dilip**

## 📄 License

Educational Purpose
