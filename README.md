# IPL Match Outcome Predictor

## 🚀 Problem
Predict IPL match outcomes using historical match data.

## ❗ Key Insight
Initial approach treated this as a multi-class classification problem (predicting exact winner), which resulted in low accuracy (~43%).

Reframing the problem as a binary classification task:
"Will team1 win?" significantly improved performance to ~81%.

## 🧠 Approach

### Data Processing
- Cleaned dataset and handled missing values
- Selected relevant features (teams, toss, venue)

### Feature Engineering
- Created team strength features based on historical wins
- Encoded toss outcome and decision

### Modeling
- Logistic Regression (baseline)
- Random Forest (non-linear model)

### Evaluation
- Multi-class accuracy: ~43%
- Binary classification accuracy: ~81%

## 📊 Key Findings
- Team strength is the most influential factor (~60% importance)
- Toss-related features contribute minimally (~5%)
- Problem formulation had the biggest impact on performance

## 🛠 Tech Stack
Python, Pandas, NumPy, Scikit-learn

## 🔮 Future Improvements
- Add player-level data
- Include recent form (rolling averages)
- Try advanced models (XGBoost)


# IPL Match Outcome Predictor

> Predict IPL match outcomes using historical data and machine learning.

## Problem Statement

Given two IPL teams and match conditions (venue, toss), predict which team is more likely to win.

## Dataset

- **Source**: [Kaggle - IPL Complete Dataset (2008-2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- **Size**: 1,090 matches across 17 IPL seasons
- **Features**: 20 columns including teams, venue, toss, result, etc.

## Approach

### 1. Data Cleaning
- Standardized team names (e.g., "Delhi Daredevils" → "Delhi Capitals")
- Handled missing values (dropped no-result matches, filled city/player NaNs)
- Removed non-predictive columns (umpires, method, id)

### 2. Feature Engineering
| Feature | Description |
|---------|-------------|
| `team1_total_wins` | Historical total wins for team 1 |
| `team2_total_wins` | Historical total wins for team 2 |
| `team1_won_toss` | Whether team 1 won the toss (binary) |
| `toss_decision_bat` | Whether toss winner chose to bat (binary) |
| `team1_venue_win_rate` | Team 1's win rate at the specific venue |
| `team2_venue_win_rate` | Team 2's win rate at the specific venue |
| `team1_h2h_win_rate` | Team 1's head-to-head win rate vs team 2 |

### 3. Model Training
- **Logistic Regression** (selected as best)
- **Random Forest Classifier** (200 trees, max_depth=8)
- 5-fold cross-validation for model selection

## Results

| Model | Test Accuracy | CV Accuracy |
|-------|:------------:|:-----------:|
| **Logistic Regression** | **81.19%** | **78.81%** |
| Random Forest | 75.69% | 77.06% |

### Sample Predictions

| Match | Venue | Predicted Winner | Win Probability |
|-------|-------|:----------------:|:---------------:|
| MI vs CSK | Wankhede Stadium | Mumbai Indians | ~60% |
| KKR vs RCB | Eden Gardens | Kolkata Knight Riders | ~71% |
| DC vs RR | Neutral | Rajasthan Royals | ~53% |
| SRH vs PBKS | Neutral | Sunrisers Hyderabad | ~61% |

## Project Structure

```
ipl-match-predictor/
├── data/
│   ├── matches.csv          # IPL match data (2008-2024)
│   └── deliveries.csv       # Ball-by-ball data (for future use)
├── notebooks/
│   ├── generate_eda_plots.py
│   └── *.png                # EDA visualizations
├── src/
│   └── ipl_predictor.py     # Core ML pipeline
└── README.md
```

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the full pipeline (clean → engineer → train → predict)
python src/ipl_predictor.py

# Generate EDA visualizations
python notebooks/generate_eda_plots.py
```

## Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation & cleaning
- **NumPy** — numerical operations
- **Scikit-learn** — ML models, evaluation, cross-validation
- **Matplotlib / Seaborn** — data visualizations

## Key Insights

- Winning the toss gives only a slight advantage (~51% win rate)
- Venue performance and head-to-head record are strong predictors
- Team historical strength (total wins) is the single most important feature
- Logistic Regression outperformed Random Forest, suggesting linear separability in the feature space

## Future Work

- Add player-level features (top batsmen/bowlers availability)
- Incorporate deliveries.csv for deeper team performance metrics
- Try gradient boosting (XGBoost / LightGBM)
- Build a Streamlit web app for interactive predictions
- Add recent form (last 5 match results) as a feature

## Author

Dilip

## License

KMIT
