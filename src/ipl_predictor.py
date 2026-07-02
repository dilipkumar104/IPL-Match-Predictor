"""
IPL Match Outcome Predictor
============================
Given two teams + match conditions → who is more likely to win?

This module handles:
- Data loading & cleaning
- Feature engineering
- Model training & evaluation
- Match outcome prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# TEAM NAME STANDARDIZATION MAP
# Teams changed names over the years — we unify them
# ============================================================
TEAM_NAME_MAP = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
}


def load_and_clean(path: str) -> pd.DataFrame:
    """
    STEP 4 — Data Cleaning
    -----------------------
    Load matches.csv and return a clean DataFrame.
    """
    df = pd.read_csv(path)

    # --- standardize team names across all team columns ---
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        df[col] = df[col].str.strip().replace(TEAM_NAME_MAP)

    # --- drop columns that add zero predictive value ---
    drop_cols = ['id', 'method', 'umpire1', 'umpire2']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- handle missing values ---
    # 'winner' is NaN for no-result / tied matches → drop those rows
    df.dropna(subset=['winner'], inplace=True)

    # fill remaining NaNs (use assignment — inplace on chained ops is deprecated in pandas >=2.x)
    if 'city' in df.columns:
        df['city'] = df['city'].fillna('Unknown')
    if 'player_of_match' in df.columns:
        df['player_of_match'] = df['player_of_match'].fillna('Unknown')
    if 'result_margin' in df.columns:
        df['result_margin'] = df['result_margin'].fillna(0)

    # --- parse date ---
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 5 — Feature Engineering
    -----------------------------
    Create the features that actually make a model useful.
    """
    # 1) Overall team win counts → proxy for team strength
    win_counts = df['winner'].value_counts().to_dict()
    df['team1_total_wins'] = df['team1'].map(win_counts).fillna(0)
    df['team2_total_wins'] = df['team2'].map(win_counts).fillna(0)

    # 2) Toss advantage — did team1 win the toss?
    df['team1_won_toss'] = (df['toss_winner'] == df['team1']).astype(int)

    # 3) Toss decision encoded
    df['toss_decision_bat'] = (df['toss_decision'] == 'bat').astype(int)

    # 4) Venue-level win rate for each team
    # NOTE: pandas >=2.2 excludes groupby keys from the sub-group DataFrame
    #       inside apply(), causing KeyError on 'team1'/'team2'.
    #       Fix: pre-compute win flags as plain columns, then use groupby()[col].mean().
    df['_t1_win'] = (df['winner'] == df['team1']).astype(float)
    df['_t2_win'] = (df['winner'] == df['team2']).astype(float)

    venue_wins_t1 = (
        df.groupby(['venue', 'team1'])['_t1_win']
        .mean()
        .reset_index()
        .rename(columns={'_t1_win': 'team1_venue_win_rate'})
    )
    df = df.merge(venue_wins_t1, on=['venue', 'team1'], how='left')
    df['team1_venue_win_rate'] = df['team1_venue_win_rate'].fillna(0.5)

    venue_wins_t2 = (
        df.groupby(['venue', 'team2'])['_t2_win']
        .mean()
        .reset_index()
        .rename(columns={'_t2_win': 'team2_venue_win_rate'})
    )
    df = df.merge(venue_wins_t2, on=['venue', 'team2'], how='left')
    df['team2_venue_win_rate'] = df['team2_venue_win_rate'].fillna(0.5)

    # 5) Head-to-head win rate (team1 vs team2)
    h2h = (
        df.groupby(['team1', 'team2'])['_t1_win']
        .mean()
        .reset_index()
        .rename(columns={'_t1_win': 'team1_h2h_win_rate'})
    )
    df = df.merge(h2h, on=['team1', 'team2'], how='left')
    df['team1_h2h_win_rate'] = df['team1_h2h_win_rate'].fillna(0.5)

    # Drop temporary helper columns
    df.drop(columns=['_t1_win', '_t2_win'], inplace=True)

    # 6) Binary target: did team1 win?
    df['team1_win'] = (df['winner'] == df['team1']).astype(int)

    return df


# ============================================================
# Feature columns used by the model
# ============================================================
FEATURE_COLS = [
    'team1_total_wins',
    'team2_total_wins',
    'team1_won_toss',
    'toss_decision_bat',
    'team1_venue_win_rate',
    'team2_venue_win_rate',
    'team1_h2h_win_rate',
]


def train_model(df: pd.DataFrame):
    """
    STEP 6 & 7 — Model Training + Evaluation
    ------------------------------------------
    Trains Logistic Regression & Random Forest, returns best.
    """
    X = df[FEATURE_COLS]
    y = df['team1_win']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    # --- Cross-validation for both ---
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring='accuracy').mean()
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()

    results = {
        'logistic_regression': {
            'model': lr,
            'test_accuracy': lr_acc,
            'cv_accuracy': lr_cv,
            'predictions': lr_preds,
        },
        'random_forest': {
            'model': rf,
            'test_accuracy': rf_acc,
            'cv_accuracy': rf_cv,
            'predictions': rf_preds,
        },
    }

    best_name = max(results, key=lambda k: results[k]['cv_accuracy'])
    best = results[best_name]

    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nLogistic Regression  -- Test acc: {lr_acc:.4f}  |  CV acc: {lr_cv:.4f}")
    print(f"Random Forest        -- Test acc: {rf_acc:.4f}  |  CV acc: {rf_cv:.4f}")
    print(f"\n>> Best model: {best_name} (CV accuracy {best['cv_accuracy']:.4f})")
    print("\nClassification Report (best model on test set):")
    print(classification_report(y_test, best['predictions'], target_names=['Team 2 wins', 'Team 1 wins']))

    return best['model'], results, (X_test, y_test)


def predict_match(model, df: pd.DataFrame, team_a: str, team_b: str, venue: str = None):
    """
    STEP 8 — Predict outcome of a new match
    ------------------------------------------
    Team A is placed as team1, Team B as team2.
    """
    # apply name mapping
    team_a = TEAM_NAME_MAP.get(team_a, team_a)
    team_b = TEAM_NAME_MAP.get(team_b, team_b)

    win_counts = df['winner'].value_counts().to_dict()
    t1_wins = win_counts.get(team_a, 0)
    t2_wins = win_counts.get(team_b, 0)

    # head-to-head
    h2h_matches = df[
        ((df['team1'] == team_a) & (df['team2'] == team_b)) |
        ((df['team1'] == team_b) & (df['team2'] == team_a))
    ]
    if len(h2h_matches) > 0:
        h2h_rate = (h2h_matches['winner'] == team_a).mean()
    else:
        h2h_rate = 0.5

    # venue win rates
    if venue:
        venue_matches_a = df[(df['venue'] == venue) & ((df['team1'] == team_a) | (df['team2'] == team_a))]
        venue_rate_a = (venue_matches_a['winner'] == team_a).mean() if len(venue_matches_a) > 0 else 0.5
        venue_matches_b = df[(df['venue'] == venue) & ((df['team1'] == team_b) | (df['team2'] == team_b))]
        venue_rate_b = (venue_matches_b['winner'] == team_b).mean() if len(venue_matches_b) > 0 else 0.5
    else:
        venue_rate_a, venue_rate_b = 0.5, 0.5

    # Predict for both toss outcomes
    results = []
    for toss_winner, toss_bat in [(1, 0), (1, 1), (0, 0), (0, 1)]:
        features = pd.DataFrame([{
            'team1_total_wins': t1_wins,
            'team2_total_wins': t2_wins,
            'team1_won_toss': toss_winner,
            'toss_decision_bat': toss_bat,
            'team1_venue_win_rate': venue_rate_a,
            'team2_venue_win_rate': venue_rate_b,
            'team1_h2h_win_rate': h2h_rate,
        }])
        prob = model.predict_proba(features)[0]
        results.append({
            'toss_winner': team_a if toss_winner else team_b,
            'toss_decision': 'bat' if toss_bat else 'field',
            f'{team_a}_win_prob': f"{prob[1]:.1%}",
            f'{team_b}_win_prob': f"{prob[0]:.1%}",
            'predicted_winner': team_a if prob[1] > 0.5 else team_b,
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN — run end-to-end
# ============================================================
if __name__ == '__main__':
    import os

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'matches.csv')

    print("[IPL] Match Outcome Predictor")
    print("=" * 60)

    # STEP 4 — Clean
    print("\n[1] Loading & cleaning data...")
    df = load_and_clean(DATA_PATH)
    print(f"   Loaded {len(df)} matches  |  {df['year'].min()} - {df['year'].max()}")
    print(f"   Teams: {df['team1'].nunique()} unique")

    # STEP 5 — Features
    print("\n[2] Engineering features...")
    df = engineer_features(df)

    # STEP 6 & 7 — Train + Evaluate
    print("\n[3] Training models...\n")
    best_model, results, (X_test, y_test) = train_model(df)

    # STEP 8 — Demo predictions
    print("\n" + "=" * 60)
    print(">>> MATCH PREDICTIONS")
    print("=" * 60)

    matchups = [
        ('Mumbai Indians', 'Chennai Super Kings', 'Wankhede Stadium'),
        ('Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Eden Gardens'),
        ('Delhi Capitals', 'Rajasthan Royals', None),
        ('Sunrisers Hyderabad', 'Punjab Kings', None),
    ]

    for team_a, team_b, venue in matchups:
        print(f"\n{'-' * 50}")
        v_str = f" at {venue}" if venue else ""
        print(f"  {team_a} vs {team_b}{v_str}")
        print(f"{'-' * 50}")
        pred_df = predict_match(best_model, df, team_a, team_b, venue)
        print(pred_df.to_string(index=False))
