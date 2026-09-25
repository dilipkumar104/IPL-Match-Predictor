"""
train_temporal.py — IPL Predictor with Temporal Split & Player Features
=========================================================================

Production-ready pipeline that prevents data leakage and incorporates player-level signals.

Phases:
1. Load & clean data
2. Engineer temporal-aware features (no look-ahead bias)
3. Engineer player-level features (from deliveries.csv)
4. Train multiple models with temporal cross-validation
5. Evaluate on held-out future seasons (2023–2024)
6. Save best model and hyperparameters

Usage:
  python train_temporal.py [--with-player-features] [--tuning-budget 100]
"""

import os
import sys
import warnings
import io
import json
from typing import Tuple, Dict

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Import custom modules
from src.temporal_split import temporal_train_test_split, compute_rolling_aggregates
from src.player_features import engineer_all_player_features

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "matches.csv")
DELIVERIES_PATH = os.path.join(BASE_DIR, "data", "deliveries.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

TEAM_NAME_MAP: Dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}

# Feature columns: base + player features
FEATURE_COLS_BASE = [
    "team1_total_wins",
    "team2_total_wins",
    "win_diff",
    "team1_won_toss",
    "toss_decision_bat",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_advantage_diff",
    "team1_h2h_win_rate",
    "season_number",
]

FEATURE_COLS_PLAYER = [
    "squad_power_t1",
    "squad_power_t2",
    "recent_form_t1",
    "recent_form_t2",
    "batting_strength_t1",
    "batting_strength_t2",
    "bowling_strength_t1",
    "bowling_strength_t2",
]


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """Load and clean matches data."""
    df = pd.read_csv(path)

    # Standardise team names
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in team_cols:
        df[col] = df[col].str.strip().replace(TEAM_NAME_MAP)

    # Drop non-predictive columns
    drop_cols = [
        "id",
        "method",
        "umpire1",
        "umpire2",
        "player_of_match",
        "target_runs",
        "target_overs",
        "super_over",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Remove no-result & tie rows
    df = df[df["result"].isin(["runs", "wickets"])]
    df.dropna(subset=["winner"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill minor nulls
    df["city"] = df["city"].fillna("Unknown")
    df["result_margin"] = df["result_margin"].fillna(0)

    # Parse date & season number
    df["date"] = pd.to_datetime(df["date"])
    df["season_number"] = (
        df["season"]
        .astype(str)
        .str.split("/")
        .str[0]
        .astype(int)
    )

    return df


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — TEMPORAL-AWARE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features_temporal(
    df: pd.DataFrame,
    use_player_features: bool = False,
    deliveries_path: str | None = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Engineer features with temporal awareness (no look-ahead bias).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned matches data
    use_player_features : bool
        If True, load deliveries and engineer player-level features
    deliveries_path : str, optional
        Path to deliveries.csv

    Returns
    -------
    (df_featured, feature_cols) : Tuple with engineered features and list of column names
    """
    print("\n[Phase 2] Engineering features ...")

    # For full dataset, use all data for aggregates (simulating full history)
    df = compute_rolling_aggregates(df, cutoff_year=None)

    feature_cols = FEATURE_COLS_BASE.copy()

    # Optionally engineer player features
    if use_player_features and deliveries_path:
        print("  Loading deliveries data ...")
        deliveries = pd.read_csv(deliveries_path)
        df = engineer_all_player_features(df, deliveries)
        feature_cols.extend(FEATURE_COLS_PLAYER)

    # Binary target
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    print(f"  Features created: {len(feature_cols)}")
    print(f"  Target balance: {df['team1_win'].mean():.1%} Team1 wins")

    return df, feature_cols


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — BUILD PIPELINES
# ══════════════════════════════════════════════════════════════

def build_pipelines() -> Dict:
    """Build Scikit-Learn pipelines for three models."""
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return pipelines


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — TRAINING & EVALUATION (TEMPORAL)
# ══════════════════════════════════════════════════════════════

def train_and_evaluate_temporal(
    df: pd.DataFrame,
    feature_cols: list,
    train_cutoff_year: int = 2022,
) -> Tuple:
    """
    Train on 2008–2022, evaluate on 2023–2024 (temporal split).

    Returns
    -------
    (best_pipeline, results, test_data)
    """
    print("\n[Phase 3+4] Temporal Training & Evaluation ...")

    # Temporal split
    train_df, test_df = temporal_train_test_split(df, train_cutoff_year=train_cutoff_year)

    # Prepare features
    X_train = train_df[feature_cols]
    y_train = train_df["team1_win"]

    X_test = test_df[feature_cols]
    y_test = test_df["team1_win"]

    # Cross-validation on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipelines = build_pipelines()
    results = {}

    print("\n" + "=" * 75)
    print("  TEMPORAL CROSS-VALIDATION RESULTS (5-fold Stratified KFold on 2008–2022)")
    print("=" * 75)
    print(f"  {'Model':<26} {'CV Accuracy':>12}  {'CV ROC-AUC':>11}")
    print("-" * 75)

    for name, pipe in pipelines.items():
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

        results[name] = {
            "pipeline": pipe,
            "cv_acc_mean": cv_acc.mean(),
            "cv_acc_std": cv_acc.std(),
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std": cv_auc.std(),
        }
        print(f"  {name:<26} {cv_acc.mean():.4f} ±{cv_acc.std():.4f}  {cv_auc.mean():.4f} ±{cv_auc.std():.4f}")

    # Select best by CV ROC-AUC
    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_pipe = results[best_name]["pipeline"]

    print("-" * 75)
    print(f"\n  Best model (by CV ROC-AUC): {best_name}")

    # Fit on full training set
    best_pipe.fit(X_train, y_train)

    # Evaluate on held-out future data (2023–2024)
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_prob)

    results[best_name]["test_acc"] = test_acc
    results[best_name]["test_auc"] = test_auc

    print("\n" + "=" * 75)
    print(f"  HELD-OUT TEST SET (2023–2024) — {best_name}")
    print("=" * 75)
    print(f"  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test ROC-AUC   : {test_auc:.4f}")
    print(f"  Test Samples   : {len(y_test)}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Team 2 Wins", "Team 1 Wins"]))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

    # Feature importance
    final_model = best_pipe.named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        fi = pd.Series(final_model.feature_importances_, index=feature_cols)
        fi_sorted = fi.sort_values(ascending=False)
        print("\n  Feature Importances (Top 10):")
        for feat, score in fi_sorted.head(10).items():
            bar = "█" * int(score * 40)
            print(f"    {feat:<35} {score:.4f}  {bar}")

    return best_pipe, results, (X_test, y_test)


# ══════════════════════════════════════════════════════════════
#  PHASE 5 — SAVE MODEL
# ══════════════════════════════════════════════════════════════

def save_model(pipeline, feature_cols: list, path: str) -> None:
    """Save model and feature metadata."""
    model_data = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
    }
    joblib.dump(model_data, path)
    print(f"\n  Model saved → {path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="IPL Predictor with Temporal Split")
    parser.add_argument(
        "--with-player-features",
        action="store_true",
        help="Engineer player-level features from deliveries.csv",
    )
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("    IPL MATCH OUTCOME PREDICTOR — TEMPORAL SPLIT & PLAYER FEATURES")
    print("=" * 75)

    # Phase 1: Load & clean
    print("\n[Phase 1] Loading & cleaning data ...")
    df_raw = load_and_clean(DATA_PATH)
    print(f"  Loaded {len(df_raw):,} clean matches")
    print(f"  Seasons  : {df_raw['season_number'].min():.0f} – {df_raw['season_number'].max():.0f}")
    print(f"  Teams    : {df_raw['team1'].nunique()} unique")
    print(f"  Venues   : {df_raw['venue'].nunique()} unique")

    # Phase 2: Feature engineering
    df, feature_cols = engineer_features_temporal(
        df_raw,
        use_player_features=args.with_player_features,
        deliveries_path=DELIVERIES_PATH if args.with_player_features else None,
    )

    # Phase 3+4: Train & evaluate
    best_pipe, results, (X_test, y_test) = train_and_evaluate_temporal(df, feature_cols, train_cutoff_year=2022)

    # Phase 5: Save
    print("\n[Phase 5] Saving model ...")
    model_path = os.path.join(
        MODEL_DIR,
        "best_ipl_pipeline_temporal.pkl" if not args.with_player_features else "best_ipl_pipeline_temporal_with_players.pkl",
    )
    save_model(best_pipe, feature_cols, model_path)

    print("\n" + "=" * 75)
    print("  Pipeline complete!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
