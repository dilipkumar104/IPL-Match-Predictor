"""
train_optimized.py — IPL Predictor with Tuned Hyperparameters & Ensemble
==========================================================================

Uses Optuna-tuned hyperparameters and combines multiple models via weighted ensemble.

Phases:
1. Load & clean data
2. Engineer temporal-aware features
3. Train multiple models with tuned hyperparameters
4. Combine via weighted ensemble
5. Apply calibration
6. Evaluate on held-out future seasons (2023–2024)
"""

import os
import sys
import warnings
import io
import json
from typing import Tuple, Dict, List

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.temporal_split import temporal_train_test_split, compute_rolling_aggregates

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "matches.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TUNING_RESULTS_PATH = os.path.join(MODEL_DIR, "tuning_results.json")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

TEAM_NAME_MAP: Dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}

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
        "id", "method", "umpire1", "umpire2", "player_of_match",
        "target_runs", "target_overs", "super_over",
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
#  PHASE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features_temporal(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Engineer temporal-aware features."""
    print("\n[Phase 2] Engineering features ...")

    # Temporal aggregates (no look-ahead bias)
    df = compute_rolling_aggregates(df, cutoff_year=None)

    # Binary target
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    feature_cols = FEATURE_COLS_BASE.copy()

    print(f"  Features created: {len(feature_cols)}")
    print(f"  Target balance: {df['team1_win'].mean():.1%} Team1 wins")

    return df, feature_cols


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — BUILD PIPELINES WITH TUNED HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════

def load_tuned_hyperparams() -> Dict:
    """Load tuned hyperparameters from JSON file."""
    if not os.path.exists(TUNING_RESULTS_PATH):
        print(f"[WARNING] Tuning results not found at {TUNING_RESULTS_PATH}")
        print("          Using default hyperparameters instead")
        return {}

    with open(TUNING_RESULTS_PATH, "r") as f:
        return json.load(f)


def build_pipelines_tuned(tuned_params: Dict) -> Dict:
    """Build pipelines using tuned hyperparameters."""
    pipelines = {}

    # ── Logistic Regression ──
    lr_params = tuned_params.get("Logistic Regression", {}).get("best_params", {})
    pipelines["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=lr_params.get("C", 1.0),
            solver=lr_params.get("solver", "lbfgs"),
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])

    # ── Random Forest ──
    rf_params = tuned_params.get("Random Forest", {}).get("best_params", {})
    pipelines["Random Forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=int(rf_params.get("n_estimators", 300)),
            max_depth=int(rf_params.get("max_depth", 8)) if rf_params.get("max_depth") else None,
            min_samples_leaf=int(rf_params.get("min_samples_leaf", 5)),
            min_samples_split=int(rf_params.get("min_samples_split", 2)),
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # ── Gradient Boosting ──
    gb_params = tuned_params.get("Gradient Boosting", {}).get("best_params", {})
    pipelines["Gradient Boosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=int(gb_params.get("n_estimators", 200)),
            learning_rate=gb_params.get("learning_rate", 0.05),
            max_depth=int(gb_params.get("max_depth", 4)),
            subsample=gb_params.get("subsample", 0.8),
            min_samples_leaf=int(gb_params.get("min_samples_leaf", 1)),
            random_state=RANDOM_STATE,
        )),
    ])

    # ── XGBoost (if available) ──
    if XGBOOST_AVAILABLE:
        xgb_params = tuned_params.get("XGBoost", {}).get("best_params", {})
        xgb_model = xgb.XGBClassifier(
            max_depth=int(xgb_params.get("max_depth", 5)),
            learning_rate=xgb_params.get("learning_rate", 0.1),
            n_estimators=int(xgb_params.get("n_estimators", 100)),
            subsample=xgb_params.get("subsample", 1.0),
            colsample_bytree=xgb_params.get("colsample_bytree", 1.0),
            reg_lambda=xgb_params.get("lambda", 1.0),
            reg_alpha=xgb_params.get("alpha", 0.0),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pipelines["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb_model),
        ])

    # ── LightGBM (if available) ──
    if LIGHTGBM_AVAILABLE:
        lgb_params = tuned_params.get("LightGBM", {}).get("best_params", {})
        lgb_model = lgb.LGBMClassifier(
            num_leaves=int(lgb_params.get("num_leaves", 31)),
            learning_rate=lgb_params.get("learning_rate", 0.1),
            n_estimators=int(lgb_params.get("n_estimators", 100)),
            subsample=lgb_params.get("subsample", 1.0),
            colsample_bytree=lgb_params.get("feature_fraction", 1.0),
            reg_lambda=lgb_params.get("lambda_l2", 0.0),
            reg_alpha=lgb_params.get("lambda_l1", 0.0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        pipelines["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", lgb_model),
        ])

    return pipelines


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — TRAINING & ENSEMBLE
# ══════════════════════════════════════════════════════════════

def train_and_evaluate_ensemble(
    df: pd.DataFrame,
    feature_cols: list,
    train_cutoff_year: int = 2022,
) -> Tuple:
    """
    Train multiple models, evaluate via CV, and combine via weighted ensemble.

    Returns
    -------
    (ensemble_pipeline, best_single, results, test_data)
    """
    print("\n[Phase 3+4] Training & Ensemble ...")

    # Temporal split
    train_df, test_df = temporal_train_test_split(df, train_cutoff_year=train_cutoff_year)

    X_train = train_df[feature_cols]
    y_train = train_df["team1_win"]

    X_test = test_df[feature_cols]
    y_test = test_df["team1_win"]

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Load tuned hyperparameters and build pipelines
    tuned_params = load_tuned_hyperparams()
    pipelines = build_pipelines_tuned(tuned_params)

    results = {}

    print("\n" + "=" * 80)
    print("  CROSS-VALIDATION RESULTS (5-fold Stratified KFold on 2008–2022)")
    print("=" * 80)
    print(f"  {'Model':<26} {'CV Accuracy':>12}  {'CV ROC-AUC':>11}")
    print("-" * 80)

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

    # Select best single model by CV ROC-AUC
    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_pipe = results[best_name]["pipeline"]

    print("-" * 80)
    print(f"\n  Best single model (by CV ROC-AUC): {best_name}")
    print(f"    CV ROC-AUC: {results[best_name]['cv_auc_mean']:.4f}")

    # Fit best model on full training set
    best_pipe.fit(X_train, y_train)

    # ── Build Weighted Ensemble ──
    print("\n  Building weighted ensemble...")

    # Use top 3 models by CV ROC-AUC
    top_3_names = sorted(results.keys(), key=lambda k: results[k]["cv_auc_mean"], reverse=True)[:3]
    top_3_auc = [results[n]["cv_auc_mean"] for n in top_3_names]

    # Normalize scores to weights (sum = 1)
    total_auc = sum(top_3_auc)
    weights = {name: auc / total_auc for name, auc in zip(top_3_names, top_3_auc)}

    print(f"    Using models: {', '.join(top_3_names)}")
    print(f"    Weights: {', '.join([f'{w}={v:.3f}' for w, v in weights.items()])}")

    # Fit ensemble models
    ensemble_models = {}
    for name in top_3_names:
        pipe_copy = results[name]["pipeline"]
        pipe_copy.fit(X_train, y_train)
        ensemble_models[name] = pipe_copy

    # ── PHASE 5: CALIBRATION ──
    print("\n  Applying calibration...")
    calibrated_models = {}
    for name, model in ensemble_models.items():
        # Use 5-fold CV calibration on the full training set
        cal_model = CalibratedClassifierCV(model, cv=5, method="isotonic")
        cal_model.fit(X_train, y_train)
        calibrated_models[name] = cal_model

    print("    Calibration applied")

    # ── Evaluate on test set ──
    print("\n" + "=" * 80)
    print("  HELD-OUT TEST SET (2023–2024) — ENSEMBLE")
    print("=" * 80)

    # Weighted ensemble prediction
    ensemble_probs = np.zeros(len(X_test))
    for name, model in calibrated_models.items():
        probs = model.predict_proba(X_test)[:, 1]
        ensemble_probs += weights[name] * probs

    ensemble_probs /= sum(weights.values())
    y_pred_ensemble = (ensemble_probs > 0.5).astype(int)

    test_acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    test_auc_ensemble = roc_auc_score(y_test, ensemble_probs)

    print(f"  Ensemble Test Accuracy : {test_acc_ensemble:.4f}  ({test_acc_ensemble*100:.2f}%)")
    print(f"  Ensemble Test ROC-AUC  : {test_auc_ensemble:.4f}")
    print(f"  Test Samples           : {len(y_test)}")

    print("\n  Classification Report (Ensemble):")
    print(classification_report(y_test, y_pred_ensemble, target_names=["Team 2 Wins", "Team 1 Wins"]))

    print("  Confusion Matrix (Ensemble):")
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

    # Also show best single model performance
    print("\n" + "-" * 80)
    print(f"  Single Best Model ({best_name}) — Test Performance")
    print("-" * 80)

    best_pipe_final = calibrated_models.get(best_name, best_pipe)
    y_pred_best = best_pipe_final.predict(X_test)
    y_prob_best = best_pipe_final.predict_proba(X_test)[:, 1]
    test_acc_best = accuracy_score(y_test, y_pred_best)
    test_auc_best = roc_auc_score(y_test, y_prob_best)

    print(f"  Test Accuracy : {test_acc_best:.4f}  ({test_acc_best*100:.2f}%)")
    print(f"  Test ROC-AUC  : {test_auc_best:.4f}")

    print("\n" + "=" * 80)

    return (ensemble_models, weights, calibrated_models), best_pipe, results, (X_test, y_test)


# ══════════════════════════════════════════════════════════════
#  SAVE MODELS
# ══════════════════════════════════════════════════════════════

def save_ensemble(
    ensemble_data: Tuple,
    feature_cols: list,
    path: str,
) -> None:
    """Save ensemble models and weights."""
    models, weights, calibrated = ensemble_data
    model_data = {
        "ensemble_models": models,
        "ensemble_weights": weights,
        "calibrated_models": calibrated,
        "feature_cols": feature_cols,
    }
    joblib.dump(model_data, path)
    print(f"\n  Ensemble saved → {path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("    IPL MATCH OUTCOME PREDICTOR — TUNED + ENSEMBLE + CALIBRATION")
    print("=" * 80)

    # Phase 1: Load & clean
    print("\n[Phase 1] Loading & cleaning data ...")
    df_raw = load_and_clean(DATA_PATH)
    print(f"  Loaded {len(df_raw):,} clean matches")
    print(f"  Seasons  : {df_raw['season_number'].min():.0f} – {df_raw['season_number'].max():.0f}")
    print(f"  Teams    : {df_raw['team1'].nunique()} unique")
    print(f"  Venues   : {df_raw['venue'].nunique()} unique")

    # Phase 2: Feature engineering
    df, feature_cols = engineer_features_temporal(df_raw)

    # Phase 3+4+5: Train, ensemble, calibrate
    ensemble_data, best_pipe, results, (X_test, y_test) = train_and_evaluate_ensemble(
        df,
        feature_cols,
        train_cutoff_year=2022,
    )

    # Save ensemble
    print("\n[Phase 6] Saving ensemble ...")
    model_path = os.path.join(MODEL_DIR, "best_ipl_ensemble_optimized.pkl")
    save_ensemble(ensemble_data, feature_cols, model_path)

    print("\n" + "=" * 80)
    print("  Pipeline complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
