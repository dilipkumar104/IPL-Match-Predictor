"""
hyperparameter_tuning.py — Bayesian Optimization with Optuna
=============================================================

Systematically tunes hyperparameters for 4 models:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost

Uses Optuna for efficient Bayesian hyperparameter search.
"""

import os
import sys
import warnings
import json
from typing import Dict, Tuple, Callable

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] Optuna not installed. Install with: pip install optuna")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not installed. Install with: pip install lightgbm")

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


# ──────────────────────────────────────────────────────────────
# TUNING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: object,
    n_trials: int = 50,
) -> Dict:
    """
    Tune Logistic Regression via Optuna.

    Parameters to tune:
    - C: inverse regularisation strength
    - solver: 'lbfgs', 'liblinear', 'newton-cg'
    """
    if not OPTUNA_AVAILABLE:
        print("[SKIP] Optuna not available for LR tuning")
        return {}

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(),
    )

    def objective(trial):
        C = trial.suggest_float("C", 0.001, 100, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "newton-cg"])

        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                )),
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()
        except Exception:
            return 0.5

    print("  Tuning Logistic Regression...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print(f"    Best ROC-AUC: {best_score:.4f}")
    print(f"    Best params: C={best_params.get('C', 1.0):.4f}, solver={best_params.get('solver', 'lbfgs')}")

    return {
        "best_params": best_params,
        "best_score": best_score,
    }


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: object,
    n_trials: int = 50,
) -> Dict:
    """
    Tune Random Forest via Optuna.

    Parameters to tune:
    - n_estimators
    - max_depth
    - min_samples_leaf
    - min_samples_split
    """
    if not OPTUNA_AVAILABLE:
        return {}

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(),
    )

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()
        except Exception:
            return 0.5

    print("  Tuning Random Forest...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print(f"    Best ROC-AUC: {best_score:.4f}")
    print(f"    Best params: n_est={best_params.get('n_estimators', 100)}, depth={best_params.get('max_depth', 8)}")

    return {
        "best_params": best_params,
        "best_score": best_score,
    }


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: object,
    n_trials: int = 50,
) -> Dict:
    """
    Tune Gradient Boosting via Optuna.

    Parameters to tune:
    - n_estimators
    - learning_rate
    - max_depth
    - subsample
    - min_samples_leaf
    """
    if not OPTUNA_AVAILABLE:
        return {}

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(),
    )

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                )),
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()
        except Exception:
            return 0.5

    print("  Tuning Gradient Boosting...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print(f"    Best ROC-AUC: {best_score:.4f}")
    print(f"    Best params: n_est={best_params.get('n_estimators', 100)}, lr={best_params.get('learning_rate', 0.05):.4f}")

    return {
        "best_params": best_params,
        "best_score": best_score,
    }


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: object,
    n_trials: int = 50,
) -> Dict:
    """
    Tune XGBoost via Optuna.
    """
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        print("  [SKIP] XGBoost tuning (Optuna or XGBoost not available)")
        return {}

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(),
    )

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 2, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        lambda_reg = trial.suggest_float("lambda", 0.0, 10.0)
        alpha_reg = trial.suggest_float("alpha", 0.0, 10.0)

        try:
            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=lambda_reg,
                reg_alpha=alpha_reg,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()
        except Exception:
            return 0.5

    print("  Tuning XGBoost...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print(f"    Best ROC-AUC: {best_score:.4f}")
    print(f"    Best params: depth={best_params.get('max_depth', 5)}, lr={best_params.get('learning_rate', 0.1):.4f}")

    return {
        "best_params": best_params,
        "best_score": best_score,
    }


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: object,
    n_trials: int = 50,
) -> Dict:
    """
    Tune LightGBM via Optuna.
    """
    if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
        print("  [SKIP] LightGBM tuning (Optuna or LightGBM not available)")
        return {}

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(),
    )

    def objective(trial):
        num_leaves = trial.suggest_int("num_leaves", 10, 100)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.5, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 10.0)
        lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 10.0)

        try:
            model = lgb.LGBMClassifier(
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=feature_fraction,
                reg_lambda=lambda_l2,
                reg_alpha=lambda_l1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            return scores.mean()
        except Exception:
            return 0.5

    print("  Tuning LightGBM...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    print(f"    Best ROC-AUC: {best_score:.4f}")
    print(f"    Best params: leaves={best_params.get('num_leaves', 31)}, lr={best_params.get('learning_rate', 0.1):.4f}")

    return {
        "best_params": best_params,
        "best_score": best_score,
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 75)
    print("  HYPERPARAMETER TUNING — Optuna Bayesian Optimization")
    print("=" * 75)

    if not OPTUNA_AVAILABLE:
        print("\n[ERROR] Optuna not installed.")
        print("Install with: pip install optuna>=4.0.0 xgboost>=2.0.0 lightgbm>=4.0.0")
        sys.exit(1)

    # Load and prepare data
    print("\n[Loading Data]")
    from src.temporal_split import compute_rolling_aggregates

    df_raw = pd.read_csv(DATA_PATH)

    # Clean
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in team_cols:
        df_raw[col] = df_raw[col].str.strip().replace(TEAM_NAME_MAP)

    drop_cols = ["id", "method", "umpire1", "umpire2", "player_of_match", "target_runs", "target_overs", "super_over"]
    df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns], inplace=True)

    df_raw = df_raw[df_raw["result"].isin(["runs", "wickets"])]
    df_raw.dropna(subset=["winner"], inplace=True)

    df_raw["city"] = df_raw["city"].fillna("Unknown")
    df_raw["result_margin"] = df_raw["result_margin"].fillna(0)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw["season_number"] = df_raw["season"].astype(str).str.split("/").str[0].astype(int)

    # Feature engineering (base features only for now)
    print("  Engineering features...")
    df = compute_rolling_aggregates(df_raw, cutoff_year=None)
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    # Use training data (2008–2022) for tuning
    train_df = df[df["season_number"] <= 2022].copy()

    X_train = train_df[FEATURE_COLS_BASE]
    y_train = train_df["team1_win"]

    print(f"  Training set: {len(train_df)} matches")
    print(f"  Features: {len(FEATURE_COLS_BASE)}")

    # Create CV splitter
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Tune all models
    print("\n[Phase 3] Hyperparameter Tuning\n")

    tuning_results = {}

    tuning_results["Logistic Regression"] = tune_logistic_regression(X_train, y_train, skf, n_trials=50)
    tuning_results["Random Forest"] = tune_random_forest(X_train, y_train, skf, n_trials=50)
    tuning_results["Gradient Boosting"] = tune_gradient_boosting(X_train, y_train, skf, n_trials=50)
    if XGBOOST_AVAILABLE:
        tuning_results["XGBoost"] = tune_xgboost(X_train, y_train, skf, n_trials=50)
    if LIGHTGBM_AVAILABLE:
        tuning_results["LightGBM"] = tune_lightgbm(X_train, y_train, skf, n_trials=50)

    # Summary
    print("\n" + "=" * 75)
    print("  TUNING SUMMARY (sorted by ROC-AUC)")
    print("=" * 75)

    sorted_results = sorted(
        [(name, info["best_score"]) for name, info in tuning_results.items() if info],
        key=lambda x: x[1],
        reverse=True,
    )

    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"  {rank}. {name:<25} ROC-AUC: {score:.4f}")

    # Save tuning results
    save_path = os.path.join(MODEL_DIR, "tuning_results.json")
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for model_name, info in tuning_results.items():
        if info:
            results_serializable[model_name] = {
                "best_score": float(info["best_score"]),
                "best_params": {
                    k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                    for k, v in info["best_params"].items()
                },
            }

    with open(save_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n  Tuning results saved → {save_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
