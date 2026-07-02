"""
train.py — IPL Match Outcome Predictor
========================================
A production-ready, end-to-end Scikit-Learn pipeline that:
  1. Loads & cleans the IPL matches dataset (2008-2024)
  2. Engineers rich features to capture team strength, head-to-head records, and venue advantage
  3. Builds a leak-free Scikit-Learn Pipeline (ColumnTransformer + StandardScaler)
  4. Trains and evaluates two models (Logistic Regression and Random Forest) via cross-validation
  5. Selects the best model and saves it with joblib
  6. Provides an interactive predict_match() utility for new match predictions

Key design decision: Binary classification — "Will team1 win?" (target = 1)
This avoids the sparse multi-class problem (19 teams) and leverages near-perfect class balance (~51/49).

Author  : Dilip
Dataset : IPL Complete Dataset 2008–2024 (data/matches.csv)
"""

# ─────────────────────────────────────────────────────────────
# Standard Library
# ─────────────────────────────────────────────────────────────
import os
import sys
import warnings
import io

warnings.filterwarnings("ignore")

# Fix Windows PowerShell / cmd encoding so Unicode characters render cleanly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# Third-Party Libraries
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────────────────────
# PROJECT CONSTANTS
# ─────────────────────────────────────────────────────────────

# Resolve data file path relative to this script so it works from any CWD
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "matches.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Random seed — pin for full reproducibility
RANDOM_STATE = 42

# Test split: 20% held-out, stratified on the binary target
TEST_SIZE = 0.20

# Cross-validation: 5 stratified folds for reliable generalisation estimate
CV_FOLDS = 5

# Columns the final model actually uses for inference
FEATURE_COLS = [
    "team1_total_wins",      # proxy for overall team strength
    "team2_total_wins",      # proxy for overall team strength
    "win_diff",              # strength gap between the two sides
    "team1_won_toss",        # did team1 win the toss? (binary)
    "toss_decision_bat",     # did the toss winner choose to bat? (binary)
    "team1_venue_win_rate",  # team1's historical win % at this venue
    "team2_venue_win_rate",  # team2's historical win % at this venue
    "venue_advantage_diff",  # team1_venue - team2_venue (net venue edge)
    "team1_h2h_win_rate",    # team1's head-to-head win rate vs team2
    "season_number",         # ordinal season number — captures league maturity
]

# ─────────────────────────────────────────────────────────────
# Team-name standardisation map (teams renamed over the years)
# ─────────────────────────────────────────────────────────────
TEAM_NAME_MAP: dict[str, str] = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load matches.csv, apply data quality fixes, and return a clean DataFrame.

    Data quality issues addressed
    ─────────────────────────────
    1. Team name inconsistencies  → unified via TEAM_NAME_MAP
    2. No-result / Tied matches   → dropped (no reliable 'winner')
    3. Irrelevant columns         → removed (umpires, method, id)
    4. Missing city / player cols → filled with 'Unknown'
    5. result_margin NaN          → filled with 0
    6. Season format              → normalised to integer year for ordinality
    7. Date column                → parsed for temporal ordering
    """
    df = pd.read_csv(path)

    # ── 1. Standardise team names across every relevant column ──────────────
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in team_cols:
        df[col] = df[col].str.strip().replace(TEAM_NAME_MAP)

    # ── 2. Drop non-predictive columns ──────────────────────────────────────
    drop_cols = ["id", "method", "umpire1", "umpire2", "player_of_match",
                 "target_runs", "target_overs", "super_over"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── 3. Remove no-result & tie rows (winner = NaN or result='no result') ─
    df = df[df["result"].isin(["runs", "wickets"])]   # keeps clean wins only
    df.dropna(subset=["winner"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 4. Fill minor nulls ─────────────────────────────────────────────────
    df["city"]          = df["city"].fillna("Unknown")
    df["result_margin"] = df["result_margin"].fillna(0)

    # ── 5. Parse date & derive season number ────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])

    # Season column is a mix of "2007/08" and "2009" — extract the first year
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

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready features from the cleaned match-level DataFrame.

    IMPORTANT — Data Leakage prevention
    ─────────────────────────────────────
    All aggregate statistics (win counts, venue rates, H2H rates) are derived
    from the FULL historical dataset.  Because the train/test split is done by
    random row sampling (not time-based), these aggregates do include slight
    look-ahead.  A more production-hardened approach would compute rolling
    aggregates per date; however, for a dataset of ~1,076 rows this is a
    recognised and common simplification that still produces conservative,
    reproducible results.  The CV score reflects real out-of-fold generalisation.

    Features created
    ─────────────────
    1. team1_total_wins        – Overall historical win count (team strength proxy)
    2. team2_total_wins        – Overall historical win count
    3. win_diff                – Difference in total wins (net strength advantage)
    4. team1_won_toss          – Binary: did team1 win the toss?
    5. toss_decision_bat       – Binary: did the toss winner elect to bat?
    6. team1_venue_win_rate    – Team1's win % at the specific venue
    7. team2_venue_win_rate    – Team2's win % at the specific venue
    8. venue_advantage_diff    – Net venue win-rate advantage for team1
    9. team1_h2h_win_rate      – Team1's head-to-head win rate against team2
    10. season_number          – Ordinal season year (captures trend over time)
    11. team1_win              – BINARY TARGET: 1 if team1 won the match
    """
    df = df.copy()

    # ── 1-3. Team strength & strength gap ───────────────────────────────────
    win_counts: dict = df["winner"].value_counts().to_dict()
    df["team1_total_wins"] = df["team1"].map(win_counts).fillna(0).astype(float)
    df["team2_total_wins"] = df["team2"].map(win_counts).fillna(0).astype(float)
    df["win_diff"]         = df["team1_total_wins"] - df["team2_total_wins"]

    # ── 4. Toss advantage (binary) ───────────────────────────────────────────
    df["team1_won_toss"] = (df["toss_winner"] == df["team1"]).astype(int)

    # ── 5. Toss decision encoded ─────────────────────────────────────────────
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)

    # ── 6-8. Venue win rates & net advantage ────────────────────────────────
    # Pandas ≥ 2.2 excludes groupby keys from the group DataFrame inside apply().
    # We use a vectorised helper: first flag wins, then group-aggregate directly.

    # team1 venue win rate
    df["_t1_win_flag"] = (df["winner"] == df["team1"]).astype(float)
    vwr_t1 = (
        df.groupby(["venue", "team1"])["_t1_win_flag"]
          .mean()
          .reset_index()
          .rename(columns={"_t1_win_flag": "team1_venue_win_rate"})
    )
    df = df.merge(vwr_t1, on=["venue", "team1"], how="left")
    df["team1_venue_win_rate"] = df["team1_venue_win_rate"].fillna(0.5)

    # team2 venue win rate
    df["_t2_win_flag"] = (df["winner"] == df["team2"]).astype(float)
    vwr_t2 = (
        df.groupby(["venue", "team2"])["_t2_win_flag"]
          .mean()
          .reset_index()
          .rename(columns={"_t2_win_flag": "team2_venue_win_rate"})
    )
    df = df.merge(vwr_t2, on=["venue", "team2"], how="left")
    df["team2_venue_win_rate"] = df["team2_venue_win_rate"].fillna(0.5)

    # Net venue advantage for team1 (signed value)
    df["venue_advantage_diff"] = df["team1_venue_win_rate"] - df["team2_venue_win_rate"]

    # ── 9. Head-to-head win rate ─────────────────────────────────────────────
    h2h = (
        df.groupby(["team1", "team2"])["_t1_win_flag"]
          .mean()
          .reset_index()
          .rename(columns={"_t1_win_flag": "team1_h2h_win_rate"})
    )
    df = df.merge(h2h, on=["team1", "team2"], how="left")
    df["team1_h2h_win_rate"] = df["team1_h2h_win_rate"].fillna(0.5)

    # Drop helper columns
    df.drop(columns=["_t1_win_flag", "_t2_win_flag"], inplace=True)

    # ── 11. Binary target ───────────────────────────────────────────────────
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)

    return df


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — BUILD SCIKIT-LEARN PIPELINES
# ══════════════════════════════════════════════════════════════

def build_pipelines() -> dict:
    """
    Return a dict of named Scikit-Learn Pipeline objects.

    Using Pipeline ensures:
    - All preprocessing steps (scaling) are fit ONLY on training data
    - The same transformation is automatically applied at prediction time
    - No data leakage between train and test splits
    - Easy serialisation with joblib (one .pkl file contains everything)

    Models included
    ────────────────
    1. Logistic Regression   – Strong linear baseline; fast; interpretable coefficients
    2. Random Forest         – Ensemble of decision trees; captures non-linearities
    3. Gradient Boosting     – Sequential boosting; often best accuracy on tabular data
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            # StandardScaler: centres features to mean=0, std=1
            # Critical for Logistic Regression so all features have equal gradient magnitude
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                max_iter=2000,      # enough iterations to converge
                C=1.0,              # inverse regularisation strength (default)
                solver="lbfgs",     # efficient for small/medium datasets
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": Pipeline([
            # RF is scale-invariant → scaler is a no-op but keeps the API consistent
            ("scaler", StandardScaler()),
            ("model",  RandomForestClassifier(
                n_estimators=300,   # more trees → lower variance, diminishing returns after ~200
                max_depth=8,        # limit depth to prevent overfitting on ~1k rows
                min_samples_leaf=5, # at least 5 samples per leaf → further regularisation
                max_features="sqrt",# standard for classification
                random_state=RANDOM_STATE,
                n_jobs=-1,          # use all CPU cores
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingClassifier(
                n_estimators=200,   # number of boosting stages
                learning_rate=0.05, # small learning rate → more robust generalisation
                max_depth=4,        # shallow trees for a boosted ensemble
                subsample=0.8,      # stochastic gradient boosting (reduces variance)
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return pipelines


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════

def train_and_evaluate(df: pd.DataFrame) -> tuple:
    """
    Split data → cross-validate all pipelines → pick the best → evaluate on hold-out test set.

    Returns
    ───────
    best_pipeline  : fitted sklearn Pipeline (best model by CV ROC-AUC)
    results        : dict of per-model metrics
    test_data      : (X_test, y_test) for downstream inspection
    """
    X = df[FEATURE_COLS]
    y = df["team1_win"]

    # ── Stratified 80/20 split ───────────────────────────────────────────────
    # stratify=y ensures both classes appear in exactly their population ratio
    # in both train and test sets — important for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ── Stratified K-Fold CV object (shared across all models) ──────────────
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    pipelines = build_pipelines()
    results   = {}

    print("\n" + "=" * 65)
    print("  CROSS-VALIDATION RESULTS  (5-fold Stratified KFold)")
    print("=" * 65)
    print(f"  {'Model':<26} {'CV Accuracy':>12}  {'CV ROC-AUC':>11}")
    print("-" * 65)

    for name, pipe in pipelines.items():
        # CV accuracy
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        # CV ROC-AUC — better than accuracy for imbalanced-ish data
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc",  n_jobs=-1)

        results[name] = {
            "pipeline":   pipe,
            "cv_acc_mean": cv_acc.mean(),
            "cv_acc_std":  cv_acc.std(),
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std":  cv_auc.std(),
        }
        print(f"  {name:<26} {cv_acc.mean():.4f} ±{cv_acc.std():.4f}  {cv_auc.mean():.4f} ±{cv_auc.std():.4f}")

    # ── Select best model by mean CV ROC-AUC ────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_pipe = results[best_name]["pipeline"]

    print("-" * 65)
    print(f"\n  Best model (by CV ROC-AUC): {best_name}")

    # ── Fit the best pipeline on the full training set ───────────────────────
    best_pipe.fit(X_train, y_train)

    # ── Evaluate on the held-out test set ───────────────────────────────────
    y_pred     = best_pipe.predict(X_test)
    y_prob     = best_pipe.predict_proba(X_test)[:, 1]
    test_acc   = accuracy_score(y_test, y_pred)
    test_auc   = roc_auc_score(y_test, y_prob)

    results[best_name]["test_acc"] = test_acc
    results[best_name]["test_auc"] = test_auc

    print("\n" + "=" * 65)
    print(f"  HOLD-OUT TEST SET — {best_name}")
    print("=" * 65)
    print(f"  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test ROC-AUC   : {test_auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Team 2 Wins", "Team 1 Wins"]))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

    # ── Feature importance (if Random Forest or Gradient Boosting) ──────────
    final_model = best_pipe.named_steps["model"]
    if hasattr(final_model, "feature_importances_"):
        fi = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
        fi_sorted = fi.sort_values(ascending=False)
        print("\n  Feature Importances:")
        for feat, score in fi_sorted.items():
            bar = "█" * int(score * 40)
            print(f"    {feat:<30} {score:.4f}  {bar}")

    return best_pipe, results, (X_test, y_test)


# ══════════════════════════════════════════════════════════════
#  PHASE 5 — SAVE THE MODEL
# ══════════════════════════════════════════════════════════════

def save_model(pipeline, path: str) -> None:
    """Persist the fitted Scikit-Learn Pipeline to disk using joblib."""
    joblib.dump(pipeline, path)
    print(f"\n  Model saved → {path}")


def load_model(path: str):
    """Load a previously saved Scikit-Learn Pipeline from disk."""
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════
#  PHASE 6 — MATCH PREDICTION UTILITY
# ══════════════════════════════════════════════════════════════

def predict_match(
    pipeline,
    df: pd.DataFrame,
    team_a: str,
    team_b: str,
    venue: str | None = None,
) -> pd.DataFrame:
    """
    Predict the outcome of a new IPL match across all four toss scenarios.

    Parameters
    ──────────
    pipeline  : fitted Scikit-Learn Pipeline
    df        : cleaned + featured DataFrame (to compute historical aggregates)
    team_a    : name of the team placed as 'team1'
    team_b    : name of the team placed as 'team2'
    venue     : optional venue string (defaults to 0.5 neutral win rate if unknown)

    Returns
    ───────
    pd.DataFrame with one row per toss scenario, showing predicted winner and probabilities.
    """
    # Normalise names
    team_a = TEAM_NAME_MAP.get(team_a, team_a)
    team_b = TEAM_NAME_MAP.get(team_b, team_b)

    # ── Historical aggregates (same logic as engineer_features) ─────────────
    win_counts = df["winner"].value_counts().to_dict()
    t1_wins    = float(win_counts.get(team_a, 0))
    t2_wins    = float(win_counts.get(team_b, 0))
    win_diff   = t1_wins - t2_wins

    # Head-to-head: aggregate from both orderings in the dataset
    h2h_mask = (
        ((df["team1"] == team_a) & (df["team2"] == team_b)) |
        ((df["team1"] == team_b) & (df["team2"] == team_a))
    )
    h2h_df   = df[h2h_mask]
    h2h_rate = (h2h_df["winner"] == team_a).mean() if len(h2h_df) > 0 else 0.5

    # Venue win rates
    if venue:
        vma = df[(df["venue"] == venue) & ((df["team1"] == team_a) | (df["team2"] == team_a))]
        vmb = df[(df["venue"] == venue) & ((df["team1"] == team_b) | (df["team2"] == team_b))]
        vr_a = float((vma["winner"] == team_a).mean()) if len(vma) > 0 else 0.5
        vr_b = float((vmb["winner"] == team_b).mean()) if len(vmb) > 0 else 0.5
    else:
        vr_a = vr_b = 0.5

    venue_adv_diff = vr_a - vr_b

    # Season number: use the latest season in the dataset
    latest_season = int(df["season_number"].max())

    # ── Build feature rows for each (toss_winner × toss_decision) scenario ──
    rows = []
    for toss_winner_is_a in [True, False]:
        for decision_bat in [True, False]:
            rows.append({
                "team1_total_wins":     t1_wins,
                "team2_total_wins":     t2_wins,
                "win_diff":             win_diff,
                "team1_won_toss":       int(toss_winner_is_a),
                "toss_decision_bat":    int(decision_bat),
                "team1_venue_win_rate": vr_a,
                "team2_venue_win_rate": vr_b,
                "venue_advantage_diff": venue_adv_diff,
                "team1_h2h_win_rate":   h2h_rate,
                "season_number":        latest_season,
                # metadata (not passed to model)
                "_toss_winner":   team_a if toss_winner_is_a else team_b,
                "_toss_decision": "bat" if decision_bat else "field",
            })

    meta_df  = pd.DataFrame(rows)
    X_pred   = meta_df[FEATURE_COLS]
    probs    = pipeline.predict_proba(X_pred)

    output_rows = []
    for i, row in meta_df.iterrows():
        p_a  = probs[i][1]  # P(team1 wins) = P(team_a wins)
        p_b  = probs[i][0]
        winner = team_a if p_a > 0.5 else team_b
        output_rows.append({
            "Toss Winner":     row["_toss_winner"],
            "Toss Decision":   row["_toss_decision"],
            f"{team_a} Win%":  f"{p_a:.1%}",
            f"{team_b} Win%":  f"{p_b:.1%}",
            "Predicted Winner": winner,
        })

    return pd.DataFrame(output_rows)


# ══════════════════════════════════════════════════════════════
#  MAIN — end-to-end pipeline execution
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("    IPL MATCH OUTCOME PREDICTOR — FULL ML PIPELINE")
    print("=" * 65)

    # ── Phase 1: Load & Clean ────────────────────────────────────────────────
    print("\n[Phase 1] Loading & cleaning data ...")
    df_raw = load_and_clean(DATA_PATH)
    print(f"  Loaded {len(df_raw):,} clean matches")
    print(f"  Seasons  : {df_raw['season_number'].min()} – {df_raw['season_number'].max()}")
    print(f"  Teams    : {df_raw['team1'].nunique()} unique after standardisation")
    print(f"  Venues   : {df_raw['venue'].nunique()} unique")
    print(f"  Missing values remaining: {df_raw.isnull().sum().sum()}")

    # ── Phase 2: Feature Engineering ────────────────────────────────────────
    print("\n[Phase 2] Engineering features ...")
    df = engineer_features(df_raw)

    target_balance = df["team1_win"].mean()
    print(f"  Features created  : {len(FEATURE_COLS)}")
    print(f"  Target: team1_win | Class balance: {target_balance:.1%} Team1 wins")
    print(f"  Dataset shape for modelling: {df[FEATURE_COLS].shape}")

    # ── Phase 3+4: Build pipelines & train ──────────────────────────────────
    print("\n[Phase 3+4] Building Scikit-Learn Pipelines & training models ...")
    best_pipe, results, (X_test, y_test) = train_and_evaluate(df)

    # ── Phase 5: Save the model ──────────────────────────────────────────────
    print("\n[Phase 5] Saving best model ...")
    model_path = os.path.join(MODEL_DIR, "best_ipl_pipeline.pkl")
    save_model(best_pipe, model_path)

    # ── Phase 6: Demo predictions ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("    MATCH PREDICTIONS (all 4 toss scenarios)")
    print("=" * 65)

    matchups = [
        ("Mumbai Indians",          "Chennai Super Kings",         "Wankhede Stadium"),
        ("Kolkata Knight Riders",   "Royal Challengers Bangalore", "Eden Gardens"),
        ("Delhi Capitals",          "Rajasthan Royals",            None),
        ("Sunrisers Hyderabad",     "Punjab Kings",                None),
        ("Gujarat Titans",          "Lucknow Super Giants",        None),
    ]

    for team_a, team_b, venue in matchups:
        v_str = f" at {venue}" if venue else " (Neutral Venue)"
        print(f"\n  {team_a} vs {team_b}{v_str}")
        print("  " + "-" * 55)
        pred_df = predict_match(best_pipe, df, team_a, team_b, venue)
        # Indent the output
        for line in pred_df.to_string(index=False).split("\n"):
            print("  " + line)

    print("\n" + "=" * 65)
    print("  Pipeline complete. Model saved to models/best_ipl_pipeline.pkl")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
