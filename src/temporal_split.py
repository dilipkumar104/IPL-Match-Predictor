"""
temporal_split.py — Time-Aware Train/Test Split for IPL Data
==============================================================

Prevents data leakage by ensuring aggregate statistics (team wins, venue rates, H2H)
are computed only from historical data available at prediction time.

Key Functions:
- temporal_train_test_split() — Time-ordered split (train on 2008–2022, test on 2023–2024)
- compute_rolling_aggregates() — Compute features without look-ahead bias
- temporal_cross_validation() — Time-stratified K-fold CV
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime


TEAM_NAME_MAP: Dict[str, str] = {
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
}


def temporal_train_test_split(
    df: pd.DataFrame,
    train_cutoff_year: int = 2022,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by date instead of random rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned matches DataFrame with 'season_number' column
    train_cutoff_year : int
        Last year to include in training set (2022 = 2008–2022 train, 2023–2024 test)

    Returns
    -------
    (train_df, test_df) : Tuple[DataFrame, DataFrame]
        Train on seasons ≤ train_cutoff_year, test on later seasons
    """
    train_df = df[df["season_number"] <= train_cutoff_year].copy()
    test_df = df[df["season_number"] > train_cutoff_year].copy()

    print(f"[Temporal Split]")
    print(f"  Train: {train_df['season_number'].min():.0f}–{train_df['season_number'].max():.0f} ({len(train_df)} matches)")
    print(f"  Test:  {test_df['season_number'].min():.0f}–{test_df['season_number'].max():.0f} ({len(test_df)} matches)")

    return train_df, test_df


def compute_rolling_aggregates(
    df: pd.DataFrame,
    cutoff_year: int | None = None,
) -> pd.DataFrame:
    """
    Compute team aggregates (wins, venue rates, H2H) from historical data only.

    If cutoff_year is provided, compute stats only from matches ≤ cutoff_year.
    This prevents look-ahead bias: we simulate making predictions with only
    data available up to that point in time.

    Parameters
    ----------
    df : pd.DataFrame
        Full or subset of matches (cleaned, with 'season_number')
    cutoff_year : int, optional
        If provided, compute aggregates only from df[df['season_number'] ≤ cutoff_year]

    Returns
    -------
    pd.DataFrame
        Copy of df with aggregate features appended (no look-ahead bias)
    """
    df = df.copy()

    # Determine which data to use for computing aggregates
    if cutoff_year is not None:
        hist_df = df[df["season_number"] <= cutoff_year]
    else:
        hist_df = df

    # ── Team strength (total wins) ──
    win_counts: dict = hist_df["winner"].value_counts().to_dict()
    df["team1_total_wins"] = df["team1"].map(win_counts).fillna(0).astype(float)
    df["team2_total_wins"] = df["team2"].map(win_counts).fillna(0).astype(float)
    df["win_diff"] = df["team1_total_wins"] - df["team2_total_wins"]

    # ── Toss features ──
    df["team1_won_toss"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["toss_decision_bat"] = (df["toss_decision"] == "bat").astype(int)

    # ── Venue win rates (historical only) ──
    df["_t1_win_flag"] = (hist_df["winner"] == df["team1"]).astype(float)
    df["_t2_win_flag"] = (hist_df["winner"] == df["team2"]).astype(float)

    # Team1 venue win rate
    vwr_t1 = (
        hist_df.assign(_t1_win_flag=(hist_df["winner"] == hist_df["team1"]).astype(float))
        .groupby(["venue", "team1"])["_t1_win_flag"]
        .mean()
        .reset_index()
        .rename(columns={"_t1_win_flag": "team1_venue_win_rate"})
    )
    df = df.merge(vwr_t1, on=["venue", "team1"], how="left")
    df["team1_venue_win_rate"] = df["team1_venue_win_rate"].fillna(0.5)

    # Team2 venue win rate
    vwr_t2 = (
        hist_df.assign(_t2_win_flag=(hist_df["winner"] == hist_df["team2"]).astype(float))
        .groupby(["venue", "team2"])["_t2_win_flag"]
        .mean()
        .reset_index()
        .rename(columns={"_t2_win_flag": "team2_venue_win_rate"})
    )
    df = df.merge(vwr_t2, on=["venue", "team2"], how="left")
    df["team2_venue_win_rate"] = df["team2_venue_win_rate"].fillna(0.5)

    df["venue_advantage_diff"] = df["team1_venue_win_rate"] - df["team2_venue_win_rate"]

    # ── Head-to-head win rate ──
    h2h = (
        hist_df.assign(_t1_win_flag=(hist_df["winner"] == hist_df["team1"]).astype(float))
        .groupby(["team1", "team2"])["_t1_win_flag"]
        .mean()
        .reset_index()
        .rename(columns={"_t1_win_flag": "team1_h2h_win_rate"})
    )
    df = df.merge(h2h, on=["team1", "team2"], how="left")
    df["team1_h2h_win_rate"] = df["team1_h2h_win_rate"].fillna(0.5)

    # Clean up helper columns
    df.drop(columns=["_t1_win_flag", "_t2_win_flag"], inplace=True, errors="ignore")

    return df


def temporal_cross_validation(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time-stratified folds for cross-validation.

    Ensures that each fold's test set is strictly later (in time) than its training set.
    Useful for detecting temporal patterns and ensuring no look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'season_number' column
    n_splits : int
        Number of folds

    Yields
    ------
    (train_fold, test_fold) : Tuple[DataFrame, DataFrame]
        Time-stratified train/test pair for each fold
    """
    df = df.sort_values("season_number").reset_index(drop=True)

    # Split into n_splits+1 temporal chunks
    total_years = df["season_number"].max() - df["season_number"].min() + 1
    test_years_per_fold = max(1, total_years // (n_splits + 1))

    folds = []
    min_year = df["season_number"].min()

    for fold_idx in range(n_splits):
        test_year_start = min_year + fold_idx * test_years_per_fold
        test_year_end = test_year_start + test_years_per_fold - 1

        train_mask = df["season_number"] < test_year_start
        test_mask = (df["season_number"] >= test_year_start) & (df["season_number"] <= test_year_end)

        train_fold = df[train_mask].reset_index(drop=True)
        test_fold = df[test_mask].reset_index(drop=True)

        if len(test_fold) > 0:  # Only yield if test fold is non-empty
            folds.append((train_fold, test_fold))

    return folds
