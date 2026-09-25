"""
player_features.py — Player-Level Feature Engineering
=======================================================

Extracts per-match player statistics from deliveries.csv to create squad-level features:
- Squad Power Index: sum of career batting/bowling averages
- Recent Form: rolling window win rate (last 5 matches)
- Bowling Strength: average economy rate of top bowlers
- Batting Strength: average strike rate of top batters
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from functools import lru_cache


def load_deliveries(path: str) -> pd.DataFrame:
    """Load and parse ball-by-ball data."""
    return pd.read_csv(path)


def extract_player_career_stats(deliveries: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute career-level player statistics from deliveries data.

    Returns a dict: {player_name: {batting_avg, bowling_economy, strike_rate, ...}}
    """
    player_stats = {}

    # ── Batting stats (per batter) ──
    batting = deliveries[deliveries["batter"].notna()].copy()
    batting["runs"] = batting["batsman_runs"] + batting["extra_runs"]

    batter_runs = batting.groupby("batter").agg({
        "runs": "sum",
        "batsman_runs": "sum",
        "match_id": "count"  # number of balls faced (roughly)
    }).rename(columns={"match_id": "balls_faced"})

    # Strike rate = (runs / balls_faced) * 100
    batter_runs["strike_rate"] = (batter_runs["runs"] / batter_runs["balls_faced"] * 100).clip(0, 200)
    batter_runs["batting_avg"] = batter_runs["runs"] / batter_runs["balls_faced"]

    # ── Bowling stats (per bowler) ──
    bowling = deliveries[deliveries["bowler"].notna()].copy()
    bowling["runs_conceded"] = bowling["batsman_runs"] + bowling["extra_runs"]

    bowler_stats = bowling.groupby("bowler").agg({
        "runs_conceded": "sum",
        "match_id": "count"  # number of balls bowled (roughly)
    }).rename(columns={"match_id": "balls_bowled"})

    # Economy rate = (runs_conceded / balls_bowled) * 6
    bowler_stats["economy"] = (bowler_stats["runs_conceded"] / bowler_stats["balls_bowled"] * 6).clip(0, 20)
    bowler_stats["bowling_avg"] = bowler_stats["runs_conceded"] / bowler_stats["balls_bowled"]

    # Combine into one dict
    for player in batter_runs.index:
        player_stats[player] = {
            "strike_rate": batter_runs.loc[player, "strike_rate"],
            "batting_avg": batter_runs.loc[player, "batting_avg"],
        }

    for player in bowler_stats.index:
        if player not in player_stats:
            player_stats[player] = {}
        player_stats[player].update({
            "economy": bowler_stats.loc[player, "economy"],
            "bowling_avg": bowler_stats.loc[player, "bowling_avg"],
        })

    return player_stats


def extract_match_playing_xi(deliveries: pd.DataFrame, matches: pd.DataFrame) -> Dict[int, Dict[str, list]]:
    """
    For each match, identify which players actually played (from deliveries data).

    Returns: {match_id: {team_name: [list of batters and bowlers who played]}}
    """
    match_players = {}

    for match_id in deliveries["match_id"].unique():
        match_deliv = deliveries[deliveries["match_id"] == match_id]

        # Inning 1 and 2
        for inning in [1, 2]:
            inning_deliv = match_deliv[match_deliv["inning"] == inning]
            if len(inning_deliv) == 0:
                continue

            team = inning_deliv["batting_team"].iloc[0]
            batters = set(inning_deliv["batter"].dropna().unique())
            bowlers = set(inning_deliv["bowler"].dropna().unique())
            players = list(batters | bowlers)

            if match_id not in match_players:
                match_players[match_id] = {}

            match_players[match_id][team] = players

    return match_players


def compute_squad_power_index(
    matches: pd.DataFrame,
    player_stats: Dict[str, Dict],
    match_players: Dict[int, Dict[str, list]],
) -> pd.DataFrame:
    """
    For each match, compute squad power index = sum of player career batting/bowling averages.

    Squadmates with missing stats default to 0.

    Returns matches DataFrame with squad_power_t1 and squad_power_t2 columns.
    """
    matches = matches.copy()
    matches["squad_power_t1"] = 0.0
    matches["squad_power_t2"] = 0.0

    for idx, row in matches.iterrows():
        match_id = row.get("match_id") or idx  # fallback to index if no match_id
        team1 = row["team1"]
        team2 = row["team2"]

        # Get players who played in this match (if available)
        if match_id in match_players:
            t1_players = match_players[match_id].get(team1, [])
            t2_players = match_players[match_id].get(team2, [])
        else:
            # Fallback: assume all-time best players (not ideal, but preserves computation)
            t1_players = []
            t2_players = []

        # Sum up player stats
        t1_power = sum(
            player_stats.get(p, {}).get("batting_avg", 0) +
            player_stats.get(p, {}).get("bowling_avg", 0)
            for p in t1_players
        )
        t2_power = sum(
            player_stats.get(p, {}).get("batting_avg", 0) +
            player_stats.get(p, {}).get("bowling_avg", 0)
            for p in t2_players
        )

        matches.loc[idx, "squad_power_t1"] = max(t1_power, 0.1)  # Avoid zero
        matches.loc[idx, "squad_power_t2"] = max(t2_power, 0.1)

    return matches


def compute_recent_form(
    matches: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute rolling win rate for each team over last N matches.

    Parameters
    ----------
    matches : pd.DataFrame
        Sorted by date
    window : int
        Number of recent matches to consider

    Returns
    -------
    pd.DataFrame with recent_form_t1, recent_form_t2 columns
    """
    matches = matches.sort_values("date").reset_index(drop=True).copy()

    matches["recent_form_t1"] = 0.5  # Default neutral
    matches["recent_form_t2"] = 0.5

    for idx in range(window, len(matches)):
        current_match = matches.iloc[idx]
        team1 = current_match["team1"]
        team2 = current_match["team2"]

        # Look back: find last N matches for each team
        prev_window = matches.iloc[max(0, idx - window) : idx]

        # Team1 recent form: win rate in last window matches
        t1_matches = prev_window[
            (prev_window["team1"] == team1) | (prev_window["team2"] == team1)
        ]
        if len(t1_matches) > 0:
            t1_wins = (t1_matches["winner"] == team1).sum()
            matches.loc[idx, "recent_form_t1"] = t1_wins / len(t1_matches)

        # Team2 recent form
        t2_matches = prev_window[
            (prev_window["team1"] == team2) | (prev_window["team2"] == team2)
        ]
        if len(t2_matches) > 0:
            t2_wins = (t2_matches["winner"] == team2).sum()
            matches.loc[idx, "recent_form_t2"] = t2_wins / len(t2_matches)

    return matches


def compute_bowling_strength(
    matches: pd.DataFrame,
    player_stats: Dict[str, Dict],
    match_players: Dict[int, Dict[str, list]],
) -> pd.DataFrame:
    """
    For each match, compute bowling strength = avg economy rate of top bowlers.

    Uses top 3 bowlers by appearance in the match (estimated from deliveries).
    """
    matches = matches.copy()
    matches["bowling_strength_t1"] = 20.0  # Max economy (neutral default)
    matches["bowling_strength_t2"] = 20.0

    for idx, row in matches.iterrows():
        match_id = row.get("match_id") or idx
        team1 = row["team1"]
        team2 = row["team2"]

        if match_id in match_players:
            # Get bowlers for each team
            t1_players = match_players[match_id].get(team1, [])
            t2_players = match_players[match_id].get(team2, [])

            # Extract bowlers (those with economy stat)
            t1_bowlers = [
                p for p in t1_players
                if "economy" in player_stats.get(p, {})
            ]
            t2_bowlers = [
                p for p in t2_players
                if "economy" in player_stats.get(p, {})
            ]

            # Average economy of top 3 bowlers (lower is better)
            if t1_bowlers:
                economies_t1 = sorted(
                    [player_stats[p]["economy"] for p in t1_bowlers[:3]]
                )
                matches.loc[idx, "bowling_strength_t1"] = np.mean(economies_t1)

            if t2_bowlers:
                economies_t2 = sorted(
                    [player_stats[p]["economy"] for p in t2_bowlers[:3]]
                )
                matches.loc[idx, "bowling_strength_t2"] = np.mean(economies_t2)

    return matches


def compute_batting_strength(
    matches: pd.DataFrame,
    player_stats: Dict[str, Dict],
    match_players: Dict[int, Dict[str, list]],
) -> pd.DataFrame:
    """
    For each match, compute batting strength = avg strike rate of top batters.

    Uses top 3 batters by appearance.
    """
    matches = matches.copy()
    matches["batting_strength_t1"] = 100.0  # Neutral SR default
    matches["batting_strength_t2"] = 100.0

    for idx, row in matches.iterrows():
        match_id = row.get("match_id") or idx
        team1 = row["team1"]
        team2 = row["team2"]

        if match_id in match_players:
            t1_players = match_players[match_id].get(team1, [])
            t2_players = match_players[match_id].get(team2, [])

            # Extract batters (those with strike_rate stat)
            t1_batters = [
                p for p in t1_players
                if "strike_rate" in player_stats.get(p, {})
            ]
            t2_batters = [
                p for p in t2_players
                if "strike_rate" in player_stats.get(p, {})
            ]

            # Average SR of top 3 batters (higher is better)
            if t1_batters:
                srs_t1 = sorted(
                    [player_stats[p]["strike_rate"] for p in t1_batters[:3]],
                    reverse=True
                )
                matches.loc[idx, "batting_strength_t1"] = np.mean(srs_t1)

            if t2_batters:
                srs_t2 = sorted(
                    [player_stats[p]["strike_rate"] for p in t2_batters[:3]],
                    reverse=True
                )
                matches.loc[idx, "batting_strength_t2"] = np.mean(srs_t2)

    return matches


def engineer_all_player_features(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Main entry point: compute all player-based features in one go.

    Parameters
    ----------
    matches : pd.DataFrame
        Cleaned matches data
    deliveries : pd.DataFrame
        Ball-by-ball data

    Returns
    -------
    matches with new columns: squad_power_t{1,2}, recent_form_t{1,2}, batting/bowling_strength_t{1,2}
    """
    print("\n[Player Features]")

    # 1. Extract player career stats
    print("  Computing career player statistics...")
    player_stats = extract_player_career_stats(deliveries)
    print(f"    {len(player_stats)} players with stats")

    # 2. Extract match-level playing XI
    print("  Extracting playing XI for each match...")
    match_players = extract_match_playing_xi(deliveries, matches)
    print(f"    {len(match_players)} matches with XI data")

    # 3. Compute squad power
    print("  Computing squad power indices...")
    matches = compute_squad_power_index(matches, player_stats, match_players)

    # 4. Compute recent form
    print("  Computing recent form (5-match window)...")
    matches = compute_recent_form(matches, window=5)

    # 5. Compute bowling strength
    print("  Computing bowling strength...")
    matches = compute_bowling_strength(matches, player_stats, match_players)

    # 6. Compute batting strength
    print("  Computing batting strength...")
    matches = compute_batting_strength(matches, player_stats, match_players)

    print(f"  ✓ All player features computed")

    return matches
