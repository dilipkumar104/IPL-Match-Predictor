"""
EDA Visualizations for IPL Match Outcome Predictor
====================================================
Generates publication-quality charts for the analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import our cleaning function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ipl_predictor import load_and_clean

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'matches.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

sns.set_theme(style='whitegrid', palette='husl', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {name}")


def main():
    df = load_and_clean(DATA_PATH)
    print(f"Loaded {len(df)} matches\n")

    # 1 -- Matches per season
    fig, ax = plt.subplots(figsize=(12, 5))
    season_counts = df.groupby('season').size()
    bars = ax.bar(season_counts.index.astype(str), season_counts.values,
                  color=sns.color_palette('viridis', len(season_counts)))
    ax.set_title('IPL Matches Per Season', fontsize=16, fontweight='bold')
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of Matches')
    ax.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars, season_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v),
                ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    save(fig, 'matches_per_season.png')

    # 2 -- Most successful teams (wins)
    fig, ax = plt.subplots(figsize=(12, 6))
    wins = df['winner'].value_counts()
    colors = sns.color_palette('Set2', len(wins))
    bars = ax.barh(wins.index[::-1], wins.values[::-1], color=colors)
    ax.set_title('Total Wins by Team', fontsize=16, fontweight='bold')
    ax.set_xlabel('Wins')
    for bar, v in zip(bars, wins.values[::-1]):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2, str(v),
                ha='left', va='center', fontsize=10)
    fig.tight_layout()
    save(fig, 'total_wins_by_team.png')

    # 3 -- Toss decision distribution
    fig, ax = plt.subplots(figsize=(7, 7))
    toss = df['toss_decision'].value_counts()
    ax.pie(toss.values, labels=toss.index.str.title(), autopct='%1.1f%%',
           colors=['#4FC3F7', '#FFB74D'], startangle=90,
           textprops={'fontsize': 14}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title('Toss Decision Distribution', fontsize=16, fontweight='bold')
    fig.tight_layout()
    save(fig, 'toss_decision_distribution.png')

    # 4 -- Does winning the toss help?
    df['toss_winner_is_match_winner'] = df['toss_winner'] == df['winner']
    toss_effect = df['toss_winner_is_match_winner'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(toss_effect.values,
           labels=['Toss Winner Lost', 'Toss Winner Won'],
           autopct='%1.1f%%', colors=['#EF5350', '#66BB6A'],
           startangle=90, textprops={'fontsize': 14},
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title('Does Winning the Toss Help?', fontsize=16, fontweight='bold')
    fig.tight_layout()
    save(fig, 'toss_effect.png')

    # 5 -- Top venues by matches
    fig, ax = plt.subplots(figsize=(12, 6))
    venues = df['venue'].value_counts().head(10)
    bars = ax.barh(venues.index[::-1], venues.values[::-1],
                   color=sns.color_palette('coolwarm', 10))
    ax.set_title('Top 10 Venues by Matches Played', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Matches')
    for bar, v in zip(bars, venues.values[::-1]):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2, str(v),
                ha='left', va='center', fontsize=10)
    fig.tight_layout()
    save(fig, 'top_venues.png')

    # 6 -- Win margin distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    runs_df = df[df['result'] == 'runs']
    wkts_df = df[df['result'] == 'wickets']
    axes[0].hist(runs_df['result_margin'], bins=20, color='#42A5F5', edgecolor='white')
    axes[0].set_title('Win by Runs Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Runs')
    axes[0].set_ylabel('Frequency')
    axes[1].hist(wkts_df['result_margin'], bins=10, color='#AB47BC', edgecolor='white')
    axes[1].set_title('Win by Wickets Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Wickets')
    axes[1].set_ylabel('Frequency')
    fig.suptitle('Win Margin Distributions', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'win_margins.png')

    print("\nAll visualizations saved to notebooks/")


if __name__ == '__main__':
    main()
