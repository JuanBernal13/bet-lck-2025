import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from src.config import PLOT_STYLE

plt.rcParams.update(PLOT_STYLE)
sns.set_palette("viridis")

BLUE  = '#3B82F6'
RED   = '#EF4444'
GREEN = '#10B981'
AMBER = '#F59E0B'
INDGO = '#6366F1'
GRAY  = 'gray'


class Visualizer:

    @staticmethod
    def plot_eda(team_rows: pd.DataFrame):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Exploratory Data Analysis — LCK 2025', fontsize=18, fontweight='bold', y=1.05)

        team_wr = team_rows.groupby('teamname')['result'].mean().sort_values()
        colors  = [RED if v < 0.5 else GREEN for v in team_wr.values]
        axes[0].barh(team_wr.index, team_wr.values, color=colors, alpha=0.9, edgecolor='white')
        axes[0].axvline(0.5, ls='--', color=GRAY, alpha=0.5)
        axes[0].set_title('Win Rate by Team', fontsize=14)

        side_wr = team_rows.groupby('side')['result'].agg(['mean', 'count'])
        bars    = axes[1].bar(side_wr.index, side_wr['mean'], color=[BLUE, RED], alpha=0.9, width=0.5, edgecolor='white')
        axes[1].set_ylim(0.40, 0.60)
        axes[1].axhline(0.5, ls='--', color=GRAY, alpha=0.5)
        axes[1].set_title('Win Rate by Side', fontsize=14)
        for bar, (_, row) in zip(bars, side_wr.iterrows()):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{row['mean']:.1%}\n(n={int(row['count'])})",
                ha='center', fontsize=11, fontweight='bold',
            )

        split_order = ['Cup', 'Rounds 1-2', 'Rounds 3-5']
        split_side  = team_rows.groupby(['split', 'side'])['result'].mean().unstack().reindex(split_order)
        split_side.plot(kind='bar', ax=axes[2], color=[BLUE, RED], alpha=0.9, rot=0, edgecolor='white')
        axes[2].axhline(0.5, ls='--', color=GRAY, alpha=0.5)
        axes[2].set_title('Meta Inertia by Split', fontsize=14)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlations(corr: pd.Series):
        plt.figure(figsize=(12, 7))
        colors = [GREEN if v > 0 else RED for v in corr.values]
        plt.barh(corr.index, corr.values, color=colors, alpha=0.9, edgecolor='white')
        plt.axvline(0, color='black', linewidth=1)
        plt.xlabel('Pearson Correlation')
        plt.title('In-Game Metrics Correlation with Victory (LCK 2025)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_calibration(y_test, raw_proba, cal_proba, model_name: str):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Probability Calibration — {model_name}', fontsize=16, fontweight='bold')

        for proba, label, style, color in [
            (raw_proba, 'Base Model',              's--', GRAY),
            (cal_proba, 'Calibrated (Platt Scaling)', 'o-', AMBER),
        ]:
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8)
            axes[0].plot(mean_pred, frac_pos, style, label=label, color=color, markersize=8, linewidth=2)

        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Calibration')
        axes[0].set_xlabel('Predicted Probability (Blue Team)', fontsize=12)
        axes[0].set_ylabel('Fraction of Actual Wins', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.2)

        axes[1].hist(raw_proba, bins=25, alpha=0.4, label='Base', color=GRAY,  density=True, edgecolor='white')
        axes[1].hist(cal_proba, bins=25, alpha=0.6, label='Calibrated', color=BLUE, density=True, edgecolor='white', hatch='//')
        axes[1].set_xlabel('Predicted Probability', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].legend(fontsize=11)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bankroll(history: list, ev_dist: pd.Series, ev_min: float):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle('Risk Management Analysis — Quarter-Kelly', fontsize=16, fontweight='bold')

        fill_color = GREEN if history[-1] >= history[0] else RED
        axes[0].plot(history, linewidth=3, color=BLUE, label='Algorithmic Bankroll')
        axes[0].axhline(history[0], ls='--', color=GRAY, alpha=0.5, label='Initial Bankroll')
        axes[0].fill_between(range(len(history)), history[0], history, alpha=0.1, color=fill_color)
        axes[0].set_xlabel('Bets Executed', fontsize=12)
        axes[0].set_ylabel('Capital ($)', fontsize=12)
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.2)

        sns.histplot(ev_dist, bins=35, kde=True, ax=axes[1], color=INDGO, alpha=0.6, edgecolor='white')
        axes[1].axvline(ev_min, color=AMBER, ls='--', linewidth=2.5, label=f'EV Threshold = {ev_min:.0%}')
        axes[1].axvline(0,      color='black', ls='-',  linewidth=1,   alpha=0.4)
        axes[1].set_xlabel('Expected Value (EV)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].legend(fontsize=11)

        plt.tight_layout()
        plt.show()
