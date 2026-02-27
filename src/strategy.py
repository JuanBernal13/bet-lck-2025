import pandas as pd
from src.config import INITIAL_BANKROLL, EV_THRESHOLD, KELLY_MULTIPLIER, MAX_BET_FRACTION


class BettingStrategy:

    @staticmethod
    def calculate_ev(model_prob: float, decimal_odds: float) -> float:
        return model_prob * (decimal_odds - 1) - (1 - model_prob)

    @staticmethod
    def kelly_fraction(
        model_prob: float,
        decimal_odds: float,
        multiplier: float = KELLY_MULTIPLIER,
        max_fraction: float = MAX_BET_FRACTION,
    ) -> float:
        b      = decimal_odds - 1.0
        f_star = (model_prob * (b + 1) - 1) / b
        f_star = max(0.0, f_star)
        f_star = min(f_star, max_fraction)
        return f_star * multiplier

    def simulate_bankroll(
        self,
        bets_df: pd.DataFrame,
        initial: float = INITIAL_BANKROLL,
        ev_min: float  = EV_THRESHOLD,
    ):
        bankroll = initial
        history  = [bankroll]
        wins, losses = 0, 0

        for _, row in bets_df[bets_df['ev'] > ev_min].iterrows():
            stake = bankroll * row['kelly']
            if stake < 1.0:
                history.append(bankroll)
                continue

            if row['actual'] == 1:
                bankroll += stake * (row['house_odds'] - 1)
                wins += 1
            else:
                bankroll -= stake
                losses  += 1

            bankroll = max(bankroll, 0)
            history.append(bankroll)

        return history, wins, losses, bets_df[bets_df['ev'] > ev_min]
