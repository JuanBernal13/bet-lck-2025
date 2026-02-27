import pandas as pd
import numpy as np
from src.config import ROLLING_COLS, WINDOWS, META_FEATS


class FeatureEngine:

    def apply_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.make_rolling_features(df, ROLLING_COLS, WINDOWS)
        df['win_streak']      = df.groupby('teamname')['result'].transform(self.compute_streak)
        df['days_since_last'] = df.groupby('teamname')['date'].transform(lambda x: x.diff().dt.days)
        df['cumulative_wr']   = df.groupby('teamname')['result'].transform(lambda x: x.shift(1).expanding().mean())
        df = self.add_meta_features(df)
        return df

    @staticmethod
    def make_rolling_features(df: pd.DataFrame, cols: list, windows: list) -> pd.DataFrame:
        for col in cols:
            for w in windows:
                df[f'{col}_roll{w}'] = df.groupby('teamname')[col].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
                )
        return df

    @staticmethod
    def compute_streak(series: pd.Series) -> pd.Series:
        streak, current = [], 0
        for val in series.shift(1):
            if pd.isna(val):
                streak.append(np.nan)
                current = 0
            elif val == 1:
                current = max(1, current + 1)
                streak.append(current)
            else:
                current = min(-1, current - 1)
                streak.append(current)
        return pd.Series(streak, index=series.index)

    @staticmethod
    def add_meta_features(df: pd.DataFrame) -> pd.DataFrame:
        patch_order = sorted(df['patch'].unique())
        df['patch_num']   = df['patch'].map({p: i for i, p in enumerate(patch_order)})
        df['is_blue']     = (df['side'] == 'Blue').astype(int)
        df['is_playoffs'] = (df['playoffs'] == 1).astype(int)
        split_map         = {'Cup': 0, 'Rounds 1-2': 1, 'Rounds 3-5': 2}
        df['split_num']   = df['split'].map(split_map)
        return df

    def build_match_df(self, df: pd.DataFrame, pre_game_feats: list) -> pd.DataFrame:
        blue = df[df['side'] == 'Blue'].copy()
        red  = df[df['side'] == 'Red'].copy()

        blue_rename = {f: f'blue_{f}' for f in pre_game_feats}
        red_rename  = {f: f'red_{f}'  for f in pre_game_feats}

        blue_sel = (
            blue[['gameid', 'date', 'split', 'teamname', 'result'] + pre_game_feats + META_FEATS]
            .rename(columns={'teamname': 'blue_team', 'result': 'blue_result', **blue_rename})
        )
        red_sel = (
            red[['gameid', 'teamname', 'result'] + pre_game_feats]
            .rename(columns={'teamname': 'red_team', 'result': 'red_result', **red_rename})
        )

        match_df = blue_sel.merge(red_sel, on='gameid').sort_values('date').reset_index(drop=True)
        match_df['target'] = match_df['blue_result']

        for f in pre_game_feats:
            bf, rf = f'blue_{f}', f'red_{f}'
            if bf in match_df.columns and rf in match_df.columns:
                match_df[f'diff_{f}'] = match_df[bf] - match_df[rf]

        return match_df
