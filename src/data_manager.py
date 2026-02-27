import pandas as pd
from src.config import DATA_PATH


class DataManager:

    def __init__(self, path: str = DATA_PATH):
        self.path     = path
        self.raw_data = None

    def load_data(self) -> pd.DataFrame:
        self.raw_data = pd.read_csv(self.path, low_memory=False)
        return self.raw_data

    def filter_lck(self, df: pd.DataFrame) -> pd.DataFrame:
        lck = df[df['league'] == 'LCK'].copy()
        lck['date'] = pd.to_datetime(lck['date'])

        team_rows = lck[lck['position'] == 'team'].copy()

        for col in ['golddiffat25', 'void_grubs', 'atakhans']:
            if col in team_rows.columns:
                team_rows[col] = team_rows[col].fillna(0)

        return team_rows.sort_values('date').reset_index(drop=True)
