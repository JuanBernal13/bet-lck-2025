RANDOM_SEED = 42
DATA_PATH = '2025_LoL_esports_match_data_from_OraclesElixir.csv'

ROLLING_COLS = [
    'result', 'golddiffat15', 'golddiffat25', 'dragons', 'barons',
    'towers', 'earned gpm', 'gspd', 'visionscore', 'ckpm', 'void_grubs'
]
WINDOWS = [3, 5, 10]

META_FEATS = ['patch_num', 'is_playoffs', 'split_num']

EV_THRESHOLD     = 0.03
KELLY_MULTIPLIER = 0.25
MAX_BET_FRACTION = 0.10
INITIAL_BANKROLL = 1000.0

PLOT_STYLE = {
    'font.family':        'sans-serif',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
}
