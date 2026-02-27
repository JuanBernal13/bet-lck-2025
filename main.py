import pandas as pd
import numpy as np
import warnings
import joblib
import json
from src.data_manager import DataManager
from src.features import FeatureEngine
from src.models import ModelTrainer
from src.strategy import BettingStrategy
from src.visualization import Visualizer
from src.config import RANDOM_SEED, INITIAL_BANKROLL, EV_THRESHOLD

warnings.filterwarnings('ignore')


def main():
    print("🚀 Iniciando Pipeline de Trading Algorítmico - LoL Esports (LCK 2025)")
    print("-" * 75)

    dm = DataManager()
    raw_data = dm.load_data()
    lck_teams = dm.filter_lck(raw_data)
    print(f"✔️ Datos cargados: {len(lck_teams)} filas de equipo LCK filtradas.")

    viz = Visualizer()
    viz.plot_eda(lck_teams)

    corr_features = ['kills', 'deaths', 'dragons', 'barons', 'towers', 'golddiffat15', 'golddiffat25']
    corr_series = (
        lck_teams[corr_features + ['result']]
        .corr()['result']
        .drop('result')
        .sort_values(key=abs, ascending=False)
    )
    viz.plot_correlations(corr_series)

    fe = FeatureEngine()
    lck_teams = fe.apply_pipeline(lck_teams)

    pregame_feats = (
        [c for c in lck_teams.columns if '_roll' in c]
        + ['win_streak', 'days_since_last', 'cumulative_wr']
    )
    match_df = fe.build_match_df(lck_teams, pregame_feats)

    feature_cols = [
        c for c in match_df.columns
        if c.startswith(('blue_', 'red_', 'diff_'))
        and c not in ['blue_team', 'red_team', 'blue_result', 'red_result']
    ]
    for col in feature_cols:
        match_df[col] = match_df[col].fillna(match_df[col].mean())

    print(f"✔️ Feature Engineering completo. Dimensión: {match_df.shape}")

    mask_cup = match_df['split'] == 'Cup'
    mask_r12  = match_df['split'] == 'Rounds 1-2'
    mask_r35  = match_df['split'] == 'Rounds 3-5'

    X_f2_train = pd.concat([match_df[mask_cup][feature_cols], match_df[mask_r12][feature_cols]])
    y_f2_train = pd.concat([match_df[mask_cup]['target'],     match_df[mask_r12]['target']])
    X_f2_test  = match_df[mask_r35][feature_cols]
    y_f2_test  = match_df[mask_r35]['target']

    trainer = ModelTrainer(seed=RANDOM_SEED)
    model_definitions = trainer.get_models()

    results = [
        trainer.train_and_evaluate(X_f2_train, y_f2_train, X_f2_test, y_f2_test, name, model)
        for name, model in model_definitions.items()
    ]
    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    print("\n--- Resultados Fold 2 (Rounds 3-5) ---")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]['Modelo']
    calibrated_model = trainer.calibrate(trainer.get_models()[best_name], X_f2_train, y_f2_train)
    cal_proba = calibrated_model.predict_proba(X_f2_test)[:, 1]
    raw_proba = model_definitions[best_name].predict_proba(X_f2_test)[:, 1]

    print(f"\n✔️ Mejor Modelo: {best_name} → Calibración Platt Scaling aplicada.")
    viz.plot_calibration(y_f2_test, raw_proba, cal_proba, best_name)

    strat = BettingStrategy()
    np.random.seed(RANDOM_SEED)
    noise = np.random.normal(0, 0.05, len(cal_proba))
    house_prob = np.clip(cal_proba + noise, 0.05, 0.95)
    house_prob = house_prob / (house_prob + (1 - house_prob) * 0.95)
    sim_odds = 1.0 / house_prob

    bet_analysis = pd.DataFrame({
        'gameid':       match_df.loc[mask_r35, 'gameid'].values,
        'model_p_blue': cal_proba,
        'house_odds':   sim_odds,
        'actual':       y_f2_test.values,
        'ev':           [strat.calculate_ev(p, o) for p, o in zip(cal_proba, sim_odds)],
        'kelly':        [strat.kelly_fraction(p, o) for p, o in zip(cal_proba, sim_odds)],
    })

    history, wins, losses, _ = strat.simulate_bankroll(bet_analysis)
    print(f"\n--- Simulación de Gestión de Capital ---")
    print(f"💰 Bankroll Inicial:     ${INITIAL_BANKROLL}")
    print(f"💰 Bankroll Final:       ${history[-1]:.2f}")
    print(f"📈 ROI Final:            {(history[-1] - INITIAL_BANKROLL) / INITIAL_BANKROLL:.1%}")
    print(f"🎯 Win Rate de Apuestas: {wins / (wins + losses):.1%}")

    viz.plot_bankroll(history, bet_analysis['ev'], EV_THRESHOLD)

    joblib.dump(calibrated_model, 'lck_betting_model.pkl')
    with open('lck_feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    print("\n✅ Pipeline finalizado. Artefactos guardados.")


if __name__ == "__main__":
    main()
