import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
from src.config import RANDOM_SEED


class ModelTrainer:

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed

    def get_models(self) -> dict:
        return {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    LogisticRegression(C=0.1, max_iter=1000, random_state=self.seed)),
            ]),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=10,
                random_state=self.seed, n_jobs=-1,
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=self.seed, eval_metric='logloss', verbosity=0,
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                num_leaves=15, min_child_samples=15,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=self.seed, verbose=-1,
            ),
        }

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, name: str, model) -> dict:
        if 'XGBoost' in name:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        return {
            'Model':    name,
            'Accuracy': accuracy_score(y_test, preds),
            'ROC-AUC':  roc_auc_score(y_test, proba),
            'Log Loss': log_loss(y_test, proba),
            'Brier':    brier_score_loss(y_test, proba),
        }

    def calibrate(self, base_model, X_train, y_train) -> CalibratedClassifierCV:
        calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        calibrated.fit(X_train, y_train)
        return calibrated
