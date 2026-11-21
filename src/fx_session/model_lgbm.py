"""LightGBM and the class prior (spec fx-session/evaluation; design D5).

Binary log-loss, fixed hyper-parameters from the config (never searched), no class weights,
fixed seed, deterministic, early stopping on the validation slice. The reference forecast is
the fit slice's up-rate applied to every evaluation day.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

EPS = 1e-6


def prior_up(y_fit: np.ndarray) -> float:
    return float(np.clip(np.mean(y_fit), EPS, 1 - EPS))


def train_lightgbm(X_fit, y_fit, X_val, y_val, params: dict):
    p = {k: v for k, v in params.items() if k not in ('n_estimators', 'early_stopping_rounds')}
    model = lgb.LGBMClassifier(n_estimators=int(params['n_estimators']), **p)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric='binary_logloss',
              callbacks=[lgb.early_stopping(int(params['early_stopping_rounds']), verbose=False)])
    return model


def proba_up(model, X) -> np.ndarray:
    return np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS)
