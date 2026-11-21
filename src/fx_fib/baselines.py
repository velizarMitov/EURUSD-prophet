"""Comparison models on the identical rows, features and early-stopping slice (spec fx-fibonacci/models).

    lightgbm     the registered baseline: fixed config hyperparameters, never searched
    logistic     L2 logistic regression            descriptive only
    mlp_small    the 64-32 MLP, first seed         descriptive only

Class weights are balanced from the fit rows, as for the MLP.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import model_mlp_cuda as M


def _sample_weight(y: np.ndarray) -> np.ndarray:
    return M.class_weights(y)[y.astype(int)]


def train_lightgbm(X_fit, y_fit, X_val, y_val, params: dict):
    p = dict(params)
    n_est = p.pop('n_estimators')
    stop = p.pop('early_stopping_rounds')
    model = lgb.LGBMClassifier(n_estimators=n_est, verbose=-1, n_jobs=-1, force_col_wise=True,
                               random_state=p.pop('seed'), **p)
    model.fit(X_fit, y_fit, sample_weight=_sample_weight(y_fit), eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss', callbacks=[lgb.early_stopping(stop, verbose=False)])
    return model, {'best_iteration': int(model.best_iteration_ or n_est)}


def train_logistic(X_fit, y_fit, params: dict):
    model = LogisticRegression(C=params['C'], max_iter=params['max_iter'], class_weight='balanced')
    model.fit(X_fit, y_fit)
    return model, {'n_iter': int(np.max(model.n_iter_))}


def train_mlp_small(X_fit, y_fit, X_val, y_val, cfg: dict):
    small = cfg['mlp_small']
    return M.train_mlp(X_fit, y_fit, X_val, y_val, cfg['mlp'], cfg['mlp']['seeds'][0],
                       hidden=small['hidden'], dropout=small['dropout'])


def proba(model, X) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    return M.predict_proba(model, X)
