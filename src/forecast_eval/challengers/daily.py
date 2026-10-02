"""Daily-cadence challengers on the production daily features (tasks 7.2, 7.3).

  DailyGBM    XGBoost classifier (direction) + pseudo-Huber regressor (return %)
  DailyLSTM   multi-task LSTM: shared trunk, return head (mse) + direction head (bce)

Both use the production preprocessing, called read-only from src/features.py:
lag PCA on the 6 autoregressive lags, then one StandardScaler -- each fitted on
the TRAINING positions only and stored with the model.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from .. import features as F
from .base import Challenger, Inputs, early_stop_split, windows


class DailyPreprocessor:
    def __init__(self, columns: list, variance: float):
        self.columns = list(columns)
        self.variance = variance

    def fit(self, X: np.ndarray):
        from sklearn.preprocessing import StandardScaler
        from src.features import LAG_COLUMNS, apply_lag_pca, fit_lag_pca, model_input_columns
        df = pd.DataFrame(X, columns=self.columns)
        self.lag_scaler, self.lag_pca = fit_lag_pca(df, lag_columns=LAG_COLUMNS,
                                                    variance_threshold=self.variance)
        red = apply_lag_pca(df, self.lag_scaler, self.lag_pca, lag_columns=LAG_COLUMNS)
        self.model_cols = model_input_columns(self.lag_pca, base_columns=self.columns,
                                              lag_columns=LAG_COLUMNS)
        self.scaler = StandardScaler().fit(red[self.model_cols])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        from src.features import LAG_COLUMNS, apply_lag_pca
        df = pd.DataFrame(X, columns=self.columns)
        red = apply_lag_pca(df, self.lag_scaler, self.lag_pca, lag_columns=LAG_COLUMNS)
        return self.scaler.transform(red[self.model_cols])


def daily_inputs(daily_df: pd.DataFrame, feature_set: str) -> Inputs:
    cols = F.daily_columns(feature_set)
    idx = pd.DatetimeIndex(daily_df.index)
    return Inputs(index=idx, close=daily_df['close'].to_numpy(float),
                  X=daily_df[cols].to_numpy(float),
                  extra={'dow': idx.dayofweek.to_numpy()})


def _valid(train_pos, *ys):
    ok = np.ones(len(train_pos), dtype=bool)
    for y in ys:
        ok &= ~np.isnan(np.asarray(y, dtype=float)[train_pos])
    return np.asarray(train_pos)[ok]


class DailyGBM(Challenger):
    cadence = 'D1'
    kind = 'direction'
    produces_returns = True

    def __init__(self, config: dict, name: str = 'daily_gbm'):
        super().__init__(config)
        self.name = name

    def build_inputs(self, daily_df: pd.DataFrame) -> Inputs:
        return daily_inputs(daily_df, self.config['feature_set'])

    def fit(self, inputs, train_pos, targets, seed, h):
        import xgboost as xgb
        cols = F.daily_columns(self.config['feature_set'])
        self.prep = DailyPreprocessor(cols, self.config['lag_pca_variance']).fit(inputs.X[train_pos])
        pos = _valid(train_pos, targets['dir'], targets['ret'])
        Xt = self.prep.transform(inputs.X[pos])
        g = self.config['gbm']
        common = dict(n_estimators=g['n_estimators'], learning_rate=g['learning_rate'],
                      max_depth=g['max_depth'], subsample=g['subsample'], tree_method='hist',
                      device='cpu', random_state=seed, n_jobs=4, verbosity=0)
        self.clf = xgb.XGBClassifier(eval_metric='logloss', **common).fit(Xt, targets['dir'][pos])
        self.reg = xgb.XGBRegressor(objective='reg:pseudohubererror', **common).fit(Xt, targets['ret'][pos])
        self.fitted = True
        return self

    def predict(self, inputs):
        Xt = self.prep.transform(inputs.X)
        return {'p_up': self.clf.predict_proba(Xt)[:, 1].astype(float),
                'ret_pct': self.reg.predict(Xt).astype(float)}

    def _save(self, path):
        joblib.dump({'prep': self.prep, 'clf': self.clf, 'reg': self.reg}, os.path.join(path, 'model.joblib'))

    def _load(self, path):
        d = joblib.load(os.path.join(path, 'model.joblib'))
        self.prep, self.clf, self.reg = d['prep'], d['clf'], d['reg']


def _multitask_lstm(time_steps, n_feat, units, dropout, lr, extra_heads=()):
    import keras
    from keras import layers
    inp = keras.Input(shape=(time_steps, n_feat))
    x = layers.LSTM(units)(inp)
    x = layers.Dropout(dropout)(x)
    outs = {'ret': layers.Dense(1, activation='linear', name='ret')(x),
            'dir': layers.Dense(1, activation='sigmoid', name='dir')(x)}
    losses = {'ret': 'mse', 'dir': 'binary_crossentropy'}
    if 'vol' in extra_heads:
        outs['vol'] = layers.Dense(1, activation='softplus', name='vol')(x)
        losses['vol'] = 'mse'
    model = keras.Model(inp, outs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=losses)
    return model


def fit_sequence_model(model, seq, rows_by_pos, train_pos, targets, heads, h, cfg):
    """Fit on windows ending at training positions; early-stop on the last
    1/8 of them with h positions purged between fit and early-stopping rows."""
    import keras
    train_pos = np.asarray([p for p in train_pos if p in rows_by_pos])
    ok = np.ones(len(train_pos), dtype=bool)
    for hd in heads:
        ok &= ~np.isnan(targets[hd][train_pos])
    train_pos = train_pos[ok]
    fit_i, es_i = early_stop_split(len(train_pos), h)
    def batch(sel):
        p = train_pos[sel]
        w = np.array([rows_by_pos[q] for q in p])
        return seq[w], {hd: targets[hd][p].astype('float32') for hd in heads}
    Xf, yf = batch(fit_i)
    Xe, ye = batch(es_i)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg['patience'],
                                       restore_best_weights=True, verbose=0)
    model.fit(Xf, yf, validation_data=(Xe, ye), epochs=cfg['epochs'],
              batch_size=cfg['batch_size'], callbacks=[es], verbose=0)
    return model


class DailyLSTM(Challenger):
    cadence = 'D1'
    kind = 'direction'
    produces_returns = True

    def __init__(self, config: dict, name: str = 'daily_lstm'):
        super().__init__(config)
        self.name = name

    def build_inputs(self, daily_df: pd.DataFrame) -> Inputs:
        return daily_inputs(daily_df, self.config['feature_set'])

    def _windows(self, inputs):
        Xs = self.prep.transform(inputs.X).astype('float32')
        seq, rows = windows(Xs, self.config['lstm']['time_steps'])
        return seq, {int(r): i for i, r in enumerate(rows)}, rows

    def fit(self, inputs, train_pos, targets, seed, h):
        import keras
        keras.utils.set_random_seed(seed)
        c = self.config['lstm']
        cols = F.daily_columns(self.config['feature_set'])
        self.prep = DailyPreprocessor(cols, self.config['lag_pca_variance']).fit(inputs.X[train_pos])
        seq, by_pos, _ = self._windows(inputs)
        self.model = _multitask_lstm(c['time_steps'], seq.shape[2], c['units'], c['dropout'], c['learning_rate'])
        fit_sequence_model(self.model, seq, by_pos, train_pos, targets, ('ret', 'dir'), h, c)
        self.fitted = True
        return self

    def predict(self, inputs):
        seq, _, rows = self._windows(inputs)
        out = self.model.predict(seq, verbose=0, batch_size=1024)
        p, r = np.full(len(inputs), np.nan), np.full(len(inputs), np.nan)
        p[rows], r[rows] = out['dir'].ravel(), out['ret'].ravel()
        return {'p_up': p, 'ret_pct': r}

    def _save(self, path):
        self.model.save(os.path.join(path, 'model.keras'))
        joblib.dump(self.prep, os.path.join(path, 'prep.joblib'))

    def _load(self, path):
        import keras
        self.model = keras.models.load_model(os.path.join(path, 'model.keras'))
        self.prep = joblib.load(os.path.join(path, 'prep.joblib'))
