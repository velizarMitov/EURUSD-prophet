"""Session challengers on the M15 cadence (task 14.6; spec horizon-study
"Session challengers on the M15 cadence"; design D15, D16).

  M15SessionGBM   XGBoost on M15 mid features, with the h1_gbm hyperparameters
                  unchanged so the two cadences stay comparable
  M15SessionLSTM  a sequence classifier over the last `time_steps` M15 bars,
                  on CPU

Both read the SAME feature builder the H1 challenger uses, plus the sub-hourly
columns (the bar's spread, its intrabar movement, its tick volume and its
position in the session). Features are trailing by construction, so a feature at
bar t never reads a later bar -- a test holds that.

`price` ('mid' or 'bid') is an instance attribute, deliberately NOT part of the
configuration: the bid-versus-mid parity check must run the SAME declared
configuration on both price definitions, and a config hash that changed with the
price definition would make `record.assert_fit_allowed` refuse the second run.

Inputs cover the FULL M15 bar grid, not just the session. The grid has to stay
contiguous for the purge arithmetic and for `close[t+h]` to be the bar h*15
minutes later; `m15_data.eligible_mask` is what restricts training and scoring to
the session.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from .. import features as F
from .base import Challenger, Inputs, early_stop_split
from .daily import _valid
from .h1 import _seq_lstm_classifier


class _M15Base(Challenger):
    cadence = 'M15'
    kind = 'direction'

    def __init__(self, config: dict, price: str = 'mid'):
        super().__init__(config)
        if price not in F.M15_PRICES:
            raise KeyError(f'unknown price definition {price!r}')
        self.price = price

    def build_inputs(self, m15: pd.DataFrame) -> Inputs:
        f = F.m15_bar_features(m15, price=self.price)
        idx = pd.DatetimeIndex(f.index)
        cols = [c for c in f.columns if c != 'close']
        from .. import m15_data as MD
        return Inputs(index=idx, close=f['close'].to_numpy(float),
                      X=f[cols].to_numpy(float),
                      extra={'dow': idx.dayofweek.to_numpy(),
                             'in_session': MD.session_mask(idx),
                             'feature_names': np.array(cols, dtype=object)})


class M15SessionGBM(_M15Base):
    name = 'm15_session_gbm'

    def fit(self, inputs, train_pos, targets, seed, h):
        import xgboost as xgb
        pos = _valid(train_pos, targets['dir'])
        if pos.size < 50:
            raise ValueError(f'only {pos.size} labelled training rows')
        y = targets['dir'][pos].astype(int)
        rate = y.mean()
        g = self.config['gbm']
        self.clf = xgb.XGBClassifier(
            n_estimators=g['n_estimators'], max_depth=g['max_depth'],
            learning_rate=g['learning_rate'], subsample=g['subsample'],
            colsample_bytree=g['colsample_bytree'], reg_lambda=g['reg_lambda'],
            objective='binary:logistic', eval_metric='logloss', tree_method='hist',
            device='cpu', scale_pos_weight=float((1 - rate) / rate) if 0 < rate < 1 else 1.0,
            random_state=seed, n_jobs=8, verbosity=0).fit(inputs.X[pos], y)
        self.fitted = True
        return self

    def predict(self, inputs):
        return {'p_up': self.clf.predict_proba(inputs.X)[:, 1].astype(float)}

    def _save(self, path):
        joblib.dump(self.clf, os.path.join(path, 'model.joblib'))

    def _load(self, path):
        self.clf = joblib.load(os.path.join(path, 'model.joblib'))


class M15SessionLSTM(_M15Base):
    name = 'm15_session_lstm'
    PREDICT_CHUNK = 20_000

    @property
    def _steps(self) -> int:
        return int(self.config['lstm']['time_steps'])

    def _windows(self, Xs: np.ndarray, rows: np.ndarray):
        """(windows ending at each row, mask of rows that have a full window).

        `sliding_window_view` is a view, so only the rows asked for are ever
        materialised -- a dense (n, steps, f) tensor over 199k M15 bars would be
        hundreds of megabytes per fit."""
        steps = self._steps
        rows = np.asarray(rows, dtype=int)
        ok = rows >= steps - 1
        if Xs.shape[0] < steps:
            return np.zeros((0, steps, Xs.shape[1]), dtype='float32'), np.zeros(len(rows), bool)
        view = np.lib.stride_tricks.sliding_window_view(Xs, steps, axis=0)   # (n-steps+1, f, steps)
        sel = rows[ok] - (steps - 1)
        return view[sel].transpose(0, 2, 1).copy(), ok

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype('float32')

    def fit(self, inputs, train_pos, targets, seed, h):
        import keras
        c = self.config['lstm']
        pos = _valid(train_pos, targets['dir'])
        pos = np.asarray(pos)[np.asarray(pos) >= self._steps - 1]
        if pos.size < 200:
            raise ValueError(f'only {pos.size} labelled training rows with a full window')
        self.mu = inputs.X[pos].mean(axis=0)
        self.sd = inputs.X[pos].std(axis=0)
        self.sd[self.sd == 0] = 1.0
        Xs = self._scale(inputs.X)

        keras.utils.set_random_seed(seed)
        self.model = _seq_lstm_classifier(self._steps, Xs.shape[1], c['units'],
                                          c['dropout'], c['learning_rate'])
        fit_i, es_i = early_stop_split(len(pos), h)
        Xf, _ = self._windows(Xs, pos[fit_i])
        Xe, _ = self._windows(Xs, pos[es_i])
        yf = {'dir': targets['dir'][pos[fit_i]].astype('float32')}
        ye = {'dir': targets['dir'][pos[es_i]].astype('float32')}
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=c['patience'],
                                          restore_best_weights=True, verbose=0)
        self.model.fit(Xf, yf, validation_data=(Xe, ye), epochs=c['epochs'],
                       batch_size=c['batch_size'], callbacks=[es], verbose=0)
        self.fitted = True
        return self

    def predict(self, inputs):
        Xs = self._scale(inputs.X)
        out = np.full(len(inputs), np.nan)
        rows = np.arange(self._steps - 1, len(inputs))
        for start in range(0, len(rows), self.PREDICT_CHUNK):
            chunk = rows[start:start + self.PREDICT_CHUNK]
            seq, ok = self._windows(Xs, chunk)
            if not ok.any():
                continue
            p = self.model.predict(seq, verbose=0, batch_size=2048)['dir'].ravel()
            out[chunk[ok]] = p
        return {'p_up': out.astype(float)}

    def _save(self, path):
        self.model.save(os.path.join(path, 'model.keras'))
        joblib.dump({'mu': self.mu, 'sd': self.sd}, os.path.join(path, 'scaler.joblib'))

    def _load(self, path):
        import keras
        self.model = keras.models.load_model(os.path.join(path, 'model.keras'))
        d = joblib.load(os.path.join(path, 'scaler.joblib'))
        self.mu, self.sd = d['mu'], d['sd']
