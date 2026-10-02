"""Volatility challenger and its baselines (task 7.7).

  VolEnsemble   5-seed multi-task LSTM (volatility softplus + return + direction
                heads) on the production price-only daily features; the forecast
                is the mean over ALL seeds -- a partial ensemble is never used
  GarchDow      GARCH(1,1) x day-of-week, the strongest known baseline
                (volatility_hypothesis_log.csv row 3), via src/calendar_volatility
                read-only. Fitted on the h-bar target directly, so its scale and
                weekday factors absorb the horizon. Train rows only.
  RandomWalkVol persistence: the realised |log return| over the last h bars
                forecasts the next h bars.

The baselines work on any cadence: they take a close series and day-of-week.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from .. import targets as T
from .base import Challenger
from .daily import DailyPreprocessor, _multitask_lstm, daily_inputs, fit_sequence_model
from .base import windows


class VolEnsemble(Challenger):
    name = 'vol_ensemble'
    cadence = 'D1'
    kind = 'volatility'
    produces_returns = True

    def build_inputs(self, daily_df):
        return daily_inputs(daily_df, self.config['feature_set'])

    def fit(self, inputs, train_pos, targets, seed, h):
        import keras
        from .. import features as F
        c = self.config['lstm']
        self.prep = DailyPreprocessor(F.daily_columns(self.config['feature_set']),
                                      0.95).fit(inputs.X[train_pos])
        Xs = self.prep.transform(inputs.X).astype('float32')
        seq, rows = windows(Xs, c['time_steps'])
        by_pos = {int(r): i for i, r in enumerate(rows)}
        self.models = []
        for s in self.config['seeds']:
            keras.utils.set_random_seed(int(s))
            m = _multitask_lstm(c['time_steps'], seq.shape[2], c['units'], c['dropout'],
                                c['learning_rate'], extra_heads=('vol',))
            fit_sequence_model(m, seq, by_pos, train_pos, targets, ('ret', 'dir', 'vol'), h, c)
            self.models.append(m)
        self.fitted = True
        return self

    def predict(self, inputs):
        c = self.config['lstm']
        if len(self.models) != len(self.config['seeds']):
            raise RuntimeError('partial ensemble: refusing to predict')
        seq, rows = windows(self.prep.transform(inputs.X).astype('float32'), c['time_steps'])
        outs = [m.predict(seq, verbose=0, batch_size=1024) for m in self.models]
        n = len(inputs)
        res = {k: np.full(n, np.nan) for k in ('p_up', 'ret_pct', 'vol_pct')}
        res['vol_pct'][rows] = np.mean([o['vol'].ravel() for o in outs], axis=0)
        res['ret_pct'][rows] = np.mean([o['ret'].ravel() for o in outs], axis=0)
        res['p_up'][rows] = np.mean([o['dir'].ravel() for o in outs], axis=0)
        return res

    def _save(self, path):
        joblib.dump(self.prep, os.path.join(path, 'prep.joblib'))
        for s, m in zip(self.config['seeds'], self.models):
            m.save(os.path.join(path, f'seed_{s}.keras'))

    def _load(self, path):
        import keras
        self.prep = joblib.load(os.path.join(path, 'prep.joblib'))
        self.models = [keras.models.load_model(os.path.join(path, f'seed_{s}.keras'))
                       for s in self.config['seeds']]


def _one_bar_returns_pct(close: np.ndarray) -> np.ndarray:
    """r_t = 100 * log(close_t / close_{t-1}) at row t (NaN at row 0)."""
    s = pd.Series(np.asarray(close, dtype=float))
    return T.return_pct_target(s, 1).shift(1).to_numpy()


class GarchDow:
    def __init__(self, use_dow: bool = True):
        self.use_dow = use_dow

    def fit(self, close, dow, target_vol_pct, train_mask):
        from src.calendar_volatility import CalendarVolatilityModel
        r = np.nan_to_num(_one_bar_returns_pct(close))
        y = np.asarray(target_vol_pct, dtype=float)
        self.model = CalendarVolatilityModel(use_dow=self.use_dow).fit(
            r, y, np.asarray(dow), np.asarray(train_mask, dtype=bool))
        return self

    def predict(self, close, dow):
        r = np.nan_to_num(_one_bar_returns_pct(close))
        return self.model.predict(r, np.asarray(dow))


class RandomWalkVol:
    def predict(self, close, h: int) -> np.ndarray:
        s = pd.Series(np.asarray(close, dtype=float))
        return T.volatility_pct_target(s, h).shift(h).to_numpy()
