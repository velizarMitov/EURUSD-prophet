"""Challengers fed by H1 bars (tasks 7.4, 7.5, 7.6).

  H1DailyEnsemble  daily cadence: XGBoost + Random Forest + SVM on the flattened
                   daily features of H1 bars, plus an LSTM on the (24, n) hourly
                   tensor; p_up is the mean of the four probabilities
  TILSTM           daily cadence: the TI-LSTM architecture (src/ti_lstm_h1_
                   experimental.build_model, imported read-only) on the (24, 8)
                   indicator tensor, trained on CPU
  H1GBM            H1 cadence: XGBoost on the served H1 next-bar features with the
                   H_dir.1 hyperparameters, on CPU

Every scaler is fitted on training positions only.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from .. import features as F
from .base import Challenger, Inputs
from .daily import _valid, fit_sequence_model


class _SeqScaler:
    """Per-feature standardisation of a (n, steps, f) tensor, fitted on train rows."""

    def fit(self, X3):
        flat = X3.reshape(-1, X3.shape[2])
        self.mu, self.sd = flat.mean(axis=0), flat.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X3):
        return ((X3 - self.mu) / self.sd).astype('float32')


def _seq_lstm_classifier(steps, n_feat, units, dropout, lr):
    import keras
    from keras import layers
    inp = keras.Input(shape=(steps, n_feat))
    x = layers.Dropout(dropout)(layers.LSTM(units)(inp))
    out = {'dir': layers.Dense(1, activation='sigmoid', name='dir')(x)}
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss={'dir': 'binary_crossentropy'})
    return m


class H1DailyEnsemble(Challenger):
    name = 'h1_daily_ensemble'
    cadence = 'D1'
    kind = 'direction'

    def build_inputs(self, h1: pd.DataFrame) -> Inputs:
        flat, seq, daily_close = F.h1_daily_features(h1)
        idx = pd.DatetimeIndex(flat.index)
        return Inputs(index=idx, close=daily_close.to_numpy(float), X=flat.to_numpy(float),
                      extra={'seq': seq, 'dow': idx.dayofweek.to_numpy()})

    def fit(self, inputs, train_pos, targets, seed, h):
        import keras
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        pos = _valid(train_pos, targets['dir'])
        X, y = inputs.X[pos], targets['dir'][pos].astype(int)
        g, rf, sv = self.config['gbm'], self.config['rf'], self.config['svm']
        self.xgb = xgb.XGBClassifier(n_estimators=g['n_estimators'], learning_rate=g['learning_rate'],
                                     max_depth=g['max_depth'], subsample=g['subsample'],
                                     tree_method='hist', device='cpu', random_state=seed,
                                     n_jobs=4, verbosity=0, eval_metric='logloss').fit(X, y)
        self.rf = RandomForestClassifier(n_estimators=rf['n_estimators'], max_depth=rf['max_depth'],
                                         random_state=seed, n_jobs=4).fit(X, y)
        self.flat_scaler = StandardScaler().fit(inputs.X[train_pos])
        # Platt-calibrated SVC; SVC(probability=True) is deprecated in sklearn 1.9.
        from sklearn.calibration import CalibratedClassifierCV
        self.svm = CalibratedClassifierCV(SVC(C=sv['C'], gamma=sv['gamma'], random_state=seed),
                                          method='sigmoid', ensemble=False
                                          ).fit(self.flat_scaler.transform(X), y)
        keras.utils.set_random_seed(seed)
        c = self.config['lstm']
        self.seq_scaler = _SeqScaler().fit(inputs.extra['seq'][train_pos])
        seq = self.seq_scaler.transform(inputs.extra['seq'])
        self.lstm = _seq_lstm_classifier(seq.shape[1], seq.shape[2], c['lstm_units'],
                                         c['lstm_dropout'], c['lstm_lr'])
        fit_sequence_model(self.lstm, seq, {i: i for i in range(len(inputs))}, train_pos,
                           targets, ('dir',), h,
                           {'patience': c['lstm_patience'], 'epochs': c['lstm_epochs'],
                            'batch_size': c['lstm_batch']})
        self.fitted = True
        return self

    def predict(self, inputs):
        X = inputs.X
        members = np.vstack([
            self.xgb.predict_proba(X)[:, 1],
            self.rf.predict_proba(X)[:, 1],
            self.svm.predict_proba(self.flat_scaler.transform(X))[:, 1],
            self.lstm.predict(self.seq_scaler.transform(inputs.extra['seq']), verbose=0,
                              batch_size=1024)['dir'].ravel(),
        ])
        return {'p_up': members.mean(axis=0).astype(float)}

    def _save(self, path):
        joblib.dump({'xgb': self.xgb, 'rf': self.rf, 'svm': self.svm, 'flat_scaler': self.flat_scaler,
                     'seq_scaler': self.seq_scaler}, os.path.join(path, 'members.joblib'))
        self.lstm.save(os.path.join(path, 'lstm.keras'))

    def _load(self, path):
        import keras
        d = joblib.load(os.path.join(path, 'members.joblib'))
        self.xgb, self.rf, self.svm = d['xgb'], d['rf'], d['svm']
        self.flat_scaler, self.seq_scaler = d['flat_scaler'], d['seq_scaler']
        self.lstm = keras.models.load_model(os.path.join(path, 'lstm.keras'))


class TILSTM(Challenger):
    name = 'ti_lstm'
    cadence = 'D1'
    kind = 'direction'
    produces_returns = True

    def build_inputs(self, h1: pd.DataFrame) -> Inputs:
        X, index, daily_close = F.ti_daily_sequences(h1)
        idx = pd.DatetimeIndex(index)
        return Inputs(index=idx, close=daily_close.to_numpy(float), X=X,
                      extra={'dow': idx.dayofweek.to_numpy()})

    def fit(self, inputs, train_pos, targets, seed, h):
        import keras
        from src.ti_lstm_h1_experimental import build_model
        keras.utils.set_random_seed(seed)
        c = self.config
        self.scaler = _SeqScaler().fit(inputs.X[train_pos])
        seq = self.scaler.transform(inputs.X)
        base = build_model(c['n_layers'], c['units'], seq.shape[2], dropout=c['dropout'], lr=c['lr'])
        # Re-key the production model's two heads to the study's names.
        self.model = keras.Model(base.input, {'ret': base.outputs[0], 'dir': base.outputs[1]})
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=c['lr']),
                           loss={'ret': 'mse', 'dir': 'binary_crossentropy'})
        fit_sequence_model(self.model, seq, {i: i for i in range(len(inputs))}, train_pos, targets,
                           ('ret', 'dir'), h, c)
        self.fitted = True
        return self

    def predict(self, inputs):
        out = self.model.predict(self.scaler.transform(inputs.X), verbose=0, batch_size=1024)
        return {'p_up': out['dir'].ravel().astype(float), 'ret_pct': out['ret'].ravel().astype(float)}

    def _save(self, path):
        self.model.save(os.path.join(path, 'model.keras'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))

    def _load(self, path):
        import keras
        self.model = keras.models.load_model(os.path.join(path, 'model.keras'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))


class H1GBM(Challenger):
    name = 'h1_gbm'
    cadence = 'H1'
    kind = 'direction'

    def build_inputs(self, h1: pd.DataFrame) -> Inputs:
        f = F.h1_bar_features(h1)
        idx = pd.DatetimeIndex(f.index)
        cols = [c for c in f.columns if c != 'close']
        return Inputs(index=idx, close=f['close'].to_numpy(float), X=f[cols].to_numpy(float),
                      extra={'dow': idx.dayofweek.to_numpy()})

    def fit(self, inputs, train_pos, targets, seed, h):
        import xgboost as xgb
        pos = _valid(train_pos, targets['dir'])
        y = targets['dir'][pos].astype(int)
        pos_rate = y.mean()
        g = self.config['gbm']
        self.clf = xgb.XGBClassifier(
            n_estimators=g['n_estimators'], max_depth=g['max_depth'], learning_rate=g['learning_rate'],
            subsample=g['subsample'], colsample_bytree=g['colsample_bytree'], reg_lambda=g['reg_lambda'],
            objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cpu',
            scale_pos_weight=float((1 - pos_rate) / pos_rate) if 0 < pos_rate < 1 else 1.0,
            random_state=seed, n_jobs=8, verbosity=0).fit(inputs.X[pos], y)
        self.fitted = True
        return self

    def predict(self, inputs):
        return {'p_up': self.clf.predict_proba(inputs.X)[:, 1].astype(float)}

    def _save(self, path):
        joblib.dump(self.clf, os.path.join(path, 'model.joblib'))

    def _load(self, path):
        self.clf = joblib.load(os.path.join(path, 'model.joblib'))
