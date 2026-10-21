"""Gold challengers and baselines (design D6; spec "Each challenger is scored
against its own honest baseline", "Neural challengers are seed ensembles, all
or nothing").

    cell           model                                         outputs
    d1_gbm/h1_gbm  XGB classifier + pseudo-Huber regressor       p_up, ret_pct
    d1_lstm/h1_lstm multi-task LSTM (Huber / BCE / MAE heads),   p_up, ret_pct, vol_pct
                   mean of five seeds, all or nothing
    d1_garch       GARCH(1,1), train-only fit                    vol_pct
    h1_garch_hod   GARCH(1,1) on returns / train-only hour-of-   vol_pct
                   day mean |r|, rescaled by the next hour's

Shared interface: src/forecast_eval/challengers/base.Challenger (fit on
training positions only, full-length prediction arrays, save/load guarded by a
config hash). Every fit uses only rows that are feature-complete, eligible and
target-defined; every transform is fitted on those training rows.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from src.forecast_eval import splits as FS
from src.forecast_eval import targets as T
from src.forecast_eval.challengers.base import Challenger, ChallengerUnavailable, Inputs, windows
from src.forecast_eval.tuning import grid_points, with_params

from . import features as F
from . import targets as G

ABS_MOMENT = float(np.sqrt(2.0 / np.pi))        # E|N(0,1)|: sigma -> E|r|


# ── inputs ──────────────────────────────────────────────────────────────────

def build_inputs(df: pd.DataFrame, cadence: str) -> Inputs:
    feats = F.build(df, cadence)
    X = feats.to_numpy(float)
    return Inputs(index=pd.DatetimeIndex(df.index), close=df['close'].to_numpy(float), X=X,
                  extra={'eligible': G.eligible(df.index, cadence),
                         'complete': ~np.isnan(X).any(axis=1)})


def usable(inputs: Inputs, pos, targets: dict, heads) -> np.ndarray:
    """Positions that are feature-complete, eligible and have every head's target."""
    pos = np.asarray(pos, dtype=np.int64)
    ok = inputs.extra['complete'][pos] & inputs.extra['eligible'][pos]
    for hd in heads:
        ok &= ~np.isnan(np.asarray(targets[hd], dtype=float)[pos])
    return pos[ok]


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(y) & ~np.isnan(p)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float('nan')
    return float(roc_auc_score(y[m], p[m]))


def _gross_sign_sharpe(p_up, ret_pct):
    """Per-trade Sharpe of sign(p_up - 0.5) * realised return, GROSS of cost.
    Recorded only so the trial log can feed the Deflated Sharpe; never a P&L."""
    m = ~np.isnan(p_up) & ~np.isnan(ret_pct)
    if m.sum() < 10:
        return float('nan')
    x = np.where(p_up[m] > 0.5, 1.0, -1.0) * ret_pct[m]
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float('nan')


# ── gradient boosting ───────────────────────────────────────────────────────

class GoldGBM(Challenger):
    kind = 'direction'
    produces_returns = True
    heads = ('dir', 'ret')

    def __init__(self, config: dict, name: str):
        super().__init__(config)
        self.name, self.cadence = name, config['cadence']
        self.tuning = []

    def _xgb(self, params: dict, seed: int):
        return dict(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'], subsample=self.config['gbm']['subsample'],
                    tree_method='hist', device='cpu', random_state=seed, n_jobs=4, verbosity=0)

    def _clf(self, X, y, params, seed):
        import xgboost as xgb
        return xgb.XGBClassifier(eval_metric='logloss', **self._xgb(params, seed)).fit(X, y)

    def tune(self, inputs, train_pos, targets, seed):
        """Nested walk-forward INSIDE the training block over the declared grid;
        best mean inner ROC-AUC wins (the EUR/USD study's rule). One trial per
        grid point is kept in self.tuning for the trial log."""
        grid = self.config['gbm']['grid']
        block = np.arange(int(np.min(train_pos)), int(np.max(train_pos)) + 1)
        folds = FS.walk_forward(len(block), h=1, n_splits=self.config['gbm']['cv_splits'],
                                min_train=max(2, len(block) // 2))
        scores = []
        for params in grid_points(grid):
            with_params({'gbm': {}}, params, grid)              # refuses undeclared values
            aucs, sharpes = [], []
            for sp in folds:
                tr = usable(inputs, block[sp.train], targets, self.heads)
                te = usable(inputs, block[sp.test], targets, self.heads)
                if tr.size < 50 or te.size == 0:
                    continue
                clf = self._clf(inputs.X[tr], targets['dir'][tr], params, seed)
                p = clf.predict_proba(inputs.X[te])[:, 1]
                aucs.append(_auc(targets['dir'][te], p))
                sharpes.append(_gross_sign_sharpe(p, targets['ret'][te]))
            score = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else -np.inf
            scores.append((score, params))
            self.tuning.append({'params': params, 'inner_auc': score,
                                'sharpe': float(np.nanmean(sharpes)) if np.isfinite(sharpes).any() else float('nan'),
                                'n_obs': int(sum(len(block[sp.test]) for sp in folds))})
        return max(scores, key=lambda t: t[0])[1]

    def fit(self, inputs, train_pos, targets, seed, h=1):
        import xgboost as xgb
        F.assert_scale_free(self.config['columns'])
        self.params = self.tune(inputs, train_pos, targets, seed)
        pos = usable(inputs, train_pos, targets, self.heads)
        self.clf = self._clf(inputs.X[pos], targets['dir'][pos], self.params, seed)
        self.reg = xgb.XGBRegressor(objective='reg:pseudohubererror',
                                    **self._xgb(self.params, seed)).fit(inputs.X[pos], targets['ret'][pos])
        self.fitted = True
        return self

    def predict(self, inputs):
        ok = inputs.extra['complete']
        p, r = np.full(len(inputs), np.nan), np.full(len(inputs), np.nan)
        p[ok] = self.clf.predict_proba(inputs.X[ok])[:, 1]
        r[ok] = self.reg.predict(inputs.X[ok])
        return {'p_up': p, 'ret_pct': r}

    def trial_rows(self, study_id: str) -> list[dict]:
        from src.forecast_eval.challengers.base import config_hash
        return [{'study_id': study_id, 'model': self.name, 'cadence': self.cadence, 'horizon': 1,
                 'config_hash': config_hash({**self.config, 'params': t['params']}), 'variant': 'grid',
                 'sharpe': t['sharpe'], 'n_obs': t['n_obs'],
                 'notes': f"inner_auc={t['inner_auc']:.4f}; gross sign Sharpe; "
                          + ('SELECTED' if t['params'] == getattr(self, 'params', None) else 'not selected')}
                for t in self.tuning]

    def _save(self, path):
        joblib.dump({'clf': self.clf, 'reg': self.reg, 'params': self.params}, os.path.join(path, 'model.joblib'))

    def _load(self, path):
        d = joblib.load(os.path.join(path, 'model.joblib'))
        self.clf, self.reg, self.params = d['clf'], d['reg'], d['params']


# ── multi-task LSTM, five seeds, all or nothing ─────────────────────────────

def _multitask(time_steps, n_feat, units, dropout, lr):
    import keras
    from keras import layers
    inp = keras.Input(shape=(time_steps, n_feat))
    x = layers.Dropout(dropout)(layers.LSTM(units)(inp))
    outs = {'ret': layers.Dense(1, activation='linear', name='ret')(x),
            'dir': layers.Dense(1, activation='sigmoid', name='dir')(x),
            'vol': layers.Dense(1, activation='softplus', name='vol')(x)}
    model = keras.Model(inp, outs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss={'ret': keras.losses.Huber(), 'dir': 'binary_crossentropy', 'vol': 'mae'})
    return model


class GoldLSTM(Challenger):
    kind = 'direction'
    produces_returns = True
    heads = ('ret', 'dir', 'vol')

    def __init__(self, config: dict, name: str):
        super().__init__(config)
        self.name, self.cadence = name, config['cadence']
        self.models = []

    def _sequences(self, inputs):
        """Windows over the scaled matrix; a window is valid only if every row in
        it is feature-complete (warm-up rows are never zero-filled into a fit)."""
        steps = self.config['lstm']['time_steps']
        Xs = np.where(inputs.extra['complete'][:, None], self.scaler.transform(np.nan_to_num(inputs.X)), 0.0)
        seq, rows = windows(Xs.astype('float32'), steps)
        incomplete = (~inputs.extra['complete']).astype(int)
        bad = pd.Series(incomplete).rolling(steps, min_periods=1).sum().to_numpy()[rows] > 0
        by_pos = {int(r): i for i, (r, b) in enumerate(zip(rows, bad)) if not b}
        return seq, by_pos, rows, ~bad

    def fit(self, inputs, train_pos, targets, seed=None, h=1):
        import keras
        from sklearn.preprocessing import StandardScaler
        from src.forecast_eval.challengers.daily import fit_sequence_model
        F.assert_scale_free(self.config['columns'])
        pos = usable(inputs, train_pos, targets, self.heads)
        self.scaler = StandardScaler().fit(inputs.X[pos])
        seq, by_pos, _, _ = self._sequences(inputs)
        c = self.config['lstm']
        self.models = []
        for s in self.config['seeds']:
            keras.utils.set_random_seed(int(s))
            m = _multitask(c['time_steps'], seq.shape[2], c['units'], c['dropout'], c['learning_rate'])
            fit_sequence_model(m, seq, by_pos, pos, targets, self.heads, h, c)
            self.models.append(m)
        self.fitted = True
        return self

    def predict(self, inputs):
        if len(self.models) != len(self.config['seeds']):
            raise ChallengerUnavailable(f'{self.name}: {len(self.models)} of {len(self.config["seeds"])} '
                                        'seed models available -- a partial ensemble is never served')
        seq, _, rows, valid = self._sequences(inputs)
        outs = [m.predict(seq, verbose=0, batch_size=2048) for m in self.models]
        res = {}
        for key, head in (('p_up', 'dir'), ('ret_pct', 'ret'), ('vol_pct', 'vol')):
            mean = np.mean([o[head].ravel() for o in outs], axis=0)
            arr = np.full(len(inputs), np.nan)
            arr[rows[valid]] = mean[valid]
            res[key] = arr
        return res

    def _save(self, path):
        for s, m in zip(self.config['seeds'], self.models):
            m.save(os.path.join(path, f'seed{s}.keras'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))

    def _load(self, path):
        import keras
        missing = [s for s in self.config['seeds'] if not os.path.exists(os.path.join(path, f'seed{s}.keras'))]
        if missing:
            raise ChallengerUnavailable(f'{self.name}: seed model(s) {missing} missing -- '
                                        'a partial ensemble is never served')
        self.models = [keras.models.load_model(os.path.join(path, f'seed{s}.keras')) for s in self.config['seeds']]
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))


# ── GARCH baselines ─────────────────────────────────────────────────────────

def realised_returns_pct(close) -> np.ndarray:
    """r_t = 100 * log(close_t / close_{t-1}) at row t, known at t (NaN at row 0)."""
    return T.return_pct_target(pd.Series(np.asarray(close, dtype=float)), 1).shift(1).to_numpy()


def _garch_sigma_next(series: np.ndarray, est_end: int, mean: str):
    """GARCH(1,1) fitted on series[1:est_end] only, then FIXED and filtered over
    the whole series: entry t is sigma_{t+1|t} from information up to t."""
    from arch import arch_model
    s = pd.Series(np.nan_to_num(series[1:], nan=0.0))
    res = arch_model(s.iloc[:est_end - 1], mean=mean, vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
    fixed = arch_model(s, mean=mean, vol='GARCH', p=1, q=1, dist='normal').fix(res.params)
    var = fixed.forecast(horizon=1, start=0, reindex=True).variance['h.1'].to_numpy(float)
    out = np.full(len(series), np.nan)
    out[1:] = np.sqrt(var)
    return out, {k: float(v) for k, v in res.params.items()}


class GoldGarch(Challenger):
    """D1: E|r_{t+1}| = sigma_{t+1|t} * sqrt(2/pi), parameters from training rows only."""
    kind = 'volatility'
    heads = ('vol',)

    def __init__(self, config: dict, name: str):
        super().__init__(config)
        self.name, self.cadence = name, config['cadence']

    def fit(self, inputs, train_pos, targets=None, seed=None, h=1):
        self.est_end = int(np.max(train_pos)) + 1           # returns r_1 .. r_{last train row}
        _, self.params = _garch_sigma_next(realised_returns_pct(inputs.close), self.est_end, 'Constant')
        self.fitted = True
        return self

    def predict(self, inputs):
        from arch import arch_model
        r = pd.Series(np.nan_to_num(realised_returns_pct(inputs.close)[1:], nan=0.0))
        fixed = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal').fix(
            pd.Series(self.params))
        var = fixed.forecast(horizon=1, start=0, reindex=True).variance['h.1'].to_numpy(float)
        out = np.full(len(inputs), np.nan)
        out[1:] = np.sqrt(var) * ABS_MOMENT
        return {'p_up': np.full(len(inputs), np.nan), 'vol_pct': out}

    def _save(self, path):
        joblib.dump({'params': self.params, 'est_end': self.est_end}, os.path.join(path, 'garch.joblib'))

    def _load(self, path):
        d = joblib.load(os.path.join(path, 'garch.joblib'))
        self.params, self.est_end = d['params'], d['est_end']


class GoldGarchHOD(GoldGarch):
    """H1: z_t = r_t / s(hour_t) with s the TRAINING mean |r| per label hour;
    GARCH(1,1) on z; E|r_{t+1}| = sqrt(2/pi) * sigma_z(t+1|t) * s(hour_{t+1}).
    hour_{t+1} is hour_t + 1: every eligible row's target is the next hour."""

    def _scale(self, index):
        hours = pd.DatetimeIndex(index).hour.to_numpy()
        s = np.array([self.hod.get(int(hh), self.hod_all) for hh in hours])
        return hours, s

    def fit(self, inputs, train_pos, targets=None, seed=None, h=1):
        r = realised_returns_pct(inputs.close)
        self.est_end = int(np.max(train_pos)) + 1
        tr = pd.Series(np.abs(r[1:self.est_end]), index=pd.DatetimeIndex(inputs.index[1:self.est_end]).hour)
        self.hod = {int(k): float(v) for k, v in tr.groupby(level=0).mean().items() if v > 0}
        self.hod_all = float(tr.mean())
        _, s = self._scale(inputs.index)
        _, self.params = _garch_sigma_next(r / s, self.est_end, 'Zero')
        self.fitted = True
        return self

    def predict(self, inputs):
        from arch import arch_model
        r = realised_returns_pct(inputs.close)
        hours, s = self._scale(inputs.index)
        z = pd.Series(np.nan_to_num((r / s)[1:], nan=0.0))
        fixed = arch_model(z, mean='Zero', vol='GARCH', p=1, q=1, dist='normal').fix(pd.Series(self.params))
        var = fixed.forecast(horizon=1, start=0, reindex=True).variance['h.1'].to_numpy(float)
        s_next = np.array([self.hod.get(int((hh + 1) % 24), self.hod_all) for hh in hours])
        out = np.full(len(inputs), np.nan)
        out[1:] = np.sqrt(var) * ABS_MOMENT * s_next[1:]
        return {'p_up': np.full(len(inputs), np.nan), 'vol_pct': out}

    def _save(self, path):
        joblib.dump({'params': self.params, 'est_end': self.est_end, 'hod': self.hod,
                     'hod_all': self.hod_all}, os.path.join(path, 'garch.joblib'))

    def _load(self, path):
        d = joblib.load(os.path.join(path, 'garch.joblib'))
        self.params, self.est_end, self.hod, self.hod_all = d['params'], d['est_end'], d['hod'], d['hod_all']


# ── cells ───────────────────────────────────────────────────────────────────

CELLS = {
    'd1_gbm': (GoldGBM, 'D1'), 'd1_lstm': (GoldLSTM, 'D1'), 'd1_garch': (GoldGarch, 'D1'),
    'h1_gbm': (GoldGBM, 'H1'), 'h1_lstm': (GoldLSTM, 'H1'), 'h1_garch_hod': (GoldGarchHOD, 'H1'),
}


def cell_config(name: str, cfg: dict) -> dict:
    cls, cadence = CELLS[name]
    cols = F.D1_FEATURE_COLUMNS if cadence == 'D1' else F.h1_feature_columns()
    out = {'cell': name, 'cadence': cadence, 'columns': list(cols)}
    if cls is GoldGBM:
        out['gbm'] = cfg['gbm']
    elif cls is GoldLSTM:
        out['lstm'] = cfg['lstm'][cadence]
        out['seeds'] = list(cfg['seeds'])
    return out


def make(name: str, cfg: dict, overrides: dict | None = None) -> Challenger:
    cls, _ = CELLS[name]
    conf = cell_config(name, cfg)
    for k, v in (overrides or {}).items():
        conf[k] = {**conf[k], **v} if isinstance(conf.get(k), dict) else v
    return cls(conf, name=name)
