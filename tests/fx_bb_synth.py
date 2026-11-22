"""Synthetic six-pair H1 data for the fx-bb-reversion tests (Berlin-label clock, with spread)."""
import copy
import os
from datetime import date

import numpy as np
import pandas as pd

from src.fx_bb import load_config
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()
PAIRS = CFG['pairs']
LEVEL = {'EURUSD': 1.10, 'GBPUSD': 1.30, 'USDJPY': 110.0, 'AUDUSD': 0.75, 'USDCAD': 1.30, 'USDCHF': 0.95}
POINT = {p: (0.001 if p == 'USDJPY' else 0.00001) for p in PAIRS}


def six_pairs(first=date(2012, 1, 1), n_weeks=8, shift_values=None, seed=0, vol=0.001):
    """EURUSD is a random walk; every other pair is EURUSD (or its inverse for USD-base pairs)
    times independent noise, so all pairs co-move at lag 0. `shift_values={pair: k}` moves that
    pair's prices k bars later on the SAME labels (the weekly opens stay on the clock)."""
    shift_values = shift_values or {}
    base = synthetic_weeks(first, n_weeks, seed=seed, vol=vol)
    rng = np.random.default_rng(seed + 100)
    out = {}
    for i, p in enumerate(PAIRS):
        if p == 'EURUSD':
            f = base.copy()
        else:
            noise = np.exp(np.cumsum(rng.normal(0, vol / 3, len(base))))
            rel = base['close'].to_numpy() / 1.10
            path = (rel if p in ('GBPUSD', 'AUDUSD') else 1 / rel) * noise * LEVEL[p]
            k = shift_values.get(p, 0)
            if k:
                path = np.r_[np.full(k, path[0]), path[:-k]]
            open_ = np.r_[path[0], path[:-1]]
            f = pd.DataFrame({'open': open_, 'high': np.maximum(open_, path) * (1 + vol / 4),
                              'low': np.minimum(open_, path) * (1 - vol / 4), 'close': path}, index=base.index)
        f['spread'] = np.random.default_rng(seed + i).integers(3, 12, len(f)).astype(float)
        out[p] = f
    return out


def write_cache(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = frame.copy()
    out.index.name = 'time'
    out.to_csv(path, lineterminator='\n')


def tmp_cfg(tmp_path, starts=None):
    c = copy.deepcopy(CFG)
    c['paths'] = {'data': str(tmp_path / 'bb_data'), 'fx_session_data': str(tmp_path / 'session_data'),
                  'results': str(tmp_path / 'results'), 'artifacts': str(tmp_path / 'artifacts')}
    starts = starts or {}
    for p in PAIRS:
        c['data']['start'][p] = starts.get(p, '2012-01-01')
        c['data']['latest_allowed_start'][p] = str((pd.Timestamp(c['data']['start'][p]) + pd.Timedelta(days=30)).date())
    return c


def install_session_caches(cfg, frames):
    """Write the three fx_session-sourced pairs and their symbols.json into the tmp session dir."""
    import json
    d = cfg['paths']['fx_session_data']
    for p, s in cfg['sources'].items():
        if s == 'fx_session':
            write_cache(frames[p], os.path.join(d, f'{p}_h1.csv'))
    with open(os.path.join(d, 'symbols.json'), 'w', encoding='utf-8') as fh:
        json.dump({p: {'symbol': p, 'point': POINT[p]} for p, s in cfg['sources'].items() if s == 'fx_session'}, fh)


def install_all(cfg, frames):
    """Both cache directories, as after a successful refresh."""
    import json
    install_session_caches(cfg, frames)
    d = cfg['paths']['data']
    for p, s in cfg['sources'].items():
        if s == 'fx_bb':
            write_cache(frames[p], os.path.join(d, f'{p}_h1.csv'))
    with open(os.path.join(d, 'symbols.json'), 'w', encoding='utf-8') as fh:
        json.dump({p: {'symbol': p, 'point': POINT[p]} for p, s in cfg['sources'].items() if s == 'fx_bb'}, fh)


class FakeSource:
    def __init__(self, frames, names=None):
        self.frames, self.fetched = frames, []
        self.names = names if names is not None else list(frames)

    def symbol_names(self):
        return self.names

    def symbol_point(self, symbol):
        return POINT.get(symbol, 0.00001)

    def fetch_h1(self, symbol, bars):
        self.fetched.append(symbol)
        f = self.frames.get(symbol)
        return None if f is None else f[['open', 'high', 'low', 'close', 'spread']].copy()
