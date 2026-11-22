"""Synthetic six-pair inputs for the fx-bb-sma-exit tests: daily random walks drawn as flat Berlin-labelled H1 days."""
import copy

import numpy as np
import pandas as pd

from src.fx_bb.daily import d1_ny
from src.fx_bb_sma import bb_view, load_config
from src.fx_bb_sma import trades as T

CFG = load_config()
POINT = {p: (0.001 if p == 'USDJPY' else 0.00001) for p in CFG['pairs']}
LEVEL = {'EURUSD': 1.10, 'GBPUSD': 1.30, 'USDJPY': 110.0, 'AUDUSD': 0.75, 'USDCAD': 1.30, 'USDCHF': 0.95}


def h1_days(days, lv, spread, rng, vol):
    stamps, o, h, l_, c = [], [], [], [], []
    prev = lv[0]
    for d, x in zip(days, lv):
        hours = [d - pd.Timedelta(hours=1)] + [d + pd.Timedelta(hours=k) for k in range(23)]
        path = np.linspace(prev, x, len(hours) + 1)[1:]
        wig = np.abs(rng.normal(0, vol / 6, len(hours))) * x
        for lab, a, b, w in zip(hours, np.r_[prev, path[:-1]], path, wig):
            stamps.append(lab)
            o.append(a)
            c.append(b)
            h.append(max(a, b) + w)
            l_.append(min(a, b) - w)
        prev = x
    idx = pd.DatetimeIndex(stamps).tz_localize('UTC')
    return pd.DataFrame({'open': o, 'high': h, 'low': l_, 'close': c, 'spread': float(spread)}, index=idx)


def synth_inputs(n_days=900, start='2012-01-02', seed=0, vol=0.006, spread=8.0, cfg=None):
    cfg = cfg or tmp_cfg_like()
    days = pd.bdate_range(start, periods=n_days)
    frames, points, d1 = {}, {}, {}
    for i, p in enumerate(cfg['pairs']):
        rng = np.random.default_rng(seed + i)
        lv = LEVEL[p] * np.exp(np.cumsum(rng.normal(0, vol, n_days)))
        frames[p] = h1_days(days, lv, spread, rng, vol)
        points[p] = POINT[p]
        d1[p] = d1_ny(frames[p], points[p], cfg['pip'][p], bb_view(cfg))
    return T.contexts(frames, points, d1, cfg), d1


def tmp_cfg_like(tmp_path=None):
    c = copy.deepcopy(CFG)
    c['data']['start']['USDCHF'] = '2012-01-01'
    if tmp_path is not None:
        c['paths'] = {'results': str(tmp_path / 'results'), 'artifacts': str(tmp_path / 'artifacts')}
    c['scoring']['resamples'] = 1000
    return c
