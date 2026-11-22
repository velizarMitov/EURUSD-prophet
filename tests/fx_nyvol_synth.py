"""Synthetic EURUSD/GBPUSD/AUDUSD H1 frames with spread for the fx-ny-volatility tests."""
import copy
from datetime import date

import numpy as np
import pandas as pd

from src.fx_nyvol import load_config
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()
PAIRS = ('EURUSD', 'GBPUSD', 'AUDUSD')
POINTS = {p: 1e-5 for p in PAIRS}


def walk_frames(first=date(2012, 1, 1), n_weeks=12, seed=0, vol=0.001):
    out = {}
    for i, p in enumerate(PAIRS):
        f = synthetic_weeks(first, n_weeks, seed=seed + i, vol=vol)
        f['spread'] = np.random.default_rng(seed + 10 + i).integers(3, 12, len(f)).astype(float)
        out[p] = f
    return out


def flat_frames(first=date(2012, 1, 1), n_weeks=8, rng_=0.0010, level=1.1):
    """Every bar flat at `level` with high/low = level +/- rng_/2, spread 0."""
    out = {}
    for p in PAIRS:
        f = synthetic_weeks(first, n_weeks, seed=1)
        f['open'] = f['close'] = level
        f['high'], f['low'] = level + rng_ / 2, level - rng_ / 2
        f['spread'] = 0.0
        out[p] = f
    return out


def stamp(day, hour):
    return pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=hour)


def tmp_cfg(tmp_path=None):
    c = copy.deepcopy(CFG)
    if tmp_path is not None:
        c['paths'] = {**c['paths'], 'results': str(tmp_path / 'results'), 'artifacts': str(tmp_path / 'artifacts')}
    return c
