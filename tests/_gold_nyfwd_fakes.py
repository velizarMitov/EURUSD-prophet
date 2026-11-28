"""Synthetic GOLD / EURUSD / USDInd H1 bars and a fake read-only feed for the gold-nyrange-forward tests."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.gold_nyfwd import load_config


@pytest.fixture(autouse=True)
def common_files_in_tmp(tmp_path, monkeypatch):
    """Import this into every gold_nyfwd test module: no test may write the real MT5 chart file."""
    monkeypatch.setenv('MT5_COMMON_FILES', str(tmp_path / 'common'))
    return tmp_path / 'common'

CFG = load_config()
GOLD, EUR = CFG['mt5']['gold_symbol'], CFG['mt5']['lag_pair_symbol']


def hours_of(days, hours):
    return pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in hours], tz='UTC')


def ohlc(idx, start, vol, rets=None, seed=0):
    rng = np.random.default_rng(seed)
    r = rets if rets is not None else rng.normal(0, vol, len(idx))
    close = start * np.exp(np.cumsum(r))
    op = np.r_[start, close[:-1]]
    wig = np.abs(rng.normal(0, vol, len(idx))) * close
    return pd.DataFrame({'open': op, 'high': np.maximum(op, close) + wig, 'low': np.minimum(op, close) - wig,
                         'close': close}, index=pd.DatetimeIndex(idx, name='time'))


def market(first='2026-04-06', last='2026-07-31', contract='USDIndSep26', seed=1, dxy_lag=0):
    """Gold and the contract trade 00:00-22:00 Berlin labels on weekdays, EURUSD 00:00-23:00.
    The contract's hourly returns are minus EURUSD's at `dxy_lag` hours plus noise."""
    days = pd.bdate_range(first, last)
    rng = np.random.default_rng(seed)
    eur_idx = hours_of(days, range(24))
    eur_r = rng.normal(0, 0.0012, len(eur_idx))
    eur = ohlc(eur_idx, 1.10, 0.0012, eur_r, seed + 1)
    gold = ohlc(hours_of(days, range(23)), 2400.0, 0.002 * (1 + 0.5 * np.sin(np.arange(len(days) * 23) / 97)),
                seed=seed + 2)
    dx_idx = hours_of(days, range(23))
    er = pd.Series(eur_r, index=eur_idx).shift(dxy_lag).reindex(dx_idx).fillna(0).to_numpy()
    dx = ohlc(dx_idx, 100.0, 0.0006, -0.8 * er + rng.normal(0, 0.0004, len(dx_idx)), seed + 3)
    return {GOLD: gold, EUR: eur, contract: dx}


class FakeFeed:
    def __init__(self, bars: dict, listed: dict, now):
        self.bars, self.listed, self.now = bars, listed, pd.Timestamp(now)

    def symbols(self):
        return dict(self.listed)

    def closed_bars(self, symbol, n):
        b = self.bars[symbol]
        return b[b.index + pd.Timedelta(hours=1) <= self.now].tail(int(n)).copy()

    def label_now(self):
        return self.now


def cfg_in(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths'] = {'results': str(tmp_path / 'results'), 'artifacts': str(tmp_path / 'art')}
    return c


def listed(contract='USDIndSep26', expiry='2026-09-18'):
    return {GOLD: None, EUR: None, contract: pd.Timestamp(expiry)}


def no_gate(_cfg):
    return '2026-04-01T10:00:00+02:00'
