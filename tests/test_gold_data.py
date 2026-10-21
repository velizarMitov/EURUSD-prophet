"""Gold data acquisition (task 2.1; spec "Gold bars come from MT5 as OHLC only")."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.gold import data as D
from src.gold import load_config

DT = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'),
      ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')]


def _rates(n, start='2024-01-01 00:00', step=3600):
    t0 = int(pd.Timestamp(start, tz='UTC').timestamp())
    r = np.zeros(n, dtype=DT)
    r['time'] = t0 + step * np.arange(n)
    r['close'] = 2000 + np.arange(n, dtype=float)
    r['open'], r['high'], r['low'] = r['close'] - 1, r['close'] + 2, r['close'] - 2
    r['tick_volume'], r['spread'], r['real_volume'] = 99, 25, 0
    return r


class _Sym:
    def __init__(self, name):
        self.name = name


class _Tick:
    def __init__(self, t):
        self.time = t


class FakeMT5:
    TIMEFRAME_D1, TIMEFRAME_H1 = 16408, 16385

    def __init__(self, rates, names=('GOLD', 'EURUSD'), up=True, tick=True):
        self.rates, self.names, self.up, self.tick = rates, names, up, tick

    def initialize(self):
        return self.up

    def shutdown(self):
        pass

    def symbols_get(self):
        return [_Sym(n) for n in self.names]

    def symbol_select(self, s, on):
        return True

    def copy_rates_range(self, *a):
        return self.rates[-5:]

    def copy_rates_from_pos(self, sym, tf, pos, n):
        return self.rates[-n:]

    def symbol_info_tick(self, sym):
        # the newest bar is still forming: "now" is 10 minutes into it
        return _Tick(int(self.rates['time'][-1]) + 600) if self.tick else None


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c['paths']['h1_cache'] = str(tmp_path / 'gold_h1.csv')
    c['paths']['d1_cache'] = str(tmp_path / 'gold_d1.csv')
    c['bars']['H1'] = 500
    return c


def test_cache_holds_only_time_and_ohlc(cfg):
    df, rec = D.load('H1', mt5=FakeMT5(_rates(300)), cfg=cfg)
    cols = pd.read_csv(cfg['paths']['h1_cache'], nrows=1).columns.tolist()
    assert cols == ['time', 'open', 'high', 'low', 'close']
    assert list(df.columns) == ['open', 'high', 'low', 'close']
    assert rec['source'] == 'mt5' and rec['symbol'] == 'GOLD'


def test_forming_bar_is_dropped(cfg):
    df, _ = D.load('H1', mt5=FakeMT5(_rates(300)), cfg=cfg)
    assert len(df) == 299
    df2, _ = D.load('H1', mt5=FakeMT5(_rates(300), tick=False), cfg={**cfg, 'paths': {**cfg['paths'], 'h1_cache': cfg['paths']['h1_cache'] + '2'}})
    assert len(df2) == 299


def test_terminal_unavailable_uses_cache(cfg):
    D.load('H1', mt5=FakeMT5(_rates(300)), cfg=cfg)
    df, rec = D.load('H1', mt5=FakeMT5(_rates(300), up=False), cfg=cfg)
    assert rec['source'] == 'cache' and len(df) == 299
    assert rec['symbol'] == 'GOLD'                       # remembered from the last live fetch


def test_no_terminal_and_no_cache_stops(cfg):
    with pytest.raises(D.GoldDataUnavailable):
        D.load('H1', mt5=FakeMT5(_rates(300), up=False), cfg=cfg)


def test_cached_bars_are_never_rewritten(cfg):
    D.load('H1', mt5=FakeMT5(_rates(300)), cfg=cfg)
    revised = _rates(310)
    revised['close'][:200] += 50.0                      # broker revises old bars
    df, _ = D.load('H1', mt5=FakeMT5(revised), cfg=cfg)
    assert len(df) == 309
    assert df['close'].iloc[0] == 2000.0                # cached value kept


def _daily_index(start, end, drop=()):
    idx = pd.bdate_range(start, end, tz='UTC')
    return idx[~idx.isin(pd.DatetimeIndex(drop, tz='UTC'))]


def _frame(idx):
    return pd.DataFrame({c: 2000.0 for c in D.OHLC}, index=idx)


def test_good_friday_gap_is_exempt():
    df = _frame(_daily_index('2024-01-02', '2024-06-28', drop=['2024-03-29']))   # Good Friday 2024
    assert len(D.check_coverage(df, 'D1', 'x')) == len(df)


def test_other_four_day_gap_is_refused():
    from src.mt5_coverage import CoverageError
    df = _frame(_daily_index('2024-01-02', '2024-06-28', drop=['2024-05-10']))   # an ordinary Friday
    with pytest.raises(CoverageError):
        D.check_coverage(df, 'D1', 'x')


def test_leading_thin_h1_months_are_trimmed_interior_ones_refused():
    from src.mt5_coverage import CoverageError
    daily_like = pd.date_range('2013-09-02', '2014-11-28', freq='B', tz='UTC')           # one bar a day
    hourly = pd.date_range('2014-12-01', '2015-06-30 23:00', freq='h', tz='UTC')
    hourly = hourly[hourly.dayofweek < 5]
    trimmed = D.trim_leading_thin(_frame(daily_like.append(hourly)), 'H1')
    assert trimmed.index[0] >= pd.Timestamp('2014-12-01', tz='UTC')
    D.check_coverage(trimmed, 'H1', 'x')
    holed = hourly[(hourly < pd.Timestamp('2015-03-01', tz='UTC')) | (hourly >= pd.Timestamp('2015-04-01', tz='UTC'))
                   | (hourly.hour == 12)]                                                  # March nearly empty
    with pytest.raises(CoverageError):
        D.check_coverage(D.trim_leading_thin(_frame(holed), 'H1'), 'H1', 'x')


def test_aliases_resolve_in_order():
    assert D.resolve_gold_symbol(['GOLD', 'EURUSD'], ['GOLD', 'XAUUSD']) == 'GOLD'
    assert D.resolve_gold_symbol(['XAUUSD.a', 'EURUSD'], ['GOLD', 'XAUUSD']) == 'XAUUSD.a'
    with pytest.raises(D.GoldDataUnavailable):
        D.resolve_gold_symbol(['EURUSD'], ['GOLD', 'XAUUSD'])
