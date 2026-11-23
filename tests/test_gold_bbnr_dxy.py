"""gold-bb-nyrange, task 2.2: the dollar index caches and their clock check."""
import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import dxy as D
from src.gold_bbnr import load_config

CFG = load_config()


def pair(years=(2018, 2019), shift_year=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f'{years[0]}-01-02', f'{years[-1]}-12-28', freq='h', tz='UTC')
    idx = idx[(idx.dayofweek < 5) & (idx.hour >= 2) & (idx.hour <= 22)]
    eu_ret = rng.normal(0, 1e-3, len(idx))
    dx_ret = -0.9 * eu_ret + rng.normal(0, 3e-4, len(idx))
    eur = pd.DataFrame({'close': 1.1 * np.exp(np.cumsum(eu_ret))}, index=idx)
    dxy = pd.DataFrame({'close': 95 * np.exp(np.cumsum(dx_ret))}, index=idx)
    if shift_year:
        m = dxy.index.year == shift_year
        moved = dxy[m].copy()
        moved.index = moved.index + pd.Timedelta(hours=1)
        dxy = pd.concat([dxy[~m], moved]).sort_index()
        dxy = dxy[~dxy.index.duplicated()]
    for c in ('open', 'high', 'low'):
        dxy[c] = dxy['close']
    return dxy, eur


def test_lag_check_passes_on_aligned_series():
    dxy, eur = pair()
    out = D.lag_check(dxy, eur)
    assert set(out) == {2018, 2019} and all(min(v, key=v.get) == 0 for v in out.values())


def test_a_year_shifted_by_one_hour_is_refused():
    dxy, eur = pair(shift_year=2019)
    with pytest.raises(D.DxyClockMismatch, match='2019'):
        D.lag_check(dxy, eur)


def test_load_h1_runs_the_check(tmp_path):
    dxy, eur = pair(shift_year=2018)
    p = tmp_path / 'dxy_h1.csv'
    d = dxy.copy()
    d.index.name = 'time'
    d[D.OHLC].to_csv(p)
    with pytest.raises(D.DxyClockMismatch):
        D.load_h1(CFG, str(p), eur)


def test_fetch_module_never_names_an_order_function():
    src = open(D.__file__, encoding='utf-8').read()
    for name in ('order_send', 'order_check', 'positions_get', 'orders_get', 'history_deals', 'account_info',
                 'TRADE_ACTION'):
        assert name not in src


def test_fetch_mt5_uses_market_data_calls_only(tmp_path):
    calls = []

    class Fake:
        TIMEFRAME_H1 = 16385

        def __getattr__(self, name):
            calls.append(name)
            raise AssertionError(f'unexpected MT5 call {name}')

    class Mt5(Fake):
        def initialize(self, **k):
            calls.append('initialize'); return True

        def symbol_select(self, s, on):
            calls.append('symbol_select'); return True

        def copy_rates_range(self, s, tf, a, b):
            calls.append('copy_rates_range')
            t = pd.date_range('2017-01-03 02:00', periods=5, freq='h', tz='UTC').astype('int64') // 10**9
            r = np.zeros(5, dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'),
                                   ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')])
            r['time'], r['close'] = t, 100.0
            return r

        def shutdown(self):
            calls.append('shutdown')

    cfg = {**CFG, 'paths': {**CFG['paths'], 'data': str(tmp_path)}}
    p = D.fetch_mt5(cfg, 'X:/terminal64.exe', Mt5())
    assert calls == ['initialize', 'symbol_select', 'copy_rates_range', 'shutdown']
    got = pd.read_csv(p)
    assert list(got.columns) == ['time'] + D.OHLC and len(got) == 5


def test_yahoo_request_sends_only_the_generic_client_name(tmp_path):
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            ts = [int(pd.Timestamp('2015-01-05 05:00', tz='UTC').timestamp()),
                  int(pd.Timestamp('2015-01-06 05:00', tz='UTC').timestamp())]
            return {'chart': {'result': [{'meta': {'exchangeTimezoneName': 'America/New_York'}, 'timestamp': ts,
                                          'indicators': {'quote': [{'open': [91, 92], 'high': [92, 93], 'low': [90, 91],
                                                                    'close': [91.5, None]}]}}]}}

    def get(url, params, headers, timeout):
        seen.update(url=url, params=params, headers=headers)
        return Resp()

    cfg = {**CFG, 'paths': {**CFG['paths'], 'data': str(tmp_path)}}
    p = D.fetch_yahoo(cfg, get)
    assert seen['headers'] == {'User-Agent': 'Mozilla/5.0 (research data client)'}
    assert 'DX-Y.NYB' in seen['url'] and seen['params']['interval'] == '1d'
    d1 = D.load_d1(cfg, p)
    assert list(d1.index) == [pd.Timestamp('2015-01-05')] and d1['close'].iloc[0] == 91.5
    assert '@' not in str(seen)
