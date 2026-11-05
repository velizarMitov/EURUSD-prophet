"""Bars and spreads (openspec change gold-window-strategy, task 2.1)."""
import copy
import csv
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.gold_window import bars as B
from src import gold_window as W

DTYPE = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'),
         ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')]


class FakeMT5:
    TIMEFRAME_H1, TIMEFRAME_M15 = 'H1', 'M15'

    def __init__(self):
        self.calls = []

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def symbols_get(self):
        return [SimpleNamespace(name='GOLD'), SimpleNamespace(name='EURUSD')]

    def symbol_info(self, sym):
        return SimpleNamespace(point=0.01)

    def copy_rates_range(self, sym, tf, start, end):
        self.calls.append((tf, start, end))
        step = 3600 if tf == 'H1' else 900
        t0 = int(start.replace(tzinfo=timezone.utc).timestamp()) + 14 * 3600
        n = 3
        r = np.zeros(n, dtype=DTYPE)
        r['time'] = t0 + step * np.arange(n)
        r['close'] = 2000.0 + np.arange(n)
        r['open'] = r['high'] = r['low'] = r['close']
        r['spread'] = 25
        r['tick_volume'] = 99
        return r


def _cfg(tmp_path):
    cfg = copy.deepcopy(W.load_config())
    cfg['paths']['mt5_dir'] = str(tmp_path / 'mt5')
    cfg['paths']['artifact_manifest'] = str(tmp_path / 'artifact_manifest.csv')
    return cfg


def test_fetch_is_chunked_by_year_and_keeps_only_ohlc_and_spread():
    mt5 = FakeMT5()
    df = B.fetch(mt5, 'GOLD', 'H1', 2020, 2022)
    assert [c[1] for c in mt5.calls] == [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)]
    assert [c[2] for c in mt5.calls] == [datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1)]
    assert list(df.columns) == ['time', 'open', 'high', 'low', 'close', 'spread_usd']
    assert np.allclose(df['spread_usd'], 0.25)


def test_labels_are_stored_unchanged():
    df = B.fetch(FakeMT5(), 'GOLD', 'H1', 2021, 2021)
    # the fake stamps 14:00 "UTC" on 2021-01-01: the label must stay 14:00, never shifted
    assert df['time'].iloc[0] == pd.Timestamp('2021-01-01 14:00', tz='UTC')


def test_fetch_all_writes_files_and_a_matching_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    B.fetch_all(FakeMT5(), cfg, now=datetime(2023, 6, 1, tzinfo=timezone.utc))
    with open(cfg['paths']['artifact_manifest'], encoding='utf-8') as fh:
        rows = {r['path']: r for r in csv.DictReader(fh)}
    assert len(rows) == 2
    for tf in ('H1', 'M15'):
        rel, path = B.mt5_path(tf, cfg)
        assert rows[rel]['sha256'] == B.sha256_file(path)
    _, h1 = B.mt5_path('H1', cfg)
    assert pd.read_csv(h1)['time'].str[:4].unique().tolist() == [str(y) for y in range(2014, 2024)]


def test_same_frame_gives_the_same_digest(tmp_path):
    df = B.fetch(FakeMT5(), 'GOLD', 'M15', 2022, 2022)
    assert B.save(df, str(tmp_path / 'a.csv.gz')) == B.save(df, str(tmp_path / 'b.csv.gz'))


def test_h1_join_keeps_every_cache_row():
    t = pd.date_range('2020-01-02 14:00', periods=5, freq='h', tz='UTC')
    cache = pd.DataFrame({'time': t, 'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0})
    spreads = pd.DataFrame({'time': t[[0, 2]], 'spread_usd': [0.25, 0.5]})
    out = B.load_h1(cache=cache, spreads=spreads)
    assert len(out) == 5 and out['time'].tolist() == t.tolist()
    assert out['spread_usd'].tolist() == [0.25, 0.0, 0.5, 0.0, 0.0]
