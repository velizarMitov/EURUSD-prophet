"""rts-signals task 4.1: append-only M1 caches and a market-data-only feed."""

import os
import re

import numpy as np
import pandas as pd
import pytest

from src.rts_signals import REPO, load_config
from src.rts_signals.bars import COLS, load, top_up


class FakeFeed:
    def __init__(self, start, n):
        idx = pd.date_range(start, periods=n, freq='1min', tz='UTC', name='time')
        self.df = pd.DataFrame({'open': 1.1, 'high': 1.1002, 'low': 1.0998, 'close': 1.1001, 'spread': 6.0},
                               index=idx)

    def bars(self, symbol, start, end):
        return self.df[(self.df.index >= start) & (self.df.index < end)]


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = dict(c['paths'], artifacts=str(tmp_path / 'art'), results=str(tmp_path / 'res'))
    c['mt5']['history_start'] = '2026-11-02'
    return c


def test_repeated_top_up_stores_each_bar_once(cfg):
    feed = FakeFeed('2026-11-02 00:00', 300)
    now = pd.Timestamp('2026-11-02 02:00', tz='Europe/Berlin').tz_convert('UTC')   # label 02:00
    a = top_up(cfg, feed, now)
    assert set(a.values()) == {120}                      # minutes 00:00 .. 01:59 are closed
    b = top_up(cfg, feed, now)
    assert set(b.values()) == {0}
    c = top_up(cfg, feed, pd.Timestamp('2026-11-02 03:00', tz='Europe/Berlin').tz_convert('UTC'))
    assert set(c.values()) == {60}
    d = load(cfg, 'USDJPY')
    assert len(d) == 180 and d.index.is_unique and list(d.columns) == COLS
    assert np.isclose(d['spread'].iloc[0], 6.0)


def test_source_has_no_order_calls():
    for name in os.listdir(os.path.join(REPO, 'src', 'rts_signals')):
        if name.endswith('.py'):
            with open(os.path.join(REPO, 'src', 'rts_signals', name), encoding='utf-8') as fh:
                src = fh.read()
            for bad in ('order_send', 'order_check', 'positions_get', 'positions_total', 'orders_get',
                        'order_calc', 'TRADE_ACTION'):
                assert not re.search(rf'\b{bad}\b', src), (name, bad)
