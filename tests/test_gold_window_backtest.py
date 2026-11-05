"""The descriptive backtest report (openspec change gold-window-strategy, task 5.1)."""
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from src import gold_window as W
from src.gold_window import backtest as BT
from src.gold_window import rules as R

CFG = W.load_config()


def _cands(days, gain=4.0, gate=lambda i: True):
    rows = []
    for i, d in enumerate(days):
        rows.append(['r', d, d + pd.Timedelta(hours=15), 2000.0, 0.25, 1, d + pd.Timedelta(hours=22),
                     2000.0 + (gain(i) if callable(gain) else gain), 0.25, 1.0, 0.5, bool(gate(i))])
    return pd.DataFrame(rows, columns=R.TRADE_COLUMNS)


def _h1(days):
    return pd.DataFrame([{'time': d + pd.Timedelta(hours=h), 'close': 2000.0 + h, 'spread_usd': 0.25}
                         for d in days for h in range(24)])


@pytest.fixture(scope='module')
def report():
    days = pd.bdate_range('2021-01-04', periods=300, tz='UTC')
    rng = np.random.default_rng(1)
    noisy = _cands(days, gain=lambda i: float(rng.normal(0, 5)), gate=lambda i: i % 2 == 0)
    res = [BT.evaluate('always_profitable', _cands(days), days, BT.always_long(_h1(days), days), CFG),
           BT.evaluate('noisy', noisy, days, BT.always_long(_h1(days), days), CFG)]
    return res, BT.render(res, CFG, {'inputs': 'synthetic'})


def test_every_section_carries_the_descriptive_label(report):
    _, text = report
    sections = text.split('\n## ')
    assert len(sections) == 4                 # preamble, summary, two rules
    for sec in sections:
        assert BT.LABEL in sec


def test_no_verdict_word_and_no_trade_instruction(report):
    _, text = report
    assert 'CLEARED' not in text and 'DROP' not in text
    assert not re.search(r'\b(buy|sell)\b', text, re.I)
    assert not re.search(r'(купи|продай)', text, re.I)


def test_always_profitable_rule_shows_a_positive_interval(report):
    res, _ = report
    n = res[0]['net']
    assert n['trades'] == 300 and n['ci_lo'] > 0 and n['total_pct'] > 0 and n['hit_pct'] == 100.0


def test_placebo_share_is_a_fraction(report):
    res, _ = report
    for r in res:
        assert 0.0 <= r['placebo_share'] <= 1.0
    # a gate that is open on every row cannot be beaten or lost by shifting it
    assert res[0]['placebo_share'] == 1.0


def test_gate_always_open_comparator_uses_every_candidate(report):
    res, _ = report
    assert res[1]['ungated']['trades'] == 300 and res[1]['net']['trades'] == 150


def test_rule_days_start_after_the_gate_warm_up():
    days = pd.bdate_range('2021-01-04', periods=10, tz='UTC')
    c = _cands(days)
    c.loc[:4, 'threshold'] = np.nan
    assert list(BT.rule_days(c, days)) == list(days[5:])


def test_backtest_writes_no_registry():
    src = inspect.getsource(BT)
    assert 'hypothesis_log' not in src and 'PRE_REGISTRATION' not in src
    assert all('hypothesis_log' not in v for v in CFG['paths'].values())
