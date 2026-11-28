"""rts-signals task 4.3: the full-rules replay on synthetic paths, and the reconciliation labels."""

import numpy as np
import pandas as pd
import pytest

from src.rts_signals import load_config
from src.rts_signals.reconcile import check_trade
from src.rts_signals.replay import replay_session
from src.rts_signals.rules import load_rules
from tests.test_rts_signals_simulate import bars, rec

CFG = load_config()


def path(*legs, n=40):
    """Piecewise-linear price path through the given points."""
    out = [np.linspace(a, b, n, endpoint=False) for a, b in zip(legs[:-1], legs[1:])]
    return np.r_[np.concatenate(out), legs[-1]]


def rules_of(trades):
    return [(t['rule'], t['side'], t['exit_reason']) for t in trades]


def test_rule2_after_a_target():
    r = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900)
    tr = replay_session(CFG, r, bars('2026-11-24', path(1.0900, 1.0866, 1.0826, 1.0760), 1.0900))
    assert rules_of(tr)[:2] == [(0, 'sell', 'target'), (2, 'sell', 'target')]
    assert tr[1]['entry'] == pytest.approx(1.0826) and tr[1]['exit'] == pytest.approx(1.0766)


def test_no_rule2_on_friday():
    r = rec('SELL', 1.0900, 1.0930, 1.0866, 1.0900, date='2026-11-19')
    tr = replay_session(CFG, r, bars('2026-11-19', path(1.0900, 1.0866, 1.0826, 1.0760), 1.0900))
    assert rules_of(tr) == [(0, 'sell', 'target')]


def test_rule3_fallback_and_three_loss_cap():
    r = rec('BUY', 1.0900, 1.0890, 1.0950, 1.0900)        # 10-point stop -> Rule 3 uses 38 / 44
    p = path(1.0900, 1.0889, 1.0900, 1.0861, 1.0900, 1.0861, 1.0900, 1.0861, 1.0900, 1.0895)
    tr = replay_session(CFG, r, bars('2026-11-24', p, 1.0900))
    r3 = [t for t in tr if t['rule'] == 3]
    assert len(r3) == 2 and all(t['exit_reason'] == 'stop' for t in r3)
    assert r3[0]['exit'] == pytest.approx(1.0900 - 0.0038)


def test_rule4_reversal():
    r = rec('BUY', 1.0900, 1.0860, 1.0950, 1.0900)
    tr = replay_session(CFG, r, bars('2026-11-24', path(1.0900, 1.0860, 1.0780, 1.0725), 1.0900))
    assert rules_of(tr) == [(0, 'buy', 'stop'), (4, 'sell', 'target')]
    assert tr[1]['entry'] == pytest.approx(1.0780) and tr[1]['exit'] == pytest.approx(1.0732)


def test_rule5_opposite_gap():
    r = rec('BUY', 1.0900, 1.0860, 1.0950, 1.0900)
    tr = replay_session(CFG, r, bars('2026-11-24', path(1.0855, 1.0835, 1.0780), 1.0900))
    assert rules_of(tr) == [(5, 'sell', 'target')]
    assert tr[0]['entry'] == pytest.approx(1.0835) and tr[0]['exit'] == pytest.approx(1.0787)


def test_npr_direction_trades_nothing():
    r = dict(rec('BUY', 1.0900, 1.0860, 1.0950, 1.0900), direction='NPR')
    assert replay_session(CFG, r, bars('2026-11-24', path(1.0900, 1.0700), 1.0900)) == []


def test_reconciliation_labels():
    rules = load_rules()
    b = bars('2026-11-22', path(1.3081, 1.3020, 1.3050), 1.3081)     # session of trade date Mon 2026-11-23
    base = {'pair': 'POUND', 'entry_date': '2026-11-23', 'exit_date': '2026-11-23', 'side': 'sell', 'rule': '0'}
    ok = check_trade(CFG, dict(base, entry='1.3081', exit='1.3024', points='57'), b, rules)
    assert ok['label'] == 'confirmed'
    far = check_trade(CFG, dict(base, entry='1.3200', exit='1.3150', points='50'), b, rules)
    assert far['label'] == 'not reproducible'
    diff = check_trade(CFG, dict(base, entry='1.3081', exit='1.2950', points='131'), b, rules)
    assert diff['label'] == 'different exit'
