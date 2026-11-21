"""fx-fibonacci-mlp, Module 1c: Fischer's swing detector."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_fib import load_config
from src.fx_fib import mtf_aligner as AL
from src.fx_fib import swing_detector as SW
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()


def bars(rows):
    """(high, low, close) rows -> arrays, with a constant ATR of 1.0."""
    h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return h, l, c, np.ones(len(h))


# A clean up-leg from a low of 0.0 to a high of 3.5 at bar 3.
UP_LEG = [(1.0, 0.0, 0.5), (1.5, 0.8, 1.4), (2.5, 1.4, 2.4), (3.5, 2.4, 3.4)]


def _highs(ev):
    return ev[(ev['kind'] == 'H') & ~ev['replaces']]


# -- task 4.1: confirmation rules --------------------------------------------------

def test_close_below_the_high_bars_low_confirms():
    h, l, c, a = bars(UP_LEG + [(3.4, 3.0, 3.2), (3.3, 2.3, 2.35)])
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0)
    [row] = _highs(ev).itertuples(index=False)
    assert (row.extreme_idx, row.extreme_price, row.confirm_idx) == (3, 3.5, 5)


def test_counter_move_of_min_swing_confirms():
    h, l, c, a = bars(UP_LEG + [(3.4, 2.45, 3.3)])          # counter-move 1.05 >= 1.0, close above 2.4
    ev = SW.detect_swings(h, l, c, a, k_atr=1.0)
    assert list(_highs(ev)['confirm_idx']) == [4]


def test_retracement_of_38_2_percent_confirms():
    h, l, c, a = bars(UP_LEG + [(3.4, 2.1, 3.0)])           # retrace 1.4 / 3.5 = 40 %, counter-move < 3
    ev = SW.detect_swings(h, l, c, a, k_atr=3.0)
    assert list(_highs(ev)['confirm_idx']) == [4]
    h, l, c, a = bars(UP_LEG + [(3.4, 2.3, 3.0)])           # retrace 1.2 / 3.5 = 34 %
    assert _highs(SW.detect_swings(h, l, c, a, k_atr=3.0)).empty


def test_leg_too_small_is_never_confirmed():
    small = [(1.0, 0.0, 0.5), (1.2, 0.6, 1.1), (1.4, 0.9, 1.3), (1.5, 1.1, 1.45), (1.3, 0.2, 0.3), (0.9, 0.1, 0.2)]
    h, l, c, a = bars(small)
    assert _highs(SW.detect_swings(h, l, c, a, k_atr=2.0)).empty


def test_the_extreme_bar_never_confirms_itself():
    # Bar 3 makes the high AND closes below its own low's neighbourhood: an outside bar.
    rows = UP_LEG[:3] + [(3.5, 2.2, 2.3), (3.0, 2.5, 2.6), (2.9, 2.0, 2.1)]
    h, l, c, a = bars(rows)
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0)
    assert list(_highs(ev)['extreme_idx']) == [3]
    assert list(_highs(ev)['confirm_idx']) == [5]           # bar 5's close 2.1 < low[3] = 2.2


def test_swing_lows_mirror_swing_highs():
    h, l, c, a = bars(UP_LEG + [(3.4, 3.0, 3.2), (3.3, 2.3, 2.35)])
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0)
    mh, ml, mc, ma = -l, -h, -c, a
    mev = SW.detect_swings(mh, ml, mc, ma, k_atr=2.0)
    assert list(mev['kind']) == ['L' if k == 'H' else 'H' for k in ev['kind']]
    assert list(mev['confirm_idx']) == list(ev['confirm_idx'])


def test_a_new_extreme_moves_the_last_pivot_and_is_marked_as_a_replacement():
    rows = UP_LEG + [(3.4, 3.0, 3.2), (3.3, 2.3, 2.35), (4.0, 2.4, 3.9)]
    h, l, c, a = bars(rows)
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0)
    last = ev.iloc[-1]
    assert (last['kind'], last['extreme_idx'], last['confirm_idx'], bool(last['replaces'])) == ('H', 6, 6, True)
    assert SW.visible_pivots(ev, 5)[-1] == ('H', 3, 3.5)
    assert SW.visible_pivots(ev, 6)[-1] == ('H', 6, 4.0)


# -- task 4.2: 3-bar rule and per-timeframe thresholds -------------------------------

def test_adjacent_pivots_are_not_confirmed():
    rows = [(1.0, 0.0, 0.5), (4.0, 0.9, 3.9), (3.9, 1.0, 1.1), (3.8, 1.0, 1.2)]
    h, l, c, a = bars(rows)
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0, min_bars=CFG['swings']['min_bars_between_pivots'])
    assert 1 not in set(_highs(ev)['extreme_idx'])


def test_fast_v_turn_two_bars_apart_is_confirmed():
    # Rule B: low at bar 0, high at bar 2 -> the swing spans 3 bars counting both pivots.
    rows = [(1.0, 0.0, 0.5), (3.0, 0.9, 2.9), (4.0, 2.9, 3.9), (3.9, 1.0, 1.1), (3.8, 1.0, 1.2)]
    h, l, c, a = bars(rows)
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0, min_bars=CFG['swings']['min_bars_between_pivots'])
    assert list(_highs(ev)['extreme_idx']) == [2]


def test_stricter_spacing_would_have_dropped_the_v_turn():
    rows = [(1.0, 0.0, 0.5), (3.0, 0.9, 2.9), (4.0, 2.9, 3.9), (3.9, 1.0, 1.1), (3.8, 1.0, 1.2)]
    h, l, c, a = bars(rows)
    assert _highs(SW.detect_swings(h, l, c, a, k_atr=2.0, min_bars=3)).empty


def test_config_uses_rule_b():
    assert CFG['swings']['min_bars_between_pivots'] == 2


def test_changing_k_atr_d1_changes_only_the_d1_pivots():
    h1 = synthetic_weeks(date(2023, 1, 1), 40, seed=7, vol=0.0015)
    d1 = AL.build_d1(h1)
    cfg2 = copy.deepcopy(CFG)
    cfg2['swings']['k_atr_d1'] = 3.0
    pd.testing.assert_frame_equal(SW.swings_for(h1, 'H1', CFG), SW.swings_for(h1, 'H1', cfg2))
    assert not SW.swings_for(d1, 'D1', CFG).equals(SW.swings_for(d1, 'D1', cfg2))


# -- task 4.3: visibility, causality and the 15-bar rule -----------------------------

H1 = synthetic_weeks(date(2023, 1, 1), 26, seed=11, vol=0.0015)
EV = SW.swings_for(H1, 'H1', CFG)


def test_real_style_series_produces_alternating_confirmed_pivots():
    assert len(EV) > 20
    plain = [p[0] for p in SW.visible_pivots(EV, len(H1) - 1)]
    assert all(x != y for x, y in zip(plain, plain[1:]))


def test_a_pivot_is_invisible_before_its_confirmation_bar():
    hist = SW.pivot_history(EV, len(H1), depth=4)
    for e in EV[~EV['replaces']].itertuples(index=False):
        assert hist['idx'][e.confirm_idx, -1] == e.extreme_idx
        if e.confirm_idx > e.extreme_idx:
            for t in range(e.extreme_idx, e.confirm_idx):
                assert hist['idx'][t, -1] != e.extreme_idx or hist['kind'][t, -1] != (1 if e.kind == 'H' else -1)


def test_delayed_visibility_example():
    # The spec's example: a high made at bar 100 and confirmed at bar 104.
    pre = [(1.0 + 0.01 * (i % 2), 0.99, 1.0) for i in range(96)]          # quiet, tiny range
    leg = [(1.5, 1.0, 1.45), (2.5, 1.4, 2.45), (3.5, 2.4, 3.45), (4.5, 3.4, 4.45)]   # bars 96..99
    top = [(5.5, 4.4, 5.4)]                                                 # bar 100: the high
    after = [(5.4, 5.0, 5.2), (5.3, 4.9, 5.0), (5.2, 4.8, 4.9), (5.1, 4.2, 4.3)]     # bar 104 closes < 4.4
    h, l, c, a = bars(pre + leg + top + after)
    ev = SW.detect_swings(h, l, c, a, k_atr=2.0)
    [row] = _highs(ev).itertuples(index=False)
    assert (row.extreme_idx, row.confirm_idx) == (100, 104)
    hist = SW.pivot_history(ev, len(h))
    assert all(hist['idx'][t, -1] != 100 for t in range(100, 104))
    assert hist['idx'][104, -1] == 100


@pytest.mark.parametrize('seed', range(8))
def test_events_are_unchanged_by_truncation(seed):
    t = int(np.random.default_rng(seed).integers(200, len(H1)))
    cut = SW.swings_for(H1.iloc[:t + 1], 'H1', CFG)
    full = EV[EV['confirm_idx'] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(cut, full)


def test_pivot_history_keeps_the_newest_pivots_last():
    hist = SW.pivot_history(EV, len(H1), depth=4)
    t = len(H1) - 1
    vis = SW.visible_pivots(EV, t)[-4:]
    assert [int(i) for i in hist['idx'][t]] == [p[1] for p in vis]


def test_quiet_rule_switches_on_after_15_bars_and_off_at_the_next_pivot():
    n = 40
    rng = np.random.default_rng(0)
    high = 1.2 + rng.uniform(0, 0.01, n)
    low = 1.1 - rng.uniform(0, 0.01, n)
    last_pivot = np.full(n, 10)
    last_pivot[30:] = 28                                  # a new pivot (extreme 28) confirmed at bar 30
    q = SW.quiet_swing(high, low, last_pivot, quiet=15)
    assert not q.loc[24, 'active'] and q.loc[25, 'active']
    w = slice(11, 26)
    assert {q.loc[25, 'first_idx'], q.loc[25, 'second_idx']} == {11 + int(np.argmax(high[w])), 11 + int(np.argmin(low[w]))}
    assert q.loc[25, 'first_idx'] < q.loc[25, 'second_idx']
    assert not q.loc[30:, 'active'].any()


def test_quiet_rule_is_active_before_any_pivot_once_15_bars_exist():
    q = SW.quiet_swing(np.arange(20.0) + 1, np.arange(20.0), np.full(20, -1), quiet=15)
    assert not q.loc[13, 'active'] and q.loc[14, 'active']
