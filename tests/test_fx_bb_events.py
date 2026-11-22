"""fx-bb-reversion, group 3.1: bands, ATR, first breaches, cooldown, pair gaps and outcomes."""
import numpy as np
import pandas as pd
import pytest

from src.fx_bb import events as E
from src.fx_bb import load_config

CFG = load_config()


def d1_from_closes(closes, start='2013-01-01', dates=None):
    c = np.asarray(closes, dtype=float)
    idx = pd.DatetimeIndex(dates) if dates is not None else pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({'open': c, 'high': c * 1.001, 'low': c * 0.999, 'close_bid': c, 'close': c, 'n_h1': 24,
                         'spread_close': 0.0, 'spread_close_pips': 0.0, 'corrupt_spread': False},
                        index=pd.DatetimeIndex(idx, name='date'))


def quiet(n, seed=0, level=1.0, sd=0.0001):
    return level + np.random.default_rng(seed).normal(0, sd, n)


def test_upper_breach_is_direction_minus_one_and_a_second_day_outside_is_not_an_event():
    c = quiet(30)
    c[25], c[26] = 1.010, 1.012                  # two closes in a row above the upper band
    ind = E.indicators(d1_from_closes(c), CFG)
    assert ind['outside'].iloc[25] and ind['outside'].iloc[26]
    ev, cnt = E.detect(d1_from_closes(c), 'EURUSD', CFG)
    assert list(ev['k']) == [25] and ev['dir'].iloc[0] == -1 and ev['side'].iloc[0] == 1
    assert cnt['first_breaches'] == 1


def test_lower_breach_is_direction_plus_one():
    c = quiet(30)
    c[25] = 0.990
    ev, _ = E.detect(d1_from_closes(c), 'EURUSD', CFG)
    assert list(ev['k']) == [25] and ev['dir'].iloc[0] == 1


def test_re_breach_inside_the_cooldown_is_not_an_event():
    c = quiet(40)
    c[25], c[27] = 1.010, 1.010                  # inside on 26, first breach again on 27 (2 days later)
    ind = E.indicators(d1_from_closes(c), CFG)
    assert ind['first_breach'].iloc[25] and ind['first_breach'].iloc[27]
    ev, cnt = E.detect(d1_from_closes(c), 'EURUSD', CFG)
    assert list(ev['k']) == [25] and cnt['cooldown_removed'] == 1
    c[28], c[27] = 1.010, c[26]                 # a re-breach 3 days later is accepted
    ev, _ = E.detect(d1_from_closes(c), 'EURUSD', CFG)
    assert list(ev['k']) == [25, 28]


def test_hand_computed_outcome_of_plus_one_half():
    assert E.outcome_r(-1, 1.1000, 1.0960, 0.0080) == pytest.approx(0.5)
    c = quiet(30)
    c[25] = 1.010
    d1 = d1_from_closes(c)
    ev, _ = E.detect(d1, 'EURUSD', CFG)
    out = E.outcomes(ev, {'EURUSD': d1})
    expect = -1 * (c[28] - c[25]) / ev['atr'].iloc[0]
    assert out['r'].iloc[0] == pytest.approx(expect) and out['r'].iloc[0] > 0 and out['y'].iloc[0] == 1.0
    assert out['date_t3'].iloc[0] == d1.index[28]


def test_losing_event_has_target_zero_and_a_tie_has_none():
    ev = pd.DataFrame({'pair': ['EURUSD'] * 2, 'dir': [1, 1], 'close': [1.0, 1.0], 'atr': [0.01, 0.01],
                       'date_t3': pd.to_datetime(['2013-01-04', '2013-01-07'])})
    d1 = d1_from_closes([1.0, 1.0, 1.0, 0.997, 1.0], dates=pd.bdate_range('2013-01-01', periods=5))
    out = E.outcomes(ev, {'EURUSD': d1})
    assert out['r'].iloc[0] == pytest.approx(-0.3) and out['y'].iloc[0] == 0.0
    assert out['tie'].iloc[1] and np.isnan(out['y'].iloc[1])
    assert list(E.ties(ev, {'EURUSD': d1})) == [False, True]


def test_atr_is_the_mean_bid_true_range_of_t_minus_19_to_t():
    c = quiet(30, seed=4)
    d1 = d1_from_closes(c)
    ind = E.indicators(d1, CFG)
    prev = d1['close_bid'].shift(1)
    tr = np.maximum.reduce([d1['high'] - d1['low'], (d1['high'] - prev).abs(), (d1['low'] - prev).abs()])
    assert ind['atr'].iloc[25] == pytest.approx(np.mean(tr[6:26]))
    assert np.isnan(ind['atr'].iloc[19]) and np.isfinite(ind['atr'].iloc[20])
    assert ind['sd'].iloc[25] == pytest.approx(np.std(c[6:26], ddof=0))


def _six(dates, drop=None, seed=0):
    drop = drop or {}
    out = {}
    for i, p in enumerate(CFG['pairs']):
        c = quiet(len(dates), seed=seed + i)
        c[30] = 1.010
        d = d1_from_closes(c, dates=dates)
        out[p] = d.drop(pd.DatetimeIndex(drop.get(p, [])))
    return out


def test_gbpusd_gap_inside_the_window_excludes_the_event_and_a_common_holiday_excludes_nothing():
    dates = pd.bdate_range('2014-02-03', periods=45)
    holiday = dates[22]
    d1 = _six(dates, drop={p: [holiday] for p in CFG['pairs']})
    gap_days = dates[15:18]
    d1['GBPUSD'] = d1['GBPUSD'].drop(gap_days)
    assert holiday not in E.reference_calendar(d1, CFG)
    ev, _ = E.build_events(d1, CFG)
    spike = ev[ev['date'] == dates[30]]                     # the planted breach of every pair
    assert len(spike) == 6
    assert spike.loc[spike['pair'] == 'GBPUSD', 'reason'].iloc[0] == 'pair gap'
    assert set(spike.loc[spike['pair'] != 'GBPUSD', 'reason']) == {''}


def test_event_without_a_third_day_has_no_outcome():
    c = quiet(27)
    c[25] = 1.010
    ev, _ = E.build_events({p: d1_from_closes(c) for p in CFG['pairs']}, CFG)
    assert set(ev['reason']) == {'no outcome'}


def test_truncating_after_t_leaves_bands_atr_and_event_flag_unchanged():
    rng = np.random.default_rng(7)
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.006, 300)))
    d1 = d1_from_closes(c)
    full = E.indicators(d1, CFG)
    ev, _ = E.detect(d1, 'EURUSD', CFG)
    assert len(ev) >= 5
    for k in list(ev['k'])[:5] + [100, 200]:
        cut = E.indicators(d1.iloc[:k + 1], CFG)
        for col in ('sma', 'sd', 'upper', 'lower', 'atr', 'outside', 'first_breach'):
            assert cut[col].iloc[k] == full[col].iloc[k], (k, col)
        ev_cut, _ = E.detect(d1.iloc[:k + 1], 'EURUSD', CFG)
        assert (k in set(ev_cut['k'])) == (k in set(ev['k']))


def test_bid_variant_uses_bid_closes_for_the_bands():
    c = quiet(30)
    c[25] = 1.010
    d1 = d1_from_closes(c)
    d1['close'] = d1['close_bid'] + 0.0001
    d1.loc[d1.index[25], 'close_bid'] = c[24]           # the bid does not breach
    assert len(E.detect(d1, 'EURUSD', CFG, 'mid')[0]) == 1
    assert len(E.detect(d1, 'EURUSD', CFG, 'bid')[0]) == 0
