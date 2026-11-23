"""gold-bb-nyrange, tasks 3.1-3.2: Part 1a events, outcome and features."""
import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import bb_events as E
from src.gold_bbnr import bb_features as F
from src.gold_bbnr import load_config

CFG = load_config()


def d1_from(closes, start='2019-01-01', rng_w=2.0):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({'open': c, 'high': c + rng_w, 'low': c - rng_w, 'close': c, 'n_h1': 22, 'valid': True},
                        index=pd.DatetimeIndex(idx, name='date'))


def wiggle(n, seed=0, level=2000.0):
    rng = np.random.default_rng(seed)
    return level + np.cumsum(rng.normal(0, 3, n))


def test_upper_and_lower_breach_directions():
    c = list(wiggle(40))
    c_up = c + [c[-1] + 80, c[-1] + 80]
    ev, _ = E.detect(d1_from(c_up), CFG)
    assert ev['dir'].iloc[-1] == -1 and ev['date'].iloc[-1] == d1_from(c_up).index[40]
    c_dn = c + [c[-1] - 80]
    ev, _ = E.detect(d1_from(c_dn), CFG)
    assert ev['dir'].iloc[-1] == 1 and ev['side'].iloc[-1] == -1


def test_cooldown_removes_a_breach_two_days_later():
    c = list(wiggle(40))
    base = c[-1]
    c += [base + 80, base, base + 120]          # breach, back inside, breach again two days after the first
    ev, cnt = E.detect(d1_from(c), CFG)
    ev_all, _ = E.detect(d1_from(c), CFG, cooldown=False)
    assert cnt['cooldown_removed'] >= 1 and len(ev_all) == len(ev) + cnt['cooldown_removed']


def test_outcome_on_the_spec_example():
    assert E.outcome_r(-1, 2000.0, 1980.0, 20.0) == pytest.approx(1.0)


def test_detection_unchanged_by_truncation_and_no_outcome_at_the_end():
    c = list(wiggle(300, seed=3))
    d1 = d1_from(c)
    full, _ = E.detect(d1, CFG)
    for _, r in full.iterrows():
        cut, _ = E.detect(d1[d1.index <= r['date']], CFG)
        last = cut.iloc[-1]
        assert last['date'] == r['date'] and last['dir'] == r['dir'] and last['k'] == r['k']
    cut, _ = E.detect(d1[d1.index <= full['date'].iloc[-1]], CFG)
    assert cut['reason'].iloc[-1] == 'no outcome'


def dxy(dates, values=None, seed=1):
    v = values if values is not None else 95 + np.cumsum(np.random.default_rng(seed).normal(0, 0.3, len(dates)))
    return pd.DataFrame({'open': v, 'high': v, 'low': v, 'close': v}, index=pd.DatetimeIndex(dates, name='date'))


def test_features_columns_cut_and_previous_day_dxy():
    c = wiggle(400, seed=5)
    d1 = d1_from(c)
    dx = dxy(pd.bdate_range('2018-10-01', '2020-12-31'))
    ev, _ = E.detect(d1, CFG)
    f = F.with_features(ev, d1, dx, CFG)
    assert list(f[CFG['features_bb']['columns']].columns) == CFG['features_bb']['columns'] and len(CFG['features_bb']['columns']) == 11
    assert 'simultaneous_frac' not in f.columns
    assert (f['reason2'] == 'warm-up').sum() >= 1                    # the 60-day rank needs history
    r = f[f['reason2'] == ''].iloc[0]
    d1_cut = d1[d1.index <= r['date']]
    ev_cut, _ = E.detect(d1_cut, CFG)
    g = F.with_features(ev_cut, d1_cut, dx, CFG).iloc[-1]
    for col in CFG['features_bb']['columns']:
        assert g[col] == pytest.approx(r[col])
    spiked = dx.copy()
    spiked.loc[r['date'], 'close'] = 1e6                              # a huge DXY close on day t itself
    h = F.with_features(ev, d1, spiked, CFG).loc[r.name]
    for col in ('dxy_zscore_20d', 'dxy_ret_5d'):
        assert h[col] == pytest.approx(r[col])


def test_dxy_features_are_signed_by_direction():
    dates = pd.bdate_range('2019-01-01', periods=40)
    dx = dxy(dates, np.linspace(90, 100, 40))                          # rising dollar
    m = F.dxy_before([dates[-1]], dx, CFG)
    assert m['z'].iloc[0] > 0 and m['dxy_date'].iloc[0] == dates[-2]
