"""gold-bb-nyrange, task 5.1: Part 2 target, eligibility and the fourteen features."""
import numpy as np
import pandas as pd
import pytest

from src.gold_bbnr import bars as B
from src.gold_bbnr import ny_features as NF
from src.gold_bbnr import ny_target as NT
from src.gold_bbnr import load_config

CFG = load_config()


def gold_h1(n_days=80, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2019-01-02', periods=n_days)
    rows = []
    lvl = 1300.0
    for d in days:
        for h in range(0, 23):
            step = rng.normal(0, 1.5)
            o, c = lvl, lvl + step
            rows.append((d + pd.Timedelta(hours=h), o, max(o, c) + abs(rng.normal(0, 1)), min(o, c) - abs(rng.normal(0, 1)), c))
            lvl = c
    df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close']).set_index('time')
    df.index = df.index.tz_localize('UTC')
    df[['open', 'high', 'low', 'close']] *= scale
    return df


def dxy_h1(h1, seed=1, drop_day=None):
    rng = np.random.default_rng(seed)
    idx = h1.index[(h1.index.hour >= 2)]
    v = 95 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    d = pd.DataFrame({'open': v, 'high': v + 0.03, 'low': v - 0.03, 'close': v}, index=idx)
    if drop_day is not None:
        d = d[~((d.index.normalize() == pd.Timestamp(drop_day, tz='UTC')) & (d.index.hour >= 8))]
    return d


def build(h1, dx):
    return NF.dataset(h1, B.d1_ny(h1, CFG), dx, CFG)


def test_missing_21h_bar_makes_the_day_ineligible():
    h1 = gold_h1()
    day = pd.Timestamp('2019-03-05', tz='UTC')
    h1 = h1[~(h1.index == day + pd.Timedelta(hours=21))]
    ds = build(h1, dxy_h1(h1))
    assert ds.loc[day, 'reason'] == 'incomplete session'
    assert ds['reason'].eq('').sum() > 20


def test_doubling_prices_changes_nothing():
    a = build(gold_h1(), dxy_h1(gold_h1()))
    b = build(gold_h1(scale=2.0), dxy_h1(gold_h1()))
    cols = CFG['features_ny']['columns'] + ['y']
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False, rtol=1e-9)


def test_features_identical_on_data_cut_after_14():
    h1 = gold_h1()
    dx = dxy_h1(h1)
    full = build(h1, dx)
    day = full.index[full['reason'] == ''][-1]
    cut_h1 = h1[h1.index <= day + pd.Timedelta(hours=14)]
    cut_dx = dx[dx.index <= day + pd.Timedelta(hours=14)]
    g = NT.grid(cut_h1, full.index[full.index <= day])
    y_full = full['y'].where(full['reason'].isin(['', 'warm-up']))
    reason = NT.eligibility(NT.grid(h1, full.index), full.index, B.d1_ny(h1, CFG), CFG)
    y_avail = full['y'].where(reason == '')
    f_cut = NF.build_features(g, full.index[full.index <= day], y_avail[y_avail.index < day], cut_dx, CFG)
    for c in CFG['features_ny']['columns']:
        assert f_cut.loc[day, c] == pytest.approx(full.loc[day, c], nan_ok=True), c
    assert y_full.notna().any()


def test_dxy_day_with_few_bars_is_warm_up():
    h1 = gold_h1()
    day = '2019-04-10'
    ds = build(h1, dxy_h1(h1, drop_day=day))           # only 6 DXY bars (02:00-07:00) on that day
    assert ds.loc[pd.Timestamp(day, tz='UTC'), 'reason'] == 'warm-up'
    assert np.isnan(ds.loc[pd.Timestamp(day, tz='UTC'), 'dollar_pre_range_ratio'])


def test_label_mode_reads_no_target_and_has_the_same_reasons(monkeypatch):
    h1 = gold_h1()
    dx = dxy_h1(h1)
    full = build(h1, dx)

    def boom(*a, **k):
        raise AssertionError('the target must not be read')
    monkeypatch.setattr(NT, 'target', boom)
    lab = NF.dataset(h1, B.d1_ny(h1, CFG), dx, CFG, read_target=False)
    assert (lab['reason'] == full['reason']).all() and lab['y'].isna().all()
