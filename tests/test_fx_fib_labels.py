"""fx-fibonacci-mlp, group 6: triple-barrier labels (spec fx-fibonacci/barrier-labels)."""
import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_fib import labeling as LB
from src.fx_fib import load_config
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()


def tb(rows, atr=1.0, m=1.0, H=3):
    """rows: (high, low, close); bar 0 is the labelled bar, close 10 => barriers 11 / 9."""
    h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return LB.triple_barrier(h, l, c, np.full(len(c), atr), m, H)


def test_upper_first():
    out = tb([(10, 10, 10), (10.5, 9.5, 10), (11.2, 10, 11), (10, 8, 9), (10, 10, 10)])
    assert out.loc[0, 'label'] == 1 and out.loc[0, 'exit_offset'] == 2


def test_lower_first():
    out = tb([(10, 10, 10), (10.2, 8.9, 9), (11.5, 9, 11), (10, 10, 10), (10, 10, 10)])
    assert out.loc[0, 'label'] == 2 and out.loc[0, 'exit_offset'] == 1


def test_time_runs_out():
    out = tb([(10, 10, 10), (10.5, 9.5, 10), (10.9, 9.1, 10), (10.5, 9.5, 10), (12, 8, 10)])
    assert out.loc[0, 'label'] == 0 and out.loc[0, 'exit_offset'] == 3   # bar 4 is outside H = 3


def test_both_barriers_in_one_bar_is_a_neutral_tie():
    out = tb([(10, 10, 10), (10.5, 9.5, 10), (11.5, 8.5, 10), (10, 10, 10), (10, 10, 10)])
    assert out.loc[0, 'label'] == 0 and bool(out.loc[0, 'tie'])
    s = LB.label_summary(out.set_index(pd.date_range('2018-01-01', periods=len(out), freq='h', tz='UTC')),
                         pd.Timestamp('2030-01-01', tz='UTC'))
    assert s['ties'] >= 1


def test_the_labelled_bars_own_range_never_counts():
    out = tb([(12, 8, 10), (10.5, 9.5, 10), (10.5, 9.5, 10), (10.5, 9.5, 10), (10, 10, 10)])
    assert out.loc[0, 'label'] == 0


def test_newest_bars_get_no_label():
    out = tb([(10, 10, 10)] * 8, H=3)
    assert out['label'].iloc[:5].notna().all() and out['label'].iloc[5:].isna().all()


def test_missing_atr_gives_no_label():
    h = np.full(6, 10.0)
    out = LB.triple_barrier(h, h, h, np.array([np.nan, 1, 1, 1, 1, 1]), 1.0, 2)
    assert np.isnan(out.loc[0, 'label']) and out.loc[1, 'label'] == 0


def test_weekend_inside_the_horizon_counts_bars_not_hours():
    h1 = synthetic_weeks(date(2024, 1, 7), 5, seed=5)
    lab = LB.build_labels(h1, CFG)
    fri = h1.index.get_loc(pd.Timestamp('2024-02-02 20:00', tz='UTC'))
    H = CFG['labels']['horizon_bars']
    window = h1.iloc[fri + 1:fri + 1 + H]
    assert window.index[-1].dayofweek == 0                       # the horizon reaches into Monday
    c = h1['close'].iloc[fri]
    a = LB.atr_d1_at(h1)[fri]
    assert np.isfinite(a)
    up = (window['high'] >= c + a).to_numpy(); dn = (window['low'] <= c - a).to_numpy()
    fu = up.argmax() if up.any() else 99
    fd = dn.argmax() if dn.any() else 99
    expect = 1 if fu < fd else 2 if fd < fu else 0
    assert lab['label'].iloc[fri] == expect


def test_labels_use_the_aligned_daily_atr():
    h1 = synthetic_weeks(date(2024, 1, 7), 6, seed=6)
    lab = LB.build_labels(h1, CFG)
    a = LB.atr_d1_at(h1)
    ok = np.isfinite(a)
    assert np.allclose(lab['upper'].to_numpy()[ok], (h1['close'] + a).to_numpy()[ok])


def test_label_parameters_come_from_config():
    h1 = synthetic_weeks(date(2024, 1, 7), 6, seed=6)
    cfg = copy.deepcopy(CFG)
    cfg['labels']['horizon_bars'] = 5
    assert LB.build_labels(h1, cfg)['label'].notna().sum() > LB.build_labels(h1, CFG)['label'].notna().sum()


def test_uniqueness_of_fully_overlapping_time_outs_is_about_one_over_h():
    n, H = 200, 24
    flat = np.full(n, 1.0)
    out = LB.triple_barrier(flat, flat, flat, np.full(n, 1.0), 1.0, H)
    u = LB.mean_uniqueness(out)
    assert 1 / H <= u < 3 / H


def test_uniqueness_of_immediate_exits_is_one():
    rows = 60
    close = np.arange(rows, dtype=float)
    out = LB.triple_barrier(close + 5, close, close, np.full(rows, 1.0), 1.0, 4)
    assert LB.mean_uniqueness(out) == pytest.approx(1.0)


def test_report_has_the_gross_header_and_balance_check(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['results'] = str(tmp_path)
    cfg['folds']['first_eval_start'] = '2024-03-01'
    frames = {'EURUSD': synthetic_weeks(date(2024, 1, 7), 12, seed=8),
              'GBPUSD': synthetic_weeks(date(2024, 1, 7), 12, seed=9)}
    import src.fx_fib.data_loader as DL
    monkeypatch.setattr(DL, 'load_h1', lambda pair, cfg=None: frames[pair])
    text = open(LB.write_report(cfg, 'T'), encoding='utf-8').read()
    assert LB.GROSS_HEADER in text and '## Class-balance check' in text
    assert '| EURUSD |' in text and '| GBPUSD |' in text and '_Generated T._' in text
