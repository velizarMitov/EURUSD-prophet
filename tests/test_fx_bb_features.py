"""fx-bb-reversion, group 4: the ten oriented, causal, scale-free Cell 2 features."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.fx_bb import events as E
from src.fx_bb import features as F
from src.fx_bb import load_config
from tests.test_fx_bb_events import d1_from_closes

CFG = load_config()
COLS = CFG['features']['columns']


def walk(n=400, seed=0, level=1.1, start='2013-01-01'):
    rng = np.random.default_rng(seed)
    c = level * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    d = d1_from_closes(c, start=start)
    wig = np.random.default_rng(seed + 50).uniform(0.001, 0.004, n)
    d['high'], d['low'] = c * (1 + wig), c * (1 - wig[::-1])
    return d


def six(n=400, seed=0):
    return {p: walk(n, seed + i) for i, p in enumerate(CFG['pairs'])}


def test_hand_computed_zscore():
    c = np.r_[np.full(10, 1.098), np.full(10, 1.102)]          # SMA 1.1000, population SD 0.0020
    d = d1_from_closes(np.r_[c, 1.1050], start='2013-01-01')
    b = F.pair_frame(d, CFG)
    sma = np.mean(np.r_[c[1:], 1.1050])
    sd = np.std(np.r_[c[1:], 1.1050], ddof=0)
    assert b['zabs'].iloc[-1] == pytest.approx(abs(1.1050 - sma) / sd)
    two = d1_from_closes(np.r_[np.full(10, 1.098), np.full(10, 1.102)])
    assert F.pair_frame(two, CFG)['zabs'].iloc[-1] == pytest.approx(abs(1.102 - 1.1) / 0.002)


def test_assemble_raises_on_extra_or_missing_columns():
    f = pd.DataFrame(np.zeros((2, len(COLS))), columns=COLS)
    assert list(F.assemble(f, CFG).columns) == COLS
    with pytest.raises(F.FeatureColumnsError, match='pair_id'):
        F.assemble(f.assign(pair_id=1.0), CFG)
    with pytest.raises(F.FeatureColumnsError, match='simultaneous_frac'):
        F.assemble(f.drop(columns='simultaneous_frac'), CFG)


def _mirror(d, level=2.2):
    m = d.copy()
    for col in ('open', 'close_bid', 'close'):
        m[col] = level - d[col]
    m['high'], m['low'] = level - d['low'], level - d['high']
    return m


def test_mirrored_path_keeps_the_oriented_features_and_flips_the_side():
    d = walk(300, seed=3)
    m = _mirror(d)
    ev, _ = E.detect(d, 'EURUSD', CFG)
    evm, _ = E.detect(m, 'EURUSD', CFG)
    assert list(ev['k']) == list(evm['k']) and (ev['side'].to_numpy() == -evm['side'].to_numpy()).all()
    a = F.build_features(ev, {**six(300), 'EURUSD': d}, CFG)
    b = F.build_features(evm, {**six(300), 'EURUSD': m}, CFG)
    keep = ['rsi_14_dist', 'close_pos_in_bar', 'ret_3d_atr', 'trend_ma50_alignment', 'bb_zscore_abs',
            'bar_range_vs_atr20', 'atr_ratio_20_60']
    ok = a[keep].notna().all(axis=1)
    assert ok.sum() >= 3
    np.testing.assert_allclose(a.loc[ok, keep].to_numpy(), b.loc[ok, keep].to_numpy(), rtol=1e-9, atol=1e-9)
    assert (a['breach_side'] == -b['breach_side']).all()


def test_close_pos_in_bar_is_one_at_the_extreme():
    c = np.r_[np.full(19, 1.0) + np.random.default_rng(1).normal(0, 1e-4, 19), 0.99]
    d = d1_from_closes(np.r_[np.full(5, 1.0), c])
    d.loc[d.index[-1], 'low'] = 0.99                         # lower-band breach closing at the bid low
    ev, _ = E.detect(d, 'EURUSD', CFG)
    assert len(ev) == 1 and ev['side'].iloc[0] == -1
    f = F.build_features(ev, {p: d for p in CFG['pairs']}, CFG)
    assert f['close_pos_in_bar'].iloc[0] == pytest.approx(1.0)


def test_simultaneous_frac_on_a_five_pair_day():
    n = 60
    base = {p: np.full(n, 1.0) + np.random.default_rng(i).normal(0, 1e-4, n) for i, p in enumerate(CFG['pairs'])}
    for p in ('EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'):       # EURUSD's event, 3 of the 4 others also breach
        base[p][50] = 1.01
    d1 = {p: d1_from_closes(v, start='2014-01-01') for p, v in base.items()}
    cfg = copy.deepcopy(CFG)
    cfg['data']['start']['USDCHF'] = '2015-02-01'          # not started yet: five pairs
    ev, _ = E.detect(d1['EURUSD'], 'EURUSD', cfg)
    ev = ev[ev['k'] == 50]                                  # the planted dollar-shock day
    f = F.build_features(ev, d1, cfg)
    assert len(f) == 1 and f['simultaneous_frac'].iloc[0] == pytest.approx(0.75)


def test_scale_test_prices_times_100():
    d1 = six()
    big = {p: d.assign(**{c: d[c] * 100 for c in ('open', 'high', 'low', 'close_bid', 'close')}) for p, d in d1.items()}
    ev, _ = E.build_events(d1, CFG)
    evb, _ = E.build_events(big, CFG)
    assert len(ev) == len(evb) and len(ev) > 20
    a, b = F.build_features(ev, d1, CFG), F.build_features(evb, big, CFG)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12)


def test_warm_up_excludes_from_cell_2_only():
    d1 = six()
    ev, _ = E.build_events(d1, CFG)
    out = F.with_features(ev, d1, CFG)
    warm = out['reason2'] == 'warm-up'
    assert warm.any() and (out.loc[warm, 'reason'] == '').all()       # still Cell 1 events
    assert out.loc[warm, 'date'].max() < d1['EURUSD'].index[90]
    assert out.loc[out['reason2'] == '', COLS].notna().all().all()


def test_rsi_is_wilder_and_bounded():
    d = walk(200, seed=9)
    rsi = F.wilder_rsi(d['close'], 14)
    assert rsi.iloc[:14].isna().all() and rsi.dropna().between(0, 100).all()
    up = F.wilder_rsi(pd.Series(np.linspace(1, 2, 40)), 14)
    assert up.iloc[-1] == 100.0
