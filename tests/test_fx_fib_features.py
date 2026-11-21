"""fx-fibonacci-mlp, group 5: structure features (spec fx-fibonacci/structure-features)."""
import copy
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.fx_fib import features as FE
from src.fx_fib import fib_features as FF
from src.fx_fib import load_config
from src.fx_fib import spiral_engine as SP
from src.fx_fib import tgd as TG
from tests.fx_fib_synth import synthetic_weeks

CFG = load_config()
PAIRS = {'A': synthetic_weeks(date(2023, 1, 1), 40, seed=21, vol=0.0015),
         'B': synthetic_weeks(date(2023, 1, 1), 40, seed=22, vol=0.0015)}
FEATS = {k: FE.build_features(v, CFG) for k, v in PAIRS.items()}


def _piv(prices, idx=None):
    p = np.array([prices], dtype=float)
    i = np.array([idx if idx is not None else list(range(len(prices)))])
    return p, i


# -- 5.1 Fibonacci level features --------------------------------------------------

def test_retracement_worked_example():
    price, idx = _piv([1.0950, 1.1050, 1.1000, 1.1100])
    f, _ = FF.leg_features(price, idx, np.array([1.1050]), np.array([0.0050]), CFG, 'd1')
    assert f['d1_retracement_pct'][0] == pytest.approx(0.5)
    assert f['d1_dist_to_fib_500'][0] == pytest.approx(0.0, abs=1e-12)
    assert f['d1_swing_amp_atr'][0] == pytest.approx(2.0)
    assert f['d1_dist_to_fib_618'][0] == pytest.approx((1.1050 - (1.1100 - 0.618 * 0.01)) / 0.005)


def test_extension_and_target2_levels():
    price, idx = _piv([1.0, 1.2, 1.1, 1.3])
    f, lv = FF.leg_features(price, idx, np.array([1.3]), np.array([0.1]), CFG, 'h1')
    ext = 1.3 + 1.618 * (1.3 - 1.1)
    t2 = 1.3 + 0.618 * (1.3 - 1.0)
    assert lv['ext'][0] == pytest.approx(ext)
    assert f['h1_dist_to_ext_1618_w3'][0] == pytest.approx((1.3 - ext) / 0.1)
    assert f['h1_dist_to_target2_0618'][0] == pytest.approx((1.3 - t2) / 0.1)
    assert f['h1_price_band_width_atr'][0] == pytest.approx(abs(ext - t2) / 0.1)


def test_swing_bars_of_a_fast_v_turn():
    price, idx = _piv([1.13, 1.11, 1.11122, 1.12691], [3040, 3048, 3055, 3057])
    f, _ = FF.leg_features(price, idx, np.array([1.12]), np.array([0.006]), CFG, 'd1')
    assert f['d1_swing_bars'][0] == 2


def test_rows_with_fewer_than_four_pivots_are_warm_up():
    price = np.array([[np.nan, 1.0, 1.2, 1.1]])
    idx = np.array([[-1, 0, 5, 9]])
    f, _ = FF.leg_features(price, idx, np.array([1.15]), np.array([0.1]), CFG, 'h1')
    assert np.isnan(f['h1_dist_to_target2_0618'][0]) and np.isfinite(f['h1_retracement_pct'][0])
    assert not FE.valid_rows(pd.DataFrame({'a': [f['h1_dist_to_target2_0618'][0]]})).iloc[0]


# -- 5.2 breakouts, alignment, risk ----------------------------------------------------

def test_one_bar_breakout_up_and_down():
    high = np.array([1.0, 1.1, 1.2, 1.05])
    low = np.array([0.9, 1.0, 1.1, 0.95])
    close = np.array([0.95, 1.15, 1.25, 0.98])
    b = FF.breakouts(high, low, close)
    assert np.isnan(b['h1_breakout_1'][0])
    assert list(b['h1_breakout_1'][1:]) == [1.0, 1.0, -1.0]
    assert np.isnan(b['h1_breakout_4'][3])


def test_alignment_flag_is_zero_outside_both_zones_and_follows_the_breakout_inside():
    out = FF.alignment_flag(np.array([1.0, 1.0, -1.0]), np.array([1.10, 1.1001, 1.2001]),
                            np.array([1.15, 1.10, 1.15]), np.array([1.25, 1.25, 1.20]),
                            np.array([0.01, 0.01, 0.01]), 0.25)
    assert list(out) == [0.0, 1.0, -1.0]


def test_rr_divisor_floor():
    n = 5
    high = np.full(n, 1.12); low = np.array([1.11, 1.105, 1.10, 1.108, 1.10])
    close = np.array([1.11, 1.11, 1.11, 1.11, 1.10])           # last close equals the 5-bar low
    level50 = np.full(n, 1.12)                                  # target above -> risk to the low
    r = FF.risk_features(high, low, close, np.full(n, 0.01), level50, CFG)
    assert r['rr_to_50pct_target'][-1] == pytest.approx(0.02 / (0.1 * 0.01))
    assert r['price_square_low_atr'][-1] == 0.0


def test_close_vs_extreme_bars_uses_the_newest_leg():
    kind = np.array([[0, 0, -1, 1]]); idx = np.array([[-1, -1, 1, 3]])
    high = np.array([1.0, 1.01, 1.05, 1.20, 1.15]); low = np.array([0.99, 0.95, 1.0, 1.10, 1.12])
    out = FF.close_vs_extreme_bars(kind, idx, high, low, np.array([1.13]), np.array([0.05]))
    assert out['h1_close_vs_high_bar_low'][0] == pytest.approx((1.13 - 1.10) / 0.05)
    assert out['h1_close_vs_low_bar_high'][0] == pytest.approx((1.13 - 1.01) / 0.05)


# -- 5.3 TGD -------------------------------------------------------------------------------

def test_tgd_projection_arithmetic():
    kind = np.array([[1, -1, 1, -1]]); idx = np.array([[100, 104, 110, 113]])
    proj = TG.projections(kind, idx, np.array([False]), np.array([-1.0]), np.array([-1.0]), 1.618)
    assert proj[0, 0] == 126                       # 110 + 1.618 * 10 = 126.18
    assert proj[0, 1] == np.rint(113 + 1.618 * 9)


def test_quiet_rule_replaces_the_pivots_for_tgd():
    kind = np.array([[1, -1, 1, -1]]); idx = np.array([[100, 104, 110, 113]])
    proj = TG.projections(kind, idx, np.array([True]), np.array([120.0]), np.array([125.0]), 1.618)
    assert proj[0, 0] == np.rint(125 + 1.618 * 5) and np.isnan(proj[0, 1])


def test_confluence_is_one_at_the_level_on_the_tgd_day():
    proj = np.array([[126.0, np.nan]])
    f = TG.tgd_features(proj, np.array([126.0]), np.array([0.0]), np.array([3.0]), CFG)
    assert f['d1_bars_to_nearest_tgd'][0] == 0 and f['price_time_confluence'][0] == pytest.approx(1.0)
    assert f['d1_tgd_cluster_count'][0] == 1


def test_no_active_tgd_uses_the_cap_and_zero_confluence():
    f = TG.tgd_features(np.array([[100.0, np.nan]]), np.array([120.0]), np.array([0.0]), np.array([0.0]), CFG)
    assert f['d1_bars_to_nearest_tgd'][0] == CFG['tgd']['no_tgd_days'] and f['price_time_confluence'][0] == 0


# -- 5.4 spiral --------------------------------------------------------------------------

@pytest.mark.parametrize('k', [-0.306349, 0.0, float('nan')])
def test_non_positive_spiral_constant_is_refused(k):
    with pytest.raises(ValueError, match='0.306349'):
        SP.check_k(k)


def test_default_and_alternative_constants():
    assert SP.K_DEFAULT == pytest.approx(0.306349, abs=1e-6)
    assert SP.K_ALT == pytest.approx(0.0766, abs=1e-4)


def test_known_geometry_distances():
    # Start A at (-1, 0) around B at the origin: R0 = 1, theta0 = pi. CCW ring 2 crosses x = 0 at
    # phi = 5 pi / 2 (y = -e^{k 5pi/2}) and phi = 7 pi / 2 (y = +e^{k 7pi/2}).
    k = SP.K_DEFAULT
    xs, ys, rs = SP.sample_rings(1.0, math.pi, k, +1, [2], 720)
    lower = -math.exp(k * 5 * math.pi / 2)
    signed, ring = SP.nearest_branch(np.array([0.0]), np.array([-10.0]), xs, ys, rs)
    assert signed[0] == pytest.approx(-10.0 - lower, abs=0.02) and ring[0] == 2
    upper = math.exp(k * 7 * math.pi / 2)
    signed, _ = SP.nearest_branch(np.array([0.0]), np.array([28.0]), xs, ys, rs)
    assert signed[0] == pytest.approx(28.0 - upper, abs=0.05)


def test_no_crossing_branch_gives_ring_zero():
    xs, ys, rs = SP.sample_rings(1.0, math.pi, SP.K_DEFAULT, +1, [2], 720)
    signed, ring = SP.nearest_branch(np.array([1e6]), np.array([0.0]), xs, ys, rs)
    assert np.isnan(signed[0]) and ring[0] == 0


def test_both_spiral_sets_are_built_and_not_degenerate():
    f = FEATS['A'][FE.valid_rows(FEATS['A'])]
    for name in ('spiral306', 'spiral077'):
        d = f[f'{name}_dist_ccw_atr']
        assert (d < CFG['spiral']['distance_cap_atr']).mean() > 0.5, name
        assert f[f'{name}_ring_id_ccw'].nunique() > 1, name
    assert set(f['spiral306_ring_id_ccw'].unique()) <= {0, 1, 2}
    assert set(f['spiral077_ring_id_cw'].unique()) <= {0, 2, 3, 4}


def test_non_positive_k_in_a_variant_stops_the_build():
    cfg = copy.deepcopy(CFG)
    cfg['spiral']['variants'][1]['k'] = -0.481212
    with pytest.raises(ValueError, match='0.306349'):
        FE.build_features(PAIRS['A'], cfg)


def test_same_normalised_geometry_gives_identical_spiral_features():
    h1 = PAIRS['A']
    scaled = h1 * 1.7
    a = FE.build_features(h1, CFG).filter(like='spiral')
    b = FE.build_features(scaled, CFG).filter(like='spiral')
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-7, atol=1e-9)


# -- 5.5 assembler ---------------------------------------------------------------------

def test_columns_match_the_declared_list_in_order():
    assert list(FEATS['A'].columns) == CFG['features']['columns']
    v = FE.valid_rows(FEATS['A'])
    assert v.sum() > 0.6 * len(v)
    assert np.isfinite(FEATS['A'][v].to_numpy()).all()


def test_column_drift_fails_and_names_the_column(monkeypatch):
    real = FE.compute_blocks

    def extra(ctx, cfg):
        cols = real(ctx, cfg)
        cols['sneaky_feature'] = np.zeros(len(ctx.h1))
        return cols
    monkeypatch.setattr(FE, 'compute_blocks', extra)
    with pytest.raises(FE.FeatureColumnError, match='sneaky_feature'):
        FE.build_features(PAIRS['A'], CFG)


def test_missing_declared_column_fails():
    cfg = copy.deepcopy(CFG)
    cfg['features']['columns'] = cfg['features']['columns'] + ['not_built']
    with pytest.raises(FE.FeatureColumnError, match='not_built'):
        FE.build_features(PAIRS['A'], cfg)


@pytest.mark.parametrize('pair', ['A', 'B'])
def test_truncation_invariance_at_20_random_cut_points(pair):
    h1, full = PAIRS[pair], FEATS[pair]
    rng = np.random.default_rng(100 + ord(pair))
    for pos in rng.integers(300, len(h1), 20):
        cut = FE.build_features(h1.iloc[:pos + 1], CFG)
        pd.testing.assert_frame_equal(full.iloc[:pos + 1], cut, check_exact=False, rtol=0, atol=1e-12)


def test_scaling_prices_by_a_constant_leaves_every_feature_unchanged():
    h1 = PAIRS['B']
    a = FEATS['B']
    b = FE.build_features(h1 * 2.0, CFG)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)
