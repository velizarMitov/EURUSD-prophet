"""M15 session cells inside the study: eligibility masking, the session
breakeven and the bid-versus-mid parity check (task 14.7)."""
import os

import numpy as np
import pandas as pd
import pytest

from src.forecast_eval import m15_data as MD
from src.forecast_eval import record as R
from src.forecast_eval import study as ST
from src.forecast_eval.challengers import make
from src.forecast_eval.challengers.base import ChallengerUnavailable

HAVE_M1 = os.path.exists(MD.M1_PATH)


def _m15(days=120, seed=0, spread_points=5.0, rollover_hour=None):
    """Synthetic M15 bars. With `rollover_hour` the spread blows up in that
    label hour and the bid drops with it -- the artifact of design D17."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp('2025-01-06', tz='UTC'), periods=96 * days, freq='15min')
    idx = idx[idx.dayofweek < 5]
    close = 1.10 + np.cumsum(rng.normal(0, 2e-5, len(idx)))
    spread = np.full(len(idx), spread_points * 1e-5)
    if rollover_hour is not None:
        hit = idx.hour == rollover_hour
        spread[hit] = 25e-5
        close[hit] -= 10e-5                       # the bid falls as the spread opens
    return pd.DataFrame({'open': close, 'high': close + 3e-5, 'low': close - 3e-5,
                         'close': close, 'tick_volume': rng.integers(20, 200, len(idx)).astype(float),
                         'spread_price': spread, 'point': 1e-5}, index=idx)


@pytest.fixture
def rec(tmp_path):
    path = str(tmp_path / 'rec.json')
    r = R.new_record('m15s')
    r['development'] = {**r['development'], 'walk_forward_splits': 2, 'min_train_fraction': 0.5,
                        'cpcv': {'default': {'n_groups': 4, 'k': 2}}}
    r['tuning'] = {**r['tuning'], 'grids': {}}          # no inner tuning in the test
    R.save(r, path)
    return r, path


# ── eligibility masking ────────────────────────────────────────────────────

def test_only_session_rows_are_trained_on_and_scored(rec):
    r, _ = rec
    cfg = r['challengers']['m15_session_gbm']
    m15 = _m15(days=60)
    m = make('m15_session_gbm', cfg)
    inputs = m.build_inputs(m15)
    tg = ST.make_targets(inputs.close, 1)
    from src.forecast_eval import splits as S
    wf = S.walk_forward(len(inputs), h=1, n_splits=2, min_train=len(inputs) // 2)
    elig = MD.eligible_positions(inputs.index, 1)
    oos = ST._oos('m15_session_gbm', cfg, inputs, tg, wf, 42, 1, ('p_up',), elig)
    scored = np.flatnonzero(np.isfinite(oos['p_up']))
    assert scored.size > 100
    assert set(scored) <= set(elig.tolist()), 'nothing outside the session is scored'
    assert MD.session_mask(inputs.index)[scored].all()


def test_keep_restricts_to_eligible_and_passes_through_without_it():
    pos = np.arange(10)
    assert list(ST._keep(pos, None)) == list(pos)
    assert list(ST._keep(pos, np.array([2, 4, 20]))) == [2, 4]


def test_inner_tuning_takes_the_contiguous_block_plus_an_eligibility_mask(rec):
    """Regression: passing the already-filtered session rows made the inner
    walk-forward arithmetic impossible and failed every M15 GBM cell."""
    from src.forecast_eval import tuning as TU
    r, _ = rec
    cfg = r['challengers']['m15_session_gbm']
    grid = {'max_depth': [3, 4]}
    m15 = _m15(days=60)
    m = make('m15_session_gbm', cfg)
    inputs = m.build_inputs(m15)
    tg = ST.make_targets(inputs.close, 1)
    from src.forecast_eval import splits as S
    wf = S.walk_forward(len(inputs), h=1, n_splits=2, min_train=len(inputs) // 2)
    elig = MD.eligible_positions(inputs.index, 1)

    out = TU.inner_tune('m15_session_gbm', cfg, grid, inputs, wf[0].train, tg, 1, 42,
                        inner_folds=2, eligible=elig)
    assert out['params']['max_depth'] in (3, 4) and out['n_candidates'] == 2

    with pytest.raises(ValueError, match='contiguous'):
        TU.inner_tune('m15_session_gbm', cfg, grid, inputs, elig, tg, 1, 42, inner_folds=2)


def test_session_breakeven_uses_the_session_move(rec):
    """E|move| over the scored rows, not over all 24 hours."""
    close = np.arange(100.0)
    rows = np.array([0, 10, 20])
    assert ST.mean_abs_move_on(close, 4, rows) == pytest.approx(4.0)
    with pytest.raises(ValueError, match='complete windows'):
        ST.mean_abs_move_on(close, 4, np.array([99]))


def test_session_cost_level_is_the_in_session_median(rec):
    m15 = _m15(days=20, rollover_hour=23)
    levels = ST.m15_cost_levels(m15)
    assert levels[0].label == 'measured'
    assert levels[0].price == pytest.approx(5e-5), 'the rollover hour is outside the session'
    assert levels[1].label == 'config_round_trip'


# ── the bid-versus-mid parity check ────────────────────────────────────────

def _parity_on(m15, rec_tuple, h=4, tolerance_pp=1.0, hours=None):
    r, _ = rec_tuple
    cfg = r['challengers']['m15_session_gbm']
    mid = make('m15_session_gbm', cfg, price='mid').build_inputs(m15)
    bid = make('m15_session_gbm', cfg, price='bid').build_inputs(m15)
    assert mid.index.equals(bid.index)
    from src.forecast_eval import splits as S
    wf = S.walk_forward(len(mid), h=h, n_splits=2, min_train=len(mid) // 2)
    elig = MD.eligible_positions(mid.index, h) if hours is None else \
        np.flatnonzero(np.isin(mid.index.hour, hours))
    elig = elig[elig + h < len(mid)]
    tg = ST.make_targets(mid.close, h)
    oos = ST._oos('m15_session_gbm', cfg, mid, tg, wf, 42, h, ('p_up',), elig)
    rows = np.flatnonzero(np.isfinite(oos['p_up']) & np.isfinite(tg['dir']))
    acc_mid = float(((oos['p_up'][rows] >= 0.5).astype(float) == tg['dir'][rows]).mean())
    return ST.bid_mid_parity('m15_session_gbm', cfg, h, wf, 42, elig, acc_mid, bid, tolerance_pp)


def test_parity_holds_inside_a_clean_session(rec):
    """A constant spread shifts mid and bid by the same amount, so the direction
    is identical and only model noise separates the two accuracies."""
    p = _parity_on(_m15(days=500, rollover_hour=23), rec)
    assert p['parity_label'] == 'parity holds'
    assert abs(p['parity_diff_pp']) <= p['parity_threshold_pp']
    assert 'label' not in p, 'a clean cell keeps its own label'


def test_the_rollover_hour_is_flagged_as_a_spread_artifact(rec):
    """The hour the session rule exists to exclude. Scored there on purpose, the
    check must refuse to call it skill."""
    p = _parity_on(_m15(days=500, rollover_hour=23), rec, hours=[23])
    assert abs(p['parity_diff_pp']) > p['parity_threshold_pp']
    assert p['parity_label'] == ST.SPREAD_ARTIFACT
    assert p['label'] == ST.SPREAD_ARTIFACT, 'it overrides the skill label'


def test_a_small_sample_is_not_called_an_artifact_on_noise_alone(rec):
    """Two separately fitted models differ by sampling noise. With few eligible
    rows that noise exceeds the declared tolerance, and calling it an artifact
    would be the opposite of what the check is for."""
    p = _parity_on(_m15(days=60), rec, h=16)
    assert p['parity_noise_floor_pp'] > p['parity_tolerance_pp']
    assert p['parity_threshold_pp'] == pytest.approx(p['parity_noise_floor_pp'])


def test_parity_reports_both_definitions_and_the_difference(rec):
    p = _parity_on(_m15(days=120), rec)
    assert set(p) >= {'accuracy_bid', 'n_scored_bid', 'parity_diff_pp', 'parity_tolerance_pp',
                      'parity_noise_floor_pp', 'parity_threshold_pp', 'parity_label'}
    assert 0.0 <= p['accuracy_bid'] <= 1.0


def test_parity_refuses_to_judge_too_few_rows(rec):
    r, _ = rec
    cfg = r['challengers']['m15_session_gbm']
    m15 = _m15(days=40)
    bid = make('m15_session_gbm', cfg, price='bid').build_inputs(m15)
    from src.forecast_eval import splits as S
    wf = S.walk_forward(len(bid), h=1, n_splits=2, min_train=len(bid) // 2)
    p = ST.bid_mid_parity('m15_session_gbm', cfg, 1, wf, 42, np.arange(5), 0.52, bid, 1.0)
    assert p['parity_label'] == 'not enough bid rows' and np.isnan(p['parity_diff_pp'])


# ── wiring ─────────────────────────────────────────────────────────────────

def test_m15_models_route_to_the_m15_source():
    cfg = {'cadence': 'M15', 'kind': 'direction'}
    frame = _m15(days=5)
    assert ST.source_for('m15_session_gbm', cfg, {'m15': frame}) is frame
    with pytest.raises(ChallengerUnavailable, match='boom'):
        ST.source_for('m15_session_gbm', cfg, {'m15': None, 'm15_error': 'boom'})


def test_load_sources_isolates_a_missing_m1_file(monkeypatch):
    from src.forecast_eval import m15_data as MDmod
    monkeypatch.setattr(MDmod, 'M1_PATH', 'absent.parquet')
    monkeypatch.setattr(MDmod, 'm15_bars', lambda *a, **k: (_ for _ in ()).throw(
        FileNotFoundError('M1 source not found')))
    monkeypatch.setattr(ST.F, 'load_daily_ohlcv', lambda *a, **k: _m15(days=3))
    monkeypatch.setattr(ST.F, 'daily_features', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ST.F, 'load_h1', lambda *a, **k: pd.DataFrame())
    s = ST.load_sources(macro=False)
    assert s['m15'] is None and 'M1 source not found' in s['m15_error']


@pytest.mark.skipif(not HAVE_M1, reason='M1 parquet is not in git (DATA.md 7)')
def test_an_end_to_end_m15_cell_on_the_real_source(rec, tmp_path, monkeypatch):
    r, path = rec
    monkeypatch.setattr(ST, 'ART', str(tmp_path / 'art'))
    m15 = MD.m15_bars().iloc[-6000:]
    cfg = r['challengers']['m15_session_gbm']
    m = make('m15_session_gbm', cfg)
    inputs = m.build_inputs(m15)
    bid = make('m15_session_gbm', cfg, price='bid').build_inputs(m15)
    elig = MD.eligible_positions(inputs.index, 4)
    row, oos, manifest = ST.run_cell(
        'm15_session_gbm', cfg, 4, inputs, R.load(path), 'digest', out=str(tmp_path / 'out'),
        eligible=elig, cost_levels=ST.m15_cost_levels(m15),
        parity={'inputs': bid, 'tolerance_pp': 1.0}, save=False)
    assert row['cadence'] == 'M15' and row['price_definition'] == 'mid'
    assert row['n_eligible'] == len(elig) and row['n_scored'] > 100
    assert 0.3 < row['accuracy'] < 0.7 and 0.3 < row['accuracy_bid'] < 0.7
    assert row['breakeven_measured'] > 0.5
    assert row['block_len'] == 26
    assert np.isfinite(row['parity_diff_pp'])
