"""fx-bb-reversion, group 2: six-pair H1 data, per-pair starts, the lag check and the 17:00 New York D1."""
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.fx_bb import data_loader as DL
from src.forecast_eval.m15_data import clock_evidence
from tests.fx_bb_synth import FakeSource, install_all, install_session_caches, six_pairs, tmp_cfg


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    cfg = tmp_cfg(tmp_path)
    frames = six_pairs()
    install_session_caches(cfg, frames)
    return cfg, frames


def _own_exists(cfg):
    return [os.path.exists(DL.cache_path(p, cfg)) for p in DL.own_pairs(cfg)]


def test_refresh_writes_the_three_own_caches_with_spread_points_and_log(setup):
    cfg, frames = setup
    rep = DL.refresh(cfg, FakeSource(frames), 'now')
    assert rep['status'] == 'ok' and set(rep['pairs']) == {'USDJPY', 'USDCAD', 'USDCHF'}
    for p in DL.own_pairs(cfg):
        with open(DL.cache_path(p, cfg), encoding='utf-8') as fh:
            assert fh.readline().strip() == 'time,open,high,low,close,spread'
        assert rep['pairs'][p]['clock_match_rate'] == 1.0 and rep['pairs'][p]['lag_years'] == '2012-2012'
    assert DL.load_point('USDJPY', cfg) == 0.001
    log = pd.read_csv(os.path.join(cfg['paths']['data'], 'refresh_log.csv'))
    assert list(log['pair']) == ['USDJPY', 'USDCAD', 'USDCHF'] and set(log['status']) == {'ok'}


def test_one_failing_pair_writes_nothing_for_any_pair(setup):
    cfg, frames = setup
    f = dict(frames)
    f.pop('USDCHF')
    rep = DL.refresh(cfg, FakeSource(f, names=list(frames)), 'now')
    assert rep['status'] == 'refused' and 'no bars' in rep['pairs']['USDCHF']
    assert not any(_own_exists(cfg))
    log = pd.read_csv(os.path.join(cfg['paths']['data'], 'refresh_log.csv'))
    assert len(log) == 3 and set(log['status']) == {'refused'}


def _holed(frame, a='2012-02-07', b='2012-02-10'):
    return frame[(frame.index < pd.Timestamp(a, tz='UTC')) | (frame.index >= pd.Timestamp(b, tz='UTC'))]


def test_undeclared_interior_hole_is_refused(setup):
    cfg, frames = setup
    f = dict(frames)
    f['USDCAD'] = _holed(frames['USDCAD'])
    cfg['data']['known_gaps'] = {}
    rep = DL.refresh(cfg, FakeSource(f), 'now')
    assert rep['status'] == 'refused' and 'coverage' in rep['pairs']['USDCAD']
    assert not any(_own_exists(cfg))


def test_declared_gap_is_accepted_and_not_filled(setup):
    cfg, frames = setup
    f = dict(frames)
    f['USDCAD'] = _holed(frames['USDCAD'])
    cfg['data']['known_gaps'] = {'USDCAD': ['2012-02']}
    assert DL.refresh(cfg, FakeSource(f), 'now')['status'] == 'ok'
    assert len(DL.load_pair('USDCAD', cfg)) == len(f['USDCAD'])


def test_one_hour_shift_fails_the_lag_check_while_passing_the_weekly_open_check(setup):
    cfg, frames = setup
    f = dict(frames)
    f.update(six_pairs(shift_values={'USDCHF': 1}))
    for p in ('EURUSD', 'GBPUSD', 'AUDUSD'):
        f[p] = frames[p]
    assert clock_evidence(f['USDCHF'].index, 'Europe/Berlin')['match_rate'] == 1.0
    rep = DL.refresh(cfg, FakeSource(f), 'now')
    assert rep['status'] == 'refused'
    assert 'lag check' in rep['pairs']['USDCHF'] and '2012 peaks at lag' in rep['pairs']['USDCHF']
    assert not any(_own_exists(cfg))


def test_lag_check_on_load_refuses_a_shifted_fx_session_pair(tmp_path):
    cfg = tmp_cfg(tmp_path)
    frames = six_pairs(shift_values={'AUDUSD': 1})
    from tests.fx_bb_synth import install_all
    install_all(cfg, frames)
    with pytest.raises(DL.DataCheckFailed, match='AUDUSD.*lag check'):
        DL.load_all(cfg)


def test_usdchf_bars_before_its_start_are_never_used(tmp_path):
    cfg = tmp_cfg(tmp_path, starts={'USDCHF': '2012-01-15'})
    frames = six_pairs()
    from tests.fx_bb_synth import install_all
    install_all(cfg, frames)
    raw = DL.load_h1('USDCHF', {'paths': {'data': cfg['paths']['data']}})
    assert raw.index.min() < pd.Timestamp('2012-01-15', tz='UTC')
    loaded, _ = DL.load_all(cfg)
    assert loaded['USDCHF'].index.min() >= pd.Timestamp('2012-01-15', tz='UTC')
    rep = DL.refresh(cfg, FakeSource(frames), 'again')
    assert rep['status'] == 'ok' and rep['pairs']['USDCHF']['start'] >= '2012-01-15'


def test_the_fx_session_caches_stay_byte_identical(setup):
    cfg, frames = setup
    paths = [DL.cache_path(p, cfg) for p in ('EURUSD', 'GBPUSD', 'AUDUSD')]
    before = [_sha(p) for p in paths] + [_sha(os.path.join(cfg['paths']['fx_session_data'], 'symbols.json'))]
    DL.refresh(cfg, FakeSource(frames), 'now')
    DL.load_all(cfg)
    after = [_sha(p) for p in paths] + [_sha(os.path.join(cfg['paths']['fx_session_data'], 'symbols.json'))]
    assert before == after
    assert not os.path.exists(os.path.join(cfg['paths']['fx_session_data'], 'refresh_log.csv'))


def test_refresh_appends_only_new_bars(setup):
    cfg, frames = setup
    DL.refresh(cfg, FakeSource({k: v.iloc[:-30] for k, v in frames.items()}), 'first')
    before = open(DL.cache_path('USDJPY', cfg), 'rb').read()
    rep = DL.refresh(cfg, FakeSource(frames), 'second')
    assert open(DL.cache_path('USDJPY', cfg), 'rb').read().startswith(before)
    assert rep['pairs']['USDJPY']['appended'] == 30 and rep['pairs']['USDJPY']['overlap_mismatches'] == 0


def test_lag_table_peaks_at_zero_for_co_moving_series():
    f = six_pairs()
    tab = DL.lag_table(f['USDJPY']['close'], f['EURUSD']['close'])
    assert DL.lag_failures(tab) == [] and tab.loc[2012, 'lag_+0'] > 0.5
    shifted = six_pairs(shift_values={'USDJPY': 1})
    assert DL.lag_failures(DL.lag_table(shifted['USDJPY']['close'], f['EURUSD']['close'])) == [2012]
    assert np.isfinite(tab.loc[2012, 'lag_-1'])
