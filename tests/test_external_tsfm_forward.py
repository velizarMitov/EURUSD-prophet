"""External forward logger (openspec change tsfm-second-opinion, spec "The
logger forecasts the newest closed bar every hour", "Settlement follows each
instrument's own rules", "A failure here costs nothing elsewhere"; design D6)."""
import csv
import os

import numpy as np
import pandas as pd
import pytest

from src.external.tsfm import baseline as B
from src.external.tsfm import forward_logger as L
from src.external.tsfm import vendor as V


def _bars(end, n, seed=0, price=2000.0):
    idx = pd.date_range(end=end, periods=n * 2, freq='h', tz='UTC')
    idx = idx[idx.dayofweek < 5][-n:]
    close = price * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 1e-3, len(idx))))
    return pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)


def _frames(end='2026-11-04 10:00'):
    return {'gold': _bars(end, 3000, 1, 4000.0), 'eurusd': _bars(end, 3000, 2, 1.16)}


def _fake_call(fail=()):
    calls = []

    def call(req, timeout=None, log_path=None):
        calls.append(req)
        if req['model'] in fail:
            return {'ok': False, 'error': 'worker timed out after 120 s and was stopped'}
        return {'ok': True, 'deciles': {j['key']: list(np.linspace(-0.2, 0.2, 9)) for j in req['jobs']}}
    return call, calls


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'verify', lambda *a, **k: 'digest')
    monkeypatch.setattr(B, 'load', lambda repo=None: object())
    monkeypatch.setattr(B, 'size_pct', lambda model, frame: np.full(len(frame), 0.061))
    return str(tmp_path)


def _rows(repo, name):
    p = L.out_path(name, repo)
    if not os.path.exists(p):
        return []
    with open(p, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _files(root):
    return {os.path.relpath(os.path.join(d, f), root).replace('\\', '/') for d, _, fs in os.walk(root) for f in fs}


def test_normal_hour_logs_two_rows_per_instrument(repo):
    call, calls = _fake_call()
    st = L.run_with_health(repo, frames=_frames(), call=call, ph='dry_run')
    preds = _rows(repo, 'predictions')
    assert st['forecasts'] == 4 and len(preds) == 4
    for inst in ('gold', 'eurusd'):
        rows = [r for r in preds if r['instrument'] == inst]
        assert {r['model'] for r in rows} == {'chronos2', 'timesfm25'}
        assert all(r['as_of_bar'] == '2026-11-04 10:00:00+00:00' for r in rows)
        assert all(r['phase'] == 'dry_run' for r in rows)
    assert len(calls) == 2                                     # one worker call per model
    assert all(len(c['jobs']) == 2 for c in calls)
    eur = [r for r in preds if r['instrument'] == 'eurusd']
    assert all(float(r['baseline_size_pct']) == pytest.approx(0.061) for r in eur)
    # deciles -0.2 .. 0.2 at 0.1 .. 0.9: F(x) = 0.5 + 2x, so F(m) - F(-m) = 0.5 at m = 0.125
    assert all(float(r['size_pct']) == pytest.approx(0.125, abs=1e-6) for r in preds)
    assert _rows(repo, 'health')[-1]['status'] == 'ok'


def test_the_newest_closed_bar_is_forecast_once(repo):
    call, _ = _fake_call()
    L.run_once(repo, frames=_frames(), call=call, ph='dry_run')
    st = L.run_once(repo, frames=_frames(), call=call, ph='dry_run')
    assert st['forecasts'] == 0 and st['instruments'] == {'gold': 'nothing new', 'eurusd': 'nothing new'}


def test_a_running_gpu_job_sends_the_hourly_pass_to_the_cpu(repo):
    """The 15:01 pass on 2026-11-02 took the GPU from under the fine-tuning
    (CUDA out of memory). While a GPU job holds its lock, the pass uses the CPU."""
    from src.external.tsfm import gpu_job
    call, calls = _fake_call()
    with gpu_job(repo):
        L.run_once(repo, frames=_frames('2026-11-04 10:00'), call=call, ph='dry_run')
    L.run_once(repo, frames=_frames('2026-11-04 11:00'), call=call, ph='dry_run')
    assert [c['device'] for c in calls] == ['cpu', 'cpu', 'auto', 'auto']
    assert not os.path.exists(os.path.join(repo, 'research_models', 'external_tsfm', 'gpu_job.lock'))


def test_three_missed_hours_become_gaps_never_late_forecasts(repo):
    call, _ = _fake_call()
    L.run_once(repo, frames=_frames('2026-11-04 10:00'), call=call, ph='dry_run')
    st = L.run_once(repo, frames=_frames('2026-11-04 14:00'), call=call, ph='dry_run')
    assert st['gaps'] == 6                                      # 11:00, 12:00, 13:00 for each instrument
    gaps = _rows(repo, 'gaps')
    assert sorted({g['as_of_bar'][11:16] for g in gaps}) == ['11:00', '12:00', '13:00']
    later = [r for r in _rows(repo, 'predictions') if r['as_of_bar'] > '2026-11-04 10:00:00+00:00']
    assert {r['as_of_bar'] for r in later} == {'2026-11-04 14:00:00+00:00'}


def test_mt5_closed_gives_a_health_row_with_zero_forecasts(repo):
    call, calls = _fake_call()
    L.run_with_health(repo, frames={}, call=call, ph='dry_run')
    h = _rows(repo, 'health')[-1]
    assert h['status'] == 'mt5 unreachable' and h['forecasts'] == '0'
    assert calls == [] and _rows(repo, 'predictions') == []


def test_one_model_failing_leaves_the_other_models_rows(repo):
    call, _ = _fake_call(fail=('chronos2',))
    L.run_with_health(repo, frames=_frames(), call=call, ph='dry_run')
    assert {r['model'] for r in _rows(repo, 'predictions')} == {'timesfm25'}
    fails = _rows(repo, 'failures')
    assert {(f['model'], f['instrument']) for f in fails} == {('chronos2', 'gold'), ('chronos2', 'eurusd')}
    assert _rows(repo, 'health')[-1]['status'] == 'partial'


def test_gold_unavailable_still_forecasts_eurusd(repo):
    call, _ = _fake_call()
    L.run_once(repo, frames={'eurusd': _frames()['eurusd']}, call=call, ph='dry_run')
    assert {r['instrument'] for r in _rows(repo, 'predictions')} == {'eurusd'}


def test_missing_side_environment_is_recorded_and_the_pass_exits_cleanly(repo, monkeypatch):
    monkeypatch.setattr(L.CL, 'available', lambda repo=None: False)
    st = L.run_with_health(repo, frames=_frames(), ph='dry_run')
    assert st['forecasts'] == 0
    assert _rows(repo, 'health')[-1]['status'] == 'external environment unavailable'


def test_no_file_outside_the_external_folders_changes(repo):
    os.makedirs(os.path.join(repo, 'results', 'gold', 'forward'))
    os.makedirs(os.path.join(repo, 'results', 'forward_eval'))
    before = _files(repo)
    call, _ = _fake_call()
    L.run_with_health(repo, frames=_frames('2026-11-04 10:00'), call=call, ph='dry_run')
    L.run_with_health(repo, frames=_frames('2026-11-04 12:00'), call=call, ph='dry_run')
    new = _files(repo) - before
    assert new and all(f.startswith(('results/external_tsfm/', 'research_models/external_tsfm/')) for f in new), new


# ── settlement ──────────────────────────────────────────────────────────────

def _gold_logger_settle(tmp_path, frame, as_of):
    """The gold family's own settlement of one h1_lstm row on the same bars."""
    from src.gold import forward_logger as GL
    fwd = tmp_path / 'goldfwd'
    fwd.mkdir()
    cfg = {'paths': {'forward': str(fwd)}}
    GL.append_rows(GL.out_path(cfg, 'predictions'), [{
        'key': f'h1_lstm|H1|{as_of}', 'cell': 'h1_lstm', 'cadence': 'H1', 'as_of_bar': str(as_of),
        'as_of_close': 1, 'phase': 'dry_run', 'logged_at': 'x'}], GL.PRED_FIELDS)
    return GL.settle(cfg, {'H1': frame}, 'now')[0]


def test_gold_realised_move_equals_the_gold_loggers_settlement(repo, tmp_path):
    call, _ = _fake_call()
    frames = _frames('2026-11-04 10:00')
    L.run_once(repo, frames={'gold': frames['gold'].iloc[:-1]}, call=call, ph='dry_run')
    as_of = frames['gold'].index[-2]
    later = pd.Timestamp('2026-11-05', tz='UTC')
    rows = L.settle({'gold': frames['gold']}, repo, 'now', now=later)
    ours = [r for r in rows if r['instrument'] == 'gold']
    theirs = _gold_logger_settle(tmp_path, frames['gold'], as_of)
    assert len(ours) == 2
    for r in ours:
        assert r['realised_abs_pct'] == pytest.approx(theirs['realised_vol_pct'], abs=0)
        assert r['realised_ret_pct'] == pytest.approx(theirs['realised_ret_pct'], abs=0)
        assert r['eligible'] == theirs['eligible']


def test_eurusd_realised_move_equals_the_arbiters_target(repo):
    from src.forecast_eval import targets as T
    call, _ = _fake_call()
    frames = _frames('2026-11-04 10:00')
    L.run_once(repo, frames={'eurusd': frames['eurusd'].iloc[:-1]}, call=call, ph='dry_run')
    rows = L.settle({'eurusd': frames['eurusd']}, repo, 'now')
    target = float(T.volatility_pct_target(frames['eurusd']['close'], 1).iloc[-2])
    assert rows and all(r['realised_abs_pct'] == pytest.approx(target, abs=0) for r in rows)
    assert all(r['baseline_size_pct'] == pytest.approx(0.061) for r in rows)


def test_a_row_bordering_a_market_break_settles_ineligible(repo):
    call, _ = _fake_call()
    frames = _frames('2026-11-06 21:00')                      # Friday's last bar, then the weekend
    L.run_once(repo, frames={'eurusd': frames['eurusd']}, call=call, ph='dry_run')
    monday = frames['eurusd'].copy()
    nxt = pd.DataFrame({c: [1.17] for c in monday.columns},
                       index=pd.DatetimeIndex([pd.Timestamp('2026-11-08 23:00', tz='UTC')]))
    rows = L.settle({'eurusd': pd.concat([monday, nxt])}, repo, 'now')
    assert rows and all(r['eligible'] is False for r in rows)


def test_a_gold_row_waits_for_the_gold_baseline(repo):
    call, _ = _fake_call()
    frames = _frames('2026-11-04 10:00')
    L.run_once(repo, frames={'gold': frames['gold'].iloc[:-1]}, call=call, ph='dry_run')
    logged = pd.Timestamp(_rows(repo, 'predictions')[0]['logged_at'])
    assert L.settle({'gold': frames['gold']}, repo, 'now', now=logged + pd.Timedelta(hours=1)) == []
    rows = L.settle({'gold': frames['gold']}, repo, 'now', now=logged + pd.Timedelta(hours=7))
    assert len(rows) == 2 and all(r['baseline_size_pct'] == '' for r in rows)


def test_polling_waits_for_the_hour_and_never_on_a_weekend():
    now = pd.Timestamp('2026-11-04 11:00:30', tz='UTC')
    frames = iter([_bars('2026-11-04 09:00', 50), _bars('2026-11-04 10:00', 50)])
    slept = []
    f = L.fetch_polling(lambda: next(frames), now_fn=lambda: now, sleep=slept.append)
    assert f.index[-1] == pd.Timestamp('2026-11-04 10:00', tz='UTC') and slept == [L.POLL_S]
    weekend = pd.Timestamp('2026-11-08 12:00', tz='UTC')
    slept = []
    f = L.fetch_polling(lambda: _bars('2026-11-06 21:00', 50), now_fn=lambda: weekend, sleep=slept.append)
    assert slept == []
