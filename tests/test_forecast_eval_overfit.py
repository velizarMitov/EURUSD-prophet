"""forecast_eval overfitting diagnostics (tasks 5.1-5.2)."""
import numpy as np
import pytest

from src.forecast_eval import overfit as O


def _rows(k, model='m'):
    return [{'study_id': 's', 'model': model, 'cadence': 'H1', 'horizon': 1,
             'config_hash': f'c{i}', 'variant': 'frozen', 'sharpe': 0.01 * i, 'n_obs': 100}
            for i in range(k)]


# ── 5.1 trial log + DSR ─────────────────────────────────────────────────────

def test_trials_accumulate_across_runs(tmp_path):
    log = str(tmp_path / 'trial_log.csv')
    assert O.append_trials(_rows(40), log) == 40
    assert O.append_trials(_rows(10, 'n'), log) == 50
    assert O.n_trials(log) == 50
    ids = [int(r['trial_id']) for r in O.read_trials(log)]
    assert ids == list(range(1, 51))


def test_trial_log_is_append_only(tmp_path):
    log = tmp_path / 'trial_log.csv'
    O.append_trials(_rows(3), str(log))
    before = log.read_bytes()
    O.append_trials(_rows(2, 'x'), str(log))
    after = log.read_bytes()
    assert after.startswith(before) and len(after) > len(before)


def test_trial_log_writer_never_opens_for_rewrite():
    import inspect
    src = inspect.getsource(O)
    assert "open(path, 'a'" in src
    assert "open(path, 'w'" not in src


def test_unknown_field_is_refused(tmp_path):
    with pytest.raises(ValueError):
        O.append_trials([{'model': 'm', 'bogus': 1}], str(tmp_path / 't.csv'))


def test_dsr_deflates_with_more_trials():
    r = np.random.default_rng(0).normal(0.05, 1.0, 2000)
    one = O.deflated_sharpe(r, 1, float('nan'))
    many = O.deflated_sharpe(r, 50, 0.002)
    assert one['sr0'] == 0.0
    assert many['sr0'] > 0 and many['dsr'] < one['dsr']
    assert many['n_trials'] == 50


def test_dsr_uses_the_cumulative_count(tmp_path):
    log = str(tmp_path / 'trial_log.csv')
    O.append_trials(_rows(40), log)
    O.append_trials(_rows(10, 'n'), log)
    r = np.random.default_rng(1).normal(0.02, 1, 500)
    d = O.deflated_sharpe(r, O.n_trials(log), O.trial_sharpe_variance(log))
    assert d['n_trials'] == 50


# ── 5.2 PBO ─────────────────────────────────────────────────────────────────

def test_pbo_split_count_at_s16():
    M = np.random.default_rng(5).normal(0, 1, (1600, 4))
    assert O.pbo_cscv(M, n_submatrices=16)['n_splits'] == 12870


def test_pbo_near_half_for_pure_noise():
    """PBO on ONE noise matrix is itself noisy (all splits share the same
    data; sd ~0.16 across realizations), so the property is tested on the mean
    over realizations, not on a single draw."""
    vals = [O.pbo_cscv(np.random.default_rng(s).normal(0, 1, (800, 10)), n_submatrices=10)['pbo']
            for s in range(20)]
    assert 0.4 <= np.mean(vals) <= 0.7


def test_pbo_near_zero_with_one_real_edge():
    rng = np.random.default_rng(6)
    M = rng.normal(0, 1, (1600, 10))
    M[:, 3] += 0.25
    r = O.pbo_cscv(M, candidates=[f'cfg{i}' for i in range(10)], n_submatrices=16)
    assert r['pbo'] < 0.05
    assert r['candidates'][3] == 'cfg3' and len(r['candidates']) == 10
