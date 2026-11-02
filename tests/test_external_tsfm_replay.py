"""Clean-window replay and the frozen EUR/USD baseline (openspec change
tsfm-second-opinion, spec "Nothing is scored inside the training window" and
"The replay fixes one configuration before forward scoring", design D4/D5)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.external.tsfm import REPO
from src.external.tsfm import baseline as B
from src.external.tsfm import data as D
from src.external.tsfm import replay as R
from src.external.tsfm import vendor as V


def _bars(start, end, seed=0):
    idx = pd.date_range(start, end, freq='h', tz='UTC')
    idx = idx[idx.dayofweek < 5]
    close = 1.1 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 6e-4, len(idx))))
    return pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)


# ── nothing before the clean window ─────────────────────────────────────────

def test_a_request_before_the_clean_window_is_refused_and_names_its_start():
    with pytest.raises(D.BoundaryViolation, match='2025-11-01'):
        D.check_window_request('2025-10-01')
    with pytest.raises(D.BoundaryViolation, match='clean-window start 2025-11-01'):
        R.positions(_bars('2025-09-01', '2025-11-30'), start='2025-10-15')
    D.check_window_request('2025-11-01')


def test_the_frozen_windows_exist_with_recorded_digests():
    m = V.read_manifest()
    for inst in ('gold', 'eurusd'):
        V.verify(f'window_{inst}', m)
        win = D.read_frozen(D.window_path(inst))
        assert win.index[0] >= D.clean_start()
        hist = D.read_frozen(D.history_path(inst))
        assert hist.index[-1] < win.index[0]


def test_write_window_keeps_one_bar_after_the_window_for_the_last_target(tmp_path):
    df = _bars('2025-10-20', '2026-11-04')
    out = D.write_window('eurusd', df, 'test', repo=str(tmp_path))
    win = D.read_frozen(D.window_path('eurusd', str(tmp_path)))
    assert win.index[0] >= D.clean_start()
    assert win.index[-1] == df.index[df.index >= pd.Timestamp('2026-11-01', tz='UTC')][0]
    assert out['rows'] == len(win)


# ── the frozen baseline ─────────────────────────────────────────────────────

def test_the_baseline_fit_sees_no_clean_window_bar(tmp_path):
    hist = _bars('2020-06-01', '2025-10-31 23:00')
    with pytest.raises(D.BoundaryViolation):
        B.fit_frame(pd.concat([hist, _bars('2025-11-03', '2025-11-04')]))
    frame = B.fit_frame(hist)
    assert frame.index[0] >= pd.Timestamp(B.FIT_START, tz='UTC')
    assert frame.index[-1] < D.clean_start()


def test_the_real_baseline_was_fitted_inside_its_window():
    model = B.load()
    hist = D.read_frozen(D.history_path('eurusd'))
    frame = B.fit_frame(hist)
    assert model.est_end == len(frame)            # every estimation row is a pre-window row


def test_a_reloaded_baseline_gives_identical_forecasts():
    frame = D.read_frozen(D.window_path('eurusd')).iloc[:500]
    a = B.size_pct(B.load(), frame)
    b = B.size_pct(B.load(), frame)
    assert np.array_equal(a, b, equal_nan=True)
    assert np.nanmin(a) > 0


# ── the replay's grid and choice ────────────────────────────────────────────

def test_grid_is_eight_configurations_or_four_without_a_fine_tune():
    for model in ('chronos2', 'timesfm25'):
        full = R.grid(model, True)
        assert len(full) == 8 and len(set(full)) == 8
        assert len(R.grid(model, False)) == 4
        assert all(':zero_shot:' in c for c in R.grid(model, False))
    assert R.config_id('timesfm25', 'zero_shot', 'returns', 16256) in R.grid('timesfm25', False)


def test_choice_is_the_lowest_first_half_error():
    res = {'chronos2_gold': {'a': {'first_half_mae': 0.08}, 'b': {'first_half_mae': 0.07},
                             'c': {'first_half_mae': None}}}
    c = R.choose(res, None)['chronos2_gold']
    assert c == {'chosen': 'b', 'replay_argmin': 'b', 'frozen_by_registration': False}


def test_a_rerun_after_registration_keeps_the_frozen_choice(tmp_path):
    os.makedirs(tmp_path / 'results' / 'external_tsfm')
    with open(tmp_path / 'results' / 'external_tsfm' / 'PRE_REGISTRATION.json', 'w') as fh:
        json.dump({'cells': [{'model': 'chronos2', 'instrument': 'gold', 'config_id': 'a'}]}, fh)
    reg = R.registered_configs(str(tmp_path))
    res = {'chronos2_gold': {'a': {'first_half_mae': 0.09}, 'b': {'first_half_mae': 0.07}}}
    c = R.choose(res, reg)['chronos2_gold']
    assert c['chosen'] == 'a' and c['replay_argmin'] == 'b' and c['frozen_by_registration'] is True


def test_replay_runs_the_grid_and_labels_itself_descriptive(tmp_path, monkeypatch):
    """End to end on injected bars with a fake worker: every configuration is
    scored, the report is labelled descriptive, and REPLAY.md is written."""
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, 'results', 'external_tsfm'))
    hist = _bars('2025-06-02', '2025-10-31 23:00', 1)
    win = _bars('2025-11-03', '2026-11-03', 2)
    D.write_history('eurusd', hist, 'test', repo)
    D.write_window('eurusd', win, 'test', repo=repo)
    monkeypatch.setattr(R, 'comparators', lambda inst, full, pos, repo=None: {'baseline': np.full(len(pos), 0.06)})

    def fake_call(req, timeout=None, log_path=None):
        pos = np.load(req['positions_path'])
        s = np.load(req['series_path'])
        if s[-1] > 0.5:                                      # closes: deciles around the last close
            last = s[pos]
            dec = last[:, None] * np.exp(np.linspace(-0.1, 0.1, 9)[None, :] / 100)
        else:
            dec = np.tile(np.linspace(-0.1, 0.1, 9), (len(pos), 1))
        np.save(req['out_path'], dec)
        return {'ok': True}
    monkeypatch.setattr(V, 'verify', lambda *a, **k: 'x')
    rep = R.replay(repo=repo, call=fake_call, instruments=('eurusd',), models=('chronos2',))
    cfgs = rep['instruments']['eurusd']['models']['chronos2']['configs']
    assert len(cfgs) == 4                                  # no fine-tune kept in this repo
    assert all(r['first_half_mae'] is not None and r['second_half_mae'] is not None for r in cfgs.values())
    assert rep['label'].startswith('DESCRIPTIVE')
    md = open(os.path.join(repo, 'results', 'external_tsfm', 'replay', 'REPLAY.md'), encoding='utf-8').read()
    assert 'DESCRIPTIVE' in md and '(chosen)' in md
    first = pd.Timestamp(rep['instruments']['eurusd']['first_as_of'])
    assert first >= D.clean_start()
