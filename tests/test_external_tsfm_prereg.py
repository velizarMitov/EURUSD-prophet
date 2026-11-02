"""Registration, the phase rule and the one-shot score (openspec change
tsfm-second-opinion, spec "The family is registered before it is scored",
design D7)."""
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.external.tsfm import forward_logger as L
from src.external.tsfm import forward_report as FR
from src.external.tsfm import prereg as P


def _git(repo, *args):
    subprocess.run(['git', '-C', repo, '-c', 'user.name=t', '-c', 'user.email=t@example.com', *args],
                   check=True, capture_output=True)


@pytest.fixture
def gitrepo(tmp_path):
    repo = str(tmp_path)
    _git(repo, 'init', '-q')
    os.makedirs(os.path.join(repo, 'results', 'external_tsfm'))
    with open(os.path.join(repo, 'README'), 'w') as fh:
        fh.write('x')
    _git(repo, 'add', 'README')
    _git(repo, 'commit', '-q', '-m', 'init')
    return repo


def _write_prereg(repo, text='registered'):
    for name in ('PRE_REGISTRATION.md', 'PRE_REGISTRATION.json'):
        with open(os.path.join(repo, 'results', 'external_tsfm', name), 'w') as fh:
            fh.write(text if name.endswith('.md') else json.dumps({'cells': []}))


def test_rows_are_dry_run_until_both_files_are_committed_unmodified(gitrepo):
    assert L.phase(gitrepo) == 'dry_run'                         # no files
    _write_prereg(gitrepo)
    assert L.phase(gitrepo) == 'dry_run'                         # written, not committed
    _git(gitrepo, 'add', 'results')
    _git(gitrepo, 'commit', '-q', '-m', 'register')
    assert L.phase(gitrepo) == 'registered'
    with open(os.path.join(gitrepo, 'results', 'external_tsfm', 'PRE_REGISTRATION.md'), 'a') as fh:
        fh.write('\nedited')
    assert L.phase(gitrepo) == 'dry_run'                         # modified after the commit


def test_write_refuses_once_the_registration_is_committed(gitrepo):
    _write_prereg(gitrepo)
    _git(gitrepo, 'add', 'results')
    _git(gitrepo, 'commit', '-q', '-m', 'register')
    with pytest.raises(P.AlreadyRegistered):
        P.write(gitrepo)


def test_cells_need_a_replay_choice(monkeypatch):
    monkeypatch.setattr(P.V, 'verify', lambda *a, **k: 'd')
    monkeypatch.setattr(P, '_gold_baseline_digest', lambda repo: 'g')
    with pytest.raises(ValueError, match='run the replay first'):
        P.cells(replay={'choice': {}})
    choice = {f'{m}_{i}': {'chosen': f'{m}:fine_tuned:returns:2048' if i == 'gold' else f'{m}:zero_shot:closes:2048'}
              for m in ('chronos2', 'timesfm25') for i in ('gold', 'eurusd')}
    cells = P.cells(replay={'choice': choice})
    assert len(cells) == 4
    assert {c['cell'] for c in cells} == {'external_tsfm:chronos2_gold', 'external_tsfm:chronos2_eurusd',
                                          'external_tsfm:timesfm25_gold', 'external_tsfm:timesfm25_eurusd'}
    gold = [c for c in cells if c['instrument'] == 'gold']
    assert all(len(c['artifacts']) == 2 for c in gold)           # base weights + the adapter
    assert all(c['baseline']['name'] == 'h1_garch_hod' for c in gold)
    assert all(c['baseline']['name'] == 'baseline_eurusd' for c in cells if c['instrument'] == 'eurusd')


def test_registration_constants():
    assert P.ALPHA == pytest.approx(0.0125)
    assert (P.BLOCK, P.N_BOOT, P.SEED, P.N_TARGET, P.N_MIN, P.CAP) == (24, 2000, 42, 1500, 500, '2027-04-30')


def test_power_statement_from_the_gold_reference():
    p = P.power_statement()
    assert p['available']
    assert 0.005 < p['gold']['detectable_edge_pp'] < 0.02
    assert p['reference_edge_pp'] == pytest.approx(0.01133, abs=1e-4)
    assert 'borderline' in p['statement']


# ── the one-shot score ──────────────────────────────────────────────────────

REG = {'cells': [{'cell': 'external_tsfm:timesfm25_eurusd', 'model': 'timesfm25', 'instrument': 'eurusd',
                  'config_id': 'timesfm25:zero_shot:returns:2048'}],
       'alpha': 0.0125, 'n_target': 1500, 'n_min': 500, 'cap': '2027-04-30',
       'test': {'bootstrap': {'block_len': 24, 'n_boot': 2000, 'seed': 42}}}


def _settlements(n, edge, seed=0, start='2026-11-10'):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq='h', tz='UTC')
    y = np.abs(rng.normal(0, 0.08, n))
    base = np.full(n, 0.07)
    model = np.clip(y + rng.normal(0, 0.05, n), 0, None) if edge else np.full(n, 0.07) + rng.normal(0, 0.001, n)
    return [{'key': f'timesfm25|eurusd|{ts}', 'model': 'timesfm25', 'instrument': 'eurusd', 'as_of_bar': str(ts),
             'realised_abs_pct': y[i], 'size_pct': model[i], 'baseline_size_pct': base[i], 'eligible': 'True',
             'phase': 'registered'} for i, ts in enumerate(t)]


def _repo_with(tmp_path, settlements):
    repo = str(tmp_path)
    os.makedirs(os.path.join(repo, 'results', 'external_tsfm'), exist_ok=True)
    with open(os.path.join(repo, 'results', 'external_tsfm', 'PRE_REGISTRATION.json'), 'w') as fh:
        json.dump(REG, fh)
    L.append_rows(L.out_path('settlements', repo), settlements, L.SETTLE_FIELDS + ['key'])
    return repo


def test_scoring_refuses_before_registration(tmp_path):
    with pytest.raises(FR.NotRegistered):
        FR.score(str(tmp_path), phase_fn=lambda repo: 'dry_run')


def test_a_clear_edge_is_kept(tmp_path):
    repo = _repo_with(tmp_path, _settlements(1600, edge=True))
    v = FR.score(repo, today=pd.Timestamp('2027-01-20', tz='UTC'), phase_fn=lambda r: 'registered')
    assert len(v) == 1 and v[0]['verdict'] == 'KEEP' and v[0]['n'] == 1500 and v[0]['ci_low'] > 0


def test_no_edge_is_dropped(tmp_path):
    repo = _repo_with(tmp_path, _settlements(1600, edge=False))
    v = FR.score(repo, today=pd.Timestamp('2027-01-20', tz='UTC'), phase_fn=lambda r: 'registered')
    assert v[0]['verdict'] == 'DROP'


def test_too_few_at_the_cap_is_underpowered(tmp_path):
    repo = _repo_with(tmp_path, _settlements(430, edge=True))
    assert FR.score(repo, today=pd.Timestamp('2027-04-29', tz='UTC'), phase_fn=lambda r: 'registered') == []
    v = FR.score(repo, today=pd.Timestamp('2027-04-30', tz='UTC'), phase_fn=lambda r: 'registered')
    assert v[0]['verdict'] == 'UNDERPOWERED — NO DECISION' and v[0]['n'] == 430


def test_a_scored_cell_is_never_rescored(tmp_path):
    repo = _repo_with(tmp_path, _settlements(1600, edge=True))
    first = FR.score(repo, today=pd.Timestamp('2027-01-20', tz='UTC'), phase_fn=lambda r: 'registered')
    L.append_rows(L.out_path('settlements', repo), _settlements(2000, edge=False, seed=3, start='2027-03-01'),
                  L.SETTLE_FIELDS)
    assert FR.score(repo, today=pd.Timestamp('2027-05-01', tz='UTC'), phase_fn=lambda r: 'registered') == []
    assert FR.read_verdicts(repo) == first


def test_dry_run_and_ineligible_rows_are_never_scored(tmp_path):
    rows = _settlements(1600, edge=True)
    for r in rows[:200]:
        r['phase'] = 'dry_run'
    for r in rows[200:300]:
        r['eligible'] = 'False'
    repo = _repo_with(tmp_path, rows)
    v = FR.score(repo, today=pd.Timestamp('2027-04-30', tz='UTC'), phase_fn=lambda r: 'registered')
    assert v[0]['n'] == 1300 and v[0]['reached'] == 'cap'
