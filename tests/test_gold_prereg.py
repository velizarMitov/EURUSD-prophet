"""Gold pre-registration (task 5.1; spec "Three gold families with registered
composition", "Pre-registration precedes the first validation score")."""
import copy
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.gold import load_config
from src.gold import prereg as P

CFG = load_config()


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, 'config', 'user.email', 't@example.invalid')
    _git(tmp_path, 'config', 'user.name', 't')
    _git(tmp_path, 'config', 'core.hooksPath', '/dev/null')
    cfg = copy.deepcopy(CFG)
    cfg['paths']['prereg_md'] = str(tmp_path / 'PRE_REGISTRATION.md')
    cfg['paths']['prereg_json'] = str(tmp_path / 'PRE_REGISTRATION.json')
    return tmp_path, cfg


def test_direction_family_admits_only_powered_cells():
    admitted, alpha, table, history = P.direction_family({'D1': 346, 'H1': 6392}, 0.03, 0.80, 0.05)
    assert admitted == ['h1_gbm', 'h1_lstm']
    assert alpha == pytest.approx(0.025)
    assert {r['cell']: r['status'] for r in table}['d1_gbm'] == P.UNDERPOWERED
    assert history[0]['family_size'] == 4 and history[-1]['family_size'] == 2


def test_no_cell_powered_admits_nothing_and_spends_no_alpha_twice():
    admitted, alpha, _, _ = P.direction_family({'D1': 10, 'H1': 10}, 0.03, 0.80, 0.05)
    assert admitted == [] and alpha == pytest.approx(0.05)


def test_scoring_gate_requires_committed_unmodified_files(repo):
    path, cfg = repo
    assert not P.scoring_allowed(cfg, repo=str(path))                 # nothing exists
    (path / 'PRE_REGISTRATION.md').write_text('x')
    (path / 'PRE_REGISTRATION.json').write_text('{}')
    assert not P.scoring_allowed(cfg, repo=str(path))                 # uncommitted
    _git(path, 'add', '.')
    _git(path, 'commit', '-q', '-m', 'register')
    assert P.scoring_allowed(cfg, repo=str(path))                     # committed and clean
    (path / 'PRE_REGISTRATION.md').write_text('edited')
    assert not P.scoring_allowed(cfg, repo=str(path))                 # modified after commit


def test_generate_refuses_to_overwrite(repo):
    path, cfg = repo
    (path / 'PRE_REGISTRATION.json').write_text('{}')
    with pytest.raises(FileExistsError):
        P.generate(cfg)


def _frame(n=400):
    idx = pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC')
    c = 1500 + np.arange(n, dtype=float)
    return pd.DataFrame({'open': c, 'high': c + 1, 'low': c - 1, 'close': c}, index=idx)


def test_pinned_frame_detects_rewritten_rows_and_ignores_appended_ones():
    df = _frame()
    reg = {'data': {'D1': {'last_bar': str(df.index[-1]), 'rows': len(df), 'sha256': P.frame_digest(df)}}}
    grown = pd.concat([df, _frame(410).iloc[400:]])
    assert len(P.pinned_frame(grown, reg, 'D1')) == 400
    rewritten = df.copy()
    rewritten.iloc[10, 3] += 1.0
    with pytest.raises(ValueError):
        P.pinned_frame(rewritten, reg, 'D1')


def test_rendered_registration_states_gross_of_cost_and_expected_outcomes():
    df = _frame(600)
    reg = P.build_registration(CFG, {'D1': df, 'H1': df}, {'D1': {'first': 'a'}, 'H1': {'first': 'a'}},
                               {'n_weeks': 1, 'match_rate': 1.0})
    md = P.render_markdown(reg)
    assert 'GROSS of cost' in md and 'Expected outcomes' in md
    assert reg['ci_method']['block_length'] == {'D1': 5, 'H1': 24}
