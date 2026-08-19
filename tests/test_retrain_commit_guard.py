"""The commit-time guard that stops a retrain arriving under an unrelated message.

Three commits titled "Refactor code structure for improved readability and
maintainability" each contained a production retrain; `f2645a0` also re-read the
one-shot test block in `models/volatility/vol_metrics.json` while doing it. The
checksum fixtures caught the moved artifacts, but only after the fact.

`.githooks/commit-msg` refuses any commit that stages a path under `models/`
unless the message carries a `RETRAIN:` declaration. These tests cover the
decision logic directly, and then prove end to end that a real `git commit` is
actually blocked -- a hook that is merely correct in the abstract is worth
nothing if git never runs it.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK_DIR = os.path.join(REPO, '.githooks')
CHECKER = os.path.join(HOOK_DIR, 'check_retrain_declaration.py')


def _load():
    spec = importlib.util.spec_from_file_location('check_retrain_declaration', CHECKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guard = _load()


# ---------------------------------------------------------------------------
# 1. The files exist and are wired together
# ---------------------------------------------------------------------------

def test_hook_and_checker_are_committed_not_just_local():
    """.git/hooks/ is not version controlled, so the hook lives in .githooks/."""
    assert os.path.isfile(CHECKER)
    assert os.path.isfile(os.path.join(HOOK_DIR, 'commit-msg'))


def test_shim_invokes_the_checker():
    with open(os.path.join(HOOK_DIR, 'commit-msg'), encoding='utf-8') as fh:
        shim = fh.read()
    assert 'check_retrain_declaration.py' in shim
    assert '"$1"' in shim, 'the commit-message path must reach the checker'


# ---------------------------------------------------------------------------
# 2. Which paths the rule watches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('path', [
    'models/volatility/vol_metrics.json',
    'models/baseline/lstm_multitask_eurusd.keras',
    'models/with_macro/global_scaler.pkl',
    'models/calendar/calendar_volatility.json',
])
def test_paths_under_models_are_watched(path):
    assert guard.watched([path]) == [path]


@pytest.mark.parametrize('path', [
    'src/volatility.py',
    'results/eurusd_features.csv',
    'DATA.md',
    'tests/fixtures/h1_production_protected_sha256.json',
    'notebooks/00_final_report.ipynb',
])
def test_ordinary_paths_are_not_watched(path):
    assert guard.watched([path]) == []


def test_watched_picks_the_models_paths_out_of_a_mixed_commit():
    staged = ['DATA.md', 'models/volatility/vol_metrics.json', 'src/volatility.py',
              'models/baseline/lag_pca.pkl']
    assert guard.watched(staged) == ['models/baseline/lag_pca.pkl',
                                     'models/volatility/vol_metrics.json']


def test_a_directory_merely_starting_with_models_is_not_a_false_positive():
    assert guard.watched(['models_scratch/foo.pkl', 'src/models_report.py']) == []


# ---------------------------------------------------------------------------
# 3. What counts as a declaration
# ---------------------------------------------------------------------------

def test_a_declaration_with_content_is_accepted():
    assert guard.declares_retrain(
        'Refit the volatility ensemble\n\nRETRAIN: 5-seed ensemble on the 8,605-row set\n')


def test_a_declaration_is_found_anywhere_in_the_body():
    assert guard.declares_retrain('subject\n\nbody line\n\nRETRAIN: h1 LSTM, new bars\n\nmore\n')


def test_a_message_without_a_declaration_is_refused():
    assert not guard.declares_retrain(
        'Refactor code structure for improved readability and maintainability\n')


def test_a_bare_token_does_not_count():
    """The point is a declaration a reader can act on, not a hook silencer."""
    assert not guard.declares_retrain('subject\n\nRETRAIN:\n')
    assert not guard.declares_retrain('subject\n\nRETRAIN:   \n')


def test_a_declaration_inside_a_comment_line_does_not_count():
    """git strips `#` lines, so such a declaration would vanish from the commit."""
    assert not guard.declares_retrain('subject\n\n# RETRAIN: models were retrained\n')


def test_the_word_retrain_in_ordinary_prose_does_not_count():
    assert not guard.declares_retrain(
        'subject\n\nThis does not retrain anything. RETRAIN: is the required marker.\n')


def test_the_marker_is_case_sensitive():
    assert not guard.declares_retrain('subject\n\nretrain: lowercase\n')


# ---------------------------------------------------------------------------
# 4. End to end: git actually refuses the commit
# ---------------------------------------------------------------------------

def _git(*args, cwd, **kw):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, **kw)


@pytest.fixture
def sandbox(tmp_path):
    """A real repository with the real hook installed."""
    if shutil.which('git') is None:
        pytest.skip('git unavailable')
    root = tmp_path / 'repo'
    (root / '.githooks').mkdir(parents=True)
    shutil.copy(CHECKER, root / '.githooks' / 'check_retrain_declaration.py')
    shutil.copy(os.path.join(HOOK_DIR, 'commit-msg'), root / '.githooks' / 'commit-msg')

    assert _git('init', '-q', cwd=root).returncode == 0
    for k, v in (('user.email', 't@example.com'), ('user.name', 'test'),
                 ('commit.gpgsign', 'false'), ('core.hooksPath', '.githooks')):
        _git('config', k, v, cwd=root)

    (root / 'seed.txt').write_text('seed\n', encoding='utf-8')
    _git('add', 'seed.txt', cwd=root)
    assert _git('commit', '-q', '-m', 'seed', cwd=root).returncode == 0
    (root / 'models' / 'volatility').mkdir(parents=True)
    return root


def _head_subject(root):
    return _git('log', '-1', '--format=%s', cwd=root).stdout.strip()


def test_git_refuses_a_models_commit_with_a_refactor_message(sandbox):
    """The exact failure this guard was built for."""
    (sandbox / 'models' / 'volatility' / 'vol_metrics.json').write_text('{}\n', encoding='utf-8')
    _git('add', 'models/volatility/vol_metrics.json', cwd=sandbox)
    out = _git('commit', '-m',
               'Refactor code structure for improved readability and maintainability',
               cwd=sandbox)

    assert out.returncode != 0, 'the undeclared retrain was allowed through'
    assert 'COMMIT REFUSED' in out.stderr
    assert 'vol_metrics.json' in out.stderr, 'the offending path must be named'
    assert _head_subject(sandbox) == 'seed', 'nothing may reach the branch'


def test_git_accepts_the_same_commit_once_it_declares_the_retrain(sandbox):
    (sandbox / 'models' / 'volatility' / 'vol_metrics.json').write_text('{}\n', encoding='utf-8')
    _git('add', 'models/volatility/vol_metrics.json', cwd=sandbox)
    out = _git('commit', '-m',
               'Refit the volatility ensemble\n\nRETRAIN: 5-seed ensemble, 8,605-row set',
               cwd=sandbox)

    assert out.returncode == 0, out.stderr
    assert _head_subject(sandbox) == 'Refit the volatility ensemble'


def test_git_leaves_commits_that_do_not_touch_models_alone(sandbox):
    (sandbox / 'DATA.md').write_text('docs\n', encoding='utf-8')
    _git('add', 'DATA.md', cwd=sandbox)
    out = _git('commit', '-m', 'Refactor code structure for improved readability', cwd=sandbox)

    assert out.returncode == 0, out.stderr
    assert _head_subject(sandbox) == 'Refactor code structure for improved readability'


def test_a_mixed_commit_is_refused_on_the_models_paths(sandbox):
    """A retrain hiding inside a large documentation commit is the realistic case."""
    (sandbox / 'DATA.md').write_text('docs\n', encoding='utf-8')
    (sandbox / 'models' / 'volatility' / 'seed42.keras').write_text('weights\n', encoding='utf-8')
    _git('add', 'DATA.md', 'models/volatility/seed42.keras', cwd=sandbox)
    out = _git('commit', '-m', 'Update the data documentation', cwd=sandbox)

    assert out.returncode != 0
    assert 'seed42.keras' in out.stderr
    assert 'DATA.md' not in out.stderr, 'only the watched paths should be listed'


def test_a_deletion_under_models_is_not_treated_as_a_retrain(sandbox):
    """Removing an artifact is not producing one; --diff-filter excludes deletes."""
    (sandbox / 'models' / 'volatility' / 'old.pkl').write_text('x\n', encoding='utf-8')
    _git('add', 'models/volatility/old.pkl', cwd=sandbox)
    assert _git('commit', '-q', '-m', 'seed artifact\n\nRETRAIN: initial fit',
                cwd=sandbox).returncode == 0

    _git('rm', '-q', 'models/volatility/old.pkl', cwd=sandbox)
    out = _git('commit', '-m', 'Drop the superseded artifact', cwd=sandbox)
    assert out.returncode == 0, out.stderr


@pytest.mark.skipif(sys.platform == 'win32' and shutil.which('sh') is None,
                    reason='POSIX sh needed to run the hook shim')
def test_no_verify_bypasses_it_which_is_why_the_checksum_fixtures_still_exist(sandbox):
    """Documented limit, asserted so it stays documented rather than assumed."""
    (sandbox / 'models' / 'volatility' / 'vol_metrics.json').write_text('{}\n', encoding='utf-8')
    _git('add', 'models/volatility/vol_metrics.json', cwd=sandbox)
    out = _git('commit', '--no-verify', '-m', 'Refactor code structure', cwd=sandbox)

    assert out.returncode == 0
    assert _head_subject(sandbox) == 'Refactor code structure'
