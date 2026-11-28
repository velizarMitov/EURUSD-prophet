"""gold-nyrange-forward, group 3: the pre-registration, its git gate and the pinned coefficients."""
import copy
import json
import shutil
import subprocess

import pandas as pd
import pytest

from src.gold_nyfwd import prereg as PR
from src.gold_nyfwd.registry import ConfigChanged
from tests._gold_nyfwd_fakes import CFG, common_files_in_tmp  # noqa: F401


def git(repo, *args):
    return subprocess.run(['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args], cwd=repo, check=True,
                          capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / 'repo'
    r.mkdir()
    git(r, 'init', '-q')
    c = copy.deepcopy(CFG)
    c['paths'] = {'results': str(r / 'results'), 'artifacts': str(r / 'art')}
    return r, c


def test_gate_refuses_missing_untracked_modified_and_other_config_and_passes_a_clean_commit(repo):
    r, c = repo
    with pytest.raises(PR.PreregNotCommitted, match='does not exist'):
        PR.gate(c, str(r))
    md, js = PR.write(c)
    with pytest.raises(PR.PreregNotCommitted, match='not committed'):
        PR.gate(c, str(r))
    git(r, 'add', '-A')
    git(r, 'commit', '-q', '-m', 'prereg')
    assert PR.gate(c, str(r))[:4] == pd.Timestamp.now().strftime('%Y')
    with open(js, 'a', encoding='utf-8') as fh:
        fh.write(' ')
    with pytest.raises(PR.PreregNotCommitted, match='differs'):
        PR.gate(c, str(r))
    git(r, 'checkout', '--', '.')
    other = copy.deepcopy(c)
    other['forward']['min_days'] = 90
    with pytest.raises(PR.PreregNotCommitted, match='another config'):
        PR.gate(other, str(r))
    with pytest.raises(ConfigChanged):
        PR.write(other)


def test_copied_coefficients_equal_gold_bbnr_and_power_is_stated(repo):
    _, c = repo
    md, js = PR.write(c)
    rec = json.load(open(js, encoding='utf-8'))
    src = PR.sources()
    ols = json.load(open(src['ols'], encoding='utf-8'))
    m = rec['model']
    assert m['coef_harx'] == ols['harx'] and m['coef_har'] == ols['har']
    assert m['columns'] == ols['columns'] and m['har_columns'] == ols['har_columns']
    assert len(m['coef_harx']) == 15 and len(m['coef_har']) == 4
    assert rec['alpha'] == 0.05 and rec['cell'] == 'gold_harx_vs_har_forward'
    assert 0.30 < rec['power']['power'] < 0.40
    text = open(md, encoding='utf-8').read()
    assert 'About 34%' in text and 'A DROP is not evidence that the model fails' in text
    assert 'UNDERPOWERED — NO DECISION' in text and '2027-04-30' in text and 'USDInd' in text


def test_an_altered_ols_json_makes_the_coefficient_loader_refuse(repo, tmp_path):
    _, c = repo
    PR.write(c)
    copy_ = tmp_path / 'ols.json'
    shutil.copy(PR.sources()['ols'], copy_)
    assert PR.coefficients(c, str(copy_))['ols_sha256']
    with open(copy_, 'a', encoding='utf-8') as fh:
        fh.write('\n')
    with pytest.raises(PR.FrozenChanged):
        PR.coefficients(c, str(copy_))
    assert PR.coefficients(c, str(tmp_path / 'gone.json'))['coef_harx']   # a lost ols.json: the copy stands


def test_build_refuses_an_ols_json_that_differs_from_the_frozen_manifest(repo, tmp_path):
    _, c = repo
    src = PR.sources()
    bad = tmp_path / 'ols.json'
    shutil.copy(src['ols'], bad)
    with open(bad, 'a', encoding='utf-8') as fh:
        fh.write(' ')
    with pytest.raises(PR.FrozenChanged):
        PR.build(c, {**src, 'ols': str(bad)})
