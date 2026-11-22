"""fx-ny-orb-hot, group 4: the power report and the pre-registration gate."""
import copy
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.fx_ny_orb import labels as LB
from src.fx_ny_orb import load_config
from src.fx_ny_orb import prereg as PR
from src.fx_ny_orb import rule as R
from src.fx_ny_orb.registry import ConfigChanged

CFG = load_config()


def synth(n_days=300, seed=0):
    """Days with flat M15 sessions and occasional planted breakouts; no outcome is ever needed."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2019-01-02', periods=n_days, tz='UTC')
    rows, bars = [], {p: {} for p in CFG['pairs']}
    for pair in CFG['pairs']:
        for d in dates:
            idx = pd.date_range(d + pd.Timedelta(hours=14), d + pd.Timedelta(hours=20, minutes=45), freq='15min')
            lv = np.full(len(idx), 1.1)
            k = rng.integers(6, 24)
            lv[k:] += rng.choice([-1, 1]) * rng.uniform(0.0003, 0.0020)
            bars[pair][d] = pd.DataFrame({'mo': lv, 'mh': lv + 0.0002, 'ml': lv - 0.0002, 'mc': lv,
                                          'spread_price': 0.00005, 'corrupt': False}, index=idx)
            p = rng.normal(-5.5, 0.3)
            rows.append({'pair': pair, 'date': d, 'p_harx': p, 'hot_threshold': -5.3, 'hot': p > -5.3, 'y': p,
                         'dst_mismatch': 0.0, 'is_nfp_day': 0.0, 'is_fomc_day': 0.0, 'close14': 1.1,
                         'F': float(np.exp(p) * 1.1), 'day_reason': ''})
    return pd.DataFrame(rows), bars


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(tmp_path / 'results')
    return c


def test_label_report_succeeds_while_the_exit_scan_raises(cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('no exit may be computed before the pre-registration')
    monkeypatch.setattr(R, 'scan_exit', boom)
    path = LB.write_report(cfg, 'T', synth())
    text = open(path, encoding='utf-8').read()
    assert text.splitlines()[2].startswith('> **Net of the recorded spread') and '## Power — before any outcome' in text
    for word in ('Win rate', 'net R |', 'take profit', 'stop loss'):
        assert word not in text


def test_label_report_refuses_a_changed_config(cfg):
    LB.write_report(cfg, 'T', synth(80))
    other = copy.deepcopy(cfg)
    other['rule']['target_f'] = 0.70
    with pytest.raises(ConfigChanged):
        LB.write_report(other, 'T2', synth(80))


def git(repo, *args):
    return subprocess.run(['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args], cwd=repo, check=True,
                          capture_output=True, text=True)


def test_git_gate_refuses_missing_untracked_and_modified_and_passes_a_clean_commit(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    c = copy.deepcopy(CFG)
    c['paths']['results'] = str(repo / 'results')
    with pytest.raises(PR.PreregNotCommitted, match='does not exist'):
        PR.gate(c, str(repo))
    LB.write_report(c, 'T', synth(60))
    md, js = PR.write(c)
    with pytest.raises(PR.PreregNotCommitted, match='not committed'):
        PR.gate(c, str(repo))
    git(repo, 'add', '-A')
    git(repo, 'commit', '-q', '-m', 'prereg')
    assert PR.gate(c, str(repo))[:4] == pd.Timestamp.now().strftime('%Y')
    with open(js, 'a', encoding='utf-8') as fh:
        fh.write(' ')
    with pytest.raises(PR.PreregNotCommitted, match='differs'):
        PR.gate(c, str(repo))
    git(repo, 'checkout', '--', '.')
    other = copy.deepcopy(c)
    other['rule']['stop_f'] = 0.30
    with pytest.raises(PR.PreregNotCommitted, match='another config'):
        PR.gate(other, str(repo))
    text = open(md, encoding='utf-8').read()
    assert 'orb_hot_net_r' in text and 'HAR-X' in text and 'were not computed' in text and os.path.exists(js)
