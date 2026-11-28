"""rts-signals tasks 5.2 / 5.3: the git gate, the forward ledger, the one-shot cell and the plain report,
in a temporary git repository."""

import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.rts_signals import load_config
from src.rts_signals import prereg, report
from src.rts_signals.ingest import REPORTS, append_csv, rpath
from src.rts_signals.ledger import FIELDS, LEDGER, read, update
from src.rts_signals.registry import REGISTRY, AlreadyScored, StageOrderError
from src.rts_signals.score import UNDER, score

PAIRS = ['POUND', 'EURO', 'AUD', 'YEN', 'SWISS', 'CAD']


def git(repo, *a):
    subprocess.run(['git', *a], cwd=repo, check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_DATE='2026-11-30T10:00:00+01:00',
                            GIT_COMMITTER_DATE='2026-11-30T10:00:00+01:00'))


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / 'repo'
    r.mkdir()
    git(r, 'init', '-q')
    git(r, 'config', 'user.email', 'test@example.invalid')
    git(r, 'config', 'user.name', 'test')
    cfg = load_config()
    cfg['paths'] = dict(cfg['paths'], results=str(r / 'results' / 'rts_signals'), artifacts=str(tmp_path / 'art'))
    return r, cfg


def commit_prereg(r, cfg):
    prereg.write(cfg)
    git(r, 'add', '-A')
    git(r, 'commit', '-q', '-m', 'prereg')


def plant(cfg, values, start='2026-12-01'):
    """Eligible ledger rows, four a day on business days from `start`, with the given net R values."""
    days = pd.bdate_range(start, periods=int(np.ceil(len(values) / 4)))
    rows = []
    for k, v in enumerate(values):
        rows.append({**{f: '' for f in FIELDS}, 'report_date': str(days[k // 4].date()), 'pair': PAIRS[k % 4],
                     'status': 'filled', 'exit_reason': 'target' if v > 0 else 'stop', 'net_r': v, 'eligible': 'true'})
    append_csv(rpath(cfg, LEDGER), FIELDS, rows)


def regbytes(cfg):
    p = rpath(cfg, REGISTRY)
    return open(p, 'rb').read() if os.path.exists(p) else None


def test_untracked_prereg_refuses_and_writes_nothing(repo):
    r, cfg = repo
    prereg.write(cfg)
    with pytest.raises(prereg.PreregNotCommitted):
        score(cfg, today='2027-07-01', gate=lambda c: prereg.gate(c, repo=str(r)))
    assert regbytes(cfg) is None


def test_planted_edge_keep_and_rescore_refuses(repo):
    r, cfg = repo
    commit_prereg(r, cfg)
    g = lambda c: prereg.gate(c, repo=str(r))          # noqa: E731
    rng = np.random.default_rng(1)
    plant(cfg, list(0.6 + rng.normal(0, 1, 300)))
    row = score(cfg, today='2027-03-15', gate=g)
    assert row['verdict'] == 'KEEP' and row['n_trades'] == 300 and row['bound'] > 0
    before = regbytes(cfg)
    with pytest.raises(AlreadyScored):
        score(cfg, today='2027-03-16', gate=g)
    assert regbytes(cfg) == before


def test_zero_edge_drop(repo):
    r, cfg = repo
    commit_prereg(r, cfg)
    plant(cfg, [1.0, -1.0] * 150)
    assert score(cfg, today='2027-03-15', gate=lambda c: prereg.gate(c, repo=str(r)))['verdict'] == 'DROP'


def test_too_early_refuses_and_underpowered_at_deadline(repo):
    r, cfg = repo
    commit_prereg(r, cfg)
    g = lambda c: prereg.gate(c, repo=str(r))          # noqa: E731
    plant(cfg, [0.5] * 120)
    with pytest.raises(StageOrderError):
        score(cfg, today='2027-02-01', gate=g)
    assert regbytes(cfg) is None
    assert score(cfg, today='2027-06-30', gate=g)['verdict'] == UNDER


def test_late_report_is_excluded(repo):
    r, cfg = repo
    recs = pd.DataFrame([{'report_date': '2026-12-03', 'pair': 'EURO', 'sha256': 'late'},
                         {'report_date': '2026-12-03', 'pair': 'YEN', 'sha256': 'ontime'}])
    base = pd.DataFrame([{'report_date': '2026-12-03', 'pair': p, 'status': 'filled', 'exit_reason': 'target',
                          'net_r': 0.5} for p in ('EURO', 'YEN')])
    append_csv(rpath(cfg, REPORTS), ['sha256', 'received_utc'],
               [{'sha256': 'late', 'received_utc': '2026-12-10T08:00:00+00:00'},
                {'sha256': 'ontime', 'received_utc': '2026-12-03T21:30:00+00:00'}])
    assert update(cfg, recs, base, pd.Timestamp('2026-11-30', tz='UTC')) == 2
    led = read(cfg).set_index('pair')
    assert led.loc['EURO', 'eligible'] == 'false' and led.loc['EURO', 'status'].startswith('received late')
    assert led.loc['YEN', 'eligible'] == 'true'
    assert update(cfg, recs, base, pd.Timestamp('2026-11-30', tz='UTC')) == 0         # never re-added or changed


def test_report_before_scoring_has_no_mean(repo):
    r, cfg = repo
    plant(cfg, [0.7, -1.0] * 40)
    t = report.build(cfg, committed=pd.Timestamp('2026-11-30', tz='UTC'), today='2027-01-20')
    assert '80 / 300' in t and 'not scored yet' in t and '| target | stop | time |' in t
    for bad in ('mean', 'Mean', 'total', 'pips', 'KEEP', 'DROP'):
        assert bad not in t.split('_Generated')[1], bad
