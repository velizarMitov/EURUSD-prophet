"""gold-nyrange-forward, groups 4.3 and 6.2: the task and indicator scripts (dry run / fake APPDATA only)."""
import os
import shutil
import subprocess
import sys

import pytest

from src.gold_nyfwd import REPO

pytestmark = pytest.mark.skipif(sys.platform != 'win32' or not shutil.which('powershell'), reason='Windows scripts')
SCRIPTS = os.path.join(REPO, 'scripts', 'gold_nyfwd')


def ps(script, *args, env=None):
    return subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           os.path.join(SCRIPTS, script), *args], capture_output=True, text=True, env=env, timeout=120)


def test_dry_run_names_exactly_the_two_tasks_at_their_times():
    out = ps('install_tasks.ps1', '-DryRun')
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.startswith('DRY RUN')]
    assert len(lines) == 2
    assert '/TN EURUSDProphet-GoldNYForecast' in lines[0] and '/ST 16:08' in lines[0]
    assert '/TN EURUSDProphet-GoldNYSettle' in lines[1] and '/ST 23:08' in lines[1]
    assert all('MON,TUE,WED,THU,FRI' in l for l in lines)
    assert all(int(l.split('(')[1].split(' ')[0]) <= 261 for l in lines)


def test_uninstall_tasks_names_only_its_own_two_tasks():
    text = open(os.path.join(SCRIPTS, 'uninstall_tasks.ps1'), encoding='utf-8').read()
    assert "@('EURUSDProphet-GoldNYForecast', 'EURUSDProphet-GoldNYSettle')" in text
    assert 'NYOrb' not in text and 'GoldLogger' not in text


def test_indicator_uninstall_leaves_the_other_indicators_and_files(tmp_path):
    term = tmp_path / 'MetaQuotes' / 'Terminal' / 'ABCDEF0123456789'
    ind = term / 'MQL5' / 'Indicators' / 'EURUSDProphet'
    ind.mkdir(parents=True)
    (term / 'origin.txt').write_text(r'C:\fake\mt5', encoding='utf-8')
    keep = ['ProphetNYOrb.mq5', 'ProphetNYOrb.ex5', 'ProphetGoldSignals.mq5', 'ProphetGoldSignals.ex5']
    for f in keep + ['ProphetGoldNYRange.mq5', 'ProphetGoldNYRange.ex5']:
        (ind / f).write_text('x', encoding='utf-8')
    common = tmp_path / 'common' / 'EURUSDProphet'
    common.mkdir(parents=True)
    for f in ('ny_orb.tsv', 'gold_signals.tsv', 'gold_ny_range.tsv'):
        (common / f).write_text('x', encoding='utf-8')
    env = {**os.environ, 'APPDATA': str(tmp_path), 'MT5_COMMON_FILES': str(tmp_path / 'common')}
    out = ps('uninstall_mt5_indicator.ps1', env=env)
    assert out.returncode == 0, out.stderr
    assert sorted(os.listdir(ind)) == sorted(keep)
    assert sorted(os.listdir(common)) == ['gold_signals.tsv', 'ny_orb.tsv']
