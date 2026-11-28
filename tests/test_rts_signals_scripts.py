"""rts-signals tasks 6.2 / 7.1 / 7.2: the indicator source, the task and indicator scripts (dry run and
fake APPDATA only) and the RUNBOOK's commands."""

import os
import re
import shutil
import subprocess
import sys

import pytest

from src.rts_signals import REPO

WIN = pytest.mark.skipif(sys.platform != 'win32' or not shutil.which('powershell'), reason='Windows scripts')
SCRIPTS = os.path.join(REPO, 'scripts', 'rts_signals')
MQ5 = os.path.join(REPO, 'mql5', 'Indicators', 'ProphetRTS.mq5')


def ps(script, *args, env=None):
    return subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                           os.path.join(SCRIPTS, script), *args], capture_output=True, text=True, env=env, timeout=120)


def test_indicator_has_no_order_functions_and_no_lot():
    src = open(MQ5, encoding='utf-8').read()
    for bad in ('OrderSend', 'OrderSendAsync', 'OrderCheck', 'PositionOpen', 'PositionClose', 'PositionModify',
                'CTrade', 'Trade.mqh', 'TRADE_ACTION', 'OrderModify', 'OrderClose'):
        assert bad not in src, bad
    assert not re.search(r'\blot', src, re.IGNORECASE)
    for sym in ('EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'USDCHF', 'USDCAD'):
        assert f'"{sym}"' in src


@WIN
def test_dry_run_is_one_task_at_20():
    out = ps('install_tasks.ps1', '-DryRun')
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.startswith('DRY RUN')]
    assert len(lines) == 1
    assert '/TN EURUSDProphet-RTSInbox' in lines[0] and '/SC HOURLY' in lines[0] and '/ST 00:20' in lines[0]
    assert int(lines[0].split('(')[1].split(' ')[0]) <= 261


def test_uninstall_tasks_names_only_its_own_task():
    text = open(os.path.join(SCRIPTS, 'uninstall_tasks.ps1'), encoding='utf-8').read()
    assert "@('EURUSDProphet-RTSInbox')" in text
    for other in ('GoldNY', 'NYOrb', 'GoldLogger', 'ForwardLogger', 'ExternalLogger'):
        assert other not in text


@WIN
def test_indicator_uninstall_leaves_the_other_indicators_and_files(tmp_path):
    term = tmp_path / 'MetaQuotes' / 'Terminal' / 'ABCDEF0123456789'
    ind = term / 'MQL5' / 'Indicators' / 'EURUSDProphet'
    ind.mkdir(parents=True)
    (term / 'origin.txt').write_text(r'C:\fake\mt5', encoding='utf-8')
    keep = ['ProphetNYOrb.mq5', 'ProphetNYOrb.ex5', 'ProphetGoldSignals.mq5', 'ProphetGoldSignals.ex5',
            'ProphetGoldNYRange.mq5', 'ProphetGoldNYRange.ex5']
    for f in keep + ['ProphetRTS.mq5', 'ProphetRTS.ex5']:
        (ind / f).write_text('x', encoding='utf-8')
    common = tmp_path / 'common' / 'EURUSDProphet'
    common.mkdir(parents=True)
    for f in ('ny_orb.tsv', 'gold_signals.tsv', 'gold_ny_range.tsv', 'rts_signals.tsv'):
        (common / f).write_text('x', encoding='utf-8')
    env = {**os.environ, 'APPDATA': str(tmp_path), 'MT5_COMMON_FILES': str(tmp_path / 'common')}
    out = ps('uninstall_mt5_indicator.ps1', env=env)
    assert out.returncode == 0, out.stderr
    assert sorted(os.listdir(ind)) == sorted(keep)
    assert sorted(os.listdir(common)) == ['gold_ny_range.tsv', 'gold_signals.tsv', 'ny_orb.tsv']


@WIN
def test_authorize_status_reports_without_secrets():
    out = ps('authorize_gmail.ps1', '-Status')
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() in ('gmail authorised', 'gmail not authorised')


@pytest.mark.parametrize('module', ['run', 'ingest', 'bars', 'simulate', 'reconcile', 'history', 'report', 'score',
                                    'prereg', 'export', 'gmail', 'rules'])
def test_runbook_modules_answer_help(module):
    out = subprocess.run([sys.executable, '-m', f'src.rts_signals.{module}', '--help'], cwd=REPO,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
