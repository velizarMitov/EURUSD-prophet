"""The `prophet` project skill (openspec change `prophet-operator-skill`).

The skill's status pass must never change the system it reports on, must grade
health and never performance, and must keep working when the server is down.
These tests pin those properties, the signal rules, and the skill's documents.
"""
import csv
import importlib.util
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(REPO, '.claude', 'skills', 'prophet')
STATUS = os.path.join(SKILL, 'scripts', 'status.py')


def _load():
    spec = importlib.util.spec_from_file_location('prophet_status', STATUS)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True          # no __pycache__ inside the skill
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


S = _load()


def _dead_url() -> str:
    """A localhost port that nothing listens on."""
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return f'http://127.0.0.1:{port}'


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)


def _mini_repo(tmp_path) -> str:
    root = str(tmp_path / 'repo')
    _write(os.path.join(root, 'api.py'), '# app\n')
    _write(os.path.join(root, 'src', 'home_summary.py'), '# src\n')
    return root


# Captured from the live app on 2026-10-29 (trimmed to the fields the pass reads).
HOME = {'generated_at': '2026-10-29 14:59',
        'tomorrow': {'available': True, 'key': 'slightly_up', 'forecasting_date': '2026-10-28',
                     'as_of_date': '2026-10-27', 'logged_at': '2026-10-28 10:22', 'outdated': True,
                     'variants_disagree': False, 'variant': 'baseline'}}
PROVENANCE = {'available': True, 'n_pinned': 44, 'undeclared': True,
              'diverged': [{'path': f'models/x{i}.pkl', 'modified_at': '2026-10-14 04:24:32 UTC'}
                           for i in range(7)],
              'missing': [], 'conflicting': [],
              'undeclared_since': '2026-10-14 04:24:32 UTC', 'undeclared_days': 15.4}
GOLD = {'market_open': True, 'hours_since_success': 0.97, 'missed_24h': 7, 'stalled': True,
        'missed_bars': ['2026-10-28 16:00:00+00:00']}


# ── 1. skeleton and the read-only guard ─────────────────────────────────────

@pytest.mark.parametrize('method', ['POST', 'PUT', 'DELETE'])
def test_only_get_is_allowed(method):
    with pytest.raises(PermissionError):
        S.fetch_json(_dead_url(), '/api/home', method=method)


@pytest.mark.parametrize('path', ['/api/paper-trading', '/api/retrain/status', '/api/predict',
                                  '/api/retrain', '/history', '/'])
def test_only_allowlisted_paths(path):
    with pytest.raises(PermissionError):
        S.fetch_json(_dead_url(), path)


def test_allowlist_is_exactly_the_three_read_only_endpoints():
    assert S.ALLOWED_PATHS == ('/api/home', '/api/provenance', '/api/gold/health')


def test_status_source_names_no_other_api_path():
    src = open(STATUS, encoding='utf-8').read()
    paths = set(re.findall(r"""['"](/api/[\w/.-]*)['"]""", src))
    assert paths <= set(S.ALLOWED_PATHS), paths - set(S.ALLOWED_PATHS)


def test_a_raising_source_is_unknown_and_the_rest_still_report():
    def boom():
        raise ValueError('bad payload')
    sig = S.guarded('gold', '/gold', boom)
    assert sig['level'] == 'unknown' and 'ValueError' in sig['summary']
    assert S.overall_level([sig, S.signal('x', 'green', 'ok', '')]) == 'unknown'


def test_overall_is_the_worst_health_level_and_ignores_info():
    lv = lambda *ls: [{'level': l} for l in ls]  # noqa: E731
    assert S.overall_level(lv('green', 'red', 'yellow')) == 'red'
    assert S.overall_level(lv('green', 'yellow', 'unknown')) == 'yellow'
    assert S.overall_level(lv('green', 'info')) == 'green'


def test_output_is_ascii_so_the_cp1252_console_cannot_crash(tmp_path, capsys):
    repo = _mini_repo(tmp_path)
    probe = tmp_path / 'probe.json'
    probe.write_text(json.dumps({'python': [], 'tasks': []}), encoding='utf-8')
    rc = S.main(['--repo', repo, '--base-url', _dead_url(), '--probe-file', str(probe), '--fast'])
    out = capsys.readouterr().out
    assert rc == 0
    out.encode('cp1252')
    assert out.isascii()
    assert json.loads(out)['overall'] == 'red'          # nothing answers -> server red


# ── 2.1 endpoint signals ────────────────────────────────────────────────────

def test_outdated_forecast_is_yellow_and_names_its_date():
    sig = S.classify_forecast(HOME)
    assert sig['level'] == 'yellow' and '2026-10-28' in sig['summary']


def test_missing_forecast_is_red():
    assert S.classify_forecast({'tomorrow': {'available': False, 'reason': 'no log'}})['level'] == 'red'


def test_current_forecast_is_green():
    home = {'tomorrow': {**HOME['tomorrow'], 'outdated': False}}
    assert S.classify_forecast(home)['level'] == 'green'


def test_undeclared_provenance_is_red_with_count_and_date():
    sig = S.classify_provenance(PROVENANCE)
    assert sig['level'] == 'red'
    assert '7 served' in sig['summary'] and '2026-10-14' in sig['summary']


def test_declared_provenance_is_green():
    assert S.classify_provenance({'available': True, 'undeclared': False, 'n_pinned': 44})['level'] == 'green'


def test_stalled_gold_logger_is_red_and_names_missed_hours():
    sig = S.classify_gold(GOLD)
    assert sig['level'] == 'red' and '7 missed' in sig['summary']


def test_gold_missed_hours_without_stall_is_yellow():
    assert S.classify_gold({**GOLD, 'stalled': False})['level'] == 'yellow'
    assert S.classify_gold({**GOLD, 'stalled': False, 'missed_24h': 0})['level'] == 'green'


@pytest.mark.parametrize('sid,path,classify', [
    ('forecast', '/api/home', 'classify_forecast'),
    ('provenance', '/api/provenance', 'classify_provenance'),
    ('gold', '/api/gold/health', 'classify_gold'),
])
def test_server_down_makes_endpoint_signals_unknown(sid, path, classify):
    sig = S._http_signal(sid, '', _dead_url(), path, getattr(S, classify))
    assert sig['level'] == 'unknown'


# ── 2.2 retrain from files ──────────────────────────────────────────────────

def _retrain(tmp_path, state, log, log_age_s=10_000):
    repo = _mini_repo(tmp_path)
    if state is not None:
        _write(os.path.join(repo, 'results', 'retrain_state.json'), json.dumps(state))
    if log is not None:
        p = os.path.join(repo, 'results', 'retrain.log')
        _write(p, log)
        now = 2_000_000_000.0
        os.utime(p, (now - log_age_s, now - log_age_s))
        return S.classify_retrain(repo, now=now)
    return S.classify_retrain(repo, now=2_000_000_000.0)


def test_completed_retrain_is_green_and_names_the_date(tmp_path):
    sig = _retrain(tmp_path, {'started_at': 1.0, 'returncode': 0, 'finished_at': 2.0},
                   '=== DONE ===\n\n=== RETRAIN EXIT rc=0 elapsed=4356.8s at=2026-10-14 08:17:59 ===\n')
    assert sig['level'] == 'green' and '2026-10-14' in sig['summary']


def test_failed_retrain_is_red(tmp_path):
    sig = _retrain(tmp_path, {'started_at': 1.0, 'returncode': None},
                   'boom\n=== RETRAIN EXIT rc=1 elapsed=3.0s at=2026-10-14 08:17:59 ===\n')
    assert sig['level'] == 'red' and 'rc=1' in sig['summary']


def test_running_retrain_is_yellow(tmp_path):
    sig = _retrain(tmp_path, {'started_at': 1.0, 'returncode': None}, 'epoch 3/50\n', log_age_s=20)
    assert sig['level'] == 'yellow'


def test_retrain_silent_without_marker_is_red(tmp_path):
    sig = _retrain(tmp_path, {'started_at': 1.0, 'returncode': None}, 'epoch 3/50\n', log_age_s=301)
    assert sig['level'] == 'red' and 'without a completion marker' in sig['summary']


def test_no_retrain_on_record_is_green(tmp_path):
    assert _retrain(tmp_path, None, None)['level'] == 'green'


def test_retrain_stall_threshold_matches_the_app():
    api = open(os.path.join(REPO, 'api.py'), encoding='utf-8').read()
    m = re.search(r'^_STALL_SECONDS\s*=\s*(\d+)', api, re.M)
    assert m and int(m.group(1)) == S.RETRAIN_STALL_SECONDS
    marker = re.search(r'^_MARKER\s*=\s*"([^"]+)"', api, re.M)
    assert marker and marker.group(1) == S._MARKER


# ── 2.3 forward logger and server log ───────────────────────────────────────

PRED_HEADER = 'key,model,logged_at\n'
FAIL_HEADER = 'model,cadence,horizon,as_of_bar,error,logged_at\n'


def _forward(tmp_path, newest, failures=()):
    repo = _mini_repo(tmp_path)
    base = os.path.join(repo, 'results', 'forward_eval')
    _write(os.path.join(base, 'predictions.csv'),
           PRED_HEADER + ''.join(f'k{i},m,{t}\n' for i, t in enumerate(['2026-10-01T00:00:00+00:00', newest])))
    _write(os.path.join(base, 'failures.csv'),
           FAIL_HEADER + ''.join(f'*,*,,,"MT5Unavailable: IPC timeout",{t}\n' for t in failures))
    return repo


def test_forward_logger_stale_on_a_weekday_is_red(tmp_path):
    repo = _forward(tmp_path, '2026-10-28T08:00:00+00:00')
    sig = S.classify_forward(repo, now=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc))   # Wednesday
    assert sig['level'] == 'red'


def test_forward_logger_quiet_on_a_saturday_is_green(tmp_path):
    repo = _forward(tmp_path, '2026-10-30T20:45:00+00:00')
    sig = S.classify_forward(repo, now=datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc))   # Saturday
    assert sig['level'] == 'green'


def test_forward_logger_recent_failures_are_yellow(tmp_path):
    repo = _forward(tmp_path, '2026-10-29T11:45:00+00:00',
                    failures=['2026-10-28T16:03:39+00:00', '2026-10-29T07:03:41+00:00',
                              '2026-10-20T07:03:41+00:00'])
    sig = S.classify_forward(repo, now=datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc))
    assert sig['level'] == 'yellow' and sig['summary'].startswith('2 ')


def test_fx_market_window():
    at = lambda d, h: datetime(2026, 10, d, h, tzinfo=timezone.utc)  # noqa: E731
    assert S.fx_market_open(at(26, 0))          # Monday 00:00
    assert S.fx_market_open(at(30, 20))         # Friday 20:00
    assert not S.fx_market_open(at(30, 21))     # Friday 21:00
    assert not S.fx_market_open(at(25, 22))     # Sunday evening


def _server_log(tmp_path, text):
    repo = _mini_repo(tmp_path)
    _write(os.path.join(repo, 'research_models', 'server', 'server.log'), text)
    return S.classify_server_log(repo)


def test_traceback_in_the_server_log_is_yellow_and_quoted(tmp_path):
    sig = _server_log(tmp_path, 'INFO: ok\nTraceback (most recent call last):\n'
                                '  File "api.py", line 1, in <module>\n    x()\nValueError: boom\nINFO: ok\n')
    assert sig['level'] == 'yellow' and 'ValueError: boom' in sig['summary']


def test_error_line_in_the_server_log_is_yellow(tmp_path):
    sig = _server_log(tmp_path, 'INFO: ok\nERROR:    Exception in ASGI application\n')
    assert sig['level'] == 'yellow' and 'ASGI' in sig['summary']


def test_clean_server_log_is_green_and_a_missing_one_unknown(tmp_path):
    assert _server_log(tmp_path, 'INFO:     127.0.0.1 - "GET / HTTP/1.1" 200 OK\n')['level'] == 'green'
    assert S.classify_server_log(_mini_repo(tmp_path / 'other'))['level'] == 'unknown'


# ── 2.4 process and task probe ──────────────────────────────────────────────

def _procs(repo, created='2026-10-29T04:26:22.4679390Z'):
    venv = os.path.join(repo, '.venv', 'Scripts', 'python.exe')
    cmd = f'"{venv}"  -u api.py '
    return [{'pid': 32132, 'ppid': 1, 'exe': venv, 'cmd': cmd, 'created': created},
            # the venv relaunch: base interpreter, same command line
            {'pid': 19052, 'ppid': 32132, 'exe': r'C:\Python313\python.exe', 'cmd': cmd,
             'created': '2026-10-29T04:26:22.5061540Z'},
            {'pid': 7, 'ppid': 1, 'exe': r'C:\Python312\python.exe', 'cmd': 'python -m json.tool',
             'created': '2026-10-29T14:00:00Z'}]


STARTED = datetime(2026, 10, 29, 4, 26, 22, tzinfo=timezone.utc).timestamp()


def test_only_the_venv_parent_is_our_server(tmp_path):
    repo = str(tmp_path)
    assert [p['pid'] for p in S.our_servers(_procs(repo), repo)] == [32132]


def test_current_server_is_green(tmp_path):
    repo = str(tmp_path)
    sig = S.classify_server({'python': _procs(repo)}, True, (STARTED - 60, 'api.py'), repo)
    assert sig['level'] == 'green'


def test_stale_code_is_yellow_and_names_the_file(tmp_path):
    repo = str(tmp_path)
    sig = S.classify_server({'python': _procs(repo)}, True, (STARTED + 3600, 'src/home_summary.py'), repo)
    assert sig['level'] == 'yellow' and 'src/home_summary.py' in sig['summary']


def test_port_answering_without_our_server_is_foreign_and_red(tmp_path):
    repo = str(tmp_path)
    sig = S.classify_server({'python': _procs(repo)[2:]}, True, (STARTED, 'api.py'), repo)
    assert sig['level'] == 'red' and 'launcher owns' in sig['summary']


def test_nothing_answering_is_red(tmp_path):
    repo = str(tmp_path)
    assert S.classify_server({'python': []}, False, (STARTED, 'api.py'), repo)['level'] == 'red'
    assert S.classify_server({'python': _procs(repo)}, False, (STARTED, 'api.py'), repo)['level'] == 'red'


def test_one_process_object_from_powershell_is_accepted(tmp_path):
    repo = str(tmp_path)
    one = _procs(repo)[0]                       # PowerShell 5.1 collapses a 1-item array
    assert S.classify_server({'python': one}, True, (STARTED - 60, 'api.py'), repo)['level'] == 'green'


def _tasks(**over):
    base = {n: {'name': n, 'missing': False, 'state': 'Ready', 'last_result': 0,
                'last_run': '2026-10-29T14:31:01Z'} for n in S.TASK_NAMES}
    for n, v in over.items():
        base[f'EURUSDProphet-{n}'].update(v)
    return {'tasks': list(base.values())}


def test_all_tasks_fine_is_green():
    assert S.classify_tasks(_tasks())['level'] == 'green'


def test_missing_task_is_red_and_named():
    sig = S.classify_tasks(_tasks(GoldLogger={'missing': True}))
    assert sig['level'] == 'red' and 'EURUSDProphet-GoldLogger' in sig['summary']


def test_disabled_task_is_red():
    assert S.classify_tasks(_tasks(Refit={'state': 'Disabled'}))['level'] == 'red'


def test_running_or_never_run_is_not_a_failure():
    assert S.classify_tasks(_tasks(Refit={'last_result': 0x41301}))['level'] == 'green'
    assert S.classify_tasks(_tasks(GoldRetrain={'last_result': 0x41303}))['level'] == 'green'


def test_failure_result_is_yellow_with_its_hex_code():
    sig = S.classify_tasks(_tasks(Refit={'last_result': 2147946720}))
    assert sig['level'] == 'yellow' and '0x800710E0' in sig['summary']


# ── 2.5 performance is info ─────────────────────────────────────────────────

def _ledger_repo(tmp_path, n=53, wins=19, net_total=-464.8):
    repo = _mini_repo(tmp_path)
    _write(os.path.join(repo, 'config.json'), json.dumps({'paper_trading': {'ledgers': {
        'baseline': {'log_path': 'results/paper_trading_log_baseline.csv'}}}}))
    path = os.path.join(repo, 'results', 'paper_trading_log_baseline.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['as_of_date', 'forecasting_date', 'direction', 'net_pips', 'outcome'])
        per = net_total / n
        for i in range(n):
            w.writerow(['2026-07-03', f'2026-07-{i % 28 + 1:02d}', 'SHORT', round(per, 6),
                        'win' if i < wins else 'loss'])
        w.writerow(['2026-10-27', '2026-10-28', 'FLAT', 0, 'flat'])     # not a position
    return repo


def test_losing_ledger_is_info_and_never_moves_overall(tmp_path):
    entry = S.ledger_entries(_ledger_repo(tmp_path))[0]
    assert entry['level'] == 'info'
    assert entry['n'] == 53 and entry['hits'] == 19 and entry['net_pips'] == -464.8
    lo, hi = entry['wilson_95']
    assert lo < 19 / 53 < hi and entry['too_early'] is False
    signals = [S.signal('x', 'green', 'ok', '')]
    assert S.overall_level(signals + [entry]) == 'green'


def test_small_ledger_is_too_early(tmp_path):
    entry = S.ledger_entries(_ledger_repo(tmp_path, n=12, wins=5, net_total=10.0))[0]
    assert entry['too_early'] is True


def test_reliability_unavailable_when_server_down():
    entry = S.reliability_entry(_dead_url())
    assert entry['level'] == 'info' and entry['available'] is False


def test_wilson_matches_the_home_page():
    from src.home_summary import wilson_lower
    for hits, n in [(19, 53), (0, 30), (30, 30), (17, 31), (520, 1000)]:
        assert abs(S.wilson_interval(hits, n)[0] - wilson_lower(hits, n)) < 1e-9


# ── 3. read-only, end to end ────────────────────────────────────────────────

def _fixture_tree(root: str) -> None:
    _write(os.path.join(root, 'api.py'), '# app\n')
    _write(os.path.join(root, 'src', 'home_summary.py'), '# src\n')
    _write(os.path.join(root, 'config.json'), json.dumps({'paper_trading': {'ledgers': {
        'baseline': {'log_path': 'results/paper_trading_log_baseline.csv'}}}}))
    _write(os.path.join(root, 'results', 'paper_trading_log_baseline.csv'),
           'as_of_date,forecasting_date,direction,net_pips,outcome\n2026-10-27,2026-10-28,SHORT,-3.0,loss\n')
    _write(os.path.join(root, 'results', 'retrain_state.json'),
           json.dumps({'pid': 1, 'started_at': 1.0, 'returncode': 0, 'finished_at': 2.0}))
    _write(os.path.join(root, 'results', 'retrain.log'),
           '=== RETRAIN EXIT rc=0 elapsed=1.0s at=2026-10-14 08:17:59 ===\n')
    _write(os.path.join(root, 'results', 'forward_eval', 'predictions.csv'),
           PRED_HEADER + 'k,m,2026-10-29T11:45:00+00:00\n')
    _write(os.path.join(root, 'results', 'forward_eval', 'failures.csv'), FAIL_HEADER)
    _write(os.path.join(root, 'research_models', 'server', 'server.log'), 'INFO: started\n')


def _snapshot(root: str) -> dict:
    import hashlib
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = os.path.join(dirpath, name)
            with open(p, 'rb') as f:
                out[os.path.relpath(p, root)] = (os.stat(p).st_mtime_ns, hashlib.sha256(f.read()).hexdigest())
    return out


def test_the_pass_only_gets_allowlisted_paths_and_changes_no_file(tmp_path):
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    seen = []
    replies = {'/api/home': HOME, '/api/provenance': PROVENANCE, '/api/gold/health': GOLD}

    class Handler(BaseHTTPRequestHandler):
        def _any(self):
            parts = urlsplit(self.path)
            seen.append((self.command, parts.path, parts.query))
            body = replies.get(parts.path)
            if parts.query == 'part=reliability':
                body = {'reliability': {'available': True, 'hits': 20, 'n': 40}}
            data = json.dumps(body or {}).encode()
            self.send_response(200 if body else 404)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        do_GET = do_POST = do_PUT = do_DELETE = _any

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        root = str(tmp_path / 'repo')
        _fixture_tree(root)
        venv = os.path.join(root, '.venv', 'Scripts', 'python.exe')
        probe = tmp_path / 'probe.json'
        probe.write_text(json.dumps({
            'python': {'pid': 5, 'ppid': 1, 'exe': venv, 'cmd': f'"{venv}" -u api.py',
                       'created': '2099-01-01T00:00:00Z'},
            'tasks': [{'name': n, 'missing': False, 'state': 'Ready', 'last_result': 0} for n in S.TASK_NAMES],
        }), encoding='utf-8')
        before = _snapshot(root)
        out = subprocess.run(
            [sys.executable, STATUS, '--repo', root, '--probe-file', str(probe),
             '--base-url', f'http://127.0.0.1:{server.server_address[1]}'],
            capture_output=True, timeout=120)
        after = _snapshot(root)
    finally:
        server.shutdown()

    assert out.returncode == 0, out.stderr.decode(errors='replace')
    assert out.stdout.decode('ascii')                     # ASCII only
    report = json.loads(out.stdout)
    levels = {s['id']: s['level'] for s in report['signals']}
    assert levels == {'server': 'green', 'forecast': 'yellow', 'provenance': 'red', 'gold': 'red',
                      'retrain': 'green', 'tasks': 'green', 'forward_logger': levels['forward_logger'],
                      'server_log': 'green'}
    assert report['overall'] == 'red'
    assert {p['id'] for p in report['performance']} == {'ledger_baseline', 'direction_hit_rate'}

    assert seen, 'the pass made no request at all'
    for method, path, _q in seen:
        assert method == 'GET' and path in S.ALLOWED_PATHS, (method, path)
    assert after == before, 'the status pass changed a file'


# ── 4. the skill documents ──────────────────────────────────────────────────

def _doc(*parts) -> str:
    return open(os.path.join(SKILL, *parts), encoding='utf-8').read()


def test_launch_doc_matches_the_pinned_launcher():
    from tests.test_launcher import _launcher
    vbs, doc = _launcher(), _doc('references', 'launch.md')
    port = re.search(r'Const URL = "http://127\.0\.0\.1:(\d+)/"', vbs).group(1)
    budget = re.search(r'Const START_BUDGET_S = (\d+)', vbs).group(1)
    assert f'127.0.0.1:{port}' in doc
    assert f'{budget} seconds' in doc and f'{budget} s' in doc
    assert 'research_models\\server\\server.log' in doc and '\\research_models\\server' in vbs
    assert 'Start EUR-USD Prophet.vbs' in doc
    assert os.path.exists(os.path.join(REPO, 'Start EUR-USD Prophet.vbs'))


def _signal_ids() -> set:
    return set(re.findall(r"""signal\('(\w+)'""", open(STATUS, encoding='utf-8').read()))


def test_every_signal_has_a_reading_section():
    doc = _doc('references', 'reading-status.md')
    headings = set(re.findall(r'^## (\w+)', doc, re.M))
    ids = _signal_ids()
    assert ids == {'server', 'forecast', 'provenance', 'gold', 'retrain', 'tasks',
                   'forward_logger', 'server_log'}
    assert ids <= headings, ids - headings
    assert '## Performance' in doc


def _skill_files() -> list:
    out = []
    for dirpath, _dirs, files in os.walk(SKILL):
        out += [os.path.join(dirpath, n) for n in files if n.endswith(('.md', '.py', '.json'))]
    return out


def test_project_rules_are_linked_not_copied():
    """No line of CLAUDE.md long enough to be a rule appears verbatim in the skill."""
    rules = [ln.strip() for ln in open(os.path.join(REPO, 'CLAUDE.md'), encoding='utf-8')]
    rules = [ln.lstrip('-*0123456789. ') for ln in rules if len(ln) >= 50]
    for path in _skill_files():
        text = open(path, encoding='utf-8').read()
        copied = [ln for ln in rules if ln and ln in text]
        assert not copied, f'{os.path.relpath(path, SKILL)} copies CLAUDE.md: {copied[:2]}'
    for sig in ('0.05/9', 'Bonferroni', 'global_scaler'):
        assert not any(sig in open(p, encoding='utf-8').read() for p in _skill_files()), sig


def _frontmatter() -> tuple[dict, str]:
    text = _doc('SKILL.md')
    m = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.S)
    assert m, 'SKILL.md must open with a --- frontmatter block'
    meta = {}
    for line in m.group(1).splitlines():
        key, _, val = line.partition(':')
        meta[key.strip()] = val.strip().strip('"')
    return meta, m.group(2)


def test_skill_frontmatter_names_and_describes_the_skill():
    meta, _body = _frontmatter()
    assert meta['name'] == os.path.basename(SKILL) == 'prophet'
    desc = meta['description']
    assert len(desc) <= 1024
    assert re.search(r'[Ѐ-ӿ]', desc), 'Bulgarian trigger phrases'
    for phrase in ('start', '/run', 'status', 'пусни', 'не тръгва', 'Do NOT use'):
        assert phrase in desc, phrase


def test_skill_body_is_short_and_its_links_exist():
    _meta, body = _frontmatter()
    assert len(body.splitlines()) <= 150
    linked = set(re.findall(r'(references/[\w-]+\.md|scripts[\\/][\w-]+\.py)', body))
    assert {'references/launch.md', 'references/reading-status.md'} <= linked
    for rel in linked:
        assert os.path.exists(os.path.join(SKILL, rel.replace('\\', '/'))), rel


def test_skill_states_the_hard_limits():
    _meta, body = _frontmatter()
    for must in ('POST /api/predict', 'POST /api/retrain', 'GET /api/paper-trading',
                 'GET /api/retrain/status', 'No trading advice'):
        assert must in body, must


# ── 5. evals ────────────────────────────────────────────────────────────────

EVALS = os.path.join(SKILL, 'evals')


def test_trigger_queries_cover_both_sides():
    with open(os.path.join(EVALS, 'trigger-queries.json'), encoding='utf-8') as f:
        items = json.load(f)
    pos = [i for i in items if i['should_trigger'] is True]
    neg = [i for i in items if i['should_trigger'] is False]
    assert len(pos) >= 8 and len(neg) >= 6 and len(pos) + len(neg) == len(items)
    assert any(re.search(r'[Ѐ-ӿ]', i['query']) for i in pos), 'Bulgarian phrasings'
    assert len({i['query'] for i in items}) == len(items)


def test_behaviour_cases_are_in_the_skill_creator_layout():
    cases = sorted(d for d in os.listdir(EVALS) if os.path.isdir(os.path.join(EVALS, d)))
    assert cases == ['stale-code-asks-before-restart', 'status-leads-with-what-is-wrong',
                     'trade-question-gets-no-advice']
    for name in cases:
        base = os.path.join(EVALS, name)
        prompt = open(os.path.join(base, 'prompt.md'), encoding='utf-8').read()
        settings = re.match(r'^---\n(.*?)\n---\n\n\S', prompt, re.S)
        assert settings, f'{name}: prompt.md needs a settings block then the prompt'
        keys = {ln.split(':')[0] for ln in settings.group(1).splitlines()}
        assert keys <= {'tags', 'max_turns', 'timeout_seconds', 'allowed_tools'}, keys
        case = open(os.path.join(base, 'case.yaml'), encoding='utf-8').read()
        assert 'schema_version: "1.1"' in case and f'name: {name}\n' in case
        graders = sorted(os.listdir(os.path.join(base, 'graders')))
        assert graders, f'{name}: a case with no grader fails to load'
        for g in graders:
            assert re.match(r'^\d\d-[a-z0-9-]+\.md$', g), g
            text = open(os.path.join(base, 'graders', g), encoding='utf-8').read()
            assert text.startswith('---\ntype: llm\nfocus: trace\nweight: 1\n---\n\n'), g


def test_mirrored_constants_match_the_home_page():
    src = open(os.path.join(REPO, 'src', 'home_summary.py'), encoding='utf-8').read()
    assert int(re.search(r'^MIN_SETTLED\s*=\s*(\d+)', src, re.M).group(1)) == S.MIN_SETTLED
    assert float(re.search(r'^WILSON_Z\s*=\s*([\d.]+)', src, re.M).group(1)) == S.WILSON_Z
