"""Read-only status pass for EUR/USD Prophet (openspec change
`prophet-operator-skill`, capability `operator-skill`).

One run prints one JSON document: a level per health signal, plus performance
numbers that are reported and never graded. Three properties are load-bearing:

  * IT CHANGES NOTHING. HTTP goes through `fetch_json`, which allows only GET and
    only the paths in ALLOWED_PATHS. `/api/paper-trading` rewrites the ledger
    CSVs and `/api/retrain/status` can hot-reload the models, so both are read
    from their files instead. No file is ever opened for writing.
  * IT DOES NOT IMPORT THE APP. `src.*` pulls in TensorFlow and the model
    loaders, which are exactly what break; the pass must still report when they
    do, and when the server is down.
  * HEALTH IS NOT PERFORMANCE. A level answers "is the machine working". Ledger
    and hit-rate numbers are `info`, so a losing ledger never reads as "broken"
    and a winning one never reads as "trade it".

Usage (from the repo root):
    .venv\\Scripts\\python.exe .claude\\skills\\prophet\\scripts\\status.py [--fast]
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

REPO_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

# The only HTTP the pass may make. Each is read-only by the app's own contract.
ALLOWED_PATHS = ('/api/home', '/api/provenance', '/api/gold/health')

TASK_NAMES = ('EURUSDProphet-ForwardLogger', 'EURUSDProphet-GoldLogger',
              'EURUSDProphet-GoldRetrain', 'EURUSDProphet-Refit')
# 0 ok, 0x41301 running, 0x41303 never run, 0x41325 queued.
NON_FAILURE_RESULTS = frozenset({0, 0x41301, 0x41303, 0x41325})

# Mirrored, not imported (see the module docstring). tests/test_prophet_skill.py
# reads the app's sources as text and fails if these drift.
RETRAIN_STALL_SECONDS = 300           # api._STALL_SECONDS
MIN_SETTLED = 30                      # src/home_summary.py MIN_SETTLED
WILSON_Z = 1.959963984540054          # src/home_summary.py WILSON_Z

FORWARD_STALE_HOURS = 2
FAILURE_WINDOW_HOURS = 24
LOG_TAIL_LINES = 200
HTTP_TIMEOUT_S = 15
RELIABILITY_TIMEOUT_S = 60

# green < unknown < yellow < red. `info` never takes part.
_RANK = {'green': 0, 'unknown': 1, 'yellow': 2, 'red': 3}
_MARKER = '=== RETRAIN EXIT'


# ── plumbing ────────────────────────────────────────────────────────────────

def fetch_json(base_url: str, path: str, params: dict | None = None,
               timeout: float = HTTP_TIMEOUT_S, method: str = 'GET'):
    """The single door to the app. Anything but a GET to an allowed path is
    refused before a socket is opened."""
    if method != 'GET':
        raise PermissionError(f'status pass is read-only: {method} refused')
    if path not in ALLOWED_PATHS:
        raise PermissionError(f'path not in the read-only allowlist: {path}')
    url = base_url.rstrip('/') + path + ('?' + urlencode(params) if params else '')
    req = Request(url, method='GET', headers={'Accept': 'application/json'})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def signal(sid: str, level: str, summary: str, see: str, **details) -> dict:
    return {'id': sid, 'level': level, 'summary': summary, 'see': see, 'details': details}


def guarded(sid: str, see: str, fn, *args, **kwargs) -> dict:
    """One failing source degrades its own signal and nothing else."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 -- a report must never die on one source
        return signal(sid, 'unknown', f'{type(e).__name__}: {e}'[:300], see)


def overall_level(signals: list[dict]) -> str:
    levels = [s['level'] for s in signals if s['level'] in _RANK]
    return max(levels, key=_RANK.__getitem__) if levels else 'unknown'


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def _parse_time(s) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def tail_lines(path: str, n: int, block: int = 65536) -> list[str]:
    """Last n lines without reading a multi-megabyte log whole."""
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        data = b''
        while pos > 0 and data.count(b'\n') <= n:
            step = min(block, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = data.decode('utf-8', errors='replace').splitlines()
    return lines[-n:]


def _csv_tail(path: str, n: int) -> list[dict]:
    """The header plus the last n rows as dicts. A half-written last row is
    dropped rather than misread."""
    with open(path, encoding='utf-8', errors='replace') as f:
        header = next(csv.reader([f.readline()]))
    rows = []
    for line in tail_lines(path, n):
        try:
            vals = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            continue
        if len(vals) == len(header) and vals != header:
            rows.append(dict(zip(header, vals)))
    return rows


def wilson_interval(hits: int, n: int, z: float = WILSON_Z):
    if n <= 0:
        return None, None
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


# ── server process ──────────────────────────────────────────────────────────

_PROBE_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
function Iso($d) { if ($d) { $d.ToUniversalTime().ToString('o') } else { $null } }
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
  [ordered]@{ pid = [int]$_.ProcessId; ppid = [int]$_.ParentProcessId; exe = $_.ExecutablePath;
              cmd = $_.CommandLine; created = (Iso $_.CreationDate) } })
$tasks = @(foreach ($n in @(%TASKS%)) {
  $t = Get-ScheduledTask -TaskName $n
  if ($t) { $i = $t | Get-ScheduledTaskInfo
    [ordered]@{ name = $n; missing = $false; state = [string]$t.State;
                last_result = [int64]$i.LastTaskResult; last_run = (Iso $i.LastRunTime) } }
  else { [ordered]@{ name = $n; missing = $true } } })
[ordered]@{ python = $py; tasks = $tasks } | ConvertTo-Json -Depth 4 -Compress
"""


def run_probe() -> dict:
    """Processes and scheduled tasks in one PowerShell start (~2 s)."""
    script = _PROBE_PS.replace('%TASKS%', ','.join(f"'{n}'" for n in TASK_NAMES))
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    out = subprocess.run(
        ['powershell', '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        capture_output=True, timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f'PowerShell probe failed (rc={out.returncode}): '
                           f'{out.stderr.decode(errors="replace")[:200]}')
    return json.loads(out.stdout.decode('utf-8', errors='replace'))


def _as_list(v) -> list:
    """PowerShell 5.1 collapses a one-item array to the item."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def port_answers(base_url: str, timeout: float = 2.0) -> bool:
    """A plain TCP connect: no HTTP request, so the allowlist is untouched."""
    parts = urlsplit(base_url)
    try:
        with socket.create_connection((parts.hostname or '127.0.0.1', parts.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def our_servers(processes: list[dict], repo: str) -> list[dict]:
    """The launcher's rule (scripts/launch.vbs OurServer): a python.exe under
    this repo's .venv whose command line runs api.py. The venv relaunches the
    base interpreter as a child with the same command line; that child's exe is
    not under .venv, so only the parent counts, and its start time is the
    server's."""
    prefix = os.path.join(repo, '.venv', '').lower()
    ours = [p for p in _as_list(processes)
            if str(p.get('exe') or '').lower().startswith(prefix)
            and 'api.py' in str(p.get('cmd') or '').lower()]
    return sorted(ours, key=lambda p: p.get('created') or '')


def newest_source(repo: str):
    """(mtime, relative path) of the newest api.py / src/**/*.py. HTML never
    counts: it is served from disk on each request."""
    best = (os.path.getmtime(os.path.join(repo, 'api.py')), 'api.py')
    for root, dirs, files in os.walk(os.path.join(repo, 'src')):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for name in files:
            if name.endswith('.py'):
                p = os.path.join(root, name)
                m = os.path.getmtime(p)
                if m > best[0]:
                    best = (m, os.path.relpath(p, repo).replace('\\', '/'))
    return best


def classify_server(probe: dict, answers: bool, newest: tuple, repo: str) -> dict:
    see = 'research_models/server/server.log'
    ours = our_servers(probe.get('python'), repo)
    if not answers and not ours:
        return signal('server', 'red', 'The app is not running: nothing answers on port 8000.', see)
    if not answers:
        p = ours[0]
        return signal('server', 'red',
                      f"The app's server process exists (pid {p['pid']}, started {p.get('created')}) "
                      'but port 8000 does not answer: still starting, or hung.', see, pid=p['pid'])
    if not ours:
        return signal('server', 'red',
                      'Port 8000 answers, but not from the project server the launcher owns '
                      '(another program, or a uvicorn dev server). The launcher will refuse to start.',
                      see)
    p = ours[0]
    started = _parse_time(p.get('created'))
    mtime, path = newest
    if started is not None and mtime > started.timestamp():
        return signal('server', 'yellow',
                      f'The server is running code older than what is on disk: {path} changed at '
                      f'{_iso(mtime)}, after the server started at {_iso(started.timestamp())}. '
                      'The launcher restarts it on the next start.', see,
                      pid=p['pid'], started=_iso(started.timestamp()), newest_file=path)
    return signal('server', 'green',
                  f"Running since {_iso(started.timestamp()) if started else '?'}, code current.", see,
                  pid=p['pid'])


def classify_tasks(probe: dict) -> dict:
    see = 'Task Scheduler (EURUSDProphet-*)'
    tasks = {t.get('name'): t for t in _as_list(probe.get('tasks'))}
    red, yellow = [], []
    for name in TASK_NAMES:
        t = tasks.get(name)
        if t is None or t.get('missing'):
            red.append(f'{name} is not registered')
        elif str(t.get('state')) == 'Disabled':
            red.append(f'{name} is disabled')
        elif int(t.get('last_result') or 0) not in NON_FAILURE_RESULTS:
            yellow.append(f"{name} last run ({t.get('last_run')}) ended with "
                          f"0x{int(t['last_result']) & 0xFFFFFFFF:08X}")
    details = {'tasks': list(tasks.values())}
    if red:
        return signal('tasks', 'red', '; '.join(red + yellow) + '.', see, **details)
    if yellow:
        return signal('tasks', 'yellow', '; '.join(yellow) + '.', see, **details)
    return signal('tasks', 'green', 'All four scheduled tasks are registered and their last runs succeeded.',
                  see, **details)


# ── app endpoints ───────────────────────────────────────────────────────────

def classify_forecast(home: dict) -> dict:
    see = '/ (home page; its refresh button makes a new forecast)'
    t = home.get('tomorrow') or {}
    if not t.get('available'):
        return signal('forecast', 'red', f"The app has no forecast to show ({t.get('reason', 'no reason given')}).",
                      see)
    date = t.get('forecasting_date')
    details = {k: t.get(k) for k in ('forecasting_date', 'as_of_date', 'logged_at', 'key',
                                     'variants_disagree', 'variant')}
    if t.get('outdated'):
        return signal('forecast', 'yellow',
                      f"The latest forecast is for {date} (made {t.get('logged_at')}) and is outdated: "
                      "no forecast has been made for the next session yet.", see, **details)
    return signal('forecast', 'green', f"The forecast for {date} is current ({t.get('key')}).", see, **details)


def classify_provenance(prov: dict) -> dict:
    see = '/advanced (provenance banner) and GET /api/provenance'
    if not prov.get('available'):
        return signal('provenance', 'unknown', f"Provenance cannot be checked: {prov.get('reason')}", see)
    if not prov.get('undeclared'):
        return signal('provenance', 'green',
                      f"All {prov.get('n_pinned')} pinned models/ artifacts match the declared set.", see)
    diverged = _as_list(prov.get('diverged'))
    missing = _as_list(prov.get('missing'))
    conflicting = _as_list(prov.get('conflicting'))
    n = len(diverged) + len(missing) + len(conflicting)
    since = prov.get('undeclared_since')
    days = prov.get('undeclared_days')
    return signal('provenance', 'red',
                  f'{n} served models/ artifact(s) differ from the declared set, the oldest since {since}'
                  + (f' ({days} days)' if days is not None else '') + '.',
                  see, diverged=[d.get('path') for d in diverged], missing=missing,
                  conflicting=conflicting, undeclared_since=since, undeclared_days=days)


def classify_gold(health: dict) -> dict:
    see = '/gold (logger warning)'
    missed = int(health.get('missed_24h') or 0)
    hours = health.get('hours_since_success')
    details = {'market_open': health.get('market_open'), 'hours_since_success': hours,
               'missed_24h': missed, 'missed_bars': health.get('missed_bars')}
    if health.get('stalled'):
        return signal('gold', 'red',
                      f'The gold logger is stalled: last success {hours} h ago, {missed} missed hour(s) in the '
                      'last 24.', see, **details)
    if missed > 0:
        return signal('gold', 'yellow', f'The gold logger missed {missed} hour(s) in the last 24.', see, **details)
    return signal('gold', 'green', 'The gold logger is keeping up.', see, **details)


def _http_signal(sid: str, see: str, base_url: str, path: str, classify) -> dict:
    try:
        payload = fetch_json(base_url, path)
    except PermissionError:
        raise
    except Exception as e:  # noqa: BLE001 -- server down or slow reads as unknown
        return signal(sid, 'unknown', f'{path} did not answer ({type(e).__name__}: {e})'[:300], see)
    return classify(payload)


# ── files ───────────────────────────────────────────────────────────────────

def classify_retrain(repo: str, now: float | None = None) -> dict:
    """api._resolve_state, rebuilt from its files: the state JSON, the exit
    marker in the log and the log's age."""
    see = 'results/retrain.log'
    now = time.time() if now is None else now
    state_path = os.path.join(repo, 'results', 'retrain_state.json')
    log_path = os.path.join(repo, 'results', 'retrain.log')
    try:
        with open(state_path, encoding='utf-8') as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    if not state.get('started_at'):
        return signal('retrain', 'green', 'No retrain on record.', see)

    rc, at = state.get('returncode'), None
    try:
        for line in reversed(tail_lines(log_path, 50)):
            if line.startswith(_MARKER):
                m = re.search(r'rc=(-?\d+)', line)
                a = re.search(r'at=(.+?) ===', line)
                if rc is None and m:
                    rc = int(m.group(1))
                at = f'{a.group(1)} (local time)' if a else None
                break
        age = max(0.0, now - os.path.getmtime(log_path))
    except OSError:
        age = None
    when = at or _iso(state.get('finished_at'))
    details = {'started': _iso(state.get('started_at')), 'finished': when, 'returncode': rc}
    if rc == 0:
        return signal('retrain', 'green', f'The last retrain completed at {when}.', see, **details)
    if rc is not None:
        return signal('retrain', 'red', f'The last retrain failed (rc={rc}) at {when}.', see, **details)
    if age is not None and age < RETRAIN_STALL_SECONDS:
        return signal('retrain', 'yellow', f'A retrain is running; its log was written {int(age)} s ago.',
                      see, **details)
    return signal('retrain', 'red',
                  'The last retrain stopped without a completion marker (killed, hung or the server restarted); '
                  'the artifacts may be from a partial run.', see, **details)


def fx_market_open(now: datetime) -> bool:
    """Monday 00:00 to Friday 21:00 UTC: narrower than the real session, so a
    Sunday evening or a late Friday can never read as a stall."""
    wd = now.weekday()
    return wd < 4 or (wd == 4 and now.hour < 21)


def classify_forward(repo: str, now: datetime | None = None) -> dict:
    see = '/forecasts and results/forward_eval/'
    now = now or datetime.now(timezone.utc)
    base = os.path.join(repo, 'results', 'forward_eval')
    rows = _csv_tail(os.path.join(base, 'predictions.csv'), 50)
    times = [t for t in (_parse_time(r.get('logged_at')) for r in rows) if t]
    newest = max(times) if times else None

    failures = []
    fpath = os.path.join(base, 'failures.csv')
    if os.path.exists(fpath):
        cutoff = now - timedelta(hours=FAILURE_WINDOW_HOURS)
        for r in _csv_tail(fpath, 500):
            t = _parse_time(r.get('logged_at'))
            if t and t >= cutoff:
                failures.append(r.get('error', '')[:160])
    details = {'newest_logged': newest.isoformat() if newest else None,
               'failures_24h': len(failures), 'recent_failures': failures[-3:]}
    stale = newest is None or (now - newest) > timedelta(hours=FORWARD_STALE_HOURS)
    if stale and fx_market_open(now):
        return signal('forward_logger', 'red',
                      f"The forward logger has logged nothing since {newest.isoformat() if newest else 'ever'} "
                      'while the FX market is open.', see, **details)
    if failures:
        return signal('forward_logger', 'yellow',
                      f'{len(failures)} logger failure(s) in the last 24 h; newest prediction '
                      f'{newest.isoformat() if newest else "none"}.', see, **details)
    return signal('forward_logger', 'green',
                  f"The forward logger is current (newest {newest.isoformat() if newest else 'none'}).",
                  see, **details)


def classify_server_log(repo: str) -> dict:
    see = 'research_models/server/server.log'
    path = os.path.join(repo, 'research_models', 'server', 'server.log')
    if not os.path.exists(path):
        return signal('server_log', 'unknown', 'There is no server.log yet.', see)
    lines = tail_lines(path, LOG_TAIL_LINES)
    last_tb = max((i for i, ln in enumerate(lines) if ln.startswith('Traceback')), default=None)
    errors = [ln for ln in lines if re.match(r'^ERROR\b|.*\bERROR:', ln)]
    if last_tb is not None:
        quote = next((ln for ln in lines[last_tb + 1:]
                      if ln and not ln[0].isspace() and not ln.startswith(('Traceback', 'During handling',
                                                                          'The above exception'))),
                     lines[last_tb])
        return signal('server_log', 'yellow', f'A traceback is in the last {LOG_TAIL_LINES} log lines: '
                      f'{quote.strip()[:200]}', see)
    if errors:
        return signal('server_log', 'yellow', f'An ERROR line is in the last {LOG_TAIL_LINES} log lines: '
                      f'{errors[-1].strip()[:200]}', see)
    return signal('server_log', 'green', f'No traceback or ERROR in the last {LOG_TAIL_LINES} log lines.', see)


# ── performance (info only) ─────────────────────────────────────────────────

def _info(pid: str, see: str, **fields) -> dict:
    return {'id': pid, 'level': 'info', 'see': see, **fields}


def _rate_fields(hits: int, n: int) -> dict:
    lo, hi = wilson_interval(hits, n)
    return {'n': n, 'hits': hits, 'rate': round(hits / n, 4) if n else None,
            'wilson_95': None if lo is None else [round(lo, 4), round(hi, 4)],
            'too_early': n < MIN_SETTLED}


def ledger_entries(repo: str) -> list[dict]:
    """Read from the persisted CSVs; /api/paper-trading would rewrite them."""
    see = '/paper-trading'
    try:
        with open(os.path.join(repo, 'config.json'), encoding='utf-8') as f:
            ledgers = json.load(f).get('paper_trading', {}).get('ledgers') or {}
    except (OSError, ValueError):
        ledgers = {}
    if not ledgers:
        ledgers = {'baseline': {'log_path': 'results/paper_trading_log_baseline.csv'},
                   'with_macro': {'log_path': 'results/paper_trading_log_macro.csv'}}
    out = []
    for variant, cfg in ledgers.items():
        path = os.path.join(repo, cfg['log_path'])
        pid = f'ledger_{variant}'
        try:
            with open(path, encoding='utf-8', newline='') as f:
                taken = [r for r in csv.DictReader(f) if r.get('direction') != 'FLAT']
            wins = sum(r.get('outcome') == 'win' for r in taken)
            net = round(sum(float(r.get('net_pips') or 0) for r in taken), 1)
            out.append(_info(pid, see, variant=variant, **_rate_fields(wins, len(taken)),
                             net_pips=net, rate_is='win rate net of spread',
                             last_forecasting_date=taken[-1]['forecasting_date'] if taken else None,
                             file_built=_iso(os.path.getmtime(path))))
        except Exception as e:  # noqa: BLE001
            out.append(_info(pid, see, variant=variant, available=False, reason=f'{type(e).__name__}: {e}'[:200]))
    return out


def reliability_entry(base_url: str) -> dict:
    see = '/ (home page reliability strip)'
    try:
        r = fetch_json(base_url, '/api/home', {'part': 'reliability'}, timeout=RELIABILITY_TIMEOUT_S)
        rel = r.get('reliability') or {}
        if not rel.get('available'):
            return _info('direction_hit_rate', see, available=False, reason=rel.get('reason'))
        return _info('direction_hit_rate', see, rate_is='direction right, before cost (baseline)',
                     **_rate_fields(int(rel['hits']), int(rel['n'])))
    except PermissionError:
        raise
    except Exception as e:  # noqa: BLE001
        return _info('direction_hit_rate', see, available=False, reason=f'{type(e).__name__}: {e}'[:200])


# ── assembly ────────────────────────────────────────────────────────────────

def build_report(repo: str, base_url: str, fast: bool = False, probe: dict | None = None,
                 probe_error: str | None = None) -> dict:
    signals = []
    if probe is None:
        why = probe_error or 'process probe unavailable'
        signals.append(signal('server', 'unknown', why, 'research_models/server/server.log'))
    else:
        signals.append(guarded('server', 'research_models/server/server.log', classify_server,
                               probe, port_answers(base_url), newest_source(repo), repo))
    signals += [
        guarded('forecast', '/', _http_signal, 'forecast', '/', base_url, '/api/home', classify_forecast),
        guarded('provenance', '/advanced', _http_signal, 'provenance', '/advanced', base_url,
                '/api/provenance', classify_provenance),
        guarded('gold', '/gold', _http_signal, 'gold', '/gold', base_url, '/api/gold/health', classify_gold),
        guarded('retrain', 'results/retrain.log', classify_retrain, repo),
    ]
    if probe is None:
        signals.append(signal('tasks', 'unknown', probe_error or 'process probe unavailable',
                              'Task Scheduler (EURUSDProphet-*)'))
    else:
        signals.append(guarded('tasks', 'Task Scheduler (EURUSDProphet-*)', classify_tasks, probe))
    signals += [
        guarded('forward_logger', 'results/forward_eval/', classify_forward, repo),
        guarded('server_log', 'research_models/server/server.log', classify_server_log, repo),
    ]
    performance = ledger_entries(repo)
    if not fast:
        performance.append(reliability_entry(base_url))
    return {'checked_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'overall': overall_level(signals), 'signals': signals, 'performance': performance}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Read-only status pass for EUR/USD Prophet.')
    ap.add_argument('--fast', action='store_true', help='skip the slow hit-rate call')
    ap.add_argument('--repo', default=REPO_DEFAULT, help='repository root (tests)')
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL, help='server address (tests)')
    ap.add_argument('--probe-file', help='recorded probe JSON instead of PowerShell (tests)')
    args = ap.parse_args(argv)
    try:
        repo = os.path.abspath(args.repo)
        probe, probe_error = None, None
        try:
            if args.probe_file:
                with open(args.probe_file, encoding='utf-8') as f:
                    probe = json.load(f)
            else:
                probe = run_probe()
        except Exception as e:  # noqa: BLE001
            probe_error = f'process probe failed: {type(e).__name__}: {e}'[:300]
        report = build_report(repo, args.base_url, fast=args.fast, probe=probe, probe_error=probe_error)
    except Exception as e:  # noqa: BLE001 -- still print something parseable
        print(json.dumps({'overall': 'unknown', 'error': f'{type(e).__name__}: {e}'}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
