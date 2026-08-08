"""Every external call in _train_pipeline.py must be bounded.

The 2026-08-07 retrain wedged at the Section 13 H1 fetch: MLflow recorded
H1_to_Daily_Ensemble as RUNNING with end_time NULL, no traceback was written,
and the run was watched for four hours. Section 12C already guarded its
subprocess with timeout=3600; the fetch one section later had no bound at all,
and `except Exception` does not catch a hang.

These tests exercise the real snippet lifted out of the pipeline source by AST,
so they cannot drift from the implementation they are guarding.
"""
import ast
import os
import subprocess
import sys
import textwrap
import time

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PIPELINE = os.path.join(REPO, '_train_pipeline.py')
sys.path.insert(0, REPO)

SOURCE = open(PIPELINE, encoding='utf-8').read()
TREE = ast.parse(SOURCE)


def _module_constant(name):
    """Resolve a module-level `NAME = <literal>` assignment."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.Assign) and node.targets \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} not found in the pipeline')


def _resolve(node):
    """A timeout may be written as a literal or as a named constant."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _module_constant(node.id)
    return None


def _subprocess_run_calls():
    """Every subprocess.run(...) in the pipeline, with its timeout kwarg."""
    found = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == 'run' \
                and isinstance(f.value, ast.Name) and f.value.id == 'subprocess':
            timeout = None
            for kw in node.keywords:
                if kw.arg == 'timeout':
                    timeout = _resolve(kw.value)
            found.append((node.lineno, timeout, node))
    return found


def _h1_fetch_snippet():
    """The inline `python -c` program the Section 13 guard actually runs."""
    for _lineno, _timeout, node in _subprocess_run_calls():
        argv = node.args[0]
        if not isinstance(argv, ast.List):
            continue
        consts = [e.value for e in argv.elts if isinstance(e, ast.Constant)]
        if any('fetch_h1_market_data' in c for c in consts if isinstance(c, str)):
            return next(c for c in consts if 'fetch_h1_market_data' in c)
    raise AssertionError('Section 13 H1 fetch subprocess not found in the pipeline')


def _h1_timeout_seconds():
    return _module_constant('H1_FETCH_TIMEOUT_S')


@pytest.fixture
def stub_repo(tmp_path):
    """A throwaway tree exposing src.live_data.fetch_h1_market_data, so the real
    snippet can be run against a fetch whose timing we control."""
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / '__init__.py').write_text('', encoding='utf-8')

    def install(body):
        (tmp_path / 'src' / 'live_data.py').write_text(
            textwrap.dedent(body), encoding='utf-8')
        return tmp_path
    return install


# ── 1. the timeout fires on a wedged fetch ─────────────────────────────────

def test_timeout_fires_when_the_fetch_never_returns(stub_repo):
    """The 2026-08-07 failure mode: a fetch that neither returns nor raises."""
    repo = stub_repo("""
        import time
        def fetch_h1_market_data(cache_path=None, **kw):
            time.sleep(120)          # the wedge
            return None, None
    """)
    t0 = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, '-c', _h1_fetch_snippet(), str(repo / 'h1.csv')],
            cwd=str(repo), capture_output=True, text=True, timeout=3,
        )
    elapsed = time.time() - t0
    assert elapsed < 30, f'timeout did not actually bound the call ({elapsed:.1f}s)'


def test_a_wedged_fetch_would_otherwise_run_unbounded(stub_repo):
    """Negative control: without the timeout the same call just keeps going."""
    repo = stub_repo("""
        import time
        def fetch_h1_market_data(cache_path=None, **kw):
            time.sleep(120)
            return None, None
    """)
    proc = subprocess.Popen(
        [sys.executable, '-c', _h1_fetch_snippet(), str(repo / 'h1.csv')],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(4)
        assert proc.poll() is None, 'stub should still be wedged -- control is invalid'
    finally:
        proc.kill()
        proc.wait()


# ── 2. a normal fetch is unaffected ────────────────────────────────────────

def test_fast_fetch_completes_and_reports_its_source(stub_repo):
    """A healthy fetch measures ~1.2s; behaviour must be unchanged."""
    repo = stub_repo("""
        def fetch_h1_market_data(cache_path=None, **kw):
            class _DF:
                def __len__(self): return 60000
            if cache_path:
                open(cache_path, 'w').write('stub')
            return _DF(), 'MT5'
    """)
    out = subprocess.run(
        [sys.executable, '-c', _h1_fetch_snippet(), str(repo / 'h1.csv')],
        cwd=str(repo), capture_output=True, text=True, timeout=_h1_timeout_seconds())

    assert out.returncode == 0, out.stderr
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith('H1FETCH|'))
    _, src, n = line.split('|', 2)
    assert src == 'MT5' and n == '60000'
    assert (repo / 'h1.csv').exists(), 'the cache the parent reads was not written'


def test_unreachable_fetch_still_degrades_gracefully(stub_repo):
    """An offline retrain must keep falling back to the cache, not abort. Only a
    *wedge* is fatal -- that distinction is the whole point of the guard."""
    repo = stub_repo("""
        def fetch_h1_market_data(cache_path=None, **kw):
            return None, None
    """)
    out = subprocess.run(
        [sys.executable, '-c', _h1_fetch_snippet(), str(repo / 'h1.csv')],
        cwd=str(repo), capture_output=True, text=True, timeout=_h1_timeout_seconds())
    assert out.returncode == 0
    assert 'H1FETCH|None|0' in out.stdout


# ── 3. the failure names itself, and cannot be swallowed ───────────────────

def test_timeout_raises_naming_the_section_and_the_call():
    handler = next(
        h for h in ast.walk(TREE)
        if isinstance(h, ast.ExceptHandler) and h.type is not None
        and 'TimeoutExpired' in ast.unparse(h.type))
    body = ast.unparse(handler)
    assert 'raise' in body, 'a timeout must raise, never print-and-continue'
    assert 'H1FetchTimeout' in body
    assert 'Section 13' in body, 'the message must name the section'
    assert 'fetch_h1_market_data' in body, 'the message must name the call'


def test_the_broad_h1_handler_cannot_swallow_the_timeout():
    """Section 13's `except Exception` exists so an offline retrain degrades
    gracefully. It must re-raise the timeout, or a wedge would still be
    reportable as a completed run -- exactly today's situation."""
    handlers = [h for h in ast.walk(TREE) if isinstance(h, ast.ExceptHandler)
                and h.type is not None and 'H1FetchTimeout' in ast.unparse(h.type)]
    assert handlers, 'no handler re-raises H1FetchTimeout'
    assert any(any(isinstance(n, ast.Raise) for n in ast.walk(h)) for h in handlers)


def test_h1_fetch_timeout_is_a_distinct_exception_type():
    cls = next((n for n in ast.walk(TREE)
                if isinstance(n, ast.ClassDef) and n.name == 'H1FetchTimeout'), None)
    assert cls is not None, 'the timeout needs its own type to survive the broad except'


# ── 4. the audit: every external call is bounded ───────────────────────────

def test_every_subprocess_call_has_a_timeout():
    calls = _subprocess_run_calls()
    assert len(calls) >= 2, 'expected the 12C TI-LSTM call and the Section 13 fetch'
    unbounded = [lineno for lineno, timeout, _ in calls if timeout is None]
    assert not unbounded, f'subprocess.run without timeout at line(s) {unbounded}'


def test_process_wide_socket_timeout_is_set():
    """fredapi and yfinance take no timeout from the pipeline and set none by
    default, so the socket default is the only bound available without editing
    those libraries."""
    call = next((n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'setdefaulttimeout'), None)
    assert call is not None, 'no socket.setdefaulttimeout -- FRED/yfinance unbounded'
    bound = next((n for n in ast.walk(TREE)
                  if isinstance(n, ast.Assign) and n.targets
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'NETWORK_SOCKET_TIMEOUT_S'), None)
    assert bound is not None and 0 < ast.literal_eval(bound.value) <= 300


def test_the_h1_bound_is_proportionate_to_a_healthy_fetch():
    """1.2s measured healthy; the bound must leave real headroom but still fire
    before the dashboard's 300s stall warning."""
    assert 60 <= _h1_timeout_seconds() <= 300


# ── 5. the failure is legible in the log ───────────────────────────────────

def test_timeout_traceback_lands_in_the_log_before_the_exit_marker(tmp_path, monkeypatch):
    """End to end with the retrain supervisor: a pipeline that dies on the H1
    timeout must leave a log whose last lines name BOTH where it stopped and how
    it ended. Answering "is it done, and did it work?" must never again require
    comparing artifact mtimes."""
    import api

    monkeypatch.setattr(api, 'RETRAIN_LOG', str(tmp_path / 'retrain.log'))
    monkeypatch.setattr(api, 'RETRAIN_STATE', str(tmp_path / 'retrain_state.json'))

    fake_pipeline = tmp_path / 'fake_pipeline.py'
    fake_pipeline.write_text(textwrap.dedent("""
        class H1FetchTimeout(RuntimeError):
            pass
        print("=== 12C. H1 TI-LSTM (observational, torch-backend subprocess) ===")
        print("=== 13. H1 -> Daily Predictor (XGBoost / RandomForest / SVM / LSTM) ===")
        raise H1FetchTimeout(
            "Section 13: H1 cache refresh (src.live_data.fetch_h1_market_data) "
            "exceeded 180s and was killed.")
    """), encoding='utf-8')

    logf = open(api.RETRAIN_LOG, 'w', encoding='utf-8')
    try:
        proc = subprocess.Popen(
            [sys.executable, '-u', str(fake_pipeline)],
            cwd=str(tmp_path), stdout=logf, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    finally:
        logf.close()
    api._watch_retrain(proc, time.time() - 42.0)

    log = open(api.RETRAIN_LOG, encoding='utf-8').read()
    lines = [ln for ln in log.strip().splitlines() if ln.strip()]

    assert lines[-1].startswith(api._MARKER), f'log must end with the marker: {lines[-1]!r}'
    assert 'rc=1' in lines[-1], f'a wedge must exit non-zero: {lines[-1]!r}'
    assert 'Section 13' in lines[-2], f'the line before must name the failure: {lines[-2]!r}'
    assert 'fetch_h1_market_data' in log
    # and the stage headers printed before the failure survived (unbuffered)
    assert '=== 13. H1 -> Daily Predictor' in log

    api._retrain.update(proc=None, started_at=None, returncode=None,
                        finished_at=None, pid=None, reloaded=False)
