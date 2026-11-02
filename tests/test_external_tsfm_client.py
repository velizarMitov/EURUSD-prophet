"""The worker call never hangs a pass and never raises (spec "The models run
apart from the application": scenarios "Side environment missing" and
"Worker hangs")."""
import json
import subprocess
import sys
from types import SimpleNamespace

from src.external.tsfm import client as C


def _python(tmp_path):
    p = tmp_path / 'python.exe'
    p.write_text('')
    return str(p)


def test_missing_environment_is_a_failure(tmp_path):
    r = C.call({'op': 'smoke'}, python=str(tmp_path / 'nope' / 'python.exe'))
    assert r['ok'] is False
    assert r['error'].startswith('external environment unavailable')


def test_timeout_is_a_failure_not_an_exception(tmp_path):
    def runner(*a, **kw):
        raise subprocess.TimeoutExpired(cmd='worker', timeout=kw['timeout'])
    r = C.call({'op': 'forecast'}, timeout=120, python=_python(tmp_path), runner=runner)
    assert r == {'ok': False, 'error': 'worker timed out after 120 s and was stopped'}


def test_crash_is_a_failure_not_an_exception(tmp_path):
    def runner(*a, **kw):
        return SimpleNamespace(returncode=3, stdout='', stderr='Traceback ...\nSegfault-ish')
    r = C.call({'op': 'forecast'}, python=_python(tmp_path), runner=runner)
    assert r['ok'] is False and 'worker crashed (exit 3)' in r['error']


def test_error_reply_is_passed_through(tmp_path):
    def runner(*a, **kw):
        return SimpleNamespace(returncode=1, stdout=json.dumps({'ok': False, 'error': 'ValueError: x'}), stderr='')
    assert C.call({'op': 'forecast'}, python=_python(tmp_path), runner=runner) == \
        {'ok': False, 'error': 'ValueError: x'}


def test_ok_reply(tmp_path):
    seen = {}

    def runner(cmd, **kw):
        seen['cmd'], seen['input'], seen['timeout'] = cmd, kw['input'], kw['timeout']
        return SimpleNamespace(returncode=0, stdout=json.dumps({'ok': True, 'deciles': {}}), stderr='')
    r = C.call({'op': 'forecast', 'model': 'chronos2'}, python=_python(tmp_path), runner=runner)
    assert r['ok'] is True
    assert seen['cmd'][1] == '-I' and seen['cmd'][2].endswith('worker.py')
    assert json.loads(seen['input'])['model'] == 'chronos2'
    assert seen['timeout'] == C.TIMEOUT_S


def test_a_real_hanging_child_is_stopped(tmp_path):
    """The real subprocess path: a child that sleeps past the limit is killed."""
    script = tmp_path / 'hang.py'
    script.write_text('import time; time.sleep(30)')

    def runner(cmd, **kw):
        return subprocess.run([sys.executable, str(script)], input=kw['input'], capture_output=True,
                              text=True, timeout=kw['timeout'])
    r = C.call({'op': 'forecast'}, timeout=1, python=_python(tmp_path), runner=runner)
    assert r['ok'] is False and 'timed out after 1 s' in r['error']
