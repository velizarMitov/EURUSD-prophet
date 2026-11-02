"""Main-environment caller of the side-environment worker (design D1).

Every call goes to `.venv-tsfm\\Scripts\\python.exe -I src/external/tsfm/worker.py`
with one JSON request on stdin. A call NEVER raises: a missing environment, a
timeout (the process is killed), a crash or an error reply all come back as
{'ok': False, 'error': ...}, so the caller records a failure and goes on with
the other model (spec "The models run apart from the application").
"""

from __future__ import annotations

import json
import os
import subprocess

from . import REPO, VENV_PYTHON, repo_path

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker.py')
TIMEOUT_S = 120


def venv_python(repo: str = REPO) -> str:
    return repo_path(VENV_PYTHON, repo)


def available(repo: str = REPO) -> bool:
    return os.path.exists(venv_python(repo))


def _tail(text: str | None, n: int = 400) -> str:
    text = (text or '').strip()
    return text[-n:] if text else 'no output'


def call(request: dict, timeout: float | None = TIMEOUT_S, python: str | None = None,
         runner=subprocess.run, log_path: str | None = None) -> dict:
    """The worker's reply, or {'ok': False, 'error': ...}. `timeout=None` is
    for the one-off GPU jobs (fine-tuning, replay), whose progress goes to
    `log_path`."""
    python = python or venv_python()
    if not os.path.exists(python):
        return {'ok': False, 'error': f'external environment unavailable: {python} not found '
                                      '(run scripts/external_tsfm/install.ps1)'}
    log = open(log_path, 'a', encoding='utf-8') if log_path else None
    try:
        proc = runner([python, '-I', WORKER], input=json.dumps(request), capture_output=log is None,
                      stdout=subprocess.PIPE if log is not None else None, stderr=log,
                      text=True, encoding='utf-8', timeout=timeout, cwd=REPO)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'worker timed out after {timeout:.0f} s and was stopped'}
    except OSError as e:
        return {'ok': False, 'error': f'worker could not start: {e}'}
    finally:
        if log is not None:
            log.close()
    try:
        reply = json.loads(proc.stdout or '')
    except ValueError:
        return {'ok': False, 'error': f'worker crashed (exit {proc.returncode}): {_tail(proc.stderr)}'}
    if proc.returncode != 0 or not isinstance(reply, dict) or not reply.get('ok'):
        err = reply.get('error') if isinstance(reply, dict) else None
        return {'ok': False, 'error': err or f'worker exit {proc.returncode}: {_tail(proc.stderr)}'}
    return reply
