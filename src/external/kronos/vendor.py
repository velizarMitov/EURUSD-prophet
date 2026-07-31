"""Resolve the upstream Kronos source at a PINNED commit.

The upstream repo is a dependency, not a copy: `from model import Kronos, ...`
must resolve against a KNOWN revision so an upstream change cannot silently
alter a result we have already reported. A hash mismatch RAISES -- it is never
warned about and never auto-corrected, because a silently different model is
exactly the failure this pin exists to prevent.

The checkout is NOT a git submodule. The core app has to install and run with no
Kronos dependency at all, and a submodule would pull 100MB of upstream on every
clone. Instead the checkout lives at KRONOS_UPSTREAM (env) or
src/external/kronos/upstream/, and `requirements-kronos.txt` documents the exact
clone command.
"""

import os
import subprocess
import sys

KRONOS_COMMIT = '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
KRONOS_REPO_URL = 'https://github.com/shiyu-coder/Kronos.git'

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UPSTREAM = os.path.join(_HERE, 'upstream')


class KronosUnavailable(RuntimeError):
    """Upstream source or its dependencies are missing. Always recoverable:
    callers degrade to an `available: false` response."""


class KronosPinMismatch(RuntimeError):
    """The checkout is not at KRONOS_COMMIT. NOT recoverable by degrading --
    it means results would be produced by unknown code."""


def upstream_path() -> str:
    return os.environ.get('KRONOS_UPSTREAM') or DEFAULT_UPSTREAM


def checkout_commit(path: str = None) -> str:
    """HEAD of the upstream checkout, or '' if it is not a git repo."""
    path = path or upstream_path()
    try:
        out = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ''
    return out.stdout.strip() if out.returncode == 0 else ''


def assert_pinned(path: str = None) -> str:
    """Verify the checkout is exactly KRONOS_COMMIT. Raises on mismatch."""
    path = path or upstream_path()
    if not os.path.isdir(os.path.join(path, 'model')):
        raise KronosUnavailable(
            'Kronos upstream not found at %s. Clone it with:\n'
            '  git clone %s %s && git -C %s checkout %s'
            % (path, KRONOS_REPO_URL, path, path, KRONOS_COMMIT))
    head = checkout_commit(path)
    if not head:
        raise KronosPinMismatch(
            'Kronos upstream at %s is not a git checkout, so its revision cannot '
            'be verified. Results from unverifiable source code are not usable.' % path)
    if head != KRONOS_COMMIT:
        raise KronosPinMismatch(
            'Kronos upstream is at %s but this harness is pinned to %s. Every '
            'reported result assumes the pinned revision. Run:\n'
            '  git -C %s checkout %s' % (head, KRONOS_COMMIT, path, KRONOS_COMMIT))
    return head


def import_upstream(path: str = None):
    """Return the upstream `model` package after verifying the pin.

    Raises KronosPinMismatch on a wrong revision, KronosUnavailable if the
    source or a third-party dependency (torch, einops, safetensors) is absent.
    """
    path = path or upstream_path()
    assert_pinned(path)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import model as kronos_model            # noqa: F401  (upstream package)
    except ImportError as e:
        raise KronosUnavailable(
            'Kronos upstream found and correctly pinned, but its dependencies '
            'are missing (%s). Install: pip install -r requirements-kronos.txt' % e)
    return kronos_model
