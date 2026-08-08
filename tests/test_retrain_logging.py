"""Regression tests for the retrain log + status contract.

These pin the three failures behind the 2026-08-07 incident, where a run that
stopped progressing after ~21 minutes was watched for four hours:

  * the child's stdout was block-buffered, so every print() after the last 8 KB
    flush was lost when the process never exited to flush it;
  * nothing guaranteed a final line, so "did it finish?" could only be answered
    by comparing artifact mtimes;
  * the run's identity lived only in an in-memory Popen handle, so a server
    restart reported "idle" for a finished run and skipped the hot-reload.
"""
import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api  # noqa: E402


@pytest.fixture
def retrain_sandbox(tmp_path, monkeypatch):
    """Point the retrain log/state at tmp_path and restore _retrain afterwards,
    so no test can clobber the real results/retrain.log."""
    monkeypatch.setattr(api, "RETRAIN_LOG", str(tmp_path / "retrain.log"))
    monkeypatch.setattr(api, "RETRAIN_STATE", str(tmp_path / "retrain_state.json"))
    saved = dict(api._retrain)
    api._retrain.update(proc=None, started_at=None, reloaded=False,
                        returncode=None, finished_at=None, pid=None)
    yield tmp_path
    api._retrain.clear()
    api._retrain.update(saved)


def _spawn(script_body, sandbox):
    """Launch a child exactly the way start_retrain does (unbuffered, stdout and
    stderr both into the log) and return the Popen handle."""
    script = sandbox / "child.py"
    script.write_text(script_body, encoding="utf-8")
    logf = open(api.RETRAIN_LOG, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(sandbox), stdout=logf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        logf.close()
    return proc


# ── 1. buffering ───────────────────────────────────────────────────────────

def test_delayed_stdout_reaches_the_log_before_the_child_exits(retrain_sandbox):
    """The exact loss mode from 2026-08-07: a small print() well after startup.

    Block-buffered stdout holds it (it is far below the 8 KB threshold) until
    the process exits -- and the incident's process never did. Unbuffered, it
    must be readable from the log while the child is still alive."""
    proc = _spawn(
        "import time\n"
        "print('stage: volatility ensemble')\n"
        "time.sleep(30)\n",
        retrain_sandbox,
    )
    try:
        deadline = time.time() + 15
        tail = ""
        while time.time() < deadline:
            tail = api._tail(api.RETRAIN_LOG)
            if "stage: volatility ensemble" in tail:
                break
            time.sleep(0.2)
        assert proc.poll() is None, "child should still be running"
        assert "stage: volatility ensemble" in tail, (
            "stdout written by a still-running child never reached the log -- "
            f"buffering regression. Log tail was: {tail!r}"
        )
    finally:
        proc.kill()
        proc.wait()


def test_start_retrain_launches_the_child_unbuffered(retrain_sandbox, monkeypatch):
    """Guards the -u / PYTHONUNBUFFERED flags on the real launch path."""
    captured = {}

    class _FakeProc:
        pid = 4242

        def poll(self):
            return None

        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env") or {}
        return _FakeProc()

    monkeypatch.setattr(api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(api.threading, "Thread",
                        lambda *a, **k: type("T", (), {"start": lambda self: None})())

    api.start_retrain()

    assert "-u" in captured["argv"], f"child must run unbuffered: {captured['argv']}"
    assert captured["env"].get("PYTHONUNBUFFERED") == "1"
    assert captured["argv"][-1] == "_train_pipeline.py"


# ── 2. the completion marker ───────────────────────────────────────────────

@pytest.mark.parametrize("exit_code", [0, 3])
def test_log_always_ends_with_the_exit_marker(retrain_sandbox, exit_code):
    """Success or failure, the last line answers "is it done?"."""
    proc = _spawn(f"import sys\nprint('working')\nsys.exit({exit_code})\n", retrain_sandbox)
    api._watch_retrain(proc, time.time())

    last = api._tail(api.RETRAIN_LOG).strip().splitlines()[-1]
    assert last.startswith(api._MARKER), f"log does not end with the marker: {last!r}"
    assert f"rc={exit_code}" in last
    assert "elapsed=" in last
    assert api._marker_returncode() == exit_code


def test_marker_is_written_even_when_the_process_is_killed(retrain_sandbox):
    """A killed run is the case where the owner most needs the log to say so --
    and the case where the child itself can write nothing."""
    proc = _spawn("import time\nprint('working')\ntime.sleep(60)\n", retrain_sandbox)
    time.sleep(1)
    proc.kill()
    api._watch_retrain(proc, time.time() - 12.0)

    last = api._tail(api.RETRAIN_LOG).strip().splitlines()[-1]
    assert last.startswith(api._MARKER)
    assert api._marker_returncode() not in (None, 0)


# ── 3. status must never call a finished process "running" ─────────────────

def test_status_reports_completed_for_an_already_exited_process(retrain_sandbox, monkeypatch):
    """Status is asked only after the child is fully reaped."""
    proc = _spawn("print('done')\n", retrain_sandbox)
    proc.wait()
    api._retrain.update(proc=proc, started_at=time.time() - 5, reloaded=True)

    body = api.retrain_status()
    assert body["state"] == "completed", body
    assert body["log_age_seconds"] is not None


def test_status_reports_failed_for_a_nonzero_exit(retrain_sandbox):
    proc = _spawn("import sys\nsys.exit(7)\n", retrain_sandbox)
    proc.wait()
    api._retrain.update(proc=proc, started_at=time.time() - 5, reloaded=True)

    body = api.retrain_status()
    assert body["state"] == "failed"
    assert body["returncode"] == 7


def test_finished_run_survives_a_server_restart(retrain_sandbox):
    """The 2026-08-07 aftermath: the server was restarted, _retrain["proc"] was
    None again, and status answered "idle" -- so the hot-reload that swaps in
    the freshly trained artifacts never fired. The exit marker plus the state
    file must carry the verdict across the restart."""
    proc = _spawn("print('done')\n", retrain_sandbox)
    api._watch_retrain(proc, time.time() - 900)

    # Simulate the restart: drop every in-memory trace, keep only what's on disk.
    api._retrain.update(proc=None, started_at=None, returncode=None,
                        finished_at=None, pid=None, reloaded=False)
    api._recover_retrain_state()

    state, returncode = api._resolve_state()
    assert state == "completed", f"restart lost a finished run: {state}"
    assert returncode == 0


def test_orphaned_run_with_a_cold_log_reads_as_interrupted(retrain_sandbox):
    """No marker and a log that stopped growing is not "running" -- that is the
    reading that cost four hours of waiting."""
    with open(api.RETRAIN_LOG, "w", encoding="utf-8") as f:
        f.write("=== [with_macro] 10. Training ===\n")
    stale = time.time() - (api._STALL_SECONDS + 600)
    os.utime(api.RETRAIN_LOG, (stale, stale))
    api._retrain.update(proc=None, started_at=stale, returncode=None, pid=999999)

    state, _ = api._resolve_state()
    assert state == "interrupted", state

    body = api.retrain_status()
    assert body["state"] == "interrupted"
    assert "partial" in body["detail"]


def test_running_status_exposes_log_age_so_a_stall_is_visible(retrain_sandbox):
    """A hung run must be distinguishable from a working one."""
    with open(api.RETRAIN_LOG, "w", encoding="utf-8") as f:
        f.write("=== [with_macro] 10. Training ===\n")
    stale = time.time() - (api._STALL_SECONDS + 600)
    os.utime(api.RETRAIN_LOG, (stale, stale))

    class _Hung:
        def poll(self):
            return None

    api._retrain.update(proc=_Hung(), started_at=stale, returncode=None)

    body = api.retrain_status()
    assert body["state"] == "running"
    assert body["stalled"] is True, body
    assert body["log_age_seconds"] >= api._STALL_SECONDS


def test_second_retrain_is_refused_while_one_is_running(retrain_sandbox):
    """Two pipelines interleaving writes into models/ is how the artifact set
    got torn once already."""
    class _Running:
        def poll(self):
            return None

    api._retrain.update(proc=_Running(), started_at=time.time())
    with pytest.raises(api.HTTPException) as exc:
        api.start_retrain()
    assert exc.value.status_code == 409
