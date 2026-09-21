"""The retrain log must survive the next launch.

`POST /api/retrain` truncates results/retrain.log (mode "w") -- deliberately,
because the supervisor reads the CURRENT run off that file. Until 2026-09-21
that also destroyed every previous run's record: the log could evidence
exactly one retrain (2026-09-11) although git showed models/ moving since
2026-06-20. src/retrain_log.py archives the file before the launcher opens it.

These tests pin: the archive is made and named by the log's own timestamp;
an empty or absent log is not archived; the directory is pruned to `keep`;
the function never raises; and -- the part that matters -- the real launch
path in api.start_retrain calls it BEFORE the truncating open.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import retrain_log as RL  # noqa: E402


def _write(path, text, mtime=None):
    path.write_text(text, encoding='utf-8')
    if mtime is not None:
        os.utime(path, (mtime, mtime))


# ── 1. the archive itself ──────────────────────────────────────────────────

def test_archives_the_previous_log_named_by_its_last_write(tmp_path):
    log = tmp_path / 'retrain.log'
    stamp = time.mktime((2026, 9, 11, 7, 13, 14, 0, 0, -1))
    _write(log, 'run output\n=== RETRAIN EXIT rc=0 elapsed=1639.7s ===\n', mtime=stamp)

    target = RL.archive_previous(str(log))

    assert target is not None
    assert os.path.dirname(target) == str(tmp_path / RL.ARCHIVE_DIRNAME)
    expected = RL.archive_name(stamp)
    assert os.path.basename(target) == expected
    assert expected.startswith('retrain_') and expected.endswith('.log')
    assert not log.exists(), 'the live log is moved, not copied, so the launcher starts clean'
    assert open(target, encoding='utf-8').read().endswith('elapsed=1639.7s ===\n')


def test_absent_and_empty_logs_are_not_archived(tmp_path):
    assert RL.archive_previous(str(tmp_path / 'nope.log')) is None
    empty = tmp_path / 'retrain.log'
    empty.write_text('', encoding='utf-8')
    assert RL.archive_previous(str(empty)) is None
    assert not (tmp_path / RL.ARCHIVE_DIRNAME).exists()


def test_two_archives_in_the_same_second_do_not_overwrite(tmp_path):
    log = tmp_path / 'retrain.log'
    stamp = time.time()
    _write(log, 'first\n', mtime=stamp)
    a = RL.archive_previous(str(log))
    _write(log, 'second\n', mtime=stamp)
    b = RL.archive_previous(str(log))
    assert a != b
    assert open(a, encoding='utf-8').read() == 'first\n'
    assert open(b, encoding='utf-8').read() == 'second\n'


def test_prunes_to_keep_oldest_first(tmp_path):
    log = tmp_path / 'retrain.log'
    base = time.time() - 10_000
    for i in range(7):
        _write(log, f'run {i}\n', mtime=base + i * 60)
        RL.archive_previous(str(log), keep=5)
    kept = RL.list_archives(str(tmp_path / RL.ARCHIVE_DIRNAME))
    assert len(kept) == 5
    contents = [open(p, encoding='utf-8').read() for p in kept]
    assert contents == [f'run {i}\n' for i in range(2, 7)], 'the two OLDEST were pruned'


def test_prune_ignores_files_that_are_not_archives(tmp_path):
    d = tmp_path / RL.ARCHIVE_DIRNAME
    d.mkdir()
    (d / 'notes.txt').write_text('keep me', encoding='utf-8')
    (d / 'retrain_20260101-000000.log').write_text('a', encoding='utf-8')
    RL.prune(str(d), keep=0)
    assert (d / 'notes.txt').exists()
    assert not (d / 'retrain_20260101-000000.log').exists()


def test_never_raises_when_the_archive_dir_cannot_be_made(tmp_path, capsys):
    log = tmp_path / 'retrain.log'
    _write(log, 'x\n')
    blocker = tmp_path / 'blocked'
    blocker.write_text('a file where the directory should go', encoding='utf-8')
    assert RL.archive_previous(str(log), archive_dir=str(blocker)) is None
    assert 'could not archive' in capsys.readouterr().err
    assert log.exists(), 'the live log is left for the launcher when archiving fails'


def test_falls_back_to_copy_when_rename_is_refused(tmp_path, monkeypatch):
    """Windows refuses to rename a file another process holds open (an orphaned
    child). The record must still be preserved."""
    log = tmp_path / 'retrain.log'
    _write(log, 'held open\n')

    def refuse(src, dst):
        raise PermissionError('in use')
    monkeypatch.setattr(RL.os, 'replace', refuse)

    target = RL.archive_previous(str(log))
    assert target is not None and open(target, encoding='utf-8').read() == 'held open\n'
    assert log.exists()


# ── 2. wired into the real launch path ─────────────────────────────────────

def test_start_retrain_archives_the_previous_log_before_truncating(tmp_path, monkeypatch):
    """The guard is worth nothing if api.start_retrain does not call it, and
    calls it before the open(..., "w")."""
    import api

    log = tmp_path / 'retrain.log'
    monkeypatch.setattr(api, 'RETRAIN_LOG', str(log))
    monkeypatch.setattr(api, 'RETRAIN_STATE', str(tmp_path / 'retrain_state.json'))
    saved = dict(api._retrain)
    api._retrain.update(proc=None, started_at=None, reloaded=False,
                        returncode=None, finished_at=None, pid=None)
    _write(log, 'PREVIOUS RUN\n=== RETRAIN EXIT rc=0 elapsed=1.0s ===\n',
           mtime=time.time() - 3600)

    class _FakeProc:
        pid = 4242

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr(api.subprocess, 'Popen', lambda argv, **kw: _FakeProc())
    monkeypatch.setattr(api.threading, 'Thread',
                        lambda *a, **k: type('T', (), {'start': lambda self: None})())
    try:
        api.start_retrain()
    finally:
        api._retrain.clear()
        api._retrain.update(saved)

    archives = RL.list_archives(str(tmp_path / RL.ARCHIVE_DIRNAME))
    assert len(archives) == 1, 'previous log was not archived by start_retrain'
    assert open(archives[0], encoding='utf-8').read().startswith('PREVIOUS RUN')
    assert log.exists() and log.read_text(encoding='utf-8') == '', \
        'the live log must be a fresh, truncated file for the new run'


def test_launcher_still_truncates_rather_than_appends():
    """Append mode would let _marker_returncode() find the PREVIOUS run's exit
    marker during a restart mid-run and hot-reload a torn models/ -- the
    2026-08-07 failure. Rotation must not have replaced the "w"."""
    import api
    src = open(api.__file__, encoding='utf-8').read()
    assert 'archive_previous(RETRAIN_LOG)' in src
    assert 'open(RETRAIN_LOG, "w", encoding="utf-8")' in src
    assert src.index('archive_previous(RETRAIN_LOG)') < src.index('open(RETRAIN_LOG, "w"')


def test_archive_directory_is_not_gitignored():
    """results/retrain.log stays tracked because it is the run's readable
    record; its archives inherit that policy. If someone ignores the directory
    the evidence trail stops at the working copy again."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ignore = open(os.path.join(repo, '.gitignore'), encoding='utf-8').read()
    assert 'retrain_logs' not in ignore
