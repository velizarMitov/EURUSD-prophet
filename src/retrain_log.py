"""Keep every retrain log instead of overwriting the last one.

Why this exists
---------------
`POST /api/retrain` opens `results/retrain.log` with mode "w". That truncation
is deliberate -- the supervisor in api.py reads the CURRENT run off that file
(last-line exit marker, mtime as liveness), and those readers assume the file
holds one run and one run only. But it also meant every launch destroyed the
previous run's record. On 2026-09-19 the question "when did the undeclared
retrains happen?" could be answered from the log for exactly one run
(2026-09-11); the four older logs in results/ survive only because someone
renamed them by hand. Git bounded models/ movement back to 2026-06-20, but the
logs that would have said what each run did were gone.

What it does
------------
Before the launcher truncates the log, `archive_previous()` moves the existing
file to `results/retrain_logs/retrain_<UTC stamp>.log` and prunes that
directory to the newest `keep` files. The stamp is the log's own mtime -- the
last write, i.e. the exit marker of the run it records -- so the archive sorts
chronologically by name and needs no other state.

What it deliberately does NOT do
--------------------------------
It does not switch the log to append mode. `_marker_returncode()` in api.py
scans the file BACKWARDS for the last exit marker; with runs appended into one
file, a server restart during a run would find the previous run's marker and
report "completed", hot-reloading a half-written models/ directory -- the exact
failure the 2026-08-07 observability program closed. Rotation preserves the
history without touching that state machine.

It never raises. A failed archive is reported on stderr and the launch goes
ahead: blocking a legitimate retrain because an archive directory is not
writable would only teach people to route around the guard.
"""

from __future__ import annotations

import os
import shutil
import sys
import time

ARCHIVE_DIRNAME = "retrain_logs"
ARCHIVE_PREFIX = "retrain_"
ARCHIVE_SUFFIX = ".log"
KEEP = 20


def archive_dir_for(log_path: str) -> str:
    """`<dir of log>/retrain_logs` -- next to the live log, so it lands in
    results/ and is tracked the same way (see .gitignore: retrain.log stays
    tracked because it is the run's readable record; so do its archives)."""
    return os.path.join(os.path.dirname(os.path.abspath(log_path)), ARCHIVE_DIRNAME)


def archive_name(mtime: float) -> str:
    return ARCHIVE_PREFIX + time.strftime("%Y%m%d-%H%M%S", time.gmtime(mtime)) + ARCHIVE_SUFFIX


def _unique(path: str) -> str:
    """Two archives in one second are unlikely but must not overwrite."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}-{n}{ext}"):
        n += 1
    return f"{stem}-{n}{ext}"


def list_archives(archive_dir: str) -> list[str]:
    """Archived logs, oldest first (the stamp sorts chronologically)."""
    try:
        names = os.listdir(archive_dir)
    except OSError:
        return []
    return sorted(
        os.path.join(archive_dir, n)
        for n in names
        if n.startswith(ARCHIVE_PREFIX) and n.endswith(ARCHIVE_SUFFIX)
    )


def prune(archive_dir: str, keep: int = KEEP) -> list[str]:
    """Delete the oldest archives beyond `keep`; return what was removed."""
    removed = []
    archives = list_archives(archive_dir)
    for path in archives[: max(0, len(archives) - keep)]:
        try:
            os.remove(path)
            removed.append(path)
        except OSError as exc:
            print(f"retrain_log: could not prune {path}: {exc}", file=sys.stderr)
    return removed


def archive_previous(log_path: str, archive_dir: str | None = None,
                     keep: int = KEEP) -> str | None:
    """Move the existing log out of the way before the next run truncates it.

    Returns the archive path, or None when there was nothing to archive (no
    log, or an empty one) or the archive could not be made. Never raises.
    """
    try:
        st = os.stat(log_path)
    except OSError:
        return None
    if st.st_size == 0:
        return None

    archive_dir = archive_dir or archive_dir_for(log_path)
    try:
        os.makedirs(archive_dir, exist_ok=True)
        target = _unique(os.path.join(archive_dir, archive_name(st.st_mtime)))
        try:
            os.replace(log_path, target)
        except OSError:
            # Windows refuses to rename a file another process still holds
            # open (an orphaned child from a run the server lost track of).
            # A copy still preserves the record; the launcher truncates the
            # original as before.
            shutil.copy2(log_path, target)
    except OSError as exc:
        print(f"retrain_log: could not archive {log_path}: {exc}", file=sys.stderr)
        return None

    prune(archive_dir, keep)
    return target
