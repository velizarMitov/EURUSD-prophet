"""Are the model artifacts being served the ones that were declared?

Why this exists
---------------
On 2026-09-11 `POST /api/retrain` rewrote 18 production artifacts under
models/. Nothing said so anywhere for eight days. The commit hook
(.githooks/check_retrain_declaration.py) fires only when someone tries to
commit, and nobody is obliged to; the checksum fixtures fire only when someone
runs the suite and reads the failure. Meanwhile the API served the new
artifacts, the volatility card kept its "validated" badge, and the forward
ledgers logged predictions from models whose gate results describe different
weights. The gap was not a bad commit -- it was undeclared artifacts in
production with no surface that showed it.

What "declared" means here
--------------------------
A models/ artifact is declared when a `tests/fixtures/*_protected_sha256.json`
fixture pins its current digest. Re-pinning is the act of declaration: it
fails the suite until done, and the commit that carries it must say `RETRAIN:`.
So "on disk != fixture" is precisely "served but not declared". Git is not
consulted -- the check must work in a container without a repository, and the
fixtures are the durable, reviewable record anyway.

What it reports
---------------
`ProvenanceMonitor.status()` hashes every models/ path pinned by any fixture
(the union; 44 paths, ~24 MB, ~0.3 s) and returns which ones differ from or
are missing against their pin, and for how long: "undeclared since" is the
earliest mtime among the diverged files, i.e. when the run that produced them
last wrote. It re-hashes only when a pinned file's (mtime, size) signature
changes, so polling it from a dashboard is cheap.

The wording is "undeclared for N days", not "error". Every legitimate retrain
shows this until it is committed; if it shouted, it would become wallpaper.

It never raises out of `status()`: missing fixtures, unreadable files and
malformed JSON produce a result with `available: False` or a `missing` list,
never a failed prediction.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time

FIXTURE_GLOB = os.path.join("tests", "fixtures", "*_protected_sha256.json")
WATCHED_PREFIX = "models/"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_digests(base_dir: str):
    """Union of every models/ pin across the fixture files.

    Returns (digests, fixtures_read, conflicts): `digests` maps repo-relative
    path -> sha256; `conflicts` maps a path to the set of digests when two
    fixtures disagree about it (the suite would already be red, but the
    monitor must not silently pick one). Entries whose value is not a string
    (the rolling-cache record in input_data_protected_sha256.json) are skipped.
    """
    seen: dict[str, set[str]] = {}
    fixtures_read: list[str] = []
    for path in sorted(glob.glob(os.path.join(base_dir, FIXTURE_GLOB))):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        fixtures_read.append(os.path.basename(path))
        for rel, digest in data.items():
            if isinstance(rel, str) and rel.startswith(WATCHED_PREFIX) and isinstance(digest, str):
                seen.setdefault(rel, set()).add(digest)
    digests = {rel: next(iter(d)) for rel, d in seen.items() if len(d) == 1}
    conflicts = {rel: d for rel, d in seen.items() if len(d) > 1}
    return digests, fixtures_read, conflicts


def _signature(base_dir: str, paths) -> tuple:
    """(path, mtime_ns, size) per pinned file; None for a missing one. A
    change here is the only thing that triggers re-hashing."""
    sig = []
    for rel in sorted(paths):
        try:
            st = os.stat(os.path.join(base_dir, rel))
            sig.append((rel, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((rel, None, None))
    return tuple(sig)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


class ProvenanceMonitor:
    """Digest check for the served models/ set, cached on file signatures."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._signature: tuple | None = None
        self._diverged: list[dict] = []
        self._missing: list[str] = []
        self.hash_runs = 0          # how many times the set was re-hashed (tests)

    def invalidate(self) -> None:
        """Force the next status() to re-hash even if signatures look unchanged
        (called after a hot-reload; belt and braces over the mtime check)."""
        self._signature = None

    def _rehash(self, digests: dict[str, str]) -> None:
        self.hash_runs += 1
        diverged, missing = [], []
        for rel in sorted(digests):
            path = os.path.join(self.base_dir, rel)
            try:
                actual = sha256_file(path)
                mtime = os.path.getmtime(path)
            except OSError:
                missing.append(rel)
                continue
            if actual != digests[rel]:
                diverged.append({
                    "path": rel,
                    "declared": digests[rel][:12],
                    "on_disk": actual[:12],
                    "modified_at": _iso(mtime),
                    "_mtime": mtime,
                })
        self._diverged, self._missing = diverged, missing

    def status(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        digests, fixtures, conflicts = declared_digests(self.base_dir)
        result = {
            "available": bool(digests),
            "checked_at": _iso(now),
            "fixtures": fixtures,
            "n_pinned": len(digests),
        }
        if not digests:
            result.update(reason="no *_protected_sha256.json fixture pins a models/ path",
                          undeclared=False, diverged=[], missing=[], conflicting=[],
                          undeclared_since=None, undeclared_days=None, summary=None)
            return result

        sig = _signature(self.base_dir, digests)
        if sig != self._signature:
            self._rehash(digests)
            self._signature = sig

        diverged = [{k: v for k, v in d.items() if not k.startswith("_")} for d in self._diverged]
        conflicting = sorted(conflicts)
        undeclared = bool(diverged or self._missing or conflicting)
        since = min((d["_mtime"] for d in self._diverged), default=None)
        days = None if since is None else max(0.0, (now - since) / 86400.0)

        result.update(
            undeclared=undeclared,
            diverged=diverged,
            missing=list(self._missing),
            conflicting=conflicting,
            undeclared_since=_iso(since),
            undeclared_days=None if days is None else round(days, 1),
            summary=self._summary(len(diverged), len(self._missing), len(conflicting),
                                  len(digests), days, since),
        )
        return result

    @staticmethod
    def _summary(n_div: int, n_miss: int, n_conf: int, n_pinned: int,
                 days: float | None, since: float | None) -> str | None:
        if not (n_div or n_miss or n_conf):
            return None
        parts = []
        if n_div:
            parts.append(f"{n_div} of {n_pinned} served model artifacts differ from the declared set")
        if n_miss:
            parts.append(f"{n_miss} pinned artifact(s) missing from disk")
        if n_conf:
            parts.append(f"{n_conf} path(s) pinned inconsistently across fixtures")
        text = "; ".join(parts)
        if days is not None:
            text += f" — undeclared for {days:.1f} days (since {_iso(since)})"
        return text
