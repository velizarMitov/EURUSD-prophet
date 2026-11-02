"""Pinned weights and the SHA-256 manifest (design D2; spec "Weights are pinned
and verified").

Each model is fetched once, at a PINNED Hugging Face revision, into
research_models/external_tsfm/<model>/. The revision is the commit that
uploaded the weights; every later commit of both repos only edits the README
(checked 2026-11-02), and the pinned upstream digests below are what that
commit serves. A download that differs is refused.

Every file the models load -- base weights, fine-tuned adapters, the frozen
EUR/USD baseline -- is recorded in results/external_tsfm/artifact_manifest.csv
and verified before every load, the same way gold's `verify_cell` does. A
mismatch RAISES: a silently different model is exactly what the pin prevents.

    python -m src.external.tsfm.vendor --fetch      download and record both models
    python -m src.external.tsfm.vendor --verify     check every manifest entry
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone

from . import ARTIFACTS, MANIFEST, REPO, repo_path

PINS = {
    'chronos2': {
        'repo': 'amazon/chronos-2',
        'revision': '95a9710e2596287d08352589f42634fa5abdf0a7',      # 2025-10-30 "Add model weights and config"
        'package': 'chronos-forecasting==2.3.2',
        'released': '2025-10-30',
        'long_context': 8192,
        # file -> (kind, upstream digest): 'lfs' is the LFS SHA-256, 'blob' the git blob SHA-1
        'files': {'config.json': ('blob', 'f3207394965100a2aa18820b87da3d2bbad3b1f0'),
                  'model.safetensors': ('lfs', 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42')},
    },
    'timesfm25': {
        'repo': 'google/timesfm-2.5-200m-pytorch',
        'revision': 'd418f3e8a8fa79d655b391c158f0ee8d68fe68c9',      # 2025-10-01 "Refreshed model with speed optimizations"
        'package': 'timesfm==2.0.2',
        'released': '2025-10-01',
        # 16,384 is the context LIMIT, which counts the 128-step output patch too
        'long_context': 16256,
        'files': {'config.json': ('blob', '94445b5415c93256bbab9cf759df2ffce2671b16'),
                  'model.safetensors': ('lfs', '2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8')},
    },
}

FIELDS = ['artifact', 'path', 'sha256', 'bytes', 'source', 'recorded_at']


class ArtifactMismatch(ValueError):
    """A pinned file is missing or differs from its recorded digest."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: str) -> str:
    with open(path, 'rb') as fh:
        data = fh.read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def weights_dir(model: str, repo: str = REPO) -> str:
    return repo_path(f'{ARTIFACTS}/{model}', repo)


# ── manifest ────────────────────────────────────────────────────────────────

def read_manifest(path: str | None = None, repo: str = REPO) -> dict:
    """{artifact: [(repo-relative path, sha256), ...]}"""
    path = path or repo_path(MANIFEST, repo)
    out: dict = {}
    if not os.path.exists(path):
        return out
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r['artifact'], []).append((r['path'], r['sha256']))
    return out


def record(artifact: str, files, source: str, manifest: str | None = None, repo: str = REPO) -> list[dict]:
    """Replace `artifact`'s rows with the current digests of `files` (absolute
    paths under the repo). Other artifacts' rows are kept as they are."""
    manifest = manifest or repo_path(MANIFEST, repo)
    keep = []
    if os.path.exists(manifest):
        with open(manifest, newline='', encoding='utf-8') as fh:
            keep = [r for r in csv.DictReader(fh) if r['artifact'] != artifact]
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = [{'artifact': artifact, 'path': os.path.relpath(f, repo).replace('\\', '/'),
             'sha256': sha256_file(f), 'bytes': os.path.getsize(f), 'source': source,
             'recorded_at': now} for f in sorted(files)]
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    tmp = manifest + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(keep + rows)
    os.replace(tmp, manifest)
    return rows


def artifact_digest(entries) -> str:
    return hashlib.sha256('|'.join(sorted(f'{p}:{d}' for p, d in entries)).encode()).hexdigest()[:16]


def verify(artifact: str, manifest: dict | None = None, repo: str = REPO) -> str:
    """The artifact's version digest, or ArtifactMismatch naming the first bad file."""
    manifest = read_manifest(repo=repo) if manifest is None else manifest
    entries = manifest.get(artifact)
    if not entries:
        raise ArtifactMismatch(f'{artifact} is not in the artifact manifest')
    for rel, digest in entries:
        full = repo_path(rel, repo)
        if not os.path.exists(full):
            raise ArtifactMismatch(f'{rel} missing')
        if sha256_file(full) != digest:
            raise ArtifactMismatch(f'{rel} differs from the manifest digest')
    return artifact_digest(entries)


# ── fetch ───────────────────────────────────────────────────────────────────

def _upstream_ok(path: str, kind: str, expected: str) -> bool:
    return (sha256_file(path) if kind == 'lfs' else git_blob_sha1(path)) == expected


def fetch(model: str, repo: str = REPO, opener=urllib.request.urlopen, manifest: str | None = None) -> list[dict]:
    """Download the pinned revision, refuse it unless every file matches the
    pinned upstream digest, then record the files in the manifest."""
    pin = PINS[model]
    out_dir = weights_dir(model, repo)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, (kind, expected) in pin['files'].items():
        dest = os.path.join(out_dir, name)
        if not (os.path.exists(dest) and _upstream_ok(dest, kind, expected)):
            url = f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/{name}"
            part = dest + '.part'
            with opener(urllib.request.Request(url, headers={'User-Agent': 'eurusdprophet-tsfm'}), timeout=60) as r, \
                    open(part, 'wb') as fh:
                for block in iter(lambda: r.read(1 << 20), b''):
                    fh.write(block)
            if not _upstream_ok(part, kind, expected):
                os.remove(part)
                raise ArtifactMismatch(f'{pin["repo"]}@{pin["revision"][:12]} {name} does not match the pinned '
                                       f'upstream digest; refusing it')
            os.replace(part, dest)
        paths.append(dest)
    return record(model, paths, f"{pin['repo']}@{pin['revision']}", manifest, repo)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Pinned TSFM weights: fetch and verify.')
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--model', choices=sorted(PINS))
    a = ap.parse_args(argv)
    out = {}
    if a.fetch:
        for m in ([a.model] if a.model else sorted(PINS)):
            out[m] = [r['path'] for r in fetch(m)]
    if a.verify or not a.fetch:
        for art in sorted(read_manifest()):
            try:
                out[art] = verify(art)
            except ArtifactMismatch as e:
                out[art] = f'MISMATCH: {e}'
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
