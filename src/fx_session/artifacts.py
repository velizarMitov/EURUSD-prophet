"""Frozen fold artifacts and their digests (spec fx-session/evaluation).

Everything is written under research_models/fx_session/ (gitignored); every file's SHA-256 is
recorded in results/fx_session/artifact_manifest.csv. Loading verifies the digest and refuses a
missing or altered file. The MLP ensemble loads all-or-nothing.
"""

from __future__ import annotations

import csv
import os

import joblib

from ..fx_fib.artifacts import sha256
from . import repo_path

MANIFEST_FIELDS = ['path', 'sha256', 'kind', 'pair', 'fold', 'member', 'written_at']


class ArtifactError(RuntimeError):
    pass


def manifest_path(cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['results']), 'artifact_manifest.csv')


def read_manifest(cfg: dict) -> dict:
    path = manifest_path(cfg)
    if not os.path.exists(path):
        return {}
    with open(path, newline='', encoding='utf-8') as fh:
        return {r['path']: r for r in csv.DictReader(fh)}


def _record(cfg, rel, kind, pair, fold, member, now_iso) -> None:
    rows = read_manifest(cfg)
    rows[rel] = {'path': rel, 'sha256': sha256(repo_path(rel)), 'kind': kind, 'pair': pair,
                 'fold': fold, 'member': member, 'written_at': now_iso}
    path = manifest_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))


def fold_dir(cfg: dict, pair: str, fold: int) -> str:
    return f"{cfg['paths']['artifacts']}/{pair}/fold_{fold:02d}"


def save(cfg, obj, pair: str, fold: int, name: str, kind: str, now_iso: str) -> str:
    rel = f'{fold_dir(cfg, pair, fold)}/{name}.joblib'
    os.makedirs(os.path.dirname(repo_path(rel)), exist_ok=True)
    joblib.dump(obj, repo_path(rel))
    _record(cfg, rel, kind, pair, fold, name, now_iso)
    return rel


def load(cfg, rel: str, manifest: dict | None = None):
    manifest = manifest if manifest is not None else read_manifest(cfg)
    path = repo_path(rel)
    if rel not in manifest or not os.path.exists(path):
        raise ArtifactError(f'missing artifact: {rel}')
    if sha256(path) != manifest[rel]['sha256']:
        raise ArtifactError(f'digest mismatch: {rel}')
    return joblib.load(path)
