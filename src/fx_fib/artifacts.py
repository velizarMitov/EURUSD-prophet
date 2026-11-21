"""Frozen fold artifacts and their digests (spec fx-fibonacci/models).

Everything is written under research_models/fx_fib/ (gitignored); the SHA-256 of every file
is recorded in results/fx_fib/artifact_manifest.csv. Loading verifies each digest and refuses
a missing or altered file. A 5-seed ensemble loads all-or-nothing.
"""

from __future__ import annotations

import csv
import hashlib
import os

import joblib
import torch

from . import load_config, repo_path
from .model_mlp_cuda import build_mlp

MANIFEST_FIELDS = ['path', 'sha256', 'kind', 'pair', 'fold', 'member', 'written_at']


class ArtifactError(RuntimeError):
    pass


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['results']), 'artifact_manifest.csv')


def read_manifest(cfg: dict) -> dict:
    path = manifest_path(cfg)
    if not os.path.exists(path):
        return {}
    with open(path, newline='', encoding='utf-8') as fh:
        return {r['path']: r for r in csv.DictReader(fh)}


def _record(cfg: dict, rel: str, kind: str, pair: str, fold: int, member: str, now_iso: str) -> None:
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


def save_mlp(cfg, model, meta: dict, pair: str, fold: int, member: str, now_iso: str) -> str:
    rel = f'{fold_dir(cfg, pair, fold)}/mlp_{member}.pt'
    os.makedirs(os.path.dirname(repo_path(rel)), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, repo_path(rel))
    _record(cfg, rel, 'mlp', pair, fold, member, now_iso)
    return rel


def save_object(cfg, obj, pair: str, fold: int, name: str, now_iso: str) -> str:
    rel = f'{fold_dir(cfg, pair, fold)}/{name}.joblib'
    os.makedirs(os.path.dirname(repo_path(rel)), exist_ok=True)
    joblib.dump(obj, repo_path(rel))
    _record(cfg, rel, name, pair, fold, name, now_iso)
    return rel


def _verified(cfg: dict, rel: str, manifest: dict) -> str:
    path = repo_path(rel)
    if rel not in manifest or not os.path.exists(path):
        raise ArtifactError(f'missing artifact: {rel}')
    if sha256(path) != manifest[rel]['sha256']:
        raise ArtifactError(f'digest mismatch: {rel}')
    return path


def load_mlp(cfg: dict, rel: str, n_features: int, manifest: dict | None = None):
    manifest = manifest if manifest is not None else read_manifest(cfg)
    blob = torch.load(_verified(cfg, rel, manifest), map_location='cpu', weights_only=False)
    meta = blob['meta']
    model = build_mlp(n_features, meta['hidden'], meta['dropout'])
    model.load_state_dict(blob['state_dict'])
    return model.eval()


def load_object(cfg: dict, rel: str, manifest: dict | None = None):
    manifest = manifest if manifest is not None else read_manifest(cfg)
    return joblib.load(_verified(cfg, rel, manifest))


def load_ensemble(cfg: dict, pair: str, fold: int, n_features: int) -> list:
    """All five seed members of a fold, or ArtifactError -- never a partial ensemble."""
    cfg = cfg or load_config()
    manifest = read_manifest(cfg)
    return [load_mlp(cfg, f'{fold_dir(cfg, pair, fold)}/mlp_seed{s}.pt', n_features, manifest)
            for s in cfg['mlp']['seeds']]
