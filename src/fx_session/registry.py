"""The family registry and the config guard (spec fx-session/evaluation).

results/fx_session/fx_session_hypothesis_log.csv holds at most four rows, one per cell, each
written once. Every row records the config and feature-list digests; once any row exists,
every label, training and scoring stage refuses a config whose digests differ.
"""

from __future__ import annotations

import csv
import os

from . import config_digest, feature_list_digest, repo_path

REGISTRY = 'fx_session_hypothesis_log.csv'
FIELDS = ['n', 'date', 'hypothesis', 'pair', 'model', 'reference', 'metric', 'n_days', 'n_folds',
          'eval_start', 'eval_end', 'logloss_model', 'logloss_reference', 'accuracy_model', 'mean_diff',
          'ci_upper', 'alpha', 'block_days', 'resamples', 'seed', 'boot_se', 'mde_80', 'verdict',
          'config_digest', 'feature_list_digest', 'artifact_manifest_sha256', 'device', 'notes']
CELLS = (('lightgbm', 'prior'), ('mlp', 'lightgbm'))


class ConfigChanged(RuntimeError):
    pass


class AlreadyScored(RuntimeError):
    pass


def registry_path(cfg: dict) -> str:
    return os.path.join(repo_path(cfg['paths']['results']), REGISTRY)


def read_registry(cfg: dict) -> list:
    path = registry_path(cfg)
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def assert_config_matches_verdicts(cfg: dict) -> None:
    rows = read_registry(cfg)
    if not rows:
        return
    if {r['config_digest'] for r in rows} != {config_digest(cfg)} or \
            {r['feature_list_digest'] for r in rows} != {feature_list_digest(cfg)}:
        raise ConfigChanged(f'the family config differs from the one recorded with the verdicts in {REGISTRY}; '
                            'a changed config is a new hypothesis, not a rerun')


def append_rows(cfg: dict, rows: list) -> None:
    """Append verdict rows. Refuses (and writes nothing) if any hypothesis is already present."""
    have = {r['hypothesis'] for r in read_registry(cfg)}
    dup = [r['hypothesis'] for r in rows if r['hypothesis'] in have]
    if dup:
        raise AlreadyScored(f'already scored, each cell is scored once: {dup}')
    path = registry_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'x' if new else 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
