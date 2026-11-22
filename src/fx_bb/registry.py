"""The family registry and the config guard (spec fx-bb-reversion/evaluation).

results/fx_bb/fx_bb_hypothesis_log.csv holds at most two rows, one per cell, each written once.

The guard: every stage (labels, walk-forward, scoring) refuses to run when the current config or
feature-list digest differs from one recorded by an earlier stage (labels.json, run.json) or by a
verdict row. A changed config is a new hypothesis, never a rerun.
"""

from __future__ import annotations

import csv
import json
import os

from . import config_digest, feature_list_digest, repo_path

REGISTRY = 'fx_bb_hypothesis_log.csv'
LABELS_JSON = 'labels.json'
RUN_JSON = 'run.json'
CELL_RULE = 'bb_rule_mean_reversion'
CELL_MODEL = 'bb_lightgbm_vs_base_rate'
CELLS = (CELL_RULE, CELL_MODEL)
FIELDS = ['n', 'date', 'hypothesis', 'cell', 'metric', 'statistic_name', 'n_events', 'n_dates', 'n_business_days',
          'first_event', 'last_event', 'n_folds', 'statistic', 'bound', 'bound_quantile', 'alpha', 'block_days',
          'resamples', 'seed', 'boot_se', 'mde_80', 'verdict', 'model_mean', 'reference_mean', 'accuracy',
          'bid_statistic', 'bid_n_events', 'parity_gap', 'parity_flag', 'config_digest', 'feature_list_digest',
          'artifact_manifest_sha256', 'notes']


class ConfigChanged(RuntimeError):
    pass


class AlreadyScored(RuntimeError):
    pass


class StageOrderError(RuntimeError):
    pass


def results_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['results'])


def registry_path(cfg: dict) -> str:
    return os.path.join(results_dir(cfg), REGISTRY)


def read_registry(cfg: dict) -> list:
    path = registry_path(cfg)
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _json(cfg: dict, name: str):
    path = os.path.join(results_dir(cfg), name)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def recorded_digests(cfg: dict) -> list:
    """(source, config digest, feature-list digest) for every earlier stage or verdict."""
    out = [(f'{REGISTRY} {r["hypothesis"]}', r['config_digest'], r['feature_list_digest']) for r in read_registry(cfg)]
    for name in (LABELS_JSON, RUN_JSON):
        rec = _json(cfg, name)
        if rec is not None:
            out.append((name, rec['config_digest'], rec['feature_list_digest']))
    return out


def guard(cfg: dict) -> None:
    now = (config_digest(cfg), feature_list_digest(cfg))
    bad = [src for src, c, f in recorded_digests(cfg) if (c, f) != now]
    if bad:
        raise ConfigChanged(f'the family config differs from the one recorded with {", ".join(bad)}; '
                            'a changed config is a new hypothesis, not a rerun')


def append_row(cfg: dict, row: dict) -> None:
    """Write one verdict row. Refuses (and writes nothing) when the cell already has a row."""
    if row['hypothesis'] in {r['hypothesis'] for r in read_registry(cfg)}:
        raise AlreadyScored(f'{row["hypothesis"]} is already scored; each cell is scored once')
    path = registry_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'x' if new else 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in FIELDS})
