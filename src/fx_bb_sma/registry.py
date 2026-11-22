"""The family registry and the config guard (spec fx-bb-sma-exit/evaluation).

results/fx_bb_sma/fx_bb_sma_hypothesis_log.csv holds at most two rows, one per cell, each written
once. Every stage (labels, walk-forward, scoring) refuses a config whose digests differ from one
recorded by an earlier stage (labels.json, run.json) or verdict row.
"""

from __future__ import annotations

import csv
import json
import os

from . import config_digest, feature_list_digest, repo_path

REGISTRY = 'fx_bb_sma_hypothesis_log.csv'
LABELS_JSON = 'labels.json'
RUN_JSON = 'run.json'
CELL_RULE = 'sma_exit_rule_net_r'
CELL_MODEL = 'sma_exit_lightgbm_vs_base_rate'
FIELDS = ['n', 'date', 'hypothesis', 'cell', 'metric', 'statistic_name', 'n_trades', 'n_dates', 'n_business_days',
          'first_entry', 'last_entry', 'n_folds', 'statistic', 'bound', 'bound_quantile', 'alpha', 'block_days',
          'resamples', 'seed', 'boot_se', 'mde_80', 'verdict', 'model_mean', 'reference_mean', 'accuracy',
          'ctrl_statistic', 'parity_gap', 'parity_flag', 'share_take_profit', 'share_stop_loss', 'share_time_stop',
          'config_digest', 'feature_list_digest', 'artifact_manifest_sha256', 'notes']


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


def read_json(cfg: dict, name: str):
    path = os.path.join(results_dir(cfg), name)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def guard(cfg: dict) -> None:
    now = (config_digest(cfg), feature_list_digest(cfg))
    seen = [(f'{REGISTRY} {r["hypothesis"]}', r['config_digest'], r['feature_list_digest']) for r in read_registry(cfg)]
    for name in (LABELS_JSON, RUN_JSON):
        rec = read_json(cfg, name)
        if rec is not None:
            seen.append((name, rec['config_digest'], rec['feature_list_digest']))
    bad = [src for src, c, f in seen if (c, f) != now]
    if bad:
        raise ConfigChanged(f'the family config differs from the one recorded with {", ".join(bad)}; '
                            'a changed config is a new hypothesis, not a rerun')


def append_row(cfg: dict, row: dict) -> None:
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
