"""The family's registry and config guard (spec gold-nyrange-forward/forward-evaluation).

    results/gold_nyfwd/gold_nyfwd_hypothesis_log.csv     one cell, gold_harx_vs_har_forward, alpha 0.05

Every stage refuses a config whose digest differs from the one recorded in the pre-registration or in
a verdict row: a changed config is a new hypothesis, not a rerun.
"""

from __future__ import annotations

import csv
import json
import os

from . import config_digest, results_dir

REGISTRY = 'gold_nyfwd_hypothesis_log.csv'
PREREG_JSON = 'PRE_REGISTRATION.json'
PREREG_MD = 'PRE_REGISTRATION.md'
FIELDS = ['n', 'date', 'hypothesis', 'n_days', 'first_day', 'last_day', 'mse_harx', 'mse_har', 'mse_ratio',
          'mean_diff', 'ci_upper', 'alpha', 'family_size', 'block_days', 'resamples', 'seed', 'boot_se', 'mde_80',
          'verdict', 'config_digest', 'prereg_sha256', 'notes']


class ConfigChanged(RuntimeError):
    pass


class AlreadyScored(RuntimeError):
    pass


class StageOrderError(RuntimeError):
    pass


def path(cfg: dict, name: str) -> str:
    return os.path.join(results_dir(cfg), name)


def read_rows(cfg: dict, name: str = REGISTRY) -> list:
    p = path(cfg, name)
    if not os.path.exists(p):
        return []
    with open(p, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def read_json(cfg: dict, name: str):
    p = path(cfg, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def guard(cfg: dict) -> None:
    now = config_digest(cfg)
    seen = [(f'{REGISTRY} {r["hypothesis"]}', r['config_digest']) for r in read_rows(cfg)]
    rec = read_json(cfg, PREREG_JSON)
    if rec is not None:
        seen.append((PREREG_JSON, rec['config_digest']))
    bad = [src for src, c in seen if c != now]
    if bad:
        raise ConfigChanged(f'the family config differs from the one recorded with {", ".join(bad)}; '
                            'a changed config is a new hypothesis, not a rerun')


def append(cfg: dict, rows: list) -> None:
    have = {r['hypothesis'] for r in read_rows(cfg)}
    dup = [r['hypothesis'] for r in rows if r['hypothesis'] in have]
    if dup:
        raise AlreadyScored(f'already scored, the cell is scored once: {dup}')
    p = path(cfg, REGISTRY)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, 'x' if new else 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerows({k: r.get(k, '') for k in FIELDS} for r in rows)


def verdict(cfg: dict) -> str:
    from . import CELL
    return next((r['verdict'] for r in read_rows(cfg) if r['hypothesis'] == CELL), '')
