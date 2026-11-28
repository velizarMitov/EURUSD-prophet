"""The family's registry and its guard (spec rts-signals/evaluation).

    results/rts_signals/rts_signals_hypothesis_log.csv     one cell, rts_spot_net_r, alpha 0.05

Every stage refuses a config or rule book whose digest differs from the one recorded in the
pre-registration or in a verdict row: a changed number is a new hypothesis, not a rerun.
"""

from __future__ import annotations

import csv
import json
import os

from . import CELL, config_digest, results_dir

REGISTRY = 'rts_signals_hypothesis_log.csv'
PREREG_JSON = 'PRE_REGISTRATION.json'
PREREG_MD = 'PRE_REGISTRATION.md'
FIELDS = ['n', 'date', 'hypothesis', 'n_trades', 'first_day', 'last_day', 'mean_net_r', 'bound', 'boot_se', 'mde_80',
          'alpha', 'family_size', 'block_days', 'resamples', 'seed', 'verdict', 'config_digest', 'rules_digest',
          'prereg_sha256', 'notes']


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


def read_json(cfg: dict, name: str = PREREG_JSON):
    p = path(cfg, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def guard(cfg: dict, rules_dig: str | None = None) -> None:
    from .rules import rules_digest
    now_c, now_r = config_digest(cfg), rules_dig or rules_digest()
    seen = [(f'{REGISTRY} {r["hypothesis"]}', r['config_digest'], r['rules_digest']) for r in read_rows(cfg)]
    rec = read_json(cfg)
    if rec is not None:
        seen.append((PREREG_JSON, rec['config_digest'], rec['rules_digest']))
    bad = [src for src, c, r in seen if c != now_c or r != now_r]
    if bad:
        raise ConfigChanged(f'the family config or rule book differs from the one recorded with {", ".join(bad)}; '
                            'a changed number is a new hypothesis, not a rerun')


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
    return next((r['verdict'] for r in read_rows(cfg) if r['hypothesis'] == CELL), '')
