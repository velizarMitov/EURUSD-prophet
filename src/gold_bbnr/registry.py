"""Paths, the three registries and the config guard (spec gold-bb-nyrange/evaluation).

    results/gold_bbnr/gold_bb_hypothesis_log.csv       gold_bb_rule_mean_reversion, gold_bb_lightgbm_vs_base_rate
    results/gold_bbnr/gold_bb_sma_hypothesis_log.csv   gold_bb_sma_rule_mean_r, gold_bb_sma_lightgbm_vs_base_rate
    results/gold_bbnr/gold_nyvol_hypothesis_log.csv    gold_harx_vs_har, gold_lgbm_vs_harx, gold_mlp_vs_harx

Every stage refuses a config whose digest differs from one recorded with labels.json, the
pre-registration or a verdict row: a changed config is a new hypothesis.
"""

from __future__ import annotations

import csv
import json
import os

from . import config_digest, repo_path

REGISTRIES = {'bb': 'gold_bb_hypothesis_log.csv', 'bb_sma': 'gold_bb_sma_hypothesis_log.csv',
              'nyvol': 'gold_nyvol_hypothesis_log.csv'}
CELLS = {'bb': ['gold_bb_rule_mean_reversion', 'gold_bb_lightgbm_vs_base_rate'],
         'bb_sma': ['gold_bb_sma_rule_mean_r', 'gold_bb_sma_lightgbm_vs_base_rate'],
         'nyvol': ['gold_harx_vs_har', 'gold_lgbm_vs_harx', 'gold_mlp_vs_harx']}
FIELDS = ['n', 'date', 'hypothesis', 'family', 'metric', 'statistic', 'model_mean', 'reference_mean', 'n_rows',
          'n_dates', 'first_date', 'last_date', 'bound', 'bound_quantile', 'alpha', 'block_days', 'resamples', 'seed',
          'boot_se', 'mde_80', 'verdict', 'config_digest', 'prereg_sha256', 'notes']
LABELS_JSON = 'labels.json'
LABELS_MD = 'labels.md'
PREREG_JSON = 'PRE_REGISTRATION.json'
PREREG_MD = 'PRE_REGISTRATION.md'


class ConfigChanged(RuntimeError):
    pass


class AlreadyScored(RuntimeError):
    pass


class StageOrderError(RuntimeError):
    pass


def results_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['results'])


def path(cfg: dict, name: str) -> str:
    return os.path.join(results_dir(cfg), name)


def read_rows(cfg: dict, family: str) -> list:
    p = path(cfg, REGISTRIES[family])
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


def write_json(cfg: dict, name: str, rec: dict) -> str:
    p = path(cfg, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    return p


def guard(cfg: dict) -> None:
    now = config_digest(cfg)
    seen = [(f'{REGISTRIES[f]} {r["hypothesis"]}', r['config_digest']) for f in REGISTRIES for r in read_rows(cfg, f)]
    for name in (LABELS_JSON, PREREG_JSON):
        rec = read_json(cfg, name)
        if rec is not None:
            seen.append((name, rec['config_digest']))
    bad = [src for src, c in seen if c != now]
    if bad:
        raise ConfigChanged(f'the family config differs from the one recorded with {", ".join(bad)}; '
                            'a changed config is a new hypothesis, not a rerun')


def scored(cfg: dict, family: str) -> dict:
    return {r['hypothesis']: r for r in read_rows(cfg, family)}


def append(cfg: dict, family: str, rows: list) -> None:
    have = scored(cfg, family)
    dup = [r['hypothesis'] for r in rows if r['hypothesis'] in have]
    if dup:
        raise AlreadyScored(f'already scored, each cell is scored once: {dup}')
    p = path(cfg, REGISTRIES[family])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, 'n': len(have) + i + 1})
