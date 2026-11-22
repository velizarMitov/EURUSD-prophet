"""The two registries and the config guard (spec fx-ny-orb-hot/evaluation, range-forward).

historical  results/fx_ny_orb/fx_ny_orb_hypothesis_log.csv                  orb_hot_net_r, orb_hot_minus_cold
forward     results/fx_ny_orb/forward/fx_ny_orb_forward_hypothesis_log.csv  <PAIR>_harx_vs_har

Every stage refuses a config whose digest differs from one recorded with labels.json, the
pre-registration or a verdict row. A changed config is a new hypothesis.
"""

from __future__ import annotations

import csv
import json
import os

from . import config_digest, repo_path

HIST = 'fx_ny_orb_hypothesis_log.csv'
FWD = os.path.join('forward', 'fx_ny_orb_forward_hypothesis_log.csv')
LABELS_JSON = 'labels.json'
PREREG_JSON = 'PRE_REGISTRATION.json'
PREREG_MD = 'PRE_REGISTRATION.md'
CELL_HOT = 'orb_hot_net_r'
CELL_DIFF = 'orb_hot_minus_cold'
HIST_FIELDS = ['n', 'date', 'hypothesis', 'family', 'metric', 'n_trades_hot', 'n_trades_cold', 'n_dates_hot',
               'window_start', 'window_end', 'statistic', 'hot_mean', 'cold_mean', 'bound', 'bound_quantile', 'alpha',
               'block_days', 'resamples', 'seed', 'boot_se', 'mde_80', 'dropped_resamples', 'verdict',
               'config_digest', 'prereg_sha256', 'notes']
FWD_FIELDS = ['n', 'date', 'hypothesis', 'pair', 'n_days', 'first_day', 'last_day', 'mse_harx', 'mse_har',
              'rel_mse_reduction', 'mean_diff', 'ci_upper', 'alpha', 'block_days', 'resamples', 'seed', 'boot_se',
              'mde_80', 'verdict', 'config_digest', 'frozen_manifest_sha256', 'notes']


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


def read_rows(cfg: dict, name: str) -> list:
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
    seen = [(f'{name} {r["hypothesis"]}', r['config_digest']) for name in (HIST, FWD) for r in read_rows(cfg, name)]
    for name in (LABELS_JSON, PREREG_JSON):
        rec = read_json(cfg, name)
        if rec is not None:
            seen.append((name, rec['config_digest']))
    bad = [src for src, c in seen if c != now]
    if bad:
        raise ConfigChanged(f'the family config differs from the one recorded with {", ".join(bad)}; '
                            'a changed config is a new hypothesis, not a rerun')


def append(cfg: dict, name: str, fields: list, rows: list) -> None:
    have = {r['hypothesis'] for r in read_rows(cfg, name)}
    dup = [r['hypothesis'] for r in rows if r['hypothesis'] in have]
    if dup:
        raise AlreadyScored(f'already scored, each cell is scored once: {dup}')
    p = path(cfg, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new = not os.path.exists(p)
    with open(p, 'x' if new else 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows({k: r.get(k, '') for k in fields} for r in rows)


def hist_verdict(cfg: dict, cell: str = CELL_HOT) -> str:
    """The historical verdict of a cell, or '' when not scored."""
    return next((r['verdict'] for r in read_rows(cfg, HIST) if r['hypothesis'] == cell), '')
