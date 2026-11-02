"""The ONLY place this package names another family's files, and it only reads
them (spec "A failure here costs nothing elsewhere"; tested by
tests/test_external_tsfm_isolation.py, which also refuses any write call here).

    gold      the H1 cache, the forward predictions (h1_lstm, h1_garch_hod) and
              the large-move alert threshold
    EUR/USD   the forward arbiter's predictions (h1_gbm, horizon 1)
"""

from __future__ import annotations

import csv
import json
import os

from . import REPO, repo_path

GOLD_H1_CACHE = 'results/gold/gold_h1.csv'
GOLD_PREDICTIONS = 'results/gold/forward/predictions.csv'
GOLD_SETTLEMENTS = 'results/gold/forward/settlements.csv'
GOLD_CONFIG_V2 = 'src/gold/gold_config_v2.json'
EURUSD_PREDICTIONS = 'results/forward_eval/predictions.csv'


def read_csv_rows(rel: str, repo: str = REPO) -> list[dict]:
    path = repo_path(rel, repo)
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def gold_rows(cell: str, repo: str = REPO, rel: str = GOLD_PREDICTIONS) -> dict:
    """{as_of_bar: row} for one gold H1 cell's forward predictions."""
    return {r['as_of_bar']: r for r in read_csv_rows(rel, repo)
            if r.get('cell') == cell and r.get('cadence') == 'H1'}


def gold_settlements(cell: str, repo: str = REPO) -> dict:
    return {r['as_of_bar']: r for r in read_csv_rows(GOLD_SETTLEMENTS, repo) if r.get('cell') == cell}


def eurusd_h1_gbm(repo: str = REPO) -> dict:
    """{as_of_bar: row} for the EUR/USD arbiter's one-hour h1_gbm forecasts."""
    return {r['as_of_bar']: r for r in read_csv_rows(EURUSD_PREDICTIONS, repo)
            if r.get('model') == 'h1_gbm' and str(r.get('horizon')) == '1' and r.get('cadence') == 'H1'}


def gold_alert_threshold(repo: str = REPO) -> float | None:
    try:
        with open(repo_path(GOLD_CONFIG_V2, repo), encoding='utf-8') as fh:
            return float(json.load(fh)['alerts']['threshold_vol_pct'])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def gold_h1_cache_path(repo: str = REPO) -> str:
    return repo_path(GOLD_H1_CACHE, repo)
