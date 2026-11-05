"""gold_accel -- does time-of-day acceleration help gold models call the next
hour's direction? (openspec change gold-acceleration-direction, spec
gold-forecasting/acceleration-direction).

Acceleration features (causal, scale-free, no fit), shadow copies of the
registered h1_gbm / h1_lstm configurations with those columns added, a new GBM
on acceleration alone, a yearly expanding walk-forward and one report.

DESCRIPTIVE ONLY. Every year has been seen and the owner saw the acceleration
statistics first, so nothing here is evidence: no verdict, no registration, no
registry row, no forward logger.

ISOLATION CONTRACT
  * A separate package, not src/gold: its one cost check reads the spread,
    which gold's contract forbids, and new columns inside src/gold would risk
    the pinned round-one digests. src/gold, gold_config.json, the registries,
    versions and loggers are read-only; models are built with
    src.gold.challengers.make and a column override, never registered.
  * Writes only under the gitignored research_models/gold/accel/ and under
    results/gold/accel/. Never writes under models/, never reads config.json.
  * Spread is a cost only, never a model input. No volume anywhere.
  * Simulated history only: no order, broker, sizing or stop-loss code, and no
    wording that tells anyone to buy or sell.

Layout:
  features     the acceleration columns and the model Inputs
  walkforward  pinned inputs, yearly expanding refits, resumable
  backtest     scoring, the one cost check, backtest.md
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'accel_config.json')


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def repo_path(rel: str, repo: str = REPO) -> str:
    """Absolute path for a repo-relative config path (absolute paths pass through)."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(repo, *rel.split('/'))
