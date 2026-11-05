"""gold_window -- the owner-window gold strategy backtest (openspec change
gold-window-strategy, spec gold-forecasting/window-strategy).

Four fixed, model-gated rules that open and close inside 15:30-23:00 Sofia
(labels 14:30-22:00 Berlin), simulated NET of cost on the MT5 history with
the models refit year by year on the past only.

DESCRIPTIVE ONLY. Every year here has been looked at and the rules were
chosen after the descriptive statistics, so nothing this package produces is
evidence: no verdict, no registration, no registry row.

ISOLATION CONTRACT
  * A separate package, not src/gold: gold's contract uses spread nowhere,
    "not as costs", and this family is net of cost by design (design D6).
    src/gold, its registries, registrations and gold_config.json are read-only.
  * Writes only under results/gold/window/ and the gitignored
    research_models/gold/window/. Never writes under models/, never reads
    config.json, is never imported by api.py.
  * Spread and volume are costs only: never a model or gate input.
  * Simulated history only: no order, broker, position-sizing or stop-loss
    code, and no wording that tells anyone to buy or sell.

Layout:
  describe     DESCRIPTIVE.md from the MT5 caches
  bars         H1 + spread, D1 and M15 frames; the read-only MT5 fetch
  rules        the four rules, window eligibility, gates
  simulate     fills, cost floor, daily series
  walkforward  yearly expanding refits
  backtest     backtest.md
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'window_config.json')


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def repo_path(rel: str, repo: str = REPO) -> str:
    """Absolute path for a repo-relative config path (absolute paths pass through)."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(repo, *rel.split('/'))
