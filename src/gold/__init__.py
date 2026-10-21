"""gold -- gold (MT5 `GOLD`) price, direction and volatility research family.

openspec change `gold-forecasting`. Judged the way the EUR/USD models are:
chronological 70/10/20 split, validation [70:80] as the only arbiter, three
pre-registered Bonferroni families (direction, return, volatility), a one-shot
test block, and forward data as the final judge.

ISOLATION CONTRACT
  * Never imported by api.py's model path, src/inference.py, _train_pipeline.py
    or src/paper_trading.py. api.py may import only src.gold.summary, which is
    read-only.
  * Writes only under results/gold/ and the gitignored research_models/gold/.
    Never writes under models/, never reads config.json, never touches
    results/forward_eval/ (the EUR/USD forward arbiter).
  * Uses src/forecast_eval's pure functions (splits, targets, stats,
    benchmarks, overfit, power maths) read-only; never modifies them.
  * Spread, tick volume and real volume are dropped at acquisition and used
    nowhere -- not as features, not as costs. Every verdict is therefore GROSS
    of cost: statistical skill, never a trading result.

Layout (design D1):
  data            MT5 -> OHLC-only caches under results/gold/
  clock           weekly-open guard on the broker's bar-label clock
  features        scale-free D1 set; the pooled H1 set unchanged
  targets         next-bar log return %, direction, |return| %; H1 eligibility
  splits          70/10/20 chronological positions with boundary purge
  challengers     GBM, multi-task LSTM ensemble, GARCH baselines
  prereg          power table, pre-registration, scoring gate
  study           validation / test / final stages
  forward_logger  hourly forward logging and settlement
  summary         read-only payload for /api/gold
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold_config.json')


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def repo_path(rel: str) -> str:
    """Absolute path for a repo-relative config path (absolute paths pass through)."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(REPO, *rel.split('/'))
