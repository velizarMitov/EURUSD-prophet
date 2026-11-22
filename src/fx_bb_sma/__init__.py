"""fx_bb_sma -- Bollinger first breach, exit at SMA20 / symmetric stop / 10-day time stop
(openspec change `fx-bb-sma-exit`).

Same six pairs, data and events as fx_bb (fx-bb-reversion, DROP on 2026-11-22), now as a simulated
trade: enter at the breach mid close, take profit at the fixed SMA20[t], stop at the same distance
beyond the entry, scan H1 bars hour by hour (a bar touching both counts as the stop), time stop at
the D1 close of valid day t+10. Results in R = net / dist, NET OF THE RECORDED BAR SPREAD ONLY.

    Cell 1  sma_exit_rule_net_r              mean net R per trade > 0, every trade from 2012
    Cell 2  sma_exit_lightgbm_vs_base_rate   pooled LightGBM vs the fit-slice win rate, log-loss

ISOLATION CONTRACT
  * Writes only under results/fx_bb_sma/ and the gitignored research_models/fx_bb_sma/. Reads the
    fx_bb and fx_session H1 caches read-only and never refreshes them. Never writes the production
    models directory or the root production config, or any other family's files.
  * Simulation of history only: no order, broker, live or API code. Slippage, swap and commission
    are not modelled, so no verdict is a trading result.
  * The events were already seen in fx-bb-reversion; every registry note says so.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fx_bb_sma_config.json')
COST_HEADER = ('Net of the recorded bar spread only — slippage, swap and commission are not included; '
               'not a trading result.')


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def repo_path(rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(REPO, *rel.split('/'))


def config_digest(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def feature_list_digest(cfg: dict) -> str:
    blob = json.dumps(cfg['features']['columns'], separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def bb_view(cfg: dict) -> dict:
    """The config shape fx_bb's read-only helpers expect (data paths, events, features)."""
    out = {k: cfg[k] for k in ('pairs', 'reference_pair', 'sources', 'pip', 'data', 'bands', 'atr_period',
                               'reference_quorum', 'features')}
    out['paths'] = dict(cfg['paths_read'])
    out['horizon_days'] = int(cfg['trade']['time_stop_days'])
    out['cooldown_days'] = 1
    return out
