"""fx_ny_orb -- the New York opening-range breakout on hot days, its frozen range forecast and its
MT5 chart layer (openspec change `fx-ny-orb-hot`).

[B] historical: a pre-registered NY opening-range breakout on M15 mid bars (2018-08-08 .. 2026-08-07),
    taken on days the walk-forward HAR-X calls hot; SL 0.25 F, TP 0.60 F, forced exit at the 20:45
    bar; net of the recorded spread plus 0.4 bps slippage; two cells at alpha 0.05/2.
[A] forward: a frozen OLS HAR-X against HAR on live days, two cells at alpha 0.05/2.
[C] chart: Common\\Files\\EURUSDProphet\\ny_orb.tsv drawn by mql5/Indicators/ProphetNYOrb.mq5.
    A live entry sign is exported only behind a historical KEEP of `orb_hot_net_r`.

ISOLATION CONTRACT
  * Writes only under results/fx_ny_orb/, the gitignored research_models/fx_ny_orb/ and the one
    exported TSV. Never the production models directory, the root production config, another
    family's caches or results, the M1 parquets or the horizon-study M15 cache.
  * MT5 is read only. No order, broker, sizing or execution code. Not a trading result.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fx_ny_orb_config.json')
COST_HEADER = ('Net of the recorded spread plus 0.4 bps slippage (no swap: intraday) — '
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


def minutes(hhmm: str) -> int:
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)
