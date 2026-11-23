"""gold_bbnr -- Bollinger Bands and the New York session range on gold (openspec change `gold-bb-nyrange`).

Part 1a  Bollinger (20, 2 SD) first breaches on D1 bars closing 17:00 New York; the 3-day reversion in ATR
         units, and a walk-forward LightGBM against the base rate.
Part 1b  The same breaches as a trade: take profit at SMA20, a symmetric stop, a time stop at t+10, exits
         scanned on H1 bars, R; and a walk-forward LightGBM against the base rate.
Part 2   log(range of the bars labelled 15:00-21:00 Berlin / close14) from fourteen features known at the
         14:00 close: HAR, HAR-X, LightGBM and a 5-seed CUDA MLP.

Three registries under results/gold_bbnr/, seven cells, each scored once and only after
PRE_REGISTRATION.md and .json are committed.

ISOLATION CONTRACT
  * Reads results/gold/gold_h1.csv, the fx_session EURUSD cache and results/fomc_dates.csv READ-ONLY.
  * Fetches the dollar index once (MT5 market data, Yahoo daily) into results/gold_bbnr/data/.
  * Writes only under results/gold_bbnr/ and the gitignored research_models/gold_bbnr/. Never the
    production models directory, config.json, gold_config*.json, or any gold or fx registry.
  * GROSS of cost on bid prices. Nothing here is a trading result. No order code.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold_bbnr_config.json')
GROSS_HEADER = 'Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.'


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


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
