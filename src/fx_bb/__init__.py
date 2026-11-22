"""fx_bb -- Bollinger Band (20, 2 SD) mean reversion on D1 (openspec change `fx-bb-reversion`).

Six pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF). An event is the first D1 mid close
outside the bands (D1 bars close at 17:00 New York, built from the Berlin-labelled H1 caches);
the direction is counter-trend and the outcome is the 3-day mid move in ATR(20) units.

    Cell 1  bb_rule_mean_reversion     the rule itself, mean outcome > 0, every event from 2012
    Cell 2  bb_lightgbm_vs_base_rate   pooled LightGBM vs the fit-slice base rate, log-loss,
                                       walk-forward from 2015

ISOLATION CONTRACT
  * Writes only under results/fx_bb/ and the gitignored research_models/fx_bb/. Never writes the
    production models directory, never reads the root production config, never writes another
    family's caches or registry (it READS the fx_session H1 caches for EURUSD, GBPUSD, AUDUSD).
  * Reuses fx_fib and fx_session helpers read-only; their config digests are pinned by a test.
  * No broker, order or execution code. Gross of spread and slippage by owner decision: every
    verdict is a predictive result, never a trading result. Spreads appear only in the
    descriptive report.
  * Two cells at alpha 0.05/2, each scored once; no MLP.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fx_bb_config.json')
GROSS_HEADER = 'Gross of spread and slippage — not a trading result.'


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def repo_path(rel: str) -> str:
    """Absolute path for a repo-relative config path (absolute paths pass through)."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(REPO, *rel.split('/'))


def config_digest(cfg: dict) -> str:
    """SHA-256 of the canonical JSON of the config, independent of key order and whitespace."""
    blob = json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def feature_list_digest(cfg: dict) -> str:
    blob = json.dumps(cfg['features']['columns'], separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()
