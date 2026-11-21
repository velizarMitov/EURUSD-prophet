"""fx_rates -- the 2-year yield differential at a 5-day horizon (openspec change `fx-rates-2y`).

For EURUSD and GBPUSD: at the close of D1 bar t, features from the foreign-minus-US 2Y yield
spread (yields strictly from the previous FX trading day) and FX momentum; target = sign of the
5-day mid move close[t+5] - close[t]. LightGBM first; a CUDA MLP only behind a KEEP.

ISOLATION CONTRACT
  * Writes only under results/fx_rates/ and the gitignored research_models/fx_rates/.
    Never writes the production models directory, never reads the root production config,
    never writes another family's caches or registry (it READS the fx_session H1 caches).
  * Reuses fx_fib and fx_session helpers read-only; both are decided and their config digests are
    pinned by a test.
  * No broker, order or execution code. Gross of spread and slippage by owner decision
    (2026-11-21): every verdict is a predictive result, never a trading result.
  * Decided by its own purged walk-forward at alpha 0.05/4; no forward phase.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fx_rates_config.json')
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
