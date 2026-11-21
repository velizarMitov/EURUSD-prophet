"""fx_fib -- Fischer Fibonacci structure + CUDA MLP research family (openspec change
`fx-fibonacci-mlp`) for EURUSD and GBPUSD H1, with D1 as causal context.

ISOLATION CONTRACT
  * Writes only under results/fx_fib/ and the gitignored research_models/fx_fib/.
    Never writes under the production models directory, never reads the root
    production config, never rewrites the pooled H1 caches or the live EURUSD H1 cache.
  * Not imported by api.py, src/inference.py, _train_pipeline.py or
    src/paper_trading.py. No broker, order or execution code.
  * Gross of spread and slippage by owner decision (2026-11-21): every verdict is a
    predictive result, never a trading result.
  * Decided by its own purged walk-forward at alpha 0.05/4 (owner decision
    2026-11-21); earlier project results are recorded as facts, never as blockers.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fx_fib_config.json')


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
