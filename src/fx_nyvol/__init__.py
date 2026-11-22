"""fx_nyvol -- the New York session range, 16:00-23:00 Sofia (openspec change `fx-ny-volatility`).

For EURUSD and GBPUSD: at the close of the bar labelled 14:00 Berlin (16:00 Sofia), forecast
y = log((mid high - mid low) / mid close14) over the bars labelled 15:00-21:00 Berlin. The reference
is a HAR OLS on the range's own history; LightGBM (squared error) is tested against it per pair, and
a 5-seed CUDA MLP against LightGBM only behind a LightGBM KEEP. Four cells at alpha 0.05/4.

ISOLATION CONTRACT
  * Reads the fx_session H1 caches (EURUSD, GBPUSD, AUDUSD) read-only and never refreshes them.
  * Writes only under results/fx_nyvol/ and the gitignored research_models/fx_nyvol/. Never the
    production models directory, the root production config, the production volatility family or
    any other family's registry.
  * GROSS of cost: no verdict uses a spread, a trade or a position; not a trading result.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fx_nyvol_config.json')
GROSS_HEADER = 'Gross of spread and slippage — not a trading result.'


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


def session_view(cfg: dict) -> dict:
    """The config shape fx_session's read-only helpers expect (cache path, pairs, session hours)."""
    return {'paths': {'data': cfg['paths']['fx_session_data']}, 'pairs': cfg['pairs'],
            'factor_pairs': cfg['factor_pairs'], 'data': cfg['data'], 'session': cfg['session']}
