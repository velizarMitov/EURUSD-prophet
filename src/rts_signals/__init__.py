"""rts_signals -- Rockefeller Treasury Services report signals: ingest, rule book, simulation on the owner's
MT5 prices, a forward evaluation and a chart (openspec change `rts-signals`).

RTS is a third-party source. Its levels are its own suggestions, never this project's signal.

ISOLATION CONTRACT
  * Display and evaluation only. No order, position, broker, sizing or execution code.
  * Gmail is read with the gmail.readonly scope only; never sent, labelled, moved or deleted.
    Credentials live in the gitignored research_models/rts_signals/secrets/ and are never printed.
  * MT5 is read for market data only (M1 bars and their spread).
  * Writes only under results/rts_signals/, the gitignored research_models/rts_signals/ and the one chart
    file Common\\Files\\EURUSDProphet\\rts_signals.tsv. Never the production models directory, config.json
    or any other family's files or registries.
  * Raw PDFs and their text are paid third-party content: they stay in research_models/, never in git.
  * Results are net of the recorded bar spread only; slippage, swap and commission are not modelled.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PKG, 'rts_signals_config.json')
CELL = 'rts_spot_net_r'
COST_HEADER = ('Net of the recorded MT5 bar spread only; slippage, swap and commission are not modelled — '
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def results_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['results'])


def art_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['artifacts'])


def sub(cfg: dict, key: str) -> str:
    """One of the artifact subfolders: inbox, drop, secrets or cache."""
    return os.path.join(art_dir(cfg), cfg['paths'][key])


def symbol_of(cfg: dict, name: str) -> str:
    return cfg['pairs'][name]['symbol']


def pair_by_symbol(cfg: dict) -> dict:
    return {v['symbol']: dict(v, name=k) for k, v in cfg['pairs'].items()}
