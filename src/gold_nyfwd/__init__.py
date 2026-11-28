"""gold_nyfwd -- the forward test of gold_bbnr's frozen HAR-X for gold's New York session range
(openspec change `gold-nyrange-forward`).

Every weekday, after the bar labelled 14:00 Berlin closes, the frozen OLS HAR-X and HAR (coefficients
copied into the committed pre-registration) forecast y = log(range of the bars labelled 15:00-21:00 /
close14). After the bar labelled 21:00 closes, the day is settled with gold_bbnr's own target and
eligibility code. One cell, `gold_harx_vs_har_forward`, at alpha 0.05, scored once at 60 forward days
or on 2027-04-30.

ISOLATION CONTRACT
  * Reads gold_bbnr's feature, target and D1 code and its frozen ols.json READ-ONLY; never refits.
  * MT5 is read for market data only (GOLD, EURUSD and the USDInd contracts). No order code.
  * Writes only under results/gold_nyfwd/, the gitignored research_models/gold_nyfwd/ and the one chart
    file Common\\Files\\EURUSDProphet\\gold_ny_range.tsv. Never the production models directory,
    config.json, the gold family's caches, gold_bbnr, fx_ny_orb or any other registry.
  * GROSS of cost on bid prices. Nothing here is a trading result.
"""

from __future__ import annotations

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold_nyfwd_config.json')
GROSS_HEADER = 'Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.'
CELL = 'gold_harx_vs_har_forward'


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


def minutes(hhmm: str) -> int:
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)


def results_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['results'])


def art_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['artifacts'])
