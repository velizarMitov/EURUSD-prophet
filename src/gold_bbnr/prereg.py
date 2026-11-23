"""The pre-registration and its git gate (spec gold-bb-nyrange/evaluation, dollar-index; design D10).

    python -m src.gold_bbnr.prereg --write    write PRE_REGISTRATION.md and .json from the config and labels.json
    python -m src.gold_bbnr.prereg --check    report whether both are committed and unchanged

No outcome, exit, target of a scored day or verdict is computed until both files are tracked by git and
unchanged since HEAD, the JSON's config digest equals the current config, and the three input caches
(gold H1, DXY H1, DXY daily) still have the digests recorded in it. The owner makes the commit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from . import REPO, config_digest, load_config
from .inputs import cache_digests
from .registry import (CELLS, LABELS_JSON, PREREG_JSON, PREREG_MD, REGISTRIES, StageOrderError, guard, path,
                       read_json)


class PreregNotCommitted(RuntimeError):
    pass


FACTS = [
    'Owner decisions (2026-11-22): both Bollinger variants, gross of cost (the gold cache has no spread), '
    'HAR/HAR-X/LightGBM/MLP for the range, and the dollar index (MT5 hourly for Part 2, Yahoo daily for Part 1).',
    'Gold H1 2015-2026 was already used by the gold family (validation and one-shot test blocks): its D1/H1 '
    'volatility LSTMs CLEARED against GARCH and its range intervals were DROP. Neither scored band events or the '
    'NY session range.',
    'Currency precedents: fx-bb-reversion and fx-bb-sma-exit DROP on six pairs; fx-ny-volatility LightGBM KEEP '
    'for both pairs and the descriptive HAR-X beat LightGBM and the MLP, which is why HAR-X is registered here; '
    'fx-ny-orb-hot DROP.',
    'Before this pre-registration only event counts, bar presence, the DXY clock and gaps were looked at '
    '(179 first breaches 2015-2026 measured before the plan). No gold outcome, exit or target was computed.',
    'The MT5 contract USDIndDec26 expires on 2026-12-11; its cache is pinned by digest and never refetched.',
]


def cells(cfg: dict) -> dict:
    sc = cfg['scoring']
    return {
        'bb': {'registry': REGISTRIES['bb'], 'alpha': sc['bb']['alpha'], 'block_days': sc['bb']['block_days'],
               'seed': sc['bb']['seed'],
               'cells': {CELLS['bb'][0]: 'mean r = dir * (close[t+3] - close[t]) / ATR20 > 0 over every event with an outcome',
                         CELLS['bb'][1]: 'mean log-loss of the walk-forward LightGBM below the fit-slice base rate of r > 0'}},
        'bb_sma': {'registry': REGISTRIES['bb_sma'], 'alpha': sc['bb_sma']['alpha'],
                   'block_days': sc['bb_sma']['block_days'], 'seed': sc['bb_sma']['seed'],
                   'cells': {CELLS['bb_sma'][0]: 'mean R > 0 over every opened trade with an exit',
                             CELLS['bb_sma'][1]: 'mean log-loss of the walk-forward LightGBM below the fit-slice base rate of R > 0'}},
        'nyvol': {'registry': REGISTRIES['nyvol'], 'alpha': sc['nyvol']['alpha'],
                  'block_days': sc['nyvol']['block_days'], 'seed': sc['nyvol']['seed'],
                  'cells': {CELLS['nyvol'][0]: 'mean squared error of y: HAR-X below HAR',
                            CELLS['nyvol'][1]: 'mean squared error of y: LightGBM below HAR-X',
                            CELLS['nyvol'][2]: 'mean squared error of y: 5-seed MLP below HAR-X'}}}


def build(cfg: dict) -> dict:
    lab = read_json(cfg, LABELS_JSON)
    if lab is None:
        raise StageOrderError('write the label and power report first: python -m src.gold_bbnr.labels')
    if lab['cache_digests'] != cache_digests(cfg):
        raise StageOrderError('an input cache changed since the label report; rewrite labels first')
    return {'change': 'gold-bb-nyrange', 'config_digest': config_digest(cfg), 'cache_digests': lab['cache_digests'],
            'window': [cfg['data']['window_start'], cfg['data']['window_end']], 'families': cells(cfg),
            'resamples': cfg['scoring']['resamples'], 'power_level': cfg['scoring']['power'],
            'power': {'part_1a': {k: lab['part_1a'][k] for k in ('rule_events', 'rule_dates', 'model_events', 'sd_3d_atr',
                                                                  'mde_rule', 'mde_model_accuracy')},
                      'part_1b': {k: lab['part_1b'][k] for k in ('rule_upper_bound', 'model_upper_bound',
                                                                  'mde_rule_at_upper_bound',
                                                                  'mde_model_accuracy_at_upper_bound')},
                      'part_2': {k: lab['part_2'][k] for k in ('usable', 'scored_days', 'first_train', 'mde_sd_units')}},
            'facts': FACTS, 'gross': True}


def write(cfg: dict | None = None) -> tuple:
    cfg = cfg or load_config()
    guard(cfg)
    rec = build(cfg)
    os.makedirs(os.path.dirname(path(cfg, PREREG_JSON)), exist_ok=True)
    with open(path(cfg, PREREG_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    p, t = rec['power'], cfg['trade']
    fam = rec['families']
    L = ['# gold-bb-nyrange — pre-registration', '',
         'Written before any gold outcome, exit or target is computed. Commit this file and PRE_REGISTRATION.json '
         'unchanged; every outcome stage refuses otherwise.', '',
         '> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**', '',
         f"Config digest `{rec['config_digest']}`. Window {rec['window'][0]} .. {rec['window'][1]}.", '',
         'Input caches (pinned; a changed cache stops every gated stage):', ''] + \
        [f'- `{k}` `{v}`' for k, v in rec['cache_digests'].items()] + \
        ['', '## Data', '',
         '- Gold: results/gold/gold_h1.csv (bid OHLC, Berlin label clock, read-only). D1 bars close at 17:00 New York; '
         'valid with >= 12 H1 bars on a weekday.',
         '- DXY hourly: MT5 ActivTrades USDIndDec26, from 2017-01-03, lag-0 checked against EURUSD every year.',
         '- DXY daily: Yahoo DX-Y.NYB, closes dated strictly before the gold day only.', '',
         '## Part 1a — Bollinger reversion', '',
         '- Event: the first D1 close outside SMA20 ± 2 SD20 (population SD); 3-valid-day cooldown.',
         '- Outcome r = dir × (close[t+3] − close[t]) / ATR20[t].',
         '- Model: binary LightGBM (fx_bb settings) on 11 features (9 gold, 2 previous-day DXY), 12-month folds '
         'from 2018-01-01, 10-business-day purge + embargo from t+3, folds with < 60 training events predict nothing.',
         f"- Cells at alpha {fam['bb']['alpha']} each, {fam['bb']['block_days']}-day blocks, seed {fam['bb']['seed']}:"] + \
        [f'  - `{k}`: {v}.' for k, v in fam['bb']['cells'].items()] + \
        [f"- Power: {p['part_1a']['rule_events']} rule events → minimum detectable mean {p['part_1a']['mde_rule']:.3f} ATR; "
         f"{p['part_1a']['model_events']} model events → minimum detectable accuracy {p['part_1a']['mde_model_accuracy']:.3f}.",
         '', '## Part 1b — the SMA20 trade', '',
         f"- Entry at the D1 close of every first breach; TP = SMA20[t]; stop the same distance beyond the entry; "
         f"time stop at the close of t+{t['time_stop_days']}; exits scanned on H1 bars (a bar touching both is a stop; "
         'a gap through the stop fills at the open; a gap through the target fills at the target); one open trade.',
         '- R = dir × (exit − entry) / |entry − stop|.',
         '- Model: as Part 1a, on R > 0, the purge measured from the exit date.',
         f"- Cells at alpha {fam['bb_sma']['alpha']} each, {fam['bb_sma']['block_days']}-day blocks, seed "
         f"{fam['bb_sma']['seed']}:"] + \
        [f'  - `{k}`: {v}.' for k, v in fam['bb_sma']['cells'].items()] + \
        [f"- Power (upper bounds): at most {p['part_1b']['rule_upper_bound']} trades → minimum detectable mean ≥ "
         f"{p['part_1b']['mde_rule_at_upper_bound']:.3f} R (SD 1.0 R); at most {p['part_1b']['model_upper_bound']} model "
         f"trades → minimum detectable accuracy ≥ {p['part_1b']['mde_model_accuracy_at_upper_bound']:.3f}.",
         '', '## Part 2 — the NY session range', '',
         '- y = log((max high − min low of the bars labelled 15:00–21:00 Berlin) / close of the 14:00 bar).',
         '- Fourteen features at the 14:00 close (nine gold, the DXY 00:00–14:00 range ratio, four calendar flags).',
         '- HAR (OLS on 3 lags), HAR-X (OLS on 14), LightGBM (fx_nyvol settings), 5-seed CUDA MLP; 6-month folds '
         'from 2018-01-01, 1-day embargo; all four models on the same rows.',
         f"- Three independent cells at alpha {fam['nyvol']['alpha']:.5f} each (0.05/3), "
         f"{fam['nyvol']['block_days']}-day blocks, seed {fam['nyvol']['seed']}:"] + \
        [f'  - `{k}`: {v}.' for k, v in fam['nyvol']['cells'].items()] + \
        [f"- Power: {p['part_2']['scored_days']} scored days → minimum detectable mean difference "
         f"{p['part_2']['mde_sd_units']:.4f} SD.",
         '', '## Decision rule', '',
         f"- Whole business days are resampled (circular blocks, {rec['resamples']} resamples, fixed seed). A \"> 0\" "
         'cell is KEEP when the alpha-quantile of the bootstrap means is above 0; a "below" cell is KEEP when the '
         '(1 − alpha)-quantile of the mean difference is below 0. Everything else is DROP. Each cell is scored once.',
         '', '## Facts recorded', ''] + [f'- {f}' for f in FACTS] + ['']
    with open(path(cfg, PREREG_MD), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path(cfg, PREREG_MD), path(cfg, PREREG_JSON)


def _git(args, repo):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True)


def gate(cfg: dict, repo: str = REPO, digests: dict | None = None) -> str:
    """Raise unless both files are committed and unchanged, the JSON matches the config, and the input
    caches match their pinned digests. Returns the ISO commit date of PRE_REGISTRATION.json."""
    for name in (PREREG_MD, PREREG_JSON):
        p = path(cfg, name)
        if not os.path.exists(p):
            raise PreregNotCommitted(f'{name} does not exist: write it with --write and commit it')
        rel = os.path.relpath(p, repo)
        if _git(['ls-files', '--error-unmatch', '--', rel], repo).returncode != 0:
            raise PreregNotCommitted(f'{name} is not committed')
        if _git(['diff', '--quiet', 'HEAD', '--', rel], repo).returncode != 0:
            raise PreregNotCommitted(f'{name} differs from its committed version')
    rec = read_json(cfg, PREREG_JSON)
    if rec['config_digest'] != config_digest(cfg):
        raise PreregNotCommitted('the committed pre-registration was written under another config')
    now = digests if digests is not None else cache_digests(cfg)
    bad = [k for k, v in rec['cache_digests'].items() if now.get(k) != v]
    if bad:
        raise PreregNotCommitted(f'input cache changed since the pre-registration: {bad}')
    rel = os.path.relpath(path(cfg, PREREG_JSON), repo)
    return _git(['log', '-1', '--format=%cI', '--', rel], repo).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold-bb-nyrange pre-registration')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', action='store_true')
    g.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    if a.write:
        print('\n'.join(write()))
    else:
        try:
            print(f'committed and unchanged (commit date {gate(load_config())}) at {datetime.now(timezone.utc):%Y-%m-%d}')
        except PreregNotCommitted as e:
            raise SystemExit(f'NOT READY: {e}')


if __name__ == '__main__':
    main()
