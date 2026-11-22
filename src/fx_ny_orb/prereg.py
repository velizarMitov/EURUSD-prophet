"""The pre-registration and its git gate (spec fx-ny-orb-hot/evaluation; design D5).

    python -m src.fx_ny_orb.prereg --write    write PRE_REGISTRATION.md and .json from the config and labels.json
    python -m src.fx_ny_orb.prereg --check    report whether both are committed and unchanged

No outcome is computed and no cell is scored until both files are committed to git, unchanged since
the commit, and the JSON's config digest equals the current config. The owner makes the commit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from . import REPO, config_digest, load_config
from .registry import CELL_DIFF, CELL_HOT, LABELS_JSON, PREREG_JSON, PREREG_MD, StageOrderError, guard, path, read_json


class PreregNotCommitted(RuntimeError):
    pass


FACTS = ['HAR-X (OLS on fx_nyvol\'s 14 features) was chosen after its range MSE was seen in fx-ny-volatility '
         '(2026-11-22, descriptive line, MSE 0.80x / 0.85x HAR); its walk-forward predictions are reused unchanged.',
         'The breakout outcomes on these days were not computed or looked at before this pre-registration.',
         'Earlier fx families: fx-fibonacci, fx-session-ny, fx-rates-2y, fx-bb-reversion and fx-bb-sma-exit were DROP; '
         'fx-ny-volatility LightGBM vs HAR KEEP for both pairs.']


def build(cfg: dict) -> dict:
    lab = read_json(cfg, LABELS_JSON)
    if lab is None:
        raise StageOrderError('write the label and power report first: python -m src.fx_ny_orb.labels')
    sc = cfg['scoring']
    return {'change': 'fx-ny-orb-hot', 'config_digest': config_digest(cfg), 'rule': cfg['rule'],
            'window': [cfg['data']['window_start'], cfg['data']['window_end']],
            'cells': {CELL_HOT: 'mean net R of hot-day trades > 0 (pooled EURUSD+GBPUSD)',
                      CELL_DIFF: 'mean net R of hot-day trades - mean net R of non-hot trades > 0'},
            'alpha_per_cell': sc['alpha'], 'family_size': sc['family_size'], 'block_days': sc['block_days'],
            'resamples': sc['resamples'], 'seed': sc['seed_historical'], 'declared_sd_r': sc['declared_sd_r'],
            'power': lab['power']['pooled'], 'facts': FACTS,
            'forward': {'cells': ['EURUSD_harx_vs_har', 'GBPUSD_harx_vs_har'], 'alpha_per_cell': sc['alpha'],
                        'min_days': cfg['forward']['min_days'], 'deadline': cfg['forward']['deadline'],
                        'frozen_fit_end': cfg['frozen']['fit_end']},
            'chart_gate': f'a live entry sign is exported only when {CELL_HOT} is KEEP'}


def write(cfg: dict | None = None) -> tuple:
    cfg = cfg or load_config()
    guard(cfg)
    rec = build(cfg)
    os.makedirs(os.path.dirname(path(cfg, PREREG_JSON)), exist_ok=True)
    with open(path(cfg, PREREG_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    r, p = rec['rule'], rec['power']
    L = ['# fx-ny-orb-hot — pre-registration', '',
         'Written before any breakout outcome is computed. Commit this file and PRE_REGISTRATION.json unchanged; '
         'scoring refuses otherwise.', '',
         f"Config digest `{rec['config_digest']}`.", '', '## The rule (fixed, never searched)', '',
         f"- NY opening range: the M15 bars labelled {', '.join(r['or_bars'])} Berlin (15:00–16:00 Sofia).",
         '- Hot day: the walk-forward HAR-X forecast above the causal Q75 of the previous 60 eligible days.',
         f"- Entry: the first M15 close outside the range from {r['breakout_from']}; entry at the next bar's mid open, "
         f"only if range ≤ {r['or_width_max_f']} F, breakout ≤ {r['breakout_max_f']} F beyond the level, entry before "
         f"{r['entry_before']}. Otherwise no trade that day.",
         f"- SL {r['stop_f']} F, TP {r['target_f']} F from the entry; a bar touching both is SL; forced exit at the "
         f"close of the bar labelled {r['last_bar']}; one trade per day per pair.",
         f"- Costs: half the recorded spread at entry and at exit, plus {r['slippage_bps']} bps slippage. R = SL distance.",
         '', '## Cells (historical family, alpha 0.05/2 each, scored once)', '',
         f"- `{CELL_HOT}`: {rec['cells'][CELL_HOT]}.", f"- `{CELL_DIFF}`: {rec['cells'][CELL_DIFF]}.",
         f"- Window {rec['window'][0]} .. {rec['window'][1]}; circular 5-business-day block bootstrap, "
         f"{rec['resamples']} resamples, seed {rec['seed']}.",
         f"- Power: {p['hot']['trades']} hot trades on {p['hot']['dates']} dates; minimum detectable mean "
         f"{p['hot']['mde_trades']:.3f} R on trades, {p['hot']['mde_dates']:.3f} R on dates (SD {rec['declared_sd_r']} R).",
         '', '## Forward family (alpha 0.05/2 each, scored once)', '',
         f"- `EURUSD_harx_vs_har`, `GBPUSD_harx_vs_har`: frozen OLS HAR-X vs HAR (fit to {rec['forward']['frozen_fit_end']}), "
         f"per-day squared error, after {rec['forward']['min_days']} settled forward days per pair or on "
         f"{rec['forward']['deadline']} (then UNDERPOWERED — NO DECISION if fewer). Forward days start after this commit.",
         '', '## Chart gate', '', f"- {rec['chart_gate']}.", '', '## Facts recorded', ''] + [f'- {f}' for f in FACTS] + ['']
    with open(path(cfg, PREREG_MD), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path(cfg, PREREG_MD), path(cfg, PREREG_JSON)


def _git(args, repo):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True)


def gate(cfg: dict, repo: str = REPO) -> str:
    """Raise unless both files are committed and unchanged and the JSON matches the config.
    Returns the ISO commit date of PRE_REGISTRATION.json."""
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
    rel = os.path.relpath(path(cfg, PREREG_JSON), repo)
    return _git(['log', '-1', '--format=%cI', '--', rel], repo).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description='fx-ny-orb-hot pre-registration')
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
