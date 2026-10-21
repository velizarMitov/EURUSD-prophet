"""Gold pre-registration: power table, families, the registered protocol and
the scoring gate (design D7, D9; spec "Three gold families with registered
composition", "Pre-registration precedes the first validation score").

The registration PINS the data: the caches are append-only and grow every hour,
so without a pinned last bar the 70 / 80 % boundaries would drift and a later
run would score different rows than the ones registered. It also pins the
gold_config.json digest; the study refuses to score when either has changed.

    python -m src.gold.prereg            writes PRE_REGISTRATION.md / .json and
                                         the three empty registries (refuses to
                                         overwrite an existing registration)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

from src.forecast_eval import prereg as FP
from src.forecast_eval import stats as ST

from . import CONFIG_PATH, REPO, load_config, repo_path

UNDERPOWERED = FP.UNDERPOWERED
REGISTRY_FIELDS = ['n', 'date', 'cell', 'family', 'hypothesis', 'arbiter', 'n_rows',
                   'metric_model', 'metric_baseline', 'point_delta', 'ci_low', 'ci_high',
                   'p_value', 'alpha_bonferroni', 'cleared_bar', 'verdict', 'notes']

DIRECTION_CELLS = [('d1_gbm', 'D1'), ('d1_lstm', 'D1'), ('h1_gbm', 'H1'), ('h1_lstm', 'H1')]
RETURN_CELLS = [('d1_gbm', 'D1'), ('d1_lstm', 'D1'), ('h1_gbm', 'H1'), ('h1_lstm', 'H1')]
VOLATILITY_CELLS = [('d1_lstm', 'D1', 'd1_garch'), ('h1_lstm', 'H1', 'h1_garch_hod')]

EXPECTED = {
    'direction': {'d1_gbm': f'{UNDERPOWERED} (~50 %)', 'd1_lstm': f'{UNDERPOWERED} (~50 %)',
                  'h1_gbm': ('possible CLEARED at 51-53 % (H_dir.1 GBM analogue: KEEP on EUR/USD 52.7 %, '
                             'AUD/USD, CHF/USD; DROP on GBP/USD -- h1_direction_hypothesis_log.csv)'),
                  'h1_lstm': ('DROP more likely than CLEARED (H_dir.2: the EUR/USD H1 LSTM scored 51.6 % '
                              'vs the GBM 52.7 %, DROP)')},
    'return': {c: 'DROP (Huber-loss shrinkage to the ~0 % conditional mean)' for c, _ in RETURN_CELLS},
    'volatility': {'d1_lstm': 'CLEARED (volatility clustering; EUR/USD analogue CLEARED)',
                   'h1_lstm': 'CLEARED'},
}


def sha256_file(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def frame_digest(df) -> str:
    """Digest of the pinned rows exactly as the study will read them."""
    text = df.to_csv(float_format='%.6f').encode()
    return hashlib.sha256(text).hexdigest()


def direction_family(n_by_cadence: dict, edge: float, power: float, family_alpha: float):
    """Fixed point on VALIDATION rows: start from every direction cell, admit
    those whose eligible validation n reaches n_required at the current family
    alpha, repeat until the admitted set is stable. Underpowered cells spend no
    alpha and are still reported."""
    cells = list(DIRECTION_CELLS)
    size, history = len(cells), []
    while True:
        alpha = family_alpha / size
        need = FP.n_required(alpha, edge, power)
        admitted = [c for c in cells if n_by_cadence[c[1]] >= need]
        history.append({'family_size': size, 'alpha': alpha, 'n_required': need, 'admitted': len(admitted)})
        if not admitted or len(admitted) == size:
            break
        size = len(admitted)
    alpha = family_alpha / max(len(admitted), 1)
    need = FP.n_required(alpha, edge, power)
    table = [{'cell': c, 'cadence': cad, 'n_validation': int(n_by_cadence[cad]),
              'n_required': round(need, 1),
              'status': 'ADMITTED' if (c, cad) in admitted else UNDERPOWERED}
             for c, cad in cells]
    return [c for c, _ in admitted], alpha, table, history


def build_registration(cfg: dict, frames: dict, data_records: dict, clock_ev: dict) -> dict:
    """The registration object. `frames` are the pinned D1 / H1 OHLC frames."""
    from . import challengers as C
    from . import splits as S
    from . import targets as G
    n_val, sizes = {}, {}
    for cad, df in frames.items():
        p = S.stage_positions(len(df), 'validation', cfg)
        inp = C.build_inputs(df, cad)
        tg = G.build_targets(df['close'])
        n_val[cad] = int(len(C.usable(inp, p['score'], tg, ('dir',))))
        t = S.stage_positions(len(df), 'test', cfg)
        sizes[cad] = {'rows': len(df), 'train': p['fit_end'], 'validation': len(p['score']),
                      'test': len(t['score']), 'eligible_validation_rows': n_val[cad]}
    pw = cfg['power']
    admitted, alpha_dir, table, history = direction_family(n_val, pw['edge'], pw['power'], pw['family_alpha'])
    alpha_ret = pw['family_alpha'] / len(RETURN_CELLS)
    alpha_vol = pw['family_alpha'] / len(VOLATILITY_CELLS)
    return {
        'change': 'gold-forecasting',
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'config_sha256': sha256_file(CONFIG_PATH),
        'data': {cad: {**data_records[cad], 'last_bar': str(frames[cad].index[-1]),
                       'rows': len(frames[cad]), 'sha256': frame_digest(frames[cad])} for cad in frames},
        'price_definition': 'MT5 bid OHLC; spread, tick volume and real volume used nowhere',
        'cost_basis': 'GROSS of cost -- statistical skill only, never a trading result',
        'clock': {**clock_ev, 'threshold': cfg['clock']['weekly_open_threshold']},
        'split': {**cfg['split'], 'sizes': sizes},
        'eligibility': ('H1: the as-of and target bars are consecutive hours and the as-of bar follows '
                        'its predecessor by one hour (no bar bordering a market gap); D1: every row with '
                        'a next bar. Zero moves are excluded from direction scoring.'),
        'families': {
            'direction': {'cells': [c for c, _ in DIRECTION_CELLS], 'admitted': admitted,
                          'alpha': alpha_dir, 'edge': pw['edge'], 'power': pw['power'],
                          'power_table': table, 'fixed_point': history,
                          'baselines': 'training-majority class and a 50 % coin, on the same rows',
                          'verdict': ('CLEARED iff the two-sided (1 - alpha) paired moving-block bootstrap '
                                      'CI of mean(correct_model - correct_majority) AND of '
                                      'mean(correct_model - 0.5) both have lower bound > 0; else DROP. '
                                      'Pesaran-Timmermann reported alongside, not decisive.')},
            'return': {'cells': [c for c, _ in RETURN_CELLS], 'alpha': alpha_ret,
                       'baselines': 'the zero-return forecast (driftless random walk)',
                       'verdict': 'CLEARED iff the one-sided Clark-West p-value < alpha; else DROP.'},
            'volatility': {'comparisons': [{'cell': c, 'cadence': cad, 'baseline': b} for c, cad, b in VOLATILITY_CELLS],
                           'alpha': alpha_vol,
                           'verdict': ('CLEARED iff the two-sided (1 - alpha) paired moving-block bootstrap '
                                       'CI of mean(|err_baseline| - |err_model|) has lower bound > 0; else DROP.')},
        },
        'ci_method': {'kind': 'paired circular moving-block bootstrap',
                      'block_length': {'D1': ST.block_length(1, 'D1'), 'H1': ST.block_length(1, 'H1')},
                      'n_boot': cfg['bootstrap']['n_boot'], 'seed': ST.SEED},
        'model_configs': {'gbm': cfg['gbm'], 'lstm': cfg['lstm'], 'seeds': cfg['seeds'],
                   'gbm_tuning': 'inner purged walk-forward on the training block only; best mean inner ROC-AUC',
                   'garch': {'d1_garch': 'GARCH(1,1), constant mean, normal, fit on training returns only',
                             'h1_garch_hod': ('GARCH(1,1), zero mean, on returns / training mean |r| per '
                                              'label hour; rescaled by the next hour')}},
        'descriptive_only': ['accuracy by hour of day, flagged when > 1/2 of excess accuracy comes from '
                             'the 00:00-06:00 labels (high-spread hours)',
                             'Deflated Sharpe of a gross sign strategy over the gold trial log',
                             'PBO across cells'],
        'stages': {'validation': 'fit [0:70 %], score [70:80 %] -- the only arbiter',
                   'test': 'after every validation verdict is committed: fit [0:80 %], score [80:100 %], once',
                   'final': 'fit [0:100 %] -> frozen forward artifacts'},
        'expected_outcomes': EXPECTED,
        'corrections': 'never edited here; see results/gold/RESULTS.md',
    }


def render_markdown(reg: dict) -> str:
    fam = reg['families']
    lines = [
        '# Gold pre-registration', '',
        f'Committed before any validation score exists (openspec change `{reg["change"]}`). '
        'This file is never edited after its commit; corrections go to `results/gold/RESULTS.md`.', '',
        f'- Created: {reg["created_at"]}',
        f'- Config digest (`src/gold/gold_config.json`): `{reg["config_sha256"]}`',
        f'- Price definition: {reg["price_definition"]}',
        f'- **Cost basis: {reg["cost_basis"]}**', '',
        '## Pinned data', '',
        '| cadence | symbol | rows | first bar | last bar | sha256 |', '|---|---|---|---|---|---|',
    ]
    for cad, d in reg['data'].items():
        lines.append(f'| {cad} | {d.get("symbol", "")} | {d["rows"]} | {d["first"]} | {d["last_bar"]} | `{d["sha256"][:16]}…` |')
    sz = reg['split']['sizes']
    lines += ['', '## Clock', '',
              f'{reg["clock"]["match_rate"]:.1%} of {reg["clock"]["n_weeks"]} weekly opens land on the predicted '
              f'Europe/Berlin hour (threshold {reg["clock"]["threshold"]:.0%}).', '',
              '## Split', '', '| cadence | train | validation | test | eligible validation rows |',
              '|---|---|---|---|---|']
    for cad, s in sz.items():
        lines.append(f'| {cad} | {s["train"]} | {s["validation"]} | {s["test"]} | {s["eligible_validation_rows"]} |')
    lines += ['', f'Eligibility: {reg["eligibility"]}', '',
              '## Families', '',
              f'### Direction (alpha = {fam["direction"]["alpha"]:.5f})', '',
              f'Edge {fam["direction"]["edge"]:.0%} over 50 % at {fam["direction"]["power"]:.0%} power. '
              f'Baselines: {fam["direction"]["baselines"]}.', '',
              '| cell | cadence | validation n | required n | status |', '|---|---|---|---|---|']
    for r in fam['direction']['power_table']:
        lines.append(f'| {r["cell"]} | {r["cadence"]} | {r["n_validation"]} | {r["n_required"]:.0f} | {r["status"]} |')
    lines += ['', f'Verdict: {fam["direction"]["verdict"]}', '',
              f'### Return (alpha = {fam["return"]["alpha"]:.5f})', '',
              f'Cells: {", ".join(fam["return"]["cells"])}. Baseline: {fam["return"]["baselines"]}.', '',
              f'Verdict: {fam["return"]["verdict"]}', '',
              f'### Volatility (alpha = {fam["volatility"]["alpha"]:.5f})', '']
    for c in fam['volatility']['comparisons']:
        lines.append(f'- `{c["cell"]}` vs `{c["baseline"]}` ({c["cadence"]})')
    ci = reg['ci_method']
    lines += ['', f'Verdict: {fam["volatility"]["verdict"]}', '',
              '## CI method', '',
              f'{ci["kind"]}, block length D1 = {ci["block_length"]["D1"]}, H1 = {ci["block_length"]["H1"]}, '
              f'{ci["n_boot"]} resamples, seed {ci["seed"]}.', '',
              '## Models', '',
              f'- GBM grid: `{json.dumps(reg["model_configs"]["gbm"]["grid"])}`; tuning: {reg["model_configs"]["gbm_tuning"]}.',
              f'- LSTM D1: `{json.dumps(reg["model_configs"]["lstm"]["D1"])}`',
              f'- LSTM H1: `{json.dumps(reg["model_configs"]["lstm"]["H1"])}`',
              f'- Seeds: {reg["model_configs"]["seeds"]} (mean of all five; a partial ensemble is never scored).',
              f'- d1_garch: {reg["model_configs"]["garch"]["d1_garch"]}.',
              f'- h1_garch_hod: {reg["model_configs"]["garch"]["h1_garch_hod"]}.', '',
              '## Descriptive only (never a verdict)', '']
    lines += [f'- {d}' for d in reg['descriptive_only']]
    lines += ['', '## Stages', '']
    lines += [f'- **{k}**: {v}' for k, v in reg['stages'].items()]
    lines += ['', '## Expected outcomes (written before any score)', '',
              '| family | cell | expected |', '|---|---|---|']
    for famname, cells in reg['expected_outcomes'].items():
        for c, e in cells.items():
            lines.append(f'| {famname} | {c} | {e} |')
    lines.append('')
    return '\n'.join(lines)


def init_registries(cfg: dict) -> list[str]:
    written = []
    for path in cfg['paths']['registries'].values():
        full = repo_path(path)
        if os.path.exists(full):
            continue
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w', newline='', encoding='utf-8') as fh:
            csv.DictWriter(fh, fieldnames=REGISTRY_FIELDS).writeheader()
        written.append(full)
    return written


def generate(cfg: dict | None = None) -> dict:
    from . import clock as K
    from . import data as D
    cfg = cfg or load_config()
    md, js = repo_path(cfg['paths']['prereg_md']), repo_path(cfg['paths']['prereg_json'])
    if os.path.exists(md) or os.path.exists(js):
        raise FileExistsError('a gold pre-registration already exists; it is never regenerated '
                              '(corrections go to results/gold/RESULTS.md)')
    frames, records = {}, {}
    for cad in ('D1', 'H1'):
        frames[cad], records[cad] = D.load(cad, allow_fetch=False, cfg=cfg)
    clock_ev = K.verify(frames['H1'].index, cfg['clock']['weekly_open_threshold'])
    reg = build_registration(cfg, frames, records, clock_ev)
    with open(js, 'w', encoding='utf-8') as fh:
        json.dump(reg, fh, indent=1, sort_keys=False, default=str)
    with open(md, 'w', encoding='utf-8') as fh:
        fh.write(render_markdown(reg))
    reg['registries_created'] = init_registries(cfg)
    return reg


def registration(cfg: dict | None = None) -> dict | None:
    cfg = cfg or load_config()
    js = repo_path(cfg['paths']['prereg_json'])
    if not os.path.exists(js):
        return None
    with open(js, encoding='utf-8') as fh:
        return json.load(fh)


def scoring_allowed(cfg: dict | None = None, repo: str = REPO) -> bool:
    """True only when both registration files are committed, unmodified, and
    their commit is an ancestor of HEAD."""
    cfg = cfg or load_config()
    return all(FP.committed_and_clean(repo_path(cfg['paths'][k]), repo) is not None
               for k in ('prereg_md', 'prereg_json'))


def pinned_frame(df, reg: dict, cadence: str):
    """The registered rows, or ValueError when the cache no longer holds them
    exactly (rows rewritten, removed or reordered)."""
    d = reg['data'][cadence]
    import pandas as pd
    pinned = df[df.index <= pd.Timestamp(d['last_bar'])]
    if len(pinned) != d['rows'] or frame_digest(pinned) != d['sha256']:
        raise ValueError(f'{cadence} cache no longer matches the registered rows '
                         f'({len(pinned)} rows vs {d["rows"]} registered)')
    return pinned


def assert_config_unchanged(reg: dict) -> None:
    if sha256_file(CONFIG_PATH) != reg['config_sha256']:
        raise ValueError('src/gold/gold_config.json changed after registration; '
                         'scoring would not be the registered protocol')


def main(argv=None):
    ap = argparse.ArgumentParser(description='write the gold pre-registration (once)')
    ap.parse_args(argv)
    reg = generate()
    fam = reg['families']['direction']
    print(f"direction family: admitted {fam['admitted']} at alpha {fam['alpha']:.5f}")
    for r in fam['power_table']:
        print(f"  {r['cell']:8s} {r['cadence']}  n_val={r['n_validation']:5d}  need {r['n_required']:.0f}  {r['status']}")
    print('wrote', os.path.relpath(repo_path(load_config()['paths']['prereg_md']), REPO))
    print('Review it, then commit it BY HAND, on its own, with an explicit message.')


if __name__ == '__main__':
    main()
