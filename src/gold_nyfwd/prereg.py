"""The pre-registration, its git gate and the pinned coefficients (spec gold-nyrange-forward/forward-evaluation;
design D2, D7).

    python -m src.gold_nyfwd.prereg --write    write PRE_REGISTRATION.md and .json (copies the frozen coefficients)
    python -m src.gold_nyfwd.prereg --check    report whether both are committed and unchanged

`build` copies, read-only, the HAR-X (15) and HAR (4) coefficients and their column order from gold_bbnr's
frozen research_models/gold_bbnr/ols.json, after checking its SHA-256 against gold_bbnr's frozen manifest.
It also takes gold_bbnr's historical HAR-X-vs-HAR result to state the expected power at 60 days.

Every pass predicts from the committed JSON, never from ols.json. If ols.json still exists and its
digest differs from the one recorded, `coefficients` refuses (FrozenChanged). No forward row is logged
and nothing is scored until both files are committed, unchanged, and written under the current config.
The owner makes the commit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from statistics import NormalDist

from . import CELL, GROSS_HEADER, REPO, config_digest, load_config, sha256_file
from .registry import PREREG_JSON, PREREG_MD, REGISTRY, StageOrderError, guard, path, read_json


class PreregNotCommitted(RuntimeError):
    pass


class FrozenChanged(RuntimeError):
    pass


FACTS = [
    'HAR-X and HAR are gold_bbnr\'s frozen OLS fits on every usable day up to 2026-11-20 (the range is in the '
    'model section; the hourly dollar index starts in 2017). gold-bb-nyrange was scored 2026-11-23: the historical '
    'cell gold_harx_vs_har was KEEP at 0.847 x HAR\'s MSE on 2 211 walk-forward days 2018-01-02 .. 2026-11-20.',
    'The frozen fit used all history, so the forward effect may be smaller than the walk-forward one. That is '
    'what this test measures.',
    'LightGBM and the MLP were DROP against HAR-X historically and are not forecast live.',
    'gold_bbnr\'s Bollinger cells were all DROP; nothing here concerns direction or a trade.',
]


def sources() -> dict:
    """gold_bbnr's frozen OLS file, its manifest and its range registry (read-only)."""
    from ..gold_bbnr import load_config as gb_config
    from ..gold_bbnr import repo_path as gb_path
    gb = gb_config()
    return {'ols': os.path.join(gb_path(gb['paths']['artifacts']), 'ols.json'),
            'manifest': os.path.join(gb_path(gb['paths']['results']), 'frozen_manifest.csv'),
            'registry': os.path.join(gb_path(gb['paths']['results']), 'gold_nyvol_hypothesis_log.csv'),
            'gold_bbnr_config_digest': config_digest(gb)}


def _manifest_sha(p: str, name: str = 'ols.json') -> str:
    with open(p, newline='', encoding='utf-8') as fh:
        return next(r['sha256'] for r in csv.DictReader(fh) if r['file'] == name)


def historical(p: str) -> dict:
    with open(p, newline='', encoding='utf-8') as fh:
        r = next(r for r in csv.DictReader(fh) if r['hypothesis'] == 'gold_harx_vs_har')
    return {'mean_diff': float(r['statistic']), 'boot_se': float(r['boot_se']), 'n_days': int(r['n_dates']),
            'mse_harx': float(r['model_mean']), 'mse_har': float(r['reference_mean']), 'verdict': r['verdict']}


def power(hist: dict, n: int, alpha: float) -> dict:
    """One-sided power of the forward cell at n days if the true mean difference equals the historical one.
    The block SE is scaled by sqrt(n_hist / n)."""
    se = hist['boot_se'] * (hist['n_days'] / n) ** 0.5
    z = NormalDist()
    p = z.cdf(abs(hist['mean_diff']) / se - z.inv_cdf(1 - alpha))
    return {'n_days': n, 'se_at_n': se, 'power': p,
            'calculation': f"SE_n = {hist['boot_se']:.6f} x sqrt({hist['n_days']} / {n}) = {se:.5f}; "
                           f"power = Phi({abs(hist['mean_diff']):.5f} / {se:.5f} - {z.inv_cdf(1 - alpha):.4f}) "
                           f"= {p:.3f}"}


def build(cfg: dict, src: dict | None = None) -> dict:
    src = src or sources()
    if not os.path.exists(src['ols']):
        raise StageOrderError(f"gold_bbnr's frozen {src['ols']} is missing; it cannot be copied")
    sha = sha256_file(src['ols'])
    if sha != _manifest_sha(src['manifest']):
        raise FrozenChanged("gold_bbnr's ols.json differs from its frozen manifest; restore it, never refit")
    with open(src['ols'], encoding='utf-8') as fh:
        ols = json.load(fh)
    if len(ols['harx']) != len(ols['columns']) + 1 or len(ols['har']) != len(ols['har_columns']) + 1:
        raise FrozenChanged('ols.json coefficient counts do not match its columns')
    hist = historical(src['registry'])
    sc, fw = cfg['scoring'], cfg['forward']
    return {
        'change': 'gold-nyrange-forward', 'config_digest': config_digest(cfg), 'cost': GROSS_HEADER,
        'cell': CELL, 'registry': f"{cfg['paths']['results']}/{REGISTRY}",
        'hypothesis': 'mean over forward days of (HAR-X squared error - HAR squared error) < 0',
        'alpha': sc['alpha'], 'family_size': sc['family_size'], 'block_days': sc['block_days'],
        'resamples': sc['resamples'], 'seed': sc['seed'], 'one_sided': True,
        'verdict_rule': 'KEEP if the bootstrap upper (1 - alpha) bound of the mean difference is below 0, else DROP',
        'horizon': {'min_days': fw['min_days'], 'deadline': fw['deadline'],
                    'rule': f"scored once on the first {fw['min_days']} forward days, as soon as they exist; with "
                            f"fewer on {fw['deadline']}, recorded UNDERPOWERED — NO DECISION. The horizon is never "
                            'extended.'},
        'forward_day': 'a settled ok day whose label date is after the commit date of PRE_REGISTRATION.json on the '
                       'Berlin label clock',
        'target': 'y = log((max high - min low of the gold bars labelled 15:00-21:00 Berlin) / close of the 14:00 bar), '
                  'bid prices; eligibility and target from gold_bbnr.ny_target unchanged',
        'contract_rule': f"the {cfg['mt5']['contract_prefix']} contract with the nearest expiry at least "
                         f"{cfg['mt5']['roll_margin_trading_days']} trading days after the day, lag-0 checked against "
                         f"EURUSD (lags {cfg['lag_check']['lags']}); no qualifying contract = no dollar data, the day "
                         'never counts and the horizon is not extended',
        'model': {'columns': ols['columns'], 'har_columns': ols['har_columns'], 'coef_harx': ols['harx'],
                  'coef_har': ols['har'], 'ols_sha256': sha, 'fit_first_day': ols['first_day'],
                  'fit_last_day': ols['last_day'], 'fit_n_days': ols['n_days'],
                  'gold_bbnr_config_digest': ols['config_digest']},
        'historical': hist, 'power': power(hist, int(fw['min_days']), float(sc['alpha'])),
        'drop_is_not_failure': 'The test is weak by the owner\'s choice of horizon. A DROP is not evidence that the '
                               'model fails; only a KEEP is a finding.',
        'facts': FACTS,
    }


def write(cfg: dict | None = None, src: dict | None = None) -> tuple:
    cfg = cfg or load_config()
    guard(cfg)
    rec = build(cfg, src)
    os.makedirs(os.path.dirname(path(cfg, PREREG_JSON)), exist_ok=True)
    with open(path(cfg, PREREG_JSON), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    m, p, h = rec['model'], rec['power'], rec['historical']
    coef = ['| term | HAR-X | HAR |', '|---|---:|---:|', f"| intercept | {m['coef_harx'][0]:.10g} | {m['coef_har'][0]:.10g} |"]
    for i, c in enumerate(m['columns']):
        har = m['coef_har'][1 + m['har_columns'].index(c)] if c in m['har_columns'] else None
        coef.append(f"| {c} | {m['coef_harx'][1 + i]:.10g} | {'' if har is None else f'{har:.10g}'} |")
    L = ['# gold-nyrange-forward — pre-registration', '', f"> **{GROSS_HEADER}**", '',
         'Written before any forward day. Commit this file and PRE_REGISTRATION.json unchanged; every pass and the '
         'scoring refuse otherwise.', '', f"Config digest `{rec['config_digest']}`.", '',
         '## The cell', '',
         f"- `{CELL}` in `{rec['registry']}`, alpha {rec['alpha']} (family size {rec['family_size']}), one-sided.",
         f"- Statistic: {rec['hypothesis']}.",
         f"- Test: circular {rec['block_days']}-day block bootstrap, {rec['resamples']} resamples, seed {rec['seed']}. "
         f"{rec['verdict_rule']}.",
         f"- Horizon: {rec['horizon']['rule']}",
         f"- A forward day is {rec['forward_day']}.",
         f"- Target: {rec['target']}.",
         f"- Dollar index: {rec['contract_rule']}.", '',
         '## Expected power — stated before any forward day', '',
         f"- Historical (gold_bbnr, {h['n_days']} days): mean difference {h['mean_diff']:.5f}, block SE "
         f"{h['boot_se']:.6f}, MSE {h['mse_harx']:.4f} vs {h['mse_har']:.4f}.",
         f"- At {p['n_days']} days: {p['calculation']}. **About {p['power']:.0%}.**",
         f"- {rec['drop_is_not_failure']}", '',
         '## The frozen model (copied from gold_bbnr, never refitted)', '',
         f"- Source: research_models/gold_bbnr/ols.json, SHA-256 `{m['ols_sha256']}`, fit on {m['fit_n_days']} days "
         f"{m['fit_first_day']} .. {m['fit_last_day']}.",
         '- Forecasts use these copied values. A pass refuses if ols.json exists and its digest differs.', ''] + coef + \
        ['', '## Facts recorded', ''] + [f'- {f}' for f in FACTS] + ['']
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


def coefficients(cfg: dict, ols_path: str | None = None) -> dict:
    """The pinned model from the pre-registration JSON. Refuses if gold_bbnr's ols.json exists and differs."""
    rec = read_json(cfg, PREREG_JSON)
    if rec is None:
        raise PreregNotCommitted('no pre-registration: write it with --write and commit it')
    m = rec['model']
    ols_path = ols_path or sources()['ols']
    if os.path.exists(ols_path) and sha256_file(ols_path) != m['ols_sha256']:
        raise FrozenChanged("gold_bbnr's ols.json differs from the digest in the pre-registration; restore it")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold-nyrange-forward pre-registration')
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
