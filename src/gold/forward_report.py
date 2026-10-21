"""Gold forward registration and scoring (tasks 9.1, 9.2; spec "Forward
pre-registration precedes forward scoring", "Forward verdicts measure skill,
not profit").

    python -m src.gold.forward_report --register   write results/gold/forward/PRE_REGISTRATION.{md,json}
                                                   from the dry run's measured coverage (once)
    python -m src.gold.forward_logger --report     score the REGISTERED rows

Scoring refuses until the forward registration is committed and unmodified.
Only `phase = registered`, eligible, settled rows are scored; dry-run rows never
are. Before a cell reaches its registered n and the minimum window, its figures
are labelled "interim, not adjudicating". Verdicts measure statistical skill
GROSS of cost -- the report never contains a profit, pip, Sharpe or ledger figure.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.forecast_eval import prereg as FP
from src.forecast_eval import stats as ST

from . import REPO, load_config, repo_path
from . import prereg as P

CAP_YEARS = 3.0
MIN_WINDOW_MONTHS = 6
KEEP = 'KEEP — statistical skill, gross of cost; not a trading result'
DROP = 'DROP — no statistical skill shown'
INTERIM = FP.INTERIM


def _paths(cfg):
    fwd = repo_path(cfg['paths']['forward'])
    return {'md': repo_path(cfg['paths']['forward_prereg_md']),
            'json': os.path.join(fwd, 'PRE_REGISTRATION.json'),
            'predictions': os.path.join(fwd, 'predictions.csv'),
            'settlements': os.path.join(fwd, 'settlements.csv'),
            'verdicts': os.path.join(fwd, 'forward_verdicts.json'),
            'report': os.path.join(fwd, 'forward_report.md')}


# ── registration (9.1) ─────────────────────────────────────────────────────

def build_registration(coverage: dict, eligible_per_year: dict, train_majority: dict,
                       manifest_digests: dict, cfg: dict) -> dict:
    """coverage: measured share of bar closes logged per cadence (dry run);
    eligible_per_year: scoreable rows a year per cadence at full coverage;
    train_majority: {cell: 0|1} of the final fit; manifest_digests: {cell: digest}."""
    pw = cfg['power']
    cells = [(c, cad, 1) for c, cad in P.DIRECTION_CELLS]
    cov = {cad: coverage[cad] * eligible_per_year[cad] / FP.BARS_PER_YEAR[cad] for cad in coverage}
    admitted, table = [], []
    size = len(cells)
    while True:                                       # fixed point at gold's own trade rate
        alpha = pw['family_alpha'] / size
        need = FP.n_required(alpha, pw['edge'], pw['power'])
        yrs = {c: need / (eligible_per_year[cad] * coverage[cad]) for c, cad, _ in cells}
        now = [c for c, _, _ in cells if yrs[c] <= CAP_YEARS]
        if not now or len(now) == size:
            admitted = now
            break
        size = len(now)
    alpha_dir = pw['family_alpha'] / max(len(admitted), 1)
    need = FP.n_required(alpha_dir, pw['edge'], pw['power'])
    for c, cad, _ in cells:
        per_year = eligible_per_year[cad] * coverage[cad]
        table.append({'cell': c, 'cadence': cad, 'n_required': round(need, 1), 'n_per_year': round(per_year, 1),
                      'years': round(need / per_year, 2),
                      'status': 'ADMITTED' if c in admitted else FP.UNDERPOWERED})
    return {
        'change': 'gold-forecasting', 'kind': 'forward', 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'cost_basis': 'GROSS of cost -- statistical skill only, never a trading result',
        'coverage': coverage, 'eligible_per_year': eligible_per_year, 'adjusted_coverage_vs_fx_rate': cov,
        'cap_years': CAP_YEARS, 'min_window_months': MIN_WINDOW_MONTHS,
        'frozen_artifacts': manifest_digests, 'refit': 'none during the window; a refit needs a new registration',
        'families': {
            'direction': {'admitted': admitted, 'alpha': alpha_dir, 'n_required': need, 'power_table': table,
                          'train_majority': train_majority,
                          'verdict': ('at n_required and at least the minimum window: KEEP iff the two-sided '
                                      '(1 - alpha) paired block-bootstrap CIs of accuracy minus the training '
                                      'majority AND minus 0.5 both have lower bound > 0, else DROP')},
            'return': {'cells': [c for c, _ in P.RETURN_CELLS], 'alpha': pw['family_alpha'] / len(P.RETURN_CELLS),
                       'verdict': 'after the minimum window: KEEP iff one-sided Clark-West p < alpha, else DROP'},
            'volatility': {'comparisons': [{'cell': c, 'cadence': cad, 'baseline': b} for c, cad, b in P.VOLATILITY_CELLS],
                           'alpha': pw['family_alpha'] / len(P.VOLATILITY_CELLS),
                           'verdict': ('after the minimum window: KEEP iff the two-sided (1 - alpha) block-bootstrap '
                                       'CI of MAE_baseline - MAE_model has lower bound > 0, else DROP')},
        },
        'ci_method': {'block_length': {'D1': ST.block_length(1, 'D1'), 'H1': ST.block_length(1, 'H1')},
                      'n_boot': cfg['bootstrap']['n_boot'], 'seed': ST.SEED},
        'scored_rows': 'phase = registered, settled, eligible; dry-run rows are never scored',
    }


def render_registration(reg: dict) -> str:
    d = reg['families']['direction']
    lines = ['# Gold forward pre-registration', '',
             'Committed before any forward row is scored. Never edited after its commit; corrections go to '
             '`results/gold/RESULTS.md`.', '',
             f'- Created: {reg["created_at"]}', f'- **Cost basis: {reg["cost_basis"]}**',
             f'- Window: at least {reg["min_window_months"]} months; cap {reg["cap_years"]} years',
             f'- Refit: {reg["refit"]}', f'- Scored rows: {reg["scored_rows"]}', '',
             '## Frozen artifacts', '', '| cell | digest |', '|---|---|']
    lines += [f'| {c} | `{g}` |' for c, g in reg['frozen_artifacts'].items()]
    lines += ['', f'## Direction (alpha = {d["alpha"]:.5f})', '',
              '| cell | cadence | required n | n a year | years | status |', '|---|---|---|---|---|---|']
    lines += [f'| {r["cell"]} | {r["cadence"]} | {r["n_required"]:.0f} | {r["n_per_year"]:.0f} | {r["years"]} | {r["status"]} |'
              for r in d['power_table']]
    lines += ['', f'Verdict: {d["verdict"]}', '',
              f'## Return (alpha = {reg["families"]["return"]["alpha"]:.5f})', '',
              f'Verdict: {reg["families"]["return"]["verdict"]}', '',
              f'## Volatility (alpha = {reg["families"]["volatility"]["alpha"]:.5f})', '']
    lines += [f'- `{c["cell"]}` vs `{c["baseline"]}`' for c in reg['families']['volatility']['comparisons']]
    lines += ['', f'Verdict: {reg["families"]["volatility"]["verdict"]}', '',
              f'Measured dry-run coverage: {json.dumps(reg["coverage"])}', '']
    return '\n'.join(lines)


def scoring_allowed(cfg) -> bool:
    p = _paths(cfg)
    return all(FP.committed_and_clean(p[k], REPO) is not None for k in ('md', 'json'))


# ── scoring (9.2) ──────────────────────────────────────────────────────────

def scored_rows(cfg) -> pd.DataFrame:
    p = _paths(cfg)
    if not (os.path.exists(p['predictions']) and os.path.exists(p['settlements'])):
        return pd.DataFrame()
    pr = pd.read_csv(p['predictions'])
    st = pd.read_csv(p['settlements'])
    df = pr.merge(st[['key', 'realised_dir', 'realised_ret_pct', 'realised_vol_pct', 'eligible']], on='key')
    return df[(df['phase'] == 'registered') & df['eligible'].astype(str).eq('True')]


def _window_months(df) -> float:
    if df.empty:
        return 0.0
    t = pd.to_datetime(df['as_of_bar'])
    return (t.max() - t.min()).days / 30.44


def score(cfg=None) -> dict:
    cfg = cfg or load_config()
    if not scoring_allowed(cfg):
        raise PermissionError('no committed, unmodified gold forward pre-registration: forward scoring refuses')
    with open(_paths(cfg)['json'], encoding='utf-8') as fh:
        reg = json.load(fh)
    df = scored_rows(cfg)
    n_boot = reg['ci_method']['n_boot']
    out = []
    fam = reg['families']
    for cell, cad in P.DIRECTION_CELLS:
        d = df[(df['cell'] == cell) & df['realised_dir'].notna() & df['p_up'].notna()]
        n, months = len(d), _window_months(d)
        row = {'cell': cell, 'family': 'direction', 'n': n, 'window_months': round(months, 1)}
        if cell not in fam['direction']['admitted']:
            row['verdict'] = FP.UNDERPOWERED
        if n:
            correct = ((d['p_up'] > 0.5).astype(float) == d['realised_dir']).astype(float).to_numpy()
            maj = fam['direction']['train_majority'].get(cell, 1)
            cm = (d['realised_dir'] == maj).astype(float).to_numpy()
            bl, a = reg['ci_method']['block_length'][cad], fam['direction']['alpha']
            lo_m, hi_m, dm = ST.block_bootstrap_delta(correct, cm, bl, a, n_boot)
            lo_c, hi_c, dc = ST.block_bootstrap_delta(correct, np.full_like(correct, 0.5), bl, a, n_boot)
            row.update(accuracy=float(correct.mean()), vs_majority=[lo_m, hi_m], vs_coin=[lo_c, hi_c])
            if 'verdict' not in row:
                ready = n >= fam['direction']['n_required'] and months >= reg['min_window_months']
                row['verdict'] = (KEEP if (lo_m > 0 and lo_c > 0) else DROP) if ready else INTERIM
        row.setdefault('verdict', INTERIM)
        out.append(row)
    for cell, cad in P.RETURN_CELLS:
        d = df[(df['cell'] == cell) & df['ret_pct'].notna()]
        months = _window_months(d)
        row = {'cell': cell, 'family': 'return', 'n': len(d), 'window_months': round(months, 1)}
        if len(d) > 2:
            cw = ST.clark_west(d['realised_ret_pct'].to_numpy(), d['ret_pct'].to_numpy(), None, 1)
            row['clark_west_p'] = cw['p_value']
            row['verdict'] = ((KEEP if cw['p_value'] < fam['return']['alpha'] else DROP)
                              if months >= reg['min_window_months'] else INTERIM)
        row.setdefault('verdict', INTERIM)
        out.append(row)
    for comp in fam['volatility']['comparisons']:
        m = df[df['cell'] == comp['cell']][['as_of_bar', 'vol_pct', 'realised_vol_pct']]
        b = df[df['cell'] == comp['baseline']][['as_of_bar', 'vol_pct']].rename(columns={'vol_pct': 'base'})
        j = m.merge(b, on='as_of_bar').dropna()
        months = _window_months(j)
        row = {'cell': comp['cell'], 'family': 'volatility', 'n': len(j), 'window_months': round(months, 1)}
        if len(j) > 2:
            em = (j['realised_vol_pct'] - j['vol_pct']).abs().to_numpy()
            eb = (j['realised_vol_pct'] - j['base']).abs().to_numpy()
            lo, hi, dlt = ST.block_bootstrap_delta(eb, em, reg['ci_method']['block_length'][comp['cadence']],
                                                   fam['volatility']['alpha'], n_boot)
            row.update(mae_model=float(em.mean()), mae_baseline=float(eb.mean()), ci=[lo, hi])
            row['verdict'] = ((KEEP if lo > 0 else DROP) if months >= reg['min_window_months'] else INTERIM)
        row.setdefault('verdict', INTERIM)
        out.append(row)
    return {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'cost_basis': reg['cost_basis'], 'n_scored_rows': int(len(df)), 'verdicts': out}


def render_report(res: dict) -> str:
    lines = ['# Gold forward report', '', f'**{res["cost_basis"]}.** Registered, settled, eligible rows only.', '',
             f'- Generated: {res["generated_at"]}', f'- Scored rows: {res["n_scored_rows"]}', '',
             '| family | cell | n | window (months) | figure | verdict |', '|---|---|---|---|---|---|']
    for r in res['verdicts']:
        fig = ''
        if 'accuracy' in r:
            fig = f'accuracy {r["accuracy"]:.4f}'
        elif 'clark_west_p' in r:
            fig = f'Clark-West p {r["clark_west_p"]:.4f}'
        elif 'mae_model' in r:
            fig = f'MAE {r["mae_model"]:.5f} vs {r["mae_baseline"]:.5f}'
        lines.append(f'| {r["family"]} | {r["cell"]} | {r["n"]} | {r["window_months"]} | {fig} | {r["verdict"]} |')
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold forward registration / report')
    ap.add_argument('--register', action='store_true')
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.register:
        raise SystemExit('write the forward registration after the dry run with build_registration() '
                         '(task 9.1 needs the measured coverage from results/gold/forward/dry_run_coverage.md)')
    res = score(cfg)
    p = _paths(cfg)
    with open(p['verdicts'], 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, default=float)
    with open(p['report'], 'w', encoding='utf-8') as fh:
        fh.write(render_report(res))
    print(render_report(res))


if __name__ == '__main__':
    main()
