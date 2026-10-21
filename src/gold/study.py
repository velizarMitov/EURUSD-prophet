"""The gold study: validation, one-shot test, final refit (design D7, D8; spec
"Validation is the only arbiter", "The test block is scored exactly once").

    python -m src.gold.study --stage validation   fit [0:70], score [70:80]; append registries
    python -m src.gold.study --stage test         fit [0:80], score [80:100] once; test report
    python -m src.gold.study --stage final        fit [0:100]; frozen artifacts + manifest

One code path for all three: the stage changes only the split and the outputs.
Every run checks, in order: the registration is committed and unmodified, the
config digest matches, the cache still holds the pinned rows exactly, the bar
clock passes. Per-row predictions of every scored stage are saved next to the
registries, so any verdict can be recomputed without retraining.

Every number is GROSS of cost: spread is used nowhere (owner decision).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.forecast_eval import overfit as OF
from src.forecast_eval import prereg as FP
from src.forecast_eval import stats as ST

from . import REPO, load_config, repo_path
from . import challengers as C
from . import clock as K
from . import data as D
from . import prereg as P
from . import splits as S
from . import targets as G

ARBITER = {'validation': 'validation[70:80]', 'test': 'test[80:100] one-shot'}
CELLS_BY_CADENCE = {'D1': ('d1_gbm', 'd1_lstm', 'd1_garch'), 'H1': ('h1_gbm', 'h1_lstm', 'h1_garch_hod')}
HIGH_SPREAD_HOURS = range(0, 6)          # labels 00:00-05:00, the hours after the daily break
MANIFEST_FIELDS = ['cell', 'cadence', 'path', 'sha256', 'bytes', 'config_hash', 'fit_rows', 'last_bar', 'written_at']


class StageRefused(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _log(msg):
    print(f'[{datetime.now():%H:%M:%S}] {msg}', flush=True)


# ── guards ──────────────────────────────────────────────────────────────────

def _committed(path: str) -> bool:
    return FP.committed_and_clean(path, REPO) is not None


def check_stage_allowed(stage: str, cfg: dict) -> dict:
    """The registration, or StageRefused naming the missing precondition."""
    if not P.scoring_allowed(cfg):
        raise StageRefused('no committed, unmodified gold pre-registration: scoring refuses to run')
    reg = P.registration(cfg)
    P.assert_config_unchanged(reg)
    paths = cfg['paths']
    if stage == 'test':
        if os.path.exists(repo_path(paths['test_report'])):
            raise StageRefused('a one-shot test report already exists: the test block is never scored twice')
        for path in list(paths['registries'].values()) + [paths['validation_report']]:
            full = repo_path(path)
            if not os.path.exists(full) or not _committed(full):
                raise StageRefused(f'{path} is not committed: every validation verdict must be '
                                   'committed before the test block is opened')
        if not any(sum(1 for _ in open(repo_path(p), encoding='utf-8')) > 1 for p in paths['registries'].values()):
            raise StageRefused('the validation registries are empty: run --stage validation first')
    if stage == 'final' and not _committed(repo_path(paths['test_report'])):
        raise StageRefused('the one-shot test report is not committed: the final refit follows the test')
    return reg


def load_pinned(cfg: dict, reg: dict) -> dict:
    frames = {}
    for cad in ('D1', 'H1'):
        df, _ = D.load(cad, allow_fetch=False, cfg=cfg)
        frames[cad] = P.pinned_frame(df, reg, cad)
    K.verify(frames['H1'].index, cfg['clock']['weekly_open_threshold'])
    return frames


# ── scoring ─────────────────────────────────────────────────────────────────

def _majority(dir_y: np.ndarray) -> int:
    y = dir_y[~np.isnan(dir_y)]
    return int(y.mean() >= 0.5)


def score_direction(cell, cad, pred, tg, rows, fit_rows, alpha, status, n_boot):
    y = tg['dir'][rows]
    p = pred['p_up'][rows]
    ok = ~np.isnan(p)
    y, p, idx = y[ok], p[ok], rows[ok]
    correct = ((p > 0.5).astype(float) == y).astype(float)
    maj = _majority(tg['dir'][fit_rows])
    correct_maj = (y == maj).astype(float)
    bl = ST.block_length(1, cad)
    lo_m, hi_m, d_m = ST.block_bootstrap_delta(correct, correct_maj, bl, alpha, n_boot)
    lo_c, hi_c, d_c = ST.block_bootstrap_delta(correct, np.full_like(correct, 0.5), bl, alpha, n_boot)
    pt = ST.pesaran_timmermann(y, (p > 0.5).astype(float))
    if status != 'ADMITTED':
        verdict, cleared = FP.UNDERPOWERED, ''
    else:
        cleared = bool(lo_m > 0 and lo_c > 0)
        verdict = ('CLEARED (beats the training majority and a coin: both CIs exclude 0)' if cleared
                   else 'DROP (a CI includes 0)')
    return {'cell': cell, 'family': 'direction', 'hypothesis': f'gold_{cell}_direction_vs_majority_and_coin',
            'n_rows': int(len(y)), 'metric_model': float(correct.mean()), 'metric_baseline': float(correct_maj.mean()),
            'point_delta': d_m, 'ci_low': lo_m, 'ci_high': hi_m, 'p_value': pt['p_value'], 'alpha_bonferroni': alpha,
            'cleared_bar': cleared, 'verdict': verdict,
            'notes': (f'accuracy {correct.mean():.4f}; train-majority class {maj}; vs coin delta {d_c:+.4f} '
                      f'CI [{lo_c:+.4f}, {hi_c:+.4f}]; PT stat {pt["stat"]:.3f}; block {bl}; gross of cost'),
            '_rows': idx, '_correct': correct}


def score_return(cell, cad, pred, tg, rows, alpha):
    y, f = tg['ret'][rows], pred['ret_pct'][rows]
    ok = ~np.isnan(f)
    y, f = y[ok], f[ok]
    cw = ST.clark_west(y, f, None, 1)
    mse_m, mse_0 = float(np.mean((y - f) ** 2)), float(np.mean(y ** 2))
    cleared = bool(np.isfinite(cw['p_value']) and cw['p_value'] < alpha)
    return {'cell': cell, 'family': 'return', 'hypothesis': f'gold_{cell}_return_vs_zero',
            'n_rows': int(len(y)), 'metric_model': mse_m, 'metric_baseline': mse_0,
            'point_delta': mse_0 - mse_m, 'ci_low': '', 'ci_high': '', 'p_value': cw['p_value'],
            'alpha_bonferroni': alpha, 'cleared_bar': cleared,
            'verdict': 'CLEARED (Clark-West p < alpha)' if cleared else 'DROP (Clark-West p >= alpha)',
            'notes': (f'MSPE model {mse_m:.6f} vs zero {mse_0:.6f} (%^2); CW stat {cw["stat"]:.3f}; '
                      f'mean |forecast| {np.mean(np.abs(f)):.5f} %')}


def score_volatility(cell, cad, base, pred_m, pred_b, tg, rows, alpha, n_boot):
    y, m, b = tg['vol'][rows], pred_m['vol_pct'][rows], pred_b['vol_pct'][rows]
    ok = ~np.isnan(m) & ~np.isnan(b)
    y, m, b = y[ok], m[ok], b[ok]
    err_m, err_b = np.abs(y - m), np.abs(y - b)
    bl = ST.block_length(1, cad)
    lo, hi, d = ST.block_bootstrap_delta(err_b, err_m, bl, alpha, n_boot)
    cleared = bool(lo > 0)
    return {'cell': cell, 'family': 'volatility', 'hypothesis': f'gold_{cell}_volatility_vs_{base}',
            'n_rows': int(len(y)), 'metric_model': float(err_m.mean()), 'metric_baseline': float(err_b.mean()),
            'point_delta': d, 'ci_low': lo, 'ci_high': hi, 'p_value': '', 'alpha_bonferroni': alpha,
            'cleared_bar': cleared,
            'verdict': (f'CLEARED (beats {base}: CI of MAE_baseline - MAE_model excludes 0)' if cleared
                        else f'DROP (CI includes 0 or favours {base})'),
            'notes': f'MAE model {err_m.mean():.5f} % vs {base} {err_b.mean():.5f} %; block {bl}'}


def hour_concentration(index, rows, correct) -> dict:
    """Descriptive: where the excess accuracy sits by as-of label hour."""
    hours = pd.DatetimeIndex(index)[rows].hour
    df = pd.DataFrame({'hour': hours, 'excess': correct - 0.5, 'correct': correct})
    total = float(df['excess'].sum())
    early = float(df.loc[df['hour'].isin(HIGH_SPREAD_HOURS), 'excess'].sum())
    by_hour = df.groupby('hour')['correct'].agg(['mean', 'size'])
    share = early / total if total > 0 else float('nan')
    return {'excess_correct_total': total, 'excess_from_00_06': early, 'share_00_06': share,
            'flag': bool(total > 0 and share > 0.5),
            'by_hour': {int(h): {'accuracy': round(float(r['mean']), 4), 'n': int(r['size'])}
                        for h, r in by_hour.iterrows()}}


# ── the run ─────────────────────────────────────────────────────────────────

def fit_and_predict(cad, df, stage, cfg, trial_rows, failures):
    inp = C.build_inputs(df, cad)
    tg = G.build_targets(df['close'])
    pos = S.stage_positions(len(df), stage, cfg)
    preds, models = {}, {}
    for cell in CELLS_BY_CADENCE[cad]:
        t0 = time.time()
        try:
            m = C.make(cell, cfg)
            m.fit(inp, pos['fit'], tg, seed=cfg['seeds'][0])
            preds[cell] = m.predict(inp) if pos['score'].size else None
            models[cell] = m
            if hasattr(m, 'trial_rows'):
                trial_rows.extend(m.trial_rows(f'gold-{stage}'))
            _log(f'{cell}: fitted on {len(pos["fit"])} rows in {time.time() - t0:.0f} s')
        except Exception as e:                                   # recorded, never hidden
            failures.append({'cell': cell, 'stage': stage, 'error': f'{type(e).__name__}: {e}'})
            _log(f'{cell}: FAILED {type(e).__name__}: {e}')
        finally:
            try:
                import keras
                keras.backend.clear_session()
            except Exception:
                pass
    return inp, tg, pos, preds, models


def score_stage(stage, frames, reg, cfg):
    fam = reg['families']
    status = {r['cell']: r['status'] for r in fam['direction']['power_table']}
    n_boot = cfg['bootstrap']['n_boot']
    results, failures, trial_rows, extras, pred_frames = [], [], [], {}, {}
    for cad in ('D1', 'H1'):
        df = frames[cad]
        inp, tg, pos, preds, _ = fit_and_predict(cad, df, stage, cfg, trial_rows, failures)
        fit_rows = C.usable(inp, pos['fit'], tg, ('dir',))
        out = pd.DataFrame(index=df.index[pos['score']])
        out['close'] = df['close'].to_numpy()[pos['score']]
        out['eligible'] = inp.extra['eligible'][pos['score']]
        for k in ('dir', 'ret', 'vol'):
            out[f'y_{k}'] = tg[k][pos['score']]
        for cell, pr in preds.items():
            for k, v in pr.items():
                out[f'{cell}__{k}'] = v[pos['score']]
        pred_frames[cad] = out
        corr = {}
        for cell, c_cad in P.DIRECTION_CELLS:
            if c_cad == cad and cell in preds:
                rows = C.usable(inp, pos['score'], tg, ('dir',))
                r = score_direction(cell, cad, preds[cell], tg, rows, fit_rows, fam['direction']['alpha'],
                                    status[cell], n_boot)
                corr[cell] = r
                results.append(r)
        for cell, c_cad in P.RETURN_CELLS:
            if c_cad == cad and cell in preds:
                rows = C.usable(inp, pos['score'], tg, ('ret',))
                results.append(score_return(cell, cad, preds[cell], tg, rows, fam['return']['alpha']))
        for cell, c_cad, base in P.VOLATILITY_CELLS:
            if c_cad == cad and cell in preds and base in preds:
                rows = C.usable(inp, pos['score'], tg, ('vol',))
                results.append(score_volatility(cell, cad, base, preds[cell], preds[base], tg, rows,
                                                fam['volatility']['alpha'], n_boot))
        extras[cad] = descriptive(cad, df, inp, tg, pos, preds, corr, cfg)
    for cell in sorted({f['cell'] for f in failures}):
        err = next(f['error'] for f in failures if f['cell'] == cell)
        for fam_name in families_of(cell):
            results.append({'cell': cell, 'family': fam_name, 'hypothesis': f'gold_{cell}_{fam_name}',
                            'verdict': 'FAILED — unspent (family size and alpha unchanged)', 'notes': err})
    return results, failures, trial_rows, extras, pred_frames


def families_of(cell: str) -> list[str]:
    """Every registered family a cell takes part in (a failed baseline voids its comparison)."""
    out = []
    if cell in {c for c, _ in P.DIRECTION_CELLS}:
        out.append('direction')
    if cell in {c for c, _ in P.RETURN_CELLS}:
        out.append('return')
    if any(cell in (c, b) for c, _, b in P.VOLATILITY_CELLS):
        out.append('volatility')
    return out


def descriptive(cad, df, inp, tg, pos, preds, corr, cfg):
    out = {}
    if cad == 'H1':
        out['hour_concentration'] = {cell: hour_concentration(df.index, r['_rows'], r['_correct'])
                                     for cell, r in corr.items()}
    rows = C.usable(inp, pos['score'], tg, ('dir',))
    sign_ret = {}
    for cell in corr:
        p = preds[cell]['p_up'][rows]
        ok = ~np.isnan(p)
        sign_ret[cell] = np.where(p > 0.5, 1.0, -1.0) * tg['ret'][rows]
        sign_ret[cell][~ok] = np.nan
    if len(sign_ret) >= 2:
        M = np.column_stack([np.nan_to_num(v) for v in sign_ret.values()])
        out['pbo'] = OF.pbo_cscv(M, list(sign_ret))
    return out


def deflated_sharpe_from_predictions(stage: str, cfg: dict) -> dict:
    """Descriptive DSR of each direction cell's GROSS sign strategy, from the
    saved per-row predictions, charged for every trial in the gold trial log.
    Must run AFTER this stage's grid trials are appended: computed earlier, the
    2026-10-21 validation report charged N = 1 instead of 16 (RESULTS.md)."""
    trial_log = repo_path(cfg['paths']['trial_log'])
    n_tr, var = max(OF.n_trials(trial_log), 1), OF.trial_sharpe_variance(trial_log)
    out = {}
    for cad in ('D1', 'H1'):
        path = os.path.join(repo_path(cfg['paths']['results']), f'{stage}_predictions_{cad.lower()}.csv')
        if not os.path.exists(path):
            continue
        pf = pd.read_csv(path)
        for cell, c_cad in P.DIRECTION_CELLS:
            col = f'{cell}__p_up'
            if c_cad != cad or col not in pf:
                continue
            m = pf['eligible'].astype(bool) & pf['y_dir'].notna() & pf[col].notna()
            r = np.where(pf.loc[m, col] > 0.5, 1.0, -1.0) * pf.loc[m, 'y_ret'].to_numpy()
            out[cell] = OF.deflated_sharpe(r, n_tr, var)
    return out


def dsr_lines(dsr: dict) -> list[str]:
    def f(x, d=4):
        return '' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
    return [f'- {cell}: gross sign-strategy per-trade Sharpe {f(d["sharpe"])}, SR0 {f(d["sr0"])} '
            f'over {d["n_trials"]} trials, DSR {f(d["dsr"], 3)}' for cell, d in dsr.items()]


def append_registry(path: str, rows: list[dict]) -> None:
    n = sum(1 for _ in open(path, encoding='utf-8')) - 1
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=P.REGISTRY_FIELDS, extrasaction='ignore')
        for i, r in enumerate(rows):
            w.writerow({**r, 'n': n + i + 1, 'date': datetime.now(timezone.utc).date().isoformat(),
                        'arbiter': ARBITER['validation']})


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def render_report(stage, results, failures, extras, meta) -> str:
    lines = [f'# Gold {stage} report', '',
             f'Arbiter: `{ARBITER.get(stage, stage)}`. **Gross of cost** -- statistical skill only, never a '
             'trading result. Registered protocol: `results/gold/PRE_REGISTRATION.md`.', '',
             f'- Run: {meta["finished_at"]} ({meta["elapsed_s"]:.0f} s)', '',
             '| family | cell | n | model | baseline | delta | CI | p | alpha | verdict |',
             '|---|---|---|---|---|---|---|---|---|---|']

    def f(x, d=4):
        return '' if x in ('', None) or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'
    for r in results:
        ci = f'[{f(r.get("ci_low"))}, {f(r.get("ci_high"))}]' if r.get('ci_low') not in ('', None) else ''
        lines.append(f'| {r["family"]} | {r["cell"]} | {r.get("n_rows", "")} | {f(r.get("metric_model"), 5)} | '
                     f'{f(r.get("metric_baseline"), 5)} | {f(r.get("point_delta"), 5)} | {ci} | '
                     f'{f(r.get("p_value"))} | {f(r.get("alpha_bonferroni"), 5)} | {r["verdict"]} |')
    lines += ['', '## Descriptive only (never a verdict)', '']
    for cad, ex in extras.items():
        for cell, hc in ex.get('hour_concentration', {}).items():
            lines.append(f'- {cell}: share of excess accuracy from 00:00-06:00 labels = '
                         f'{f(hc["share_00_06"], 3) or "n/a"}' + (' — **FLAG: possible bid-side artifact**' if hc['flag'] else ''))
        if 'pbo' in ex:
            lines.append(f'- {cad} PBO across {ex["pbo"]["candidates"]}: {f(ex["pbo"]["pbo"], 3)}')
    lines += dsr_lines(meta.get('deflated_sharpe_gross', {}))
    if failures:
        lines += ['', '## Failures (unspent)', ''] + [f'- {x["cell"]}: {x["error"]}' for x in failures]
    lines.append('')
    return '\n'.join(lines)


def save_artifacts(frames, cfg):
    """--stage final: refit on [0:100] and freeze every cell with digests."""
    rows, failures, trials = [], [], []
    art = repo_path(cfg['paths']['artifacts'])
    for cad in ('D1', 'H1'):
        df = frames[cad]
        _, _, pos, _, models = fit_and_predict(cad, df, 'final', cfg, trials, failures)
        for cell, m in models.items():
            path = os.path.join(art, cell)
            for f in m.save(path):
                with open(f, 'rb') as fh:
                    blob = fh.read()
                rows.append({'cell': cell, 'cadence': cad,
                             'path': os.path.relpath(f, REPO).replace('\\', '/'),
                             'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob),
                             'config_hash': m.config_hash, 'fit_rows': len(pos['fit']),
                             'last_bar': str(df.index[-1]), 'written_at': _now()})
    with open(repo_path(cfg['paths']['manifest']), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    if trials:
        OF.append_trials(trials, path=repo_path(cfg['paths']['trial_log']))
    return rows, failures


def run(stage: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    t0 = time.time()
    reg = check_stage_allowed(stage, cfg)
    frames = load_pinned(cfg, reg)
    _log(f'stage {stage}: D1 {len(frames["D1"])} rows, H1 {len(frames["H1"])} rows (pinned)')
    if stage == 'final':
        rows, failures = save_artifacts(frames, cfg)
        return {'stage': stage, 'artifacts': len(rows), 'failures': failures}
    results, failures, trial_rows, extras, pred_frames = score_stage(stage, frames, reg, cfg)
    if trial_rows:
        OF.append_trials(trial_rows, path=repo_path(cfg['paths']['trial_log']))
    meta = {'stage': stage, 'finished_at': _now(), 'elapsed_s': time.time() - t0,
            'registration_commit': FP.committed_and_clean(repo_path(cfg['paths']['prereg_json']), REPO)}
    res_dir = repo_path(cfg['paths']['results'])
    for cad, pf in pred_frames.items():
        pf.to_csv(os.path.join(res_dir, f'{stage}_predictions_{cad.lower()}.csv'), float_format='%.6g')
    meta['deflated_sharpe_gross'] = deflated_sharpe_from_predictions(stage, cfg)
    report = render_report(stage, results, failures, extras, meta)
    if stage == 'validation':
        for fam, path in cfg['paths']['registries'].items():
            append_registry(repo_path(path), [r for r in results if r['family'] == fam])
        with open(repo_path(cfg['paths']['validation_report']), 'w', encoding='utf-8') as fh:
            fh.write(report)
    else:
        with open(repo_path(cfg['paths']['test_report']), 'w', encoding='utf-8') as fh:
            json.dump(_clean({'meta': meta, 'results': results, 'failures': failures, 'descriptive': extras}),
                      fh, indent=1)
        with open(repo_path(cfg['paths']['test_report']).replace('.json', '.md'), 'w', encoding='utf-8') as fh:
            fh.write(report)
    return {'stage': stage, 'results': _clean(results), 'failures': failures}


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold study stages')
    ap.add_argument('--stage', required=True, choices=S.STAGES)
    args = ap.parse_args(argv)
    out = run(args.stage)
    for r in out.get('results', []):
        print(f"{r['family']:10s} {r['cell']:13s} {r['verdict']}")
    if out.get('failures'):
        print('FAILURES:', out['failures'])


if __name__ == '__main__':
    main()
