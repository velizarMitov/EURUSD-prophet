"""The horizon study (tasks 8.1, 8.2; spec horizon-study; design D4-D10).

For every challenger x horizon cell in the study record:

  1. purged expanding walk-forward  -> out-of-sample predictions on the later
     half of the history (the primary development view);
  2. CPCV                           -> a distribution of out-of-sample paths;
  3. GBM cells only: nested-CV tuning, then the declared frozen-vs-tuned rule;
  4. one final fit on every labelled row -> research artifact + manifest row,
     which the forward logger loads.

Everything written is DESCRIPTIVE. Nothing here reads or writes a hypothesis
log, and nothing is written under the production models/ directory.

Outputs (results/horizon_study/):
  horizon_curves.csv      one row per cell
  cpcv_paths.csv          one row per cell x path
  pbo_across_horizons.csv per model: the risk of picking the best horizon in-sample
  failures.csv            cells that could not run, with the reason
  artifact_manifest.csv   tracked digests of every research artifact
  trial_log.csv           append-only trial count (overfit.py)
  run_meta.json           date ranges, seeds, timings
Out-of-sample predictions go to research_models/horizon_study/oos/ (regenerable).

Resumable: a cell already in horizon_curves.csv for this study id is skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import benchmarks as B
from . import costs as C
from . import features as F
from . import overfit as O
from . import record as R
from . import splits as S
from . import stats as ST
from . import targets as T
from . import tuning as TU
from .challengers import make
from .challengers.base import ChallengerUnavailable, config_hash
from .challengers.volatility import GarchDow, RandomWalkVol

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'results', 'horizon_study')
ART = os.path.join(REPO, 'research_models', 'horizon_study')
SYMBOL = 'EURUSD'
BANNER = 'DESCRIPTIVE - not a verdict; describes challengers, not production models'
MANIFEST_FIELDS = ['version_id', 'study_id', 'model', 'cadence', 'horizon', 'variant', 'config_hash',
                   'path', 'files_sha256', 'train_start', 'train_end', 'feature_digest', 'created_at']


# ── data ────────────────────────────────────────────────────────────────────

def load_sources(macro: bool = True) -> dict:
    """Daily (both feature sets on the euro-era row set) and H1 frames."""
    raw = F.load_daily_ohlcv()
    macro_df, macro_sources = None, {'all': 'not requested'}
    if macro:
        macro_df, macro_sources = load_macro_window(raw.index.min(), raw.index.max(), with_sources=True)
    daily = {fs: F.daily_features(raw, macro_df, fs) for fs in F.FEATURE_SETS}
    return {'daily': daily, 'h1': F.load_h1(), 'macro_sources': macro_sources}


def load_macro_window(start, end, with_sources: bool = False):
    """Production macro chain (FRED API -> public CSV -> cache) on a sandboxed
    COPY of the caches: the chain refreshes its on-disk caches, which are
    tracked production files, so the study and the forward logger never touch
    the originals."""
    from src.macro_data import fetch_macro_features
    with open(os.path.join(REPO, 'config.json'), encoding='utf-8') as fh:
        macro_cfg = json.load(fh).get('macro', {})
    sandbox = os.path.join(ART, 'macro_cache')
    for rel in _macro_cache_paths(macro_cfg):
        src, dst = os.path.join(REPO, rel), os.path.join(sandbox, rel)
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    macro_df, sources = fetch_macro_features(start, end, macro_cfg, base_dir=sandbox)
    return (macro_df, sources) if with_sources else macro_df


def _macro_cache_paths(macro_cfg: dict) -> list:
    out = []
    def walk(d):
        for k, v in d.items():
            if k == 'cache_path' and isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                walk(v)
    walk(macro_cfg)
    return sorted(set(out))


def source_for(name: str, cfg: dict, sources: dict):
    if name.startswith(('daily_', 'vol_')):
        return sources['daily'][cfg.get('feature_set', 'price')]
    return sources['h1']


def make_targets(close: np.ndarray, h: int) -> dict:
    s = pd.Series(np.asarray(close, dtype=float))
    d, n_zero = T.direction_target(s, h)
    return {'dir': d.to_numpy(), 'ret': T.return_pct_target(s, h).to_numpy(),
            'vol': T.volatility_pct_target(s, h).to_numpy(), 'n_zero': n_zero}


def cell_horizons(name: str, cfg: dict, rec: dict) -> list[int]:
    return list(cfg.get('horizons') or rec['grid'][cfg['cadence']])


# ── fitting helpers ─────────────────────────────────────────────────────────

def _oos(name, cfg, inputs, tg, splits_, seed, h, keys):
    out = {k: np.full(len(inputs), np.nan) for k in keys}
    for sp in splits_:
        m = make(name, cfg).fit(inputs, sp.train, tg, seed, h)
        pr = m.predict(inputs)
        for k in keys:
            if k in pr:
                out[k][sp.test] = pr[k][sp.test]
    return out


def _cpcv_paths(name, cfg, inputs, tg, cv, seed, h, key):
    preds = []
    for sp in cv.splits:
        m = make(name, cfg).fit(inputs, sp.train, tg, seed, h)
        preds.append(m.predict(inputs)[key][sp.test])
    return cv.assemble(preds)


def _kronos_oos(model, inputs, horizons, stride, seed):
    from .challengers.kronos import CLEAN_WINDOW_START
    start = int(np.searchsorted(inputs.index, CLEAN_WINDOW_START))
    as_of = np.arange(start, len(inputs) - 1, stride)
    paths = model.sample(inputs, as_of, max(horizons), seed=seed)
    return paths, as_of


# ── metrics ─────────────────────────────────────────────────────────────────

def non_overlapping(rows: np.ndarray, h: int) -> np.ndarray:
    """Every h-th scored row: one trade per horizon, held to its own exit."""
    rows = np.sort(rows)
    keep, last = [], -10 ** 9
    for r in rows:
        if r >= last + h:
            keep.append(r)
            last = r
    return np.asarray(keep, dtype=int)


def strategy_stats(p_up, close, rows, h, cost):
    rows = non_overlapping(rows, h)
    rows = rows[rows + h < len(close)]
    if rows.size < 3:
        return {}, np.array([])
    pos = np.where(p_up[rows] >= 0.5, 1.0, -1.0)
    moves = close[rows + h] - close[rows]
    pnl = C.net_pnl(pos, moves, cost)
    per_trade_net = pos * moves - np.where(C.position_changes(pos), cost, 0.0)
    return pnl, per_trade_net


def direction_metrics(name, cadence, h, p_up, ret_pct, close, tg, train_majority_y, cost_levels):
    y = tg['dir']
    rows = np.flatnonzero(np.isfinite(p_up) & np.isfinite(y))
    if rows.size < 30:
        raise ValueError(f'only {rows.size} scored rows')
    yy, pp = y[rows], p_up[rows]
    pred = (pp >= 0.5).astype(float)
    correct = (pred == yy).astype(float)
    acc = float(correct.mean())
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else float('nan')
    pt = ST.pesaran_timmermann(yy, pred)
    blen = ST.block_length(h, cadence)
    lo, hi, _ = ST.block_bootstrap_ci(correct, blen, 0.05)
    s_close = pd.Series(close)
    be = {lv.label: T.breakeven_for_horizon(s_close, h, lv.price) for lv in cost_levels}
    label = ('no directional skill' if acc <= 0.5 else
             'predictive, not cost-viable' if acc <= be['measured'] else
             'above breakeven (descriptive)')
    out = {'n_scored': int(rows.size), 'n_eff': S.effective_n(rows, h, len(close)),
           'n_zero_moves': int(tg['n_zero']), 'accuracy': acc, 'acc_ci_low': lo, 'acc_ci_high': hi,
           'auc': auc, 'coin_flip': B.COIN_FLIP_ACCURACY,
           'majority_accuracy': B.majority_accuracy(train_majority_y, yy),
           'pt_stat': pt['stat'], 'pt_p': pt['p_value'],
           'breakeven_measured': be['measured'], 'breakeven_config': be['config_round_trip'],
           'label': label, 'block_len': blen}
    nets = {}
    for lv in cost_levels:
        pnl, per_trade = strategy_stats(p_up, close, rows, h, lv.price)
        if pnl:
            out[f'gross_per_trade_{lv.label}'] = pnl['gross_per_trade']
            out[f'net_per_trade_{lv.label}'] = pnl['net_per_trade']
            out[f'n_trades_{lv.label}'] = pnl['n_trades']
            nets[lv.label] = per_trade
    trade_rows = non_overlapping(rows, h)
    trade_rows = trade_rows[trade_rows + h < len(close)]
    if trade_rows.size > 3:
        pos = np.where(p_up[trade_rows] >= 0.5, 1.0, -1.0)
        rs = B.random_sign_benchmark(pos, close[trade_rows + h] - close[trade_rows],
                                     cost_levels[0].price, n_sims=500)
        out['random_sign_percentile'] = rs['percentile']
    if ret_pct is not None and np.isfinite(ret_pct).any():
        m = np.isfinite(ret_pct) & np.isfinite(tg['ret'])
        cw = ST.clark_west(tg['ret'][m], ret_pct[m], None, h)
        out['clark_west_stat'], out['clark_west_p'] = cw['stat'], cw['p_value']
    return out, nets.get('measured', np.array([]))


def baseline_vol_oos(close, dow, y, folds, use_dow):
    """GARCH(1,1)(x day-of-week) refitted per fold on that fold's training rows
    only, exactly as the challenger is -- a baseline fitted once on the first
    fold would be handicapped against a model that refits every fold."""
    out = np.full(len(close), np.nan)
    for train_mask, test_rows in folds:
        pred = GarchDow(use_dow).fit(close, dow, y, train_mask).predict(close, dow)
        out[test_rows] = pred[test_rows]
    return out


def volatility_metrics(cadence, h, vol_pred, close, dow, tg, folds):
    """folds: [(train_mask, test_rows), ...] -- the walk-forward folds, or one
    fold for a train-free model (train = before its clean window)."""
    y = tg['vol']
    rows = np.flatnonzero(np.isfinite(vol_pred) & np.isfinite(y))
    if rows.size < 30:
        raise ValueError(f'only {rows.size} scored rows')
    gd = baseline_vol_oos(close, dow, y, folds, True)
    g = baseline_vol_oos(close, dow, y, folds, False)
    rw = RandomWalkVol().predict(close, h)
    rows = rows[np.isfinite(gd[rows]) & np.isfinite(g[rows]) & np.isfinite(rw[rows])]
    err = lambda p: np.abs(p[rows] - y[rows])
    blen = ST.block_length(h, cadence)
    lo, hi, d = ST.block_bootstrap_delta(err(vol_pred), err(gd), blen, 0.05)
    return {'n_scored': int(rows.size), 'mae_model': float(err(vol_pred).mean()),
            'mae_garch_dow': float(err(gd).mean()), 'mae_garch': float(err(g).mean()),
            'mae_rw_vol': float(err(rw).mean()), 'delta_mae_vs_garch_dow': d,
            'delta_ci_low': lo, 'delta_ci_high': hi, 'block_len': blen,
            'label': ('beats strongest baseline (descriptive)' if np.isfinite(hi) and hi < 0
                      else 'does not beat GARCH x day-of-week')}


# ── artifacts ───────────────────────────────────────────────────────────────

def save_artifact(model, study_id, name, cadence, h, variant, inputs, train_pos, digest) -> dict:
    vid = f'{study_id}-{name}-h{h}-{variant}-{pd.Timestamp(inputs.index[train_pos[-1]]).strftime("%Y%m%d%H")}'
    path = os.path.join(ART, name, f'h{h}', vid)
    files = model.save(path)
    rel = {os.path.relpath(f, REPO).replace('\\', '/'): F.file_digest(f) for f in files}
    return {'version_id': vid, 'study_id': study_id, 'model': name, 'cadence': cadence,
            'horizon': h, 'variant': variant, 'config_hash': model.config_hash,
            'path': os.path.relpath(path, REPO).replace('\\', '/'),
            'files_sha256': json.dumps(rel, sort_keys=True),
            'train_start': str(inputs.index[train_pos[0]]), 'train_end': str(inputs.index[train_pos[-1]]),
            'feature_digest': digest, 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}


def _append_csv(path, row, fields=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new and fields is None:
        with open(path, newline='', encoding='utf-8') as fh:
            fields = next(csv.reader(fh))
        missing = [k for k in row if k not in fields]
        if missing:                                        # widen the header once
            old = pd.read_csv(path)
            for k in missing:
                old[k] = np.nan
            old.to_csv(path, index=False)
            fields = list(old.columns)
    fields = fields or list(row)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in fields})


def done_cells(study_id: str, out: str = OUT) -> set:
    p = os.path.join(out, 'horizon_curves.csv')
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p)
    d = d[d['study_id'] == study_id]
    return set(zip(d['model'], d['horizon'].astype(int)))


# ── one cell ────────────────────────────────────────────────────────────────

def run_cell(name, cfg, h, inputs, rec, digest, out=OUT, trial_log=None, kronos_paths=None,
             save=True) -> dict:
    trial_log = trial_log or os.path.join(out, 'trial_log.csv')
    dev = rec['development']
    seed = rec['seeds'][0]
    cadence = cfg['cadence']
    tg = make_targets(inputs.close, h)
    n = len(inputs)
    labelled = np.flatnonzero(np.isfinite(tg['dir']) | np.isfinite(tg['vol']))
    row = {'study_id': rec['study_id'], 'model': name, 'cadence': cadence, 'horizon': h,
           'config_hash': config_hash(cfg), 'first_bar': str(inputs.index[0]),
           'last_bar': str(inputs.index[-1]), 'banner': BANNER}
    cost_levels = C.cost_levels(SYMBOL)
    kind = cfg['kind']
    key = 'p_up' if kind == 'direction' else 'vol_pct'
    t0 = time.time()

    if cfg.get('train_free'):
        from .challengers.kronos import Kronos
        paths, as_of = kronos_paths
        pred = Kronos.from_paths(paths, inputs.close, h, n, kind)
        row['variant'] = 'pinned'
        train_mask = inputs.index < pd.Timestamp('2024-07-01', tz='UTC')
        if kind == 'direction':
            m, per_trade = direction_metrics(name, cadence, h, pred, None, inputs.close, tg,
                                             tg['dir'][train_mask], cost_levels)
        else:
            scored = np.flatnonzero(np.isfinite(pred))
            m, per_trade = volatility_metrics(cadence, h, pred, inputs.close, inputs.extra['dow'],
                                              tg, [(train_mask, scored)]), np.array([])
        row.update(m)
        row['history_stride'] = cfg.get('history_stride')
        O.append_trials([{'study_id': rec['study_id'], 'model': name, 'cadence': cadence, 'horizon': h,
                          'config_hash': config_hash(cfg), 'variant': 'pinned',
                          'sharpe': O.sharpe(per_trade) if per_trade.size > 2 else '',
                          'n_obs': m['n_scored']}], trial_log)
        row['fit_seconds'] = round(time.time() - t0, 1)
        return row, {key: pred}, None

    min_train = int(n * dev['min_train_fraction'])
    wf = S.walk_forward(n, h=h, n_splits=dev['walk_forward_splits'], min_train=min_train)
    keys = ('p_up', 'ret_pct', 'vol_pct')
    oos = _oos(name, cfg, inputs, tg, wf, seed, h, keys)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[wf[0].train] = True
    cv_cfg = dev['cpcv'].get(name, dev['cpcv']['default'])
    cv = S.cpcv(n, h=h, n_groups=cv_cfg['n_groups'], k=cv_cfg['k'])

    variant, chosen_cfg = 'frozen', cfg
    if kind == 'direction':
        m, per_trade = direction_metrics(name, cadence, h, oos['p_up'], oos.get('ret_pct'),
                                         inputs.close, tg, tg['dir'][wf[0].train], cost_levels)
        paths = _cpcv_paths(name, cfg, inputs, tg, cv, seed, h, 'p_up')
        path_rows = [_path_summary(p, inputs.close, tg, h, cost_levels[0].price) for p in paths]
        grid = rec['tuning']['grids'].get(name)
        if grid:
            tuned = TU.inner_tune(name, cfg, grid, inputs, wf[0].train, tg, h, seed,
                                  rec['tuning']['rule']['inner_folds'])
            t_paths = _cpcv_paths(name, tuned['config'], inputs, tg, cv, seed, h, 'p_up')
            t_rows = [_path_summary(p, inputs.close, tg, h, cost_levels[0].price) for p in t_paths]
            f_net = np.nanmedian([r['net_per_trade'] for r in path_rows])
            t_net = np.nanmedian([r['net_per_trade'] for r in t_rows])
            pair = np.column_stack([_path_series(paths[0], inputs.close, tg, h),
                                    _path_series(t_paths[0], inputs.close, tg, h)])
            pair = pair[np.isfinite(pair).all(axis=1)]
            pbo_pair = O.pbo_cscv(pair, ['frozen', 'tuned'], 16)['pbo'] if len(pair) >= 64 else float('nan')
            variant, why = TU.select_variant(f_net, t_net, pbo_pair, rec['tuning']['rule'])
            row.update({'tuned_params': json.dumps(tuned['params']), 'tuned_inner_auc': tuned['inner_auc'],
                        'cpcv_net_frozen': f_net, 'cpcv_net_tuned': t_net, 'pbo_frozen_vs_tuned': pbo_pair,
                        'variant_reason': why})
            O.append_trials(TU.variant_trials(rec['study_id'], name, cadence, h, cfg, tuned['config'],
                                              O.sharpe(per_trade) if per_trade.size > 2 else float('nan'),
                                              float('nan'), m['n_scored'], variant), trial_log)
            if variant == 'tuned':
                chosen_cfg = tuned['config']
                oos = _oos(name, chosen_cfg, inputs, tg, wf, seed, h, keys)
                m, per_trade = direction_metrics(name, cadence, h, oos['p_up'], oos.get('ret_pct'),
                                                 inputs.close, tg, tg['dir'][wf[0].train], cost_levels)
                path_rows = t_rows
        else:
            O.append_trials([{'study_id': rec['study_id'], 'model': name, 'cadence': cadence,
                              'horizon': h, 'config_hash': config_hash(cfg), 'variant': 'frozen',
                              'sharpe': O.sharpe(per_trade) if per_trade.size > 2 else '',
                              'n_obs': m['n_scored']}], trial_log)
        nets = np.array([r['net_per_trade'] for r in path_rows])
        m.update({'cpcv_paths': len(path_rows), 'cpcv_net_median': float(np.nanmedian(nets)),
                  'cpcv_net_p10': float(np.nanpercentile(nets, 10)),
                  'cpcv_net_p90': float(np.nanpercentile(nets, 90)),
                  'cpcv_acc_median': float(np.nanmedian([r['accuracy'] for r in path_rows]))})
        dsr = O.deflated_sharpe(per_trade, O.n_trials(trial_log), O.trial_sharpe_variance(trial_log))
        m.update({'sharpe_per_trade': dsr['sharpe'], 'dsr': dsr['dsr'], 'dsr_sr0': dsr['sr0'],
                  'dsr_n_trials': dsr['n_trials']})
        for i, r in enumerate(path_rows):
            _append_csv(os.path.join(out, 'cpcv_paths.csv'),
                        {'study_id': rec['study_id'], 'model': name, 'horizon': h, 'path': i, **r},
                        ['study_id', 'model', 'horizon', 'path', 'accuracy', 'net_per_trade', 'n_trades'])
    else:
        folds = []
        for sp in wf:
            fm = np.zeros(n, dtype=bool)
            fm[sp.train] = True
            folds.append((fm, sp.test))
        m = volatility_metrics(cadence, h, oos['vol_pct'], inputs.close, inputs.extra['dow'], tg, folds)
        vpaths = _cpcv_paths(name, cfg, inputs, tg, cv, seed, h, 'vol_pct')
        maes = [float(np.nanmean(np.abs(p - tg['vol']))) for p in vpaths]
        m.update({'cpcv_paths': len(maes), 'cpcv_mae_median': float(np.median(maes)),
                  'cpcv_mae_min': float(np.min(maes)), 'cpcv_mae_max': float(np.max(maes))})
        O.append_trials([{'study_id': rec['study_id'], 'model': name, 'cadence': cadence, 'horizon': h,
                          'config_hash': config_hash(cfg), 'variant': 'frozen', 'sharpe': '',
                          'n_obs': m['n_scored']}], trial_log)
    row.update(m)
    row['variant'] = variant
    row['fit_seconds'] = round(time.time() - t0, 1)

    manifest = None
    if save:
        final_pos = labelled[labelled < n - h]
        final = make(name, chosen_cfg).fit(inputs, final_pos, tg, seed, h)
        manifest = save_artifact(final, rec['study_id'], name, cadence, h, variant, inputs, final_pos, digest)
    return row, oos, manifest


def _path_series(p_up, close, tg, h):
    """Per-bar P&L of the signal held one bar (used for PBO comparisons)."""
    pos = np.where(np.isfinite(p_up), np.where(p_up >= 0.5, 1.0, -1.0), np.nan)
    nxt = np.append(close[1:] - close[:-1], np.nan)
    return pos * nxt


def _path_summary(p_up, close, tg, h, cost):
    rows = np.flatnonzero(np.isfinite(p_up) & np.isfinite(tg['dir']))
    acc = float(((p_up[rows] >= 0.5).astype(float) == tg['dir'][rows]).mean()) if rows.size else float('nan')
    pnl, _ = strategy_stats(p_up, close, rows, h, cost)
    return {'accuracy': acc, 'net_per_trade': pnl.get('net_per_trade', float('nan')),
            'n_trades': pnl.get('n_trades', 0)}


# ── whole study ─────────────────────────────────────────────────────────────

def pbo_across_horizons(study_id: str, out: str = OUT) -> pd.DataFrame:
    """Per model: PBO of picking its best horizon in-sample, from the per-bar
    P&L of each horizon's out-of-sample signal on their common rows."""
    oos_dir = os.path.join(ART, 'oos', study_id)
    rows = []
    if not os.path.isdir(oos_dir):
        return pd.DataFrame()
    by_model = {}
    for f in sorted(os.listdir(oos_dir)):
        if f.endswith('.parquet'):
            model, h = f[:-8].rsplit('_h', 1)
            by_model.setdefault(model, []).append((int(h), os.path.join(oos_dir, f)))
    for model, items in by_model.items():
        cols, names = [], []
        for h, p in sorted(items):
            d = pd.read_parquet(p)
            if 'p_up' not in d or d['p_up'].notna().sum() < 64:
                continue
            cols.append(_path_series(d['p_up'].to_numpy(), d['close'].to_numpy(), None, h))
            names.append(f'h{h}')
        if len(cols) < 2:
            continue
        M = np.column_stack(cols)
        M = M[np.isfinite(M).all(axis=1)]
        if len(M) < 64:
            continue
        r = O.pbo_cscv(M, names, 16)
        rows.append({'study_id': study_id, 'model': model, 'pbo': r['pbo'], 'candidates': ','.join(names),
                     'n_rows': len(M), 'logit_median': r['logit_median'], 'banner': BANNER})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(os.path.join(out, 'pbo_across_horizons.csv'), index=False)
    return df


def run_study(study_id: str, models=None, out: str = OUT, record_path: str | None = None,
              sources=None, save=True) -> dict:
    record_path = record_path or os.path.join(out, 'study_record.json')
    rec = R.load(record_path)
    if rec is None or rec['study_id'] != study_id:
        raise R.StudyRecordLocked(f'write the study record for {study_id} before running it')
    sources = sources or load_sources()
    digest = F.feature_code_digest()['_combined']
    R.mark_first_fit(record_path)
    rec = R.load(record_path)
    done = done_cells(study_id, out)
    timings, kronos_cache = {}, {}
    for name, cfg in rec['challengers'].items():
        if models and name not in models:
            continue
        horizons = cell_horizons(name, cfg, rec)
        try:
            model = make(name, cfg)
            inputs = model.build_inputs(source_for(name, cfg, sources))
            kpaths = None
            if cfg.get('train_free') and any((name, h) not in done for h in horizons):
                # Both Kronos channels read the SAME sampled paths (same seed per
                # bar): sampling once at the longest horizon serves every cell.
                key = (cfg['history_stride'], rec['seeds'][0], cfg['n_paths'], cfg.get('model'))
                cached = kronos_cache.get(key)
                if cached is None or cached[2] < max(horizons):
                    longest = max(max(cell_horizons(n2, c2, rec)) for n2, c2 in rec['challengers'].items()
                                  if c2.get('train_free'))
                    paths, as_of = _kronos_oos(model, inputs, [longest], cfg['history_stride'],
                                               rec['seeds'][0])
                    kronos_cache[key] = cached = (paths, as_of, longest)
                kpaths = cached[:2]
        except (ChallengerUnavailable, Exception) as e:      # noqa: BLE001 -- isolate the family
            for h in horizons:
                _fail(out, study_id, name, h, e)
            continue
        for h in horizons:
            if (name, h) in done:
                continue
            try:
                R.assert_fit_allowed(name, cfg, h, cfg['cadence'], record_path)
                row, oos, manifest = run_cell(name, cfg, h, inputs, rec, digest, out,
                                              kronos_paths=kpaths, save=save)
                _save_oos(study_id, name, h, inputs, oos)
                _append_csv(os.path.join(out, 'horizon_curves.csv'), row)
                if manifest:
                    _append_csv(os.path.join(out, 'artifact_manifest.csv'), manifest, MANIFEST_FIELDS)
                timings[f'{name}_h{h}'] = row['fit_seconds']
                print(f'[{datetime.now():%H:%M:%S}] {name} h={h}: {row.get("label")} '
                      f'({row["fit_seconds"]}s)', flush=True)
            except Exception as e:                           # noqa: BLE001 -- one cell must not stop the rest
                _fail(out, study_id, name, h, e)
    pbo = pbo_across_horizons(study_id, out)
    meta = {'study_id': study_id, 'finished_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'daily_range': [str(sources['daily']['price'].index[0]), str(sources['daily']['price'].index[-1])],
            'daily_rows': len(sources['daily']['price']),
            'h1_range': [str(sources['h1'].index[0]), str(sources['h1'].index[-1])],
            'h1_rows': len(sources['h1']), 'macro_sources': sources.get('macro_sources'),
            'seeds': rec['seeds'], 'cell_seconds': timings, 'banner': BANNER}
    meta_path = os.path.join(out, f'run_meta_{study_id}.json')
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as fh:
            meta['cell_seconds'] = {**json.load(fh).get('cell_seconds', {}), **timings}
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1, default=str)
    return {'meta': meta, 'pbo': pbo}


def _save_oos(study_id, name, h, inputs, oos):
    d = os.path.join(ART, 'oos', study_id)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame({'close': inputs.close, **{k: v for k, v in oos.items()}}, index=inputs.index)
    df.to_parquet(os.path.join(d, f'{name}_h{h}.parquet'))


def _fail(out, study_id, name, h, exc):
    _append_csv(os.path.join(out, 'failures.csv'),
                {'study_id': study_id, 'model': name, 'horizon': h,
                 'at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 'error': f'{type(exc).__name__}: {exc}',
                 'trace': traceback.format_exc(limit=3).replace('\n', ' | ')},
                ['study_id', 'model', 'horizon', 'at', 'error', 'trace'])
    print(f'FAILED {name} h={h}: {type(exc).__name__}: {exc}', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Run the horizon study (descriptive only).')
    ap.add_argument('--study-id', required=True)
    ap.add_argument('--init', action='store_true', help='write the study record (before any fit)')
    ap.add_argument('--models', nargs='*')
    a = ap.parse_args(argv)
    path = os.path.join(OUT, 'study_record.json')
    if a.init:
        R.save(R.new_record(a.study_id), path)
        print(f'study record written: {path}')
        return
    run_study(a.study_id, a.models)


if __name__ == '__main__':
    main()
