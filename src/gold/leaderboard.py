"""Live gold leaderboard (openspec change gold-forecasting-v2, spec gold-forecasting/leaderboard).

One row per (cell, model digest) from settled LIVE rows only:
  * settled count, and the phase (dry run / registered);
  * direction hit rate with a Wilson 95 % interval against a coin;
  * move error next to its GARCH baseline on the bars both forecast;
  * 80 % and 95 % range coverage;
  * whether the row comes from a shadow version (a monthly refit running beside
    the frozen registered models, change gold-forecasting-v2, model-versions).

Descriptive, never a verdict: no KEEP or DROP appears, and a row with fewer than
30 settled forecasts shows only its count. A second block compares alerted hours
with the rest. Read-only: runs no model, fetches nothing, writes nothing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import history as H
from . import load_config, repo_path

SORT_KEYS = ('n', 'hit_rate', 'vol_error_gain', 'coverage80')


def _read(path):
    return H._read(path)


def v2_rows(cfg) -> pd.DataFrame:
    """Round-two and shadow-version forecasts (predictions_v2 / settlements_v2),
    in the history schema plus `version` and `shadow`."""
    fwd = repo_path(cfg['paths']['forward'])
    pred = _read(os.path.join(fwd, 'predictions_v2.csv'))
    if pred.empty:
        return pd.DataFrame(columns=H.COLUMNS + ['version', 'shadow'])
    sett = _read(os.path.join(fwd, 'settlements_v2.csv'))
    s = sett.drop_duplicates('key').set_index('key') if not sett.empty else pd.DataFrame(
        columns=['exit_bar', 'exit_close', 'realised_dir', 'realised_ret_pct', 'realised_vol_pct', 'eligible'])
    j = s.reindex(pred['key'])
    rows = pd.DataFrame({
        'source': 'live', 'cell': pred['cell'], 'cadence': pred['cadence'], 'model_digest': pred['model_digest'],
        'as_of_bar': pred['as_of_bar'], 'as_of_close': H._num(pred['as_of_close']), 'p_up': H._num(pred['p_up']),
        'ret_pct': H._num(pred['ret_pct']), 'vol_pct': H._num(pred['vol_pct']), 'phase': pred.get('phase', ''),
        'logged_at': pred.get('logged_at', ''), 'reason': None,
        'version': pred.get('version', ''), 'shadow': pred.get('shadow', '').astype(str).isin(['1', 'True', 'true']),
    })
    rows['exit_bar'] = j['exit_bar'].to_numpy()
    rows['actual_price'] = H._num(j['exit_close']).to_numpy()
    rows['realised_dir'] = H._num(j['realised_dir']).to_numpy()
    rows['realised_ret_pct'] = H._num(j['realised_ret_pct']).to_numpy()
    rows['realised_vol_pct'] = H._num(j['realised_vol_pct']).to_numpy()
    has = j['exit_bar'].notna().to_numpy()
    elig = j['eligible'].astype(str).str.lower().eq('true').to_numpy()
    rows['status'] = np.where(~has, 'pending', np.where(elig, 'settled', 'not_scored'))
    return H._derive(rows)


def _coverage(g: pd.DataFrame, tag: str):
    col = f'{tag}_inside'
    if col not in g:
        return None, 0
    v = g[col].dropna()
    return (float(v.astype(bool).mean()) if len(v) else None), int(len(v))


def alert_comparison(cfg, live: pd.DataFrame) -> dict:
    """Mean realised move of alerted hours against all other settled h1_lstm hours."""
    fwd = repo_path(cfg['paths']['forward'])
    alerts = _read(os.path.join(fwd, 'alerts.csv'))
    h = live[(live['cell'] == 'h1_lstm') & (live['status'] == 'settled')]
    if h.empty:
        return {'alerted_n': 0, 'other_n': 0, 'kind': H.DESCRIPTIVE}
    keys = h['cell'] + '|' + h['cadence'] + '|' + h['as_of_bar']
    alerted = keys.isin(set(alerts['key'])) if not alerts.empty else pd.Series(False, index=h.index)
    a, o = h.loc[alerted.to_numpy(), 'realised_vol_pct'], h.loc[~alerted.to_numpy(), 'realised_vol_pct']
    return {'alerted_n': int(a.notna().sum()), 'other_n': int(o.notna().sum()),
            'alerted_mean_move': float(a.mean()) if a.notna().any() else None,
            'other_mean_move': float(o.mean()) if o.notna().any() else None, 'kind': H.DESCRIPTIVE}


def build(cfg=None, sort: str = 'n') -> dict:
    cfg = cfg or load_config()
    if sort not in SORT_KEYS:
        raise ValueError(f'unknown sort {sort!r}; allowed: {list(SORT_KEYS)}')
    live = H.live(cfg)
    live = live.assign(version='registered', shadow=False)
    v2 = v2_rows(cfg)
    rows = pd.concat([live, v2], ignore_index=True) if not v2.empty else live
    summ = {(s['cell'], s['model_digest'] or ''): s for s in H.summaries(rows)}
    out = []
    settled = rows[rows['status'] == 'settled']
    for (cell, digest), g in settled.groupby(['cell', 'model_digest'], dropna=False, sort=True):
        s = summ.get((cell, digest or ''), {})
        cov80, n80 = _coverage(g, 'range80')
        cov95, n95 = _coverage(g, 'range95')
        item = {'cell': cell, 'cadence': g['cadence'].iloc[0], 'model_digest': digest or None,
                'version': str(g['version'].iloc[0]), 'shadow': bool(g['shadow'].iloc[0]),
                'phase': g['phase'].mode().iloc[0] if g['phase'].notna().any() else None,
                'n': int(len(g)), 'kind': H.DESCRIPTIVE, 'state': s.get('state', 'too_few')}
        if item['state'] == 'ok':
            for k in ('n_direction', 'hit_rate', 'hit_low', 'hit_high', 'vs_coin', 'vol_mae', 'vol_baseline',
                      'n_shared', 'vol_mae_shared', 'vol_mae_baseline'):
                if k in s:
                    item[k] = s[k]
            if 'vol_mae_baseline' in s and s.get('vol_mae_shared') is not None:
                item['vol_error_gain'] = s['vol_mae_baseline'] - s['vol_mae_shared']
            if n80:
                item.update(coverage80=cov80, coverage80_n=n80, coverage95=cov95, coverage95_n=n95)
        out.append(item)
    ready = [r for r in out if r['state'] == 'ok']
    rest = [r for r in out if r['state'] != 'ok']
    ready.sort(key=lambda r: (r.get(sort) is None, -(r.get(sort) or 0)))
    return {'kind': H.DESCRIPTIVE, 'cost_basis': 'gross of cost', 'sort': sort, 'sort_keys': list(SORT_KEYS),
            'min_rows': H.MIN_SUMMARY_ROWS, 'rows': ready + rest, 'alerts': alert_comparison(cfg, live)}
