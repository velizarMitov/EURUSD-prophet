"""Scoring and report of the acceleration study (design D5-D7, spec "The report
is descriptive", "One fixed net-of-cost check").

Every comparison uses the rows both sides predicted that are eligible and
target-defined. Hit = (p_up > 0.5) == dir; p_up of exactly 0.5 is a miss.
Differences carry a paired moving-block-bootstrap 95 % interval (block 120 H1
rows; the cost check resamples days in blocks of 20).

DESCRIPTIVE ONLY: every year has been seen. The report writes no verdict, and an
interval that excludes zero is reported as such, never as an improvement.

    python -m src.gold_accel.backtest
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.gold_window.simulate import trade_returns

from . import load_config, repo_path
from .walkforward import _guard, load_predictions, pin_inputs, window_lstm_reuse


# -- statistics ---------------------------------------------------------------

def block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int, seed: int, ci: float = 0.95):
    """(mean, low, high) of the moving-block bootstrap of mean(x)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return None, None, None
    mean = float(x.mean())
    b = max(1, min(block, n))
    means = np.convolve(x, np.ones(b) / b, mode='valid')          # every block's mean
    k = math.ceil(n / b)
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(0, len(means), size=(n_boot, k))].mean(axis=1)
    a = (1 - ci) / 2
    return mean, float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def hits(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((p > 0.5) == (y > 0.5)).astype(float)


def majority_by_year(times: pd.DatetimeIndex, y: np.ndarray, eligible: np.ndarray) -> dict:
    """Each prediction year's training-majority direction: the more frequent
    class among the eligible rows labelled before that year."""
    out = {}
    s = pd.Series(np.where(eligible, y, np.nan), index=times)
    for year in sorted(set(times.year)):
        past = s[s.index < pd.Timestamp(f'{year}-01-01', tz='UTC')].dropna()
        out[year] = float(past.mean() >= 0.5) if len(past) else 1.0
    return out


# -- frames -------------------------------------------------------------------

def scored_frame(pred: pd.DataFrame, h1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Predictions joined to targets; only eligible, target-defined, finite rows."""
    from src.gold import targets as G
    t = G.build_targets(h1['close'])
    base = pd.DataFrame({'dir': t['dir'], 'eligible': G.eligible(h1.index, 'H1')}, index=h1.index)
    df = pred.set_index('time').join(base, how='inner')
    df = df[df['eligible'] & df['dir'].notna() & df['p_up'].notna()]
    hours = set(cfg['window']['as_of_labels'])
    df['window'] = df.index.hour.isin(hours)
    return df[['p_up', 'dir', 'window']]


def metrics(df: pd.DataFrame, maj: dict, st: dict) -> dict:
    if df.empty:
        return {'n': 0}
    p, y = df['p_up'].to_numpy(), df['dir'].to_numpy()
    h = hits(p, y)
    m = np.array([maj[yr] for yr in df.index.year])
    coin = block_bootstrap_mean(h - 0.5, st['block_rows'], st['n_boot'], st['seed'], st['ci'])
    vs_maj = block_bootstrap_mean(h - (y == m), st['block_rows'], st['n_boot'], st['seed'], st['ci'])
    return {'n': int(len(df)), 'hit': float(h.mean()), 'auc': auc(y, p), 'brier': float(np.mean((p - y) ** 2)),
            'vs_coin': coin, 'vs_majority': vs_maj}


def paired(a: pd.DataFrame, b: pd.DataFrame, st: dict) -> dict:
    """Model a minus model b on their common rows."""
    common = a.index.intersection(b.index)
    if not len(common):
        return {'n': 0}
    ya = a.loc[common, 'dir'].to_numpy()
    d = hits(a.loc[common, 'p_up'].to_numpy(), ya) - hits(b.loc[common, 'p_up'].to_numpy(), ya)
    mean, lo, hi = block_bootstrap_mean(d, st['block_rows'], st['n_boot'], st['seed'], st['ci'])
    return {'n': int(len(common)), 'diff': mean, 'ci': (lo, hi),
            'hit_a': float(hits(a.loc[common, 'p_up'].to_numpy(), ya).mean()),
            'hit_b': float(hits(b.loc[common, 'p_up'].to_numpy(), ya).mean())}


# -- the one cost check ---------------------------------------------------------

def cost_check(df: pd.DataFrame, h1: pd.DataFrame, spread: pd.Series, cfg: dict, floor_pct: float) -> dict:
    """Inside the window: side = sign(p_up - 0.5) at the bar's close, exit at the
    next bar's close, paying the spread (floored)."""
    w = df[df['window'] & (df['p_up'] != 0.5)]
    if w.empty:
        return {'trades': 0}
    close = h1['close']
    pos = h1.index.get_indexer(w.index)
    ok = (pos >= 0) & (pos + 1 < len(h1))
    w, pos = w[ok], pos[ok]
    nxt = h1.index[pos + 1]
    trades = pd.DataFrame({'day': w.index.normalize(), 'side': np.sign(w['p_up'].to_numpy() - 0.5),
                           'entry_price': close.iloc[pos].to_numpy(), 'exit_price': close.iloc[pos + 1].to_numpy(),
                           'entry_spread': spread.reindex(w.index).fillna(0.0).to_numpy(),
                           'exit_spread': spread.reindex(nxt).fillna(0.0).to_numpy()})
    t = trade_returns(trades, floor_pct)
    st = cfg['stats']
    g = t.groupby('day')
    days = pd.DataFrame({'net': g['net'].sum() * 100, 'n': g.size(), 'wins': g['net'].apply(lambda s: (s > 0).sum())})
    rng = np.random.default_rng(st['seed'])
    n, b = len(days), max(1, min(st['block_days'], len(days)))
    k = math.ceil(n / b)
    starts = rng.integers(0, n - b + 1, size=(st['n_boot'], k))
    idx = (starts[:, :, None] + np.arange(b)).reshape(st['n_boot'], -1)[:, :n]
    net, cnt, win = days['net'].to_numpy(), days['n'].to_numpy(float), days['wins'].to_numpy(float)
    per_trade = net[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    win_share = win[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    total = net[idx].sum(axis=1)
    a = (1 - st['ci']) / 2
    q = lambda arr: (float(np.quantile(arr, a)), float(np.quantile(arr, 1 - a)))
    return {'trades': int(len(t)), 'days': int(n),
            'mean_net_pct': float(t['net'].mean() * 100), 'mean_net_ci': q(per_trade),
            'win_share': float((t['net'] > 0).mean()), 'win_share_ci': q(win_share),
            'cum_net_pct': float(t['net'].sum() * 100), 'cum_net_ci': q(total),
            'mean_gross_pct': float(t['gross'].mean() * 100), 'mean_cost_pct': float(t['cost'].mean() * 100)}


# -- report -------------------------------------------------------------------

def _p(v, d=3, pct=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    return f'{v * 100:.{d - 2}f} %' if pct else f'{v:.{d}f}'


def _pp(v):
    return '-' if v is None else f'{v * 100:+.2f} pp'


def _ci(t):
    if not t or t[0] is None:
        return '-'
    return f'[{t[0] * 100:+.2f}, {t[1] * 100:+.2f}]'


def reading(ci) -> str:
    if not ci or ci[0] is None:
        return 'too few rows'
    if ci[0] <= 0 <= ci[1]:
        return 'not distinguishable from zero'
    return ('interval above zero (descriptive; by chance about one interval in twenty excludes zero)'
            if ci[0] > 0 else 'interval below zero (descriptive)')


def render(res: dict, cfg: dict, meta: dict, now: datetime | None = None) -> str:
    L = cfg['label']
    now = now or datetime.now(timezone.utc)
    out = [f'# Gold acceleration and next-hour direction ({L})', '',
           f'> **{L}.** Walk-forward refits on the past only, but every year has been seen and the acceleration '
           'statistics were looked at before this study. No verdict, registration or registry row.', '',
           f'Generated {now:%Y-%m-%d %H:%M} UTC. Inputs: ' +
           ', '.join(f'`{k}` {v["rows"]} rows {v["first"]} .. {v["last"]} (sha256 {v["sha256"][:12]})'
                     for k, v in meta['inputs'].items()) + '.', '',
           f'h1_lstm baseline: {meta["reuse"]["reason"]}.', '',
           'Models and years (`ok` / `failed`): ' +
           '; '.join(f'`{m}` {s}' for m, s in meta['years'].items()) + '.', '']
    for view, title in (('all', 'All hours'), ('window', "The owner's window (as-of labels 14-20, 15:30-23:00 Sofia)")):
        out += [f'## {title} ({L})', '',
                '| model | rows | hit rate | AUC | Brier | hit - coin [95 %] | hit - training majority [95 %] |',
                '|---|---|---|---|---|---|---|']
        for m, r in res['metrics'][view].items():
            if not r.get('n'):
                out.append(f'| `{m}` | 0 | - | - | - | - | - |')
                continue
            c, mj = r['vs_coin'], r['vs_majority']
            out.append(f'| `{m}` | {r["n"]:,} | {_p(r["hit"], 4)} | {_p(r["auc"], 3, False)} | '
                       f'{_p(r["brier"], 4, False)} | {_pp(c[0])} {_ci(c[1:])} | {_pp(mj[0])} {_ci(mj[1:])} |')
        out.append('')
    out += [f'## With acceleration minus without ({L})', '',
            'Each model against its pair on exactly the same rows. A hit-rate difference in percentage points.', '',
            '| model | pair | view | rows | hit | pair hit | difference [95 %] | reading |', '|---|---|---|---|---|---|---|---|']
    for (m, view), r in res['pairs'].items():
        if not r.get('n'):
            continue
        out.append(f'| `{m}` | `{cfg["models"][m]["pair"]}` | {view} | {r["n"]:,} | {_p(r["hit_a"], 4)} | '
                   f'{_p(r["hit_b"], 4)} | {_pp(r["diff"])} {_ci(r["ci"])} | {reading(r["ci"])} |')
    out += ['', f'## Per year, with acceleration minus without ({L})', '',
            '| model | year | view | rows | difference [95 %] | reading |', '|---|---|---|---|---|---|']
    for (m, year, view), r in res['per_year'].items():
        if r.get('n'):
            out.append(f'| `{m}` | {year} | {view} | {r["n"]:,} | {_pp(r["diff"])} {_ci(r["ci"])} | {reading(r["ci"])} |')
    fl, sp = cfg['costs']['floor_round_trip_pct'], cfg['costs']['stress_round_trip_pct']
    out += ['', f'## The one cost check ({L})', '',
            f'Fixed before any result: inside the window, the sign of p_up - 0.5 at a bar close, held one hour, '
            f'paying the real spread (floor {fl} %; stress {sp} %). Days resampled in blocks of '
            f'{cfg["stats"]["block_days"]}. A simulation of past bars, not a trade.', '',
            '| model | trades | mean gross / cost | mean net per trade [95 %] | winning share [95 %] | '
            'cumulative net [95 %] | cumulative net, stress cost |', '|---|---|---|---|---|---|---|']
    for m, r in res['costs'].items():
        if not r['floor'].get('trades'):
            continue
        f, s = r['floor'], r['stress']
        out.append(f'| `{m}` | {f["trades"]:,} | {f["mean_gross_pct"]:+.4f} % / {f["mean_cost_pct"]:.4f} % | '
                   f'{f["mean_net_pct"]:+.4f} % [{f["mean_net_ci"][0]:+.4f}, {f["mean_net_ci"][1]:+.4f}] | '
                   f'{f["win_share"] * 100:.1f} % [{f["win_share_ci"][0] * 100:.1f}, {f["win_share_ci"][1] * 100:.1f}] | '
                   f'{f["cum_net_pct"]:+.1f} % [{f["cum_net_ci"][0]:+.1f}, {f["cum_net_ci"][1]:+.1f}] | '
                   f'{s["cum_net_pct"]:+.1f} % |')
    n_int = sum(1 for r in res['pairs'].values() if r.get('n'))
    out += ['', f'## Reading ({L})', '',
            f'- {n_int} with-minus-without intervals are reported here (plus the per-year ones), so one excluding '
            'zero is expected by chance even with no real effect.',
            '- A hit rate is a share of next-hour directions called right; 50 % is a coin. A difference that is '
            'not distinguishable from zero means acceleration did not change how often the direction was called.',
            '- Nothing here was searched: the features, models, views and the single cost rule were fixed in the '
            'design before any result.', '']
    return '\n'.join(out)


def evaluate(preds: dict, h1: pd.DataFrame, spread: pd.Series, cfg: dict) -> dict:
    from src.gold import targets as G
    st = cfg['stats']
    t = G.build_targets(h1['close'])
    maj = majority_by_year(pd.DatetimeIndex(h1.index), t['dir'], G.eligible(h1.index, 'H1'))
    frames = {m: scored_frame(p, h1, cfg) for m, p in preds.items()}
    res = {'metrics': {'all': {}, 'window': {}}, 'pairs': {}, 'per_year': {}, 'costs': {}}
    for m, df in frames.items():
        res['metrics']['all'][m] = metrics(df, maj, st)
        res['metrics']['window'][m] = metrics(df[df['window']], maj, st)
        res['costs'][m] = {'floor': cost_check(df, h1, spread, cfg, cfg['costs']['floor_round_trip_pct']),
                           'stress': cost_check(df, h1, spread, cfg, cfg['costs']['stress_round_trip_pct'])}
    for m, spec in cfg['models'].items():
        pair = spec['pair']
        if not pair or m not in frames or pair not in frames:
            continue
        a, b = frames[m], frames[pair]
        res['pairs'][(m, 'all')] = paired(a, b, st)
        res['pairs'][(m, 'window')] = paired(a[a['window']], b[b['window']], st)
        for year in sorted(set(a.index.year)):
            ay, by = a[a.index.year == year], b[b.index.year == year]
            res['per_year'][(m, year, 'all')] = paired(ay, by, st)
            res['per_year'][(m, year, 'window')] = paired(ay[ay['window']], by[by['window']], st)
    return res


def _meta(cfg: dict, preds: dict) -> dict:
    import csv
    import json
    manifest = repo_path(cfg['paths']['artifact_manifest'])
    inputs = {}
    with open(manifest, encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            if '/inputs/' in r['path']:
                inputs[os.path.basename(r['path']).split('.')[0]] = r
    years = {}
    for m in cfg['models']:
        root = os.path.join(repo_path(cfg['paths']['models_dir']), m)
        ok, bad = [], []
        if os.path.isdir(root):
            for y in sorted(os.listdir(root)):
                d = os.path.join(root, y, 'done.json')
                if os.path.exists(d):
                    with open(d, encoding='utf-8') as fh:
                        (ok if json.load(fh)['ok'] else bad).append(y)
        years[m] = f'{ok[0]}-{ok[-1]} ok' + (f', failed {",".join(bad)}' if bad else '') if ok else \
            ('failed ' + ','.join(bad) if bad else 'not run')
    return {'inputs': inputs, 'reuse': window_lstm_reuse(cfg), 'years': years}


def run(cfg: dict | None = None, now: datetime | None = None) -> tuple[str, dict]:
    cfg = cfg or load_config()
    frames = pin_inputs(cfg)
    h1 = frames['h1']
    sp = pd.read_csv(repo_path(cfg['paths']['h1_spread']))
    sp['time'] = pd.to_datetime(sp['time'], utc=True)
    spread = sp.set_index('time')['spread_usd']
    preds = {}
    for m in cfg['models']:
        try:
            preds[m] = load_predictions(m, cfg)
        except FileNotFoundError:
            continue
    res = evaluate(preds, h1, spread, cfg)
    md = render(res, cfg, _meta(cfg, preds), now)
    path = _guard(repo_path(cfg['paths']['backtest']), cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(md)
    return md, res


if __name__ == '__main__':
    md, _ = run()
    print(md)
