"""results/fx_bb/walk_forward_report.md (spec fx-bb-reversion/evaluation; design D9).

    python -m src.fx_bb.report

Built only from saved inputs: the registry, labels.json, cell1_events.csv.gz (written when Cell 1
is scored), oos_events.csv.gz and run.json. Two builds differ only in the generation line. A cell's
descriptive section appears only after that cell has its registry row, so nothing is read before
its one-shot verdict. Everything but the verdict table is descriptive. Pips and profit factors are
shown GROSS and after the pair's mean D1 closing-bar spread charged once per event; neither is a
trading result.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_session.score import row_logloss
from . import GROSS_HEADER, load_config
from .registry import CELL_MODEL, CELL_RULE, LABELS_JSON, RUN_JSON, _json, read_registry, results_dir


def wilson(k: float, n: float, z: float = 1.959964) -> tuple:
    if n <= 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def pips(d: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Counter-trend pips from mid close t to mid close t+3, by the pair's pip size."""
    size = d['pair'].map(cfg['pip']).to_numpy(float)
    return d['dir'].to_numpy(float) * (d['close_t3'].to_numpy(float) - d['close'].to_numpy(float)) / size


def _pf(x: np.ndarray) -> float:
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float('inf')


def stats(d: pd.DataFrame, cfg: dict) -> dict:
    if not len(d):
        return {'events': 0}
    g = pips(d, cfg)
    net = g - d['spread_pips'].to_numpy(float)
    win = d['r'].to_numpy(float) > 0
    n_dates = int(d['date'].nunique())
    lo, hi = wilson(win.mean() * n_dates, n_dates)
    return {'events': len(d), 'dates': n_dates, 'win': float(win.mean()), 'lo': lo, 'hi': hi,
            'mean_r': float(d['r'].mean()), 'mean_pips': float(g.mean()), 'total_pips': float(g.sum()),
            'pf': _pf(g), 'mean_net': float(net.mean()), 'total_net': float(net.sum()), 'pf_net': _pf(net)}


STAT_HEAD = ('| Events | Dates | Win rate | Wilson 95 % (dates) | Mean r (ATR) | Mean pips | Total pips | PF | '
             'Mean pips after spread | Total after spread | PF after spread |')
STAT_SEP = '|---|---|---|---|---|---|---|---|---|---|---|'


def _f(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def stat_cells(s: dict) -> str:
    if not s.get('events'):
        return '| 0 | 0 | — | — | — | — | — | — | — | — | — |'
    return (f"| {s['events']} | {s['dates']} | {s['win']:.1%} | {s['lo']:.1%}–{s['hi']:.1%} | {s['mean_r']:+.3f} | "
            f"{s['mean_pips']:+.1f} | {s['total_pips']:+.0f} | {_f(s['pf'])} | {s['mean_net']:+.1f} | "
            f"{s['total_net']:+.0f} | {_f(s['pf_net'])} |")


def breakdown(d: pd.DataFrame, cfg: dict, title: str) -> list:
    L = [f'### {title}', '', '| Group ' + STAT_HEAD, '|---' + STAT_SEP]
    L.append('| pooled ' + stat_cells(stats(d, cfg)))
    for p in cfg['pairs']:
        L.append(f'| {p} ' + stat_cells(stats(d[d['pair'] == p], cfg)))
    for side, name in ((1, 'upper-band breach (sell)'), (-1, 'lower-band breach (buy)')):
        L.append(f'| {name} ' + stat_cells(stats(d[d['side'] == side], cfg)))
    for yr, g in d.groupby(d['date'].dt.year):
        L.append(f'| {yr} ' + stat_cells(stats(g, cfg)))
    return L + ['']


def reliability(y, p, bins=(0, .4, .45, .5, .55, .6, 1.0)) -> list:
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a)
        out.append((f'{a:.2f}–{b:.2f}', int(m.sum()), float(p[m].mean()) if m.any() else np.nan,
                    float(y[m].mean()) if m.any() else np.nan))
    return out


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    sc = cfg['scoring']
    reg = {r['hypothesis']: r for r in read_registry(cfg)}
    lab = _json(cfg, LABELS_JSON)
    L = ['# fx-bb-reversion walk-forward and rule report', '', f'> **{GROSS_HEADER}**', '',
         f"Rule: the first D1 mid close outside the {cfg['bands']['length']}-day, {cfg['bands']['width']:g}-SD Bollinger "
         f"Bands (D1 closes at {cfg['data']['day_close_hour']}:00 New York), counter-trend, outcome r over "
         f"{cfg['horizon_days']} valid days in ATR({cfg['atr_period']}) units, {cfg['cooldown_days']}-day cooldown. "
         f"Six pairs: {', '.join(cfg['pairs'])} (USDCHF from {cfg['data']['start']['USDCHF']}).", '', '## Verdicts', '']
    if reg:
        L += [f"alpha = {sc['alpha']} (0.05/{sc['family_size']}) per cell, one-sided circular block bootstrap over business "
              f"days, {sc['block_days']}-day blocks, {sc['resamples']} resamples, the same blocks for the daily sums and "
              'counts. Cell 1 is KEEP when the alpha-quantile is above 0; Cell 2 when the (1 − alpha)-quantile is below 0.',
              '', '| Cell | Statistic | Events | Dates | Window | Value | Bound | Quantile | mde_80 | Verdict |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for h in (CELL_RULE, CELL_MODEL):
            if h in reg:
                r = reg[h]
                L.append(f"| {h} | {r['statistic_name']} | {int(r['n_events']):,} | {int(r['n_dates']):,} | "
                         f"{r['first_event']}..{r['last_event']} | {float(r['statistic']):+.5f} | {float(r['bound']):+.5f} | "
                         f"{float(r['bound_quantile']):.3f} | {float(r['mde_80']):.5f} | **{r['verdict']}** |")
            else:
                L.append(f'| {h} | — | — | — | — | — | — | — | — | Not scored yet |')
    else:
        L.append('Not scored yet: run `python -m src.fx_bb.score --cell rule` and `--cell model`, once each.')

    L += ['', '## Power (from labels.md, before any model)', '']
    if lab:
        p1, p2 = lab['power']['pooled']['cell1'], lab['power']['pooled']['cell2']
        L += ['| Cell | Events | Distinct dates | MDE on events | MDE on dates |', '|---|---|---|---|---|',
              f"| Cell 1, mean r (ATR) | {p1['eligible']} | {p1['dates']} | {p1['mde_events']:.3f} | {p1['mde_dates']:.3f} |",
              f"| Cell 2, accuracy | {p2['targets']} | {p2['target_dates']} | {p2['mde_events']:.2%} | {p2['mde_dates']:.2%} |"]
    else:
        L.append('No label report yet: run `python -m src.fx_bb.labels`.')

    L += ['', '## Cell 1 — every eligible event (descriptive)', '']
    c1 = os.path.join(results_dir(cfg), 'cell1_events.csv.gz')
    if CELL_RULE in reg and os.path.exists(c1):
        from .score import load_cell1
        ev = load_cell1(cfg)
        mid, bid = ev[ev['price'] == 'mid'], ev[ev['price'] == 'bid']
        L += ['Win = r > 0. Pips: counter-trend move of the mid close from t to t+3 (pip 0.01 for USDJPY, 0.0001 '
              'otherwise). After spread: minus the pair\'s mean D1 closing-bar spread over its event period, charged '
              'once per event. Wilson intervals count distinct dates, not events.', '',
              '| Pair | Mean closing spread (pips) |', '|---|---|']
        L += [f'| {p} | {mid.loc[mid["pair"] == p, "spread_pips"].iloc[0]:.2f} |' for p in cfg['pairs']
              if (mid['pair'] == p).any()]
        L += [''] + breakdown(mid, cfg, 'Mid closes (the registered price)')
        r = reg[CELL_RULE]
        L += ['### Mid versus bid', '',
              f"Mid statistic {float(r['statistic']):+.4f} ATR on {int(r['n_events'])} events; bid statistic "
              f"{float(r['bid_statistic']):+.4f} ATR on {int(r['bid_n_events'])} events (bands, ATR and outcomes from bid "
              f"closes); gap {float(r['parity_gap']):.4f} ATR, tolerance {sc['parity_tolerance_atr']} — "
              f"{'FLAGGED (recorded; the verdict follows the mid test)' if r['parity_flag'] else 'within tolerance'}.", '']
        L += breakdown(bid, cfg, 'Bid closes (parity only)')
    else:
        L.append('Not shown until Cell 1 is scored.')

    L += ['', '## Cell 2 — pooled LightGBM out of sample (descriptive)', '']
    oos_path = os.path.join(results_dir(cfg), 'oos_events.csv.gz')
    run = _json(cfg, RUN_JSON)
    if CELL_MODEL in reg and os.path.exists(oos_path) and run:
        from .walk_forward import load_oos
        oos = load_oos(cfg).sort_values(['date', 'pair'], kind='mergesort')
        y = oos['y'].to_numpy(float)
        L += [f"Walk-forward run {run['run_at']}, config `{run['config_digest'][:12]}`, {len(run['folds'])} folds, "
              f"{len(oos):,} out-of-sample events.", '',
              '| Fold | Start | Fit | Val | Test | Base rate | Trees | Log-loss base | Log-loss LightGBM | Brier base | Brier LightGBM |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for f in run['folds']:
            d = oos[oos['fold'] == f['k']]
            yy = d['y'].to_numpy(float)
            pp, pl = d['p_prior'].to_numpy(float), d['p_lightgbm'].to_numpy(float)
            L.append(f"| {f['k']} | {f['start']} | {f['n_fit']} | {f['n_val']} | {f['n_eval']} | {f['base_rate']:.3f} | "
                     f"{f['best_iteration']} | {row_logloss(yy, pp).mean():.5f} | {row_logloss(yy, pl).mean():.5f} | "
                     f"{np.mean((pp - yy) ** 2):.5f} | {np.mean((pl - yy) ** 2):.5f} |")
        pl = oos['p_lightgbm'].to_numpy(float)
        L += ['', '### Reliability (LightGBM P(win) vs the observed win rate)', '',
              '| P(win) bin | Events | Mean P(win) | Observed |', '|---|---|---|---|']
        L += [f'| {b} | {n} | {_f(m)} | {_f(o)} |' for b, n, m, o in reliability(y, pl)]
        L += ['', '### All out-of-sample events and the filter', '', '| Group ' + STAT_HEAD, '|---' + STAT_SEP,
              '| every out-of-sample event ' + stat_cells(stats(oos, cfg)),
              '| P(win) > 0.5 ' + stat_cells(stats(oos[pl > 0.5], cfg)),
              '| P(win) ≤ 0.5 ' + stat_cells(stats(oos[pl <= 0.5], cfg)), '']
    else:
        L.append('Not shown until Cell 2 is scored.')
    L += ['', 'Only the registry rows are verdicts; every other number is descriptive and none is a trading result.',
          '', f'_Generated {now_iso}._', '']
    path = os.path.join(results_dir(cfg), 'walk_forward_report.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L))
    return path


def main(argv=None):
    argparse.ArgumentParser(description='write the fx-bb report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
