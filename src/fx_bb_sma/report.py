"""results/fx_bb_sma/walk_forward_report.md (spec fx-bb-sma-exit/trade-report; design D10).

    python -m src.fx_bb_sma.report

Built only from saved files (registry, labels.json, cell1_trades.csv.gz, oos_trades.csv.gz, run.json);
two builds differ only in the generation line. A cell's numbers appear only after its verdict.
The plain-language section is in Bulgarian for the owner; everything but the registry rows is
descriptive. Net of the recorded bar spread only: not a trading result.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..fx_bb.report import reliability, wilson
from ..fx_session.score import row_logloss
from . import COST_HEADER, load_config
from .registry import CELL_MODEL, CELL_RULE, LABELS_JSON, RUN_JSON, read_json, read_registry, results_dir

REASONS = ('take profit', 'stop loss', 'time stop')
BG_REASON = {'take profit': 'цел (SMA20)', 'stop loss': 'стоп', 'time stop': 'време (10 дни)'}


def pips(d: pd.DataFrame, cfg: dict) -> np.ndarray:
    size = d['pair'].map(cfg['pip']).to_numpy(float)
    return d['dir'].to_numpy(float) * (d['exit_price'].to_numpy(float) - d['close'].to_numpy(float)) / size


def _pf(x) -> float:
    x = np.asarray(x, dtype=float)
    loss = -x[x < 0].sum()
    return float(x[x > 0].sum() / loss) if loss > 0 else float('inf')


def _f(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def summary(d: pd.DataFrame) -> dict:
    """The plain-language numbers. A win is a trade whose NET result is above zero."""
    net, gross = d['net_pips'].to_numpy(float), d['gross_pips'].to_numpy(float)
    share = d['exit_reason'].value_counts(normalize=True)
    return {'n': len(d), **{r: float(share.get(r, 0.0)) for r in REASONS},
            'win': float((net > 0).mean()), 'avg_win_net': float(net[net > 0].mean()) if (net > 0).any() else np.nan,
            'avg_loss_net': float(net[net <= 0].mean()) if (net <= 0).any() else np.nan,
            'avg_win_gross': float(gross[gross > 0].mean()) if (gross > 0).any() else np.nan,
            'avg_loss_gross': float(gross[gross <= 0].mean()) if (gross <= 0).any() else np.nan,
            'pf_gross': _pf(gross), 'pf_net': _pf(net), 'r_gross': float(d['r_gross'].sum()),
            'r_net': float(d['r_net'].sum()), 'mean_r_net': float(d['r_net'].mean()),
            'pips_net': float(net.sum()), 'pips_gross': float(gross.sum())}


def examples(d: pd.DataFrame) -> pd.DataFrame:
    """Five trades by a fixed rule, never by result: the latest take profit, stop and time stop, then the
    latest long and the latest short not already listed."""
    d = d.sort_values(['date', 'pair'], kind='mergesort')
    picked = []
    for r in REASONS:
        g = d[(d['exit_reason'] == r) & ~d.index.isin(picked)]
        if len(g):
            picked.append(g.index[-1])
    for side in (1, -1):
        g = d[(d['dir'] == side) & ~d.index.isin(picked)]
        if len(g) and len(picked) < 5:
            picked.append(g.index[-1])
    for i in d.index[::-1]:
        if len(picked) >= 5:
            break
        if i not in picked:
            picked.append(i)
    return d.loc[picked]


def group_row(name: str, g: pd.DataFrame) -> str:
    if not len(g):
        return f'| {name} | 0 |' + ' — |' * 9
    s = summary(g)
    n_dates = int(g['date'].nunique())
    lo, hi = wilson(s['win'] * n_dates, n_dates)
    return (f"| {name} | {s['n']} | {s['take profit']:.0%} | {s['stop loss']:.0%} | {s['time stop']:.0%} | {s['win']:.1%} | "
            f"{lo:.1%}–{hi:.1%} | {s['mean_r_net']:+.3f} | {g['net_pips'].mean():+.1f} | {_f(s['pf_net'])} | "
            f"{g['hold_hours'].mean() / 24:.1f} |")


GROUP_HEAD = ('| Group | Trades | Take profit | Stop | Time stop | Win rate (net) | Wilson 95 % (dates) | Mean net R | '
              'Mean net pips | PF net | Mean hold (calendar days) |')
GROUP_SEP = '|---|---|---|---|---|---|---|---|---|---|---|'


def build(cfg: dict | None = None, now_iso: str | None = None) -> str:
    cfg = cfg or load_config()
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    sc = cfg['scoring']
    reg = {r['hypothesis']: r for r in read_registry(cfg)}
    lab = read_json(cfg, LABELS_JSON)
    L = ['# fx-bb-sma-exit report', '', f'> **{COST_HEADER}**', '',
         'Trade: entry at the D1 mid close of a first breach of the 20-day, 2-SD Bollinger Bands (D1 closes at 17:00 '
         'New York); take profit at SMA20 of the entry day; stop at the same distance beyond the entry; exits checked '
         'hour by hour on H1 bars, a bar touching both counts as the stop; time stop at the close of the 10th trading '
         'day. One open trade per pair. The events were already seen in fx-bb-reversion (DROP, 2026-11-22).', '',
         '## Verdicts', '']
    if reg:
        L += [f"alpha = {sc['alpha']} (0.05/{sc['family_size']}) per cell; one-sided circular block bootstrap over entry "
              f"business days, {sc['block_days']}-day blocks, {sc['resamples']} resamples.", '',
              '| Cell | Statistic | Trades | Dates | Window | Value | Bound | Quantile | mde_80 | Verdict |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for h in (CELL_RULE, CELL_MODEL):
            if h in reg:
                r = reg[h]
                L.append(f"| {h} | {r['statistic_name']} | {int(r['n_trades']):,} | {int(r['n_dates']):,} | "
                         f"{r['first_entry']}..{r['last_entry']} | {float(r['statistic']):+.5f} | {float(r['bound']):+.5f} | "
                         f"{float(r['bound_quantile']):.3f} | {float(r['mde_80']):.5f} | **{r['verdict']}** |")
            else:
                L.append(f'| {h} | — | — | — | — | — | — | — | — | Not scored yet |')
    else:
        L.append('Not scored yet: run `python -m src.fx_bb_sma.score --cell rule` and `--cell model`, once each.')

    L += ['', '## Power (from labels.md, before any model)', '']
    if lab:
        p1, p2 = lab['power']['pooled']['cell1'], lab['power']['pooled']['cell2']
        L += ['| Cell | Trades | Distinct dates | MDE on trades | MDE on dates |', '|---|---|---|---|---|',
              f"| Cell 1, mean net R | {p1['eligible']} | {p1['dates']} | {p1['mde_trades']:.3f} R | {p1['mde_dates']:.3f} R |",
              f"| Cell 2, accuracy | {p2['eligible']} | {p2['dates']} | {p2['mde_trades']:.2%} | {p2['mde_dates']:.2%} |"]
    else:
        L.append('No label report yet: run `python -m src.fx_bb_sma.labels`.')

    c1 = os.path.join(results_dir(cfg), 'cell1_trades.csv.gz')
    if CELL_RULE in reg and os.path.exists(c1):
        from .score import load_cell1
        d = load_cell1(cfg)
        s = summary(d)
        r = reg[CELL_RULE]
        L += ['', '## На прост език: как завършиха сделките', '',
              'Всяка сделка се отваря, когато дневното затваряне излезе извън лентите на Болинджър. Целта е средната '
              'линия (SMA20) от деня на входа. Стопът е на същото разстояние в обратната посока. Ако за 10 търговски '
              'дни не се стигне нито до целта, нито до стопа, сделката се затваря в края на 10-ия ден. Ако в един и '
              'същи час цената докосне и двете нива, броим стоп. Резултатите са след спреда, но без slippage, '
              'суап и комисиона. Това е симулация върху историята, не резултат от реална търговия.', '',
              '| Показател | Стойност |', '|---|---|',
              f"| Сделки | {s['n']:,} |",
              f"| Затворени на целта (SMA20) | {s['take profit']:.1%} |",
              f"| Затворени на стопа | {s['stop loss']:.1%} |",
              f"| Затворени по време (10 дни) | {s['time stop']:.1%} |",
              f"| Печеливши след спреда | {s['win']:.1%} |",
              f"| Средна печалба (пипса, след спреда / преди) | {s['avg_win_net']:+.1f} / {s['avg_win_gross']:+.1f} |",
              f"| Средна загуба (пипса, след спреда / преди) | {s['avg_loss_net']:+.1f} / {s['avg_loss_gross']:+.1f} |",
              f"| Profit Factor преди спреда | {_f(s['pf_gross'])} |",
              f"| Profit Factor след спреда | {_f(s['pf_net'])} |",
              f"| Общ резултат в R, преди / след спреда | {s['r_gross']:+.1f} R / {s['r_net']:+.1f} R |",
              f"| Средно на сделка, след спреда | {s['mean_r_net']:+.3f} R |",
              f"| Общо пипса, преди / след спреда | {s['pips_gross']:+,.0f} / {s['pips_net']:+,.0f} |",
              f"| Решение на теста (Клетка 1) | **{r['verdict']}** |", '',
              '1 R е разстоянието от входа до стопа. Profit Factor е сборът на печалбите, разделен на сбора на '
              'загубите; под 1 значи, че загубите са повече от печалбите.', '',
              '### Пет примерни сделки', '',
              'Избрани по фиксирано правило, без да гледаме резултата: последната сделка на целта, на стопа и по '
              'време, после последната дълга и последната къса, ако още ги няма.', '',
              '| Двойка | Посока | Вход (дата) | Цена вход | Цел (TP) | Стоп (SL) | Изход (дата) | Причина | Пипса преди / след спреда |',
              '|---|---|---|---|---|---|---|---|---|']
        for _, e in examples(d).iterrows():
            nd = 3 if e['pair'] == 'USDJPY' else 5
            L.append(f"| {e['pair']} | {'покупка' if e['dir'] > 0 else 'продажба'} | {e['date'].date()} | {e['close']:.{nd}f} | "
                     f"{e['tp']:.{nd}f} | {e['sl']:.{nd}f} | {e['exit_date'].date()} | {BG_REASON[e['exit_reason']]} | "
                     f"{e['gross_pips']:+.1f} / {e['net_pips']:+.1f} |")
        L += ['', '## Cell 1 breakdowns (descriptive)', '',
              f"Same-bar conflicts counted as stops: {int(d['conflict'].sum())} trades. Bid/ask control (long in at the ask "
              f"and out on the bid, short mirrored): mean {float(r['ctrl_statistic']):+.4f} R against the registered "
              f"{float(r['statistic']):+.4f} R; gap {float(r['parity_gap']):.4f}, tolerance {sc['parity_tolerance_r']} — "
              f"{'FLAGGED (the verdict follows the mid result)' if r['parity_flag'] else 'within tolerance'}. Win = net result "
              '> 0. Wilson intervals count distinct entry dates.', '', GROUP_HEAD, GROUP_SEP, group_row('pooled', d)]
        L += [group_row(p, d[d['pair'] == p]) for p in cfg['pairs']]
        L += [group_row('long (lower-band breach)', d[d['dir'] > 0]), group_row('short (upper-band breach)', d[d['dir'] < 0])]
        L += [group_row(str(yr), g) for yr, g in d.groupby(d['date'].dt.year)]
        L += ['', '**Net R by exit reason**', '', '| Exit | Trades | Mean gross R | Mean net R | Mean net pips |',
              '|---|---|---|---|---|']
        for reason in REASONS:
            g = d[d['exit_reason'] == reason]
            if len(g):
                L.append(f"| {reason} | {len(g)} | {g['r_gross'].mean():+.3f} | {g['r_net'].mean():+.3f} | "
                         f"{g['net_pips'].mean():+.1f} |")
    else:
        L += ['', '## На прост език: как завършиха сделките', '', 'Not shown until Cell 1 is scored.']

    L += ['', '## Cell 2 — pooled LightGBM out of sample (descriptive)', '']
    run = read_json(cfg, RUN_JSON)
    if CELL_MODEL in reg and run and os.path.exists(os.path.join(results_dir(cfg), 'oos_trades.csv.gz')):
        from .walk_forward import load_oos
        oos = load_oos(cfg).sort_values(['date', 'pair'], kind='mergesort')
        y = oos['y'].to_numpy(float)
        L += [f"Walk-forward run {run['run_at']}, config `{run['config_digest'][:12]}`, {len(run['folds'])} folds, "
              f"{len(oos):,} out-of-sample trades.", '',
              '| Fold | Start | Fit | Val | Test | Base rate | Trees | Log-loss base | Log-loss LightGBM | Brier base | Brier LightGBM |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for f in run['folds']:
            g = oos[oos['fold'] == f['k']]
            yy, pp, pl = g['y'].to_numpy(float), g['p_prior'].to_numpy(float), g['p_lightgbm'].to_numpy(float)
            L.append(f"| {f['k']} | {f['start']} | {f['n_fit']} | {f['n_val']} | {f['n_eval']} | {f['base_rate']:.3f} | "
                     f"{f['best_iteration']} | {row_logloss(yy, pp).mean():.5f} | {row_logloss(yy, pl).mean():.5f} | "
                     f"{np.mean((pp - yy) ** 2):.5f} | {np.mean((pl - yy) ** 2):.5f} |")
        pl = oos['p_lightgbm'].to_numpy(float)
        L += ['', '### Reliability (LightGBM P(win) vs the observed net win rate)', '',
              '| P(win) bin | Trades | Mean P(win) | Observed |', '|---|---|---|---|']
        L += [f'| {b} | {n} | {_f(m)} | {_f(o)} |' for b, n, m, o in reliability(y, pl)]
        L += ['', '### All out-of-sample trades and the filter', '', GROUP_HEAD, GROUP_SEP]
        oos = oos.assign(hold_hours=np.nan, conflict=False)
        for name, g in (('every out-of-sample trade', oos), ('P(win) > 0.5', oos[pl > 0.5]),
                        ('P(win) ≤ 0.5', oos[pl <= 0.5])):
            L.append(group_row(name, g).rsplit('|', 2)[0] + '| — |')
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
    argparse.ArgumentParser(description='write the fx-bb-sma report').parse_args(argv)
    print(build())


if __name__ == '__main__':
    main()
