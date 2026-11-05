"""The descriptive window backtest (spec window-strategy: "The backtest report is
descriptive"; design D5).

For each rule: its gated daily series net of cost and at the stress cost, the
same rule with its gate always open, always long from 16:00 to 23:00 Sofia on
the same days, a time-shift placebo of the gate, a 95 % block-bootstrap
interval, and a per-year table. Written to results/gold/window/backtest.md.

DESCRIPTIVE ONLY: the rules were chosen after these years were seen. Nothing
here is a verdict; no registry is written.

    python -m src.gold_window.backtest
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.forecast_eval.stats import block_bootstrap_ci

from . import load_config, repo_path
from . import bars as B
from . import rules as R
from . import simulate as S
from . import walkforward as WF

LABEL = ('_Descriptive: rules chosen after seeing these years, simulated, net of the stated costs. '
         'Not evidence of an edge._')
PLACEBO_MARGIN = 20


# ── statistics ──────────────────────────────────────────────────────────────

def summary(daily: pd.Series, trades: pd.DataFrame, cfg: dict) -> dict:
    st = cfg['stats']
    x = daily.to_numpy(float) * 100
    lo, hi, mean = block_bootstrap_ci(x, st['block'], 1 - st['ci'], n_boot=st['n_boot'], seed=st['seed'])
    eq = np.cumprod(1 + daily.to_numpy(float))
    dd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100) if len(eq) else float('nan')
    return {'days': int(len(daily)), 'trades': int(len(trades)),
            'mean_pct': mean, 'ci_lo': lo, 'ci_hi': hi,
            'total_pct': float((eq[-1] - 1) * 100) if len(eq) else float('nan'),
            'max_dd_pct': dd,
            'hit_pct': float((trades['net'] > 0).mean() * 100) if len(trades) else float('nan')}


def per_year(daily: pd.Series, trades: pd.DataFrame) -> pd.DataFrame:
    yrs = daily.groupby(daily.index.year)
    out = pd.DataFrame({'total %': yrs.apply(lambda v: (np.prod(1 + v.to_numpy()) - 1) * 100),
                        'days': yrs.size()})
    tc = trades.groupby(trades['day'].dt.year).size() if len(trades) else pd.Series(dtype=int)
    out['trades'] = tc.reindex(out.index).fillna(0).astype(int)
    return out


def placebo_share(candidates: pd.DataFrame, n_days: int, margin: int = PLACEBO_MARGIN) -> float:
    """Share of time-shifted gates (offsets margin .. N - margin over the
    candidate rows) whose mean daily net is at least the real gate's."""
    c = candidates.sort_values('entry_time')
    r = c['net'].to_numpy(float)
    g = c['gate_open'].to_numpy(bool).astype(float)
    n = len(r)
    if n <= 2 * margin or n_days == 0:
        return float('nan')
    real = float(g @ r) / n_days
    fake = np.array([np.roll(g, k) @ r for k in range(margin, n - margin + 1)]) / n_days
    return float(np.mean(fake >= real - 1e-15))


def always_long(h1: pd.DataFrame, days: pd.DatetimeIndex, entry_label: int = 14, exit_label: int = 21) -> pd.DataFrame:
    """Long from the close of `entry_label` (16:00 Sofia) to the close of `exit_label` (23:00 Sofia)."""
    h1i = h1.set_index('time')
    rows = []
    for day in pd.DatetimeIndex(days):
        a, b = R._bar(h1i, day, entry_label), R._bar(h1i, day, exit_label)
        if a is None or b is None:
            continue
        rows.append(['always_long', day, day + pd.Timedelta(hours=entry_label + 1), a[0], a[1], 1,
                     day + pd.Timedelta(hours=exit_label + 1), b[0], b[1], np.nan, np.nan, True])
    return pd.DataFrame(rows, columns=R.TRADE_COLUMNS)


def evaluate(name: str, candidates: pd.DataFrame, days: pd.DatetimeIndex, baseline: pd.DataFrame, cfg: dict) -> dict:
    """Everything the report shows for one rule, on `days`."""
    fl, sp = cfg['costs']['floor_round_trip_pct'], cfg['costs']['stress_round_trip_pct']
    cand = candidates[candidates['day'].isin(set(days))]
    gated = S.simulate(cand, days, fl, sp)
    ungated = S.simulate(cand, days, fl, sp, gate=False)
    base = S.simulate(baseline[baseline['day'].isin(set(days))], days, fl, sp)
    all_net = S.trade_returns(cand, fl)
    return {'name': name, 'span': (days.min(), days.max()) if len(days) else (None, None),
            'net': summary(gated['net'], gated['trades'], cfg),
            'stress': summary(gated['stress'], gated['trades'], cfg),
            'gross': summary(gated['gross'], gated['trades'], cfg),
            'ungated': summary(ungated['net'], ungated['trades'], cfg),
            'always_long': summary(base['net'], base['trades'], cfg),
            'placebo_share': placebo_share(all_net, len(days)),
            'per_year': per_year(gated['net'], gated['trades']),
            'trades': gated['trades']}


def rule_days(candidates: pd.DataFrame, eligible: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Eligible days from the first day whose gate has a threshold (after warm-up)."""
    have = candidates[np.isfinite(candidates['threshold'].astype(float))]
    if have.empty:
        return pd.DatetimeIndex([])
    first, last = have['day'].min(), candidates['day'].max()
    e = pd.DatetimeIndex(eligible)
    return e[(e >= first) & (e <= last)]


# ── report ─────────────────────────────────────────────────────────────────

def _f(v, fmt='{:+.3f}'):
    return fmt.format(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else '–'


def render(results: list, cfg: dict, meta: dict, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    c = cfg['costs']
    out = ['# Gold window backtest', '',
           f'Generated {now.strftime("%Y-%m-%d %H:%M UTC")} by `python -m src.gold_window.backtest`.', '',
           LABEL, '',
           '- Every trade opens and closes inside 15:30–23:00 Sofia (labels 14:30–22:00 Berlin); nothing is '
           'held overnight, so no swap applies.',
           f'- Prices are MT5 bid closes. A long entry pays its entry bar\'s MT5 spread, a short exit pays its exit '
           f'bar\'s; every round trip costs at least {c["floor_round_trip_pct"]} % of the entry price '
           f'(stress: {c["stress_round_trip_pct"]} %).',
           '- Models are refit each calendar year on the past only (walk-forward); gates compare a forecast with '
           'its own trailing 250 trading days.',
           f'- Inputs: {meta.get("inputs", "")}', '',
           '## Summary', '', LABEL, '',
           '| rule | span | days | trades | mean net %/day | 95 % interval | total net % | total stress % | '
           'max drawdown % | hit % | gate always open: total % | always long 16–23: total % | placebo share |',
           '|---' * 13 + '|']
    for r in results:
        n, s = r['net'], r['stress']
        span = f'{r["span"][0]:%Y-%m-%d} .. {r["span"][1]:%Y-%m-%d}' if r['span'][0] is not None else '–'
        out.append(f'| `{r["name"]}` | {span} | {n["days"]} | {n["trades"]} | {_f(n["mean_pct"], "{:+.4f}")} | '
                   f'[{_f(n["ci_lo"], "{:+.4f}")}, {_f(n["ci_hi"], "{:+.4f}")}] | {_f(n["total_pct"], "{:+.1f}")} | '
                   f'{_f(s["total_pct"], "{:+.1f}")} | {_f(n["max_dd_pct"], "{:.1f}")} | {_f(n["hit_pct"], "{:.1f}")} | '
                   f'{_f(r["ungated"]["total_pct"], "{:+.1f}")} | {_f(r["always_long"]["total_pct"], "{:+.1f}")} | '
                   f'{_f(r["placebo_share"], "{:.2f}")} |')
    out += ['', 'The placebo share is the fraction of time-shifted gates (the same gate decisions moved along the '
            'candidate trades) whose mean daily net is at least the real gate\'s: near 0 means the timing of the '
            'gate mattered in these years, near 0.5 or above means it did not.', '']
    for r in results:
        g = r['gross']
        out += [f'## `{r["name"]}`', '', LABEL, '',
                f'Gross (before cost) total {_f(g["total_pct"], "{:+.1f}")} %, mean {_f(g["mean_pct"], "{:+.4f}")} '
                f'%/day. Gate always open: {r["ungated"]["trades"]} trades, total '
                f'{_f(r["ungated"]["total_pct"], "{:+.1f}")} %, mean {_f(r["ungated"]["mean_pct"], "{:+.4f}")} %/day.',
                '', '| year | total net % | days | trades |', '|---|---|---|---|']
        for y, row in r['per_year'].iterrows():
            out.append(f'| {y} | {_f(row["total %"], "{:+.1f}")} | {int(row["days"])} | {int(row["trades"])} |')
        out.append('')
    return '\n'.join(out)


# ── orchestration ──────────────────────────────────────────────────────────

def run(cfg: dict | None = None, now: datetime | None = None) -> tuple[str, list]:
    cfg = cfg or load_config()
    h1, m15, d1 = B.load_h1(cfg), B.load_m15(cfg), B.load_d1()
    p_h1 = WF.load_predictions('h1_lstm', cfg)
    p_d1 = WF.load_predictions('d1_lstm', cfg)
    p_g = WF.load_predictions('d1_garch', cfg)
    look, rl = cfg['rules']['gate_lookback_days'], cfg['rules']
    h1_days = R.eligible_days_h1(h1, cfg['window']['h1_labels'])
    m15_days = R.eligible_days_m15(m15, cfg['window']['first_m15_label'], cfg['window']['last_m15_label'])
    cal = pd.DatetimeIndex(sorted(d1.loc[d1['time'].dt.dayofweek < 5, 'time'].dt.normalize().unique()))
    gate_g = R.daily_gate(p_g, cal, rl['continuation']['gate_quantile'], look)
    gate_b = R.daily_gate(p_g, cal, rl['us_open_breakout']['gate_quantile'], look)

    def within(days, pred):
        lo, hi = pred['time'].min().normalize(), pred['time'].max().normalize() + pd.Timedelta(days=1)
        return days[(days >= lo) & (days <= hi)]

    specs = [
        ('h1_gated_direction', R.h1_gated_direction(h1, p_h1, within(h1_days, p_h1), cfg), h1_days),
        ('continuation', R.continuation(h1, gate_g, within(h1_days, p_g), cfg), h1_days),
        ('us_open_breakout', R.us_open_breakout(m15, gate_b, within(m15_days, p_g), cfg), m15_days),
        ('d1_gated_direction', R.d1_gated_direction(h1, p_d1, within(h1_days, p_d1), cfg), h1_days),
    ]
    results = []
    for name, cand, elig in specs:
        days = rule_days(cand, elig)
        base = always_long(h1, days)
        results.append(evaluate(name, cand, days, base, cfg))
    meta = {'inputs': f'H1 cache to {h1["time"].max():%Y-%m-%d %H:%M}, MT5 M15 from {m15["time"].min():%Y-%m-%d}, '
                      f'walk-forward predictions h1_lstm {p_h1["time"].min():%Y} on, d1_lstm and d1_garch '
                      f'{p_d1["time"].min():%Y} on.'}
    text = render(results, cfg, meta, now)
    path = repo_path(cfg['paths']['backtest'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    return path, results


if __name__ == '__main__':
    p, res = run()
    print(p)
    for r in res:
        n = r['net']
        print(f'{r["name"]:20} trades {n["trades"]:5d} total {n["total_pct"]:+7.1f}% '
              f'mean {n["mean_pct"]:+.4f} [{n["ci_lo"]:+.4f}, {n["ci_hi"]:+.4f}] placebo {r["placebo_share"]:.2f}')
