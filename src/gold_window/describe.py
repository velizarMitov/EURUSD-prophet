"""Descriptive statistics of gold on the seen history (spec window-strategy:
"A reproducible descriptive report on the seen history").

Reads the MT5 caches results/gold/gold_{h1,d1}.csv and writes
results/gold/window/DESCRIPTIVE.md. Every table is DESCRIPTIVE ONLY: these
years have all been looked at, so nothing here is evidence for a rule.

Clock: labels are Berlin wall time; Sofia = Berlin + 1 h all year, so an H1
bar labelled h (Berlin) closes at h + 2 Sofia.

    python -m src.gold_window.describe
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import load_config, repo_path

SPANS = [('2015-18', '2015-01-01', '2019-01-01'),
         ('2019-22.10', '2019-01-01', '2022-10-30'),
         ('2022.11-26', '2022-10-30', '2027-01-01')]
SESSIONS = [('Asia 01-09', range(2, 10)), ('London 09-16', range(10, 17)),
            ('window 16-23', range(17, 24)), ('late 23-01', (0, 1))]
WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
LABEL = '_Descriptive only: these years have been looked at, so nothing here is evidence for a rule._'


def _span_of(times: pd.Series) -> pd.Series:
    edges = [pd.Timestamp(s, tz='UTC') for _, s, _ in SPANS] + [pd.Timestamp(SPANS[-1][2], tz='UTC')]
    return pd.cut(times, edges, labels=[n for n, _, _ in SPANS], right=False)


def hourly_returns(h1: pd.DataFrame) -> pd.DataFrame:
    """H1 log returns close to close, flagged when the bar follows the weekend."""
    h = h1.sort_values('time').reset_index(drop=True)
    h['lr'] = np.log(h['close']).diff()
    h['week_open'] = h['time'].diff().dt.total_seconds() / 3600 > 6
    h = h.dropna(subset=['lr'])
    h = h[h['time'] >= pd.Timestamp(SPANS[0][1], tz='UTC')].copy()
    h['sofia_end'] = (h['time'].dt.hour + 2) % 24
    h['span'] = _span_of(h['time'])
    return h


def span_years(h: pd.DataFrame) -> dict:
    return {n: (h.loc[h.span == n, 'time'].max() - h.loc[h.span == n, 'time'].min()).days / 365.25
            for n, _, _ in SPANS}


def per_year(h: pd.DataFrame, key: str, yrs: dict) -> pd.DataFrame:
    """Log return in % per year, by `key` and span."""
    t = h.pivot_table(index=key, columns='span', values='lr', aggfunc='sum', observed=False) * 100
    for n in yrs:
        t[n] = t[n] / yrs[n]
    return t


def by_hour(h: pd.DataFrame, yrs: dict) -> pd.DataFrame:
    x = h[~h.week_open]
    t = per_year(x, 'sofia_end', yrs)
    mv = x.pivot_table(index='sofia_end', columns='span', values='lr', observed=False,
                       aggfunc=lambda v: np.abs(v).mean() * 100)
    up = x.pivot_table(index='sofia_end', columns='span', values='lr', observed=False,
                       aggfunc=lambda v: (v > 0).mean() * 100)
    out = pd.concat({'return %/yr': t, 'mean abs move %': mv, 'up %': up}, axis=1)
    out[('bars', '')] = x.groupby('sofia_end').size()
    return out


def by_session(h: pd.DataFrame, yrs: dict) -> pd.DataFrame:
    x = h[~h.week_open].copy()
    lookup = {hr: name for name, hrs in SESSIONS for hr in hrs}
    x['session'] = x['sofia_end'].map(lookup)
    t = per_year(x, 'session', yrs)
    return t.reindex([n for n, _ in SESSIONS])


def weekend_gap(h: pd.DataFrame, yrs: dict) -> dict:
    return {n: float(h.loc[(h.span == n) & h.week_open, 'lr'].sum() * 100 / yrs[n]) for n in yrs}


def daily_returns(d1: pd.DataFrame) -> pd.Series:
    d = d1.set_index('time')['close']
    d = d[d.index.dayofweek < 5]
    r = np.log(d).diff().dropna() * 100
    return r[r.index >= pd.Timestamp(SPANS[0][1], tz='UTC')]


def by_weekday(dr: pd.Series) -> pd.DataFrame:
    span = _span_of(pd.Series(dr.index, index=dr.index))
    t = dr.groupby([dr.index.dayofweek, span], observed=False).mean().unstack()
    return t.rename(index=dict(enumerate(WEEKDAYS)))


def by_month(d1: pd.DataFrame) -> pd.DataFrame:
    d = d1.set_index('time')['close']
    d = d[d.index >= pd.Timestamp(SPANS[0][1], tz='UTC')]
    mo = np.log(d.resample('ME').last()).diff().dropna() * 100
    g = mo.groupby(mo.index.month)
    return pd.DataFrame({'mean %': g.mean(), 'up years': g.apply(lambda v: f'{(v > 0).sum()}/{len(v)}')})


def by_year(d1: pd.DataFrame) -> dict:
    d = d1.set_index('time')['close']
    d = d[d.index >= pd.Timestamp(SPANS[0][1], tz='UTC')]
    return {int(y): float((v.iloc[-1] / v.iloc[0] - 1) * 100) for y, v in d.groupby(d.index.year)}


def concentration(dr: pd.Series) -> dict:
    srt = dr.sort_values(ascending=False)
    total = float(dr.sum())
    out = {'total': total}
    for k in (10, 25, 50, 100):
        out[f'without best {k}'] = total - float(srt.iloc[:k].sum())
    big = dr[dr.abs() > 2]
    out['share of days with abs r > 2 %'] = float((dr.abs() > 2).mean() * 100)
    out['big days up %'] = float((big > 0).mean() * 100)
    return out


def size_clustering(dr: pd.Series) -> pd.DataFrame:
    a = dr.abs()
    q = pd.qcut(a.shift(1), 5, labels=['Q1 calm', 'Q2', 'Q3', 'Q4', 'Q5 stormy'])
    return pd.DataFrame({'mean abs r today %': a.groupby(q, observed=False).mean(),
                         'mean r today %': dr.groupby(q, observed=False).mean()}).T


def _cell(v, fmt: str) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return fmt.format(v) if np.isfinite(v) else ''
    return str(v)


def _md(df: pd.DataFrame, fmt: str = '{:+.2f}', formats: dict | None = None) -> str:
    """Markdown table; `formats` maps a column's top-level name to its own format."""
    formats = formats or {}
    cols = [' '.join(str(p) for p in c if str(p)) if isinstance(c, tuple) else str(c) for c in df.columns]
    tops = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    lines = ['| | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 1) + '|']
    for idx, row in df.iterrows():
        cells = [_cell(v, formats.get(t, fmt)) for t, v in zip(tops, row.to_numpy(dtype=object))]
        lines.append(f'| {idx} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def render(h1: pd.DataFrame, d1: pd.DataFrame, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    h = hourly_returns(h1)
    yrs = span_years(h)
    dr = daily_returns(d1)
    hours = by_hour(h, yrs)
    hours.index = [f'{i:02d}:00' for i in hours.index]
    c = concentration(dr)
    gap = weekend_gap(h, yrs)
    out = [
        '# Gold descriptive statistics',
        '',
        f'Generated {now.strftime("%Y-%m-%d %H:%M UTC")} by `python -m src.gold_window.describe` '
        f'from `results/gold/gold_h1.csv` ({h1["time"].min():%Y-%m-%d} .. {h1["time"].max():%Y-%m-%d %H:%M}) '
        f'and `results/gold/gold_d1.csv`.',
        '',
        LABEL,
        '',
        'Hours are **Sofia** time, at which the H1 bar closes (a bar labelled h Berlin closes at '
        'h + 2 Sofia). Returns are log % per year of each span, on bid closes.',
        '',
        '## Total by span',
        '',
        LABEL,
        '',
        _md(pd.DataFrame({'all hours %/yr': pd.Series({n: h.loc[h.span == n, 'lr'].sum() * 100 / yrs[n]
                                                       for n in yrs}),
                          'weekend gap %/yr': pd.Series(gap)}), '{:+.1f}'),
        '',
        '## By session',
        '',
        LABEL,
        '',
        _md(by_session(h, yrs), '{:+.1f}'),
        '',
        '## By hour',
        '',
        LABEL,
        '',
        '**Bid-price limitation.** The cache holds bid closes only, and MT5 keeps one spread value per '
        'bar (0.00 for 2015-2017, a flat 0.25 or 0.50 later), which cannot show the widening at the '
        'daily break. The fall into 00:00 and the rise into 02:00 Sofia may therefore be partly a bid '
        'artifact; only bid/ask data can separate the two.',
        '',
        _md(hours, formats={'return %/yr': '{:+.1f}', 'mean abs move %': '{:.3f}', 'up %': '{:.0f}',
                           'bars': '{:.0f}'}),
        '',
        '## By weekday (D1 close to close, mean % per day)',
        '',
        LABEL,
        '',
        _md(by_weekday(dr), '{:+.3f}'),
        '',
        '## By month (mean monthly log return %)',
        '',
        LABEL,
        '',
        _md(by_month(d1).T, '{:+.2f}'),
        '',
        '## By calendar year (%)',
        '',
        LABEL,
        '',
        _md(pd.DataFrame({'return %': pd.Series(by_year(d1))}), '{:+.1f}'),
        '',
        '## Concentration in the best days (D1, log %)',
        '',
        LABEL,
        '',
        _md(pd.DataFrame({'value': pd.Series(c)}), '{:+.1f}'),
        '',
        '## Size clustering: today by quintile of yesterday\'s absolute move',
        '',
        LABEL,
        '',
        _md(size_clustering(dr), '{:+.3f}'),
        '',
    ]
    return '\n'.join(out)


def load_caches() -> tuple[pd.DataFrame, pd.DataFrame]:
    h1 = pd.read_csv(repo_path('results/gold/gold_h1.csv'), parse_dates=['time'])
    d1 = pd.read_csv(repo_path('results/gold/gold_d1.csv'), parse_dates=['time'])
    return h1, d1


def write(path: str | None = None, now: datetime | None = None) -> str:
    cfg = load_config()
    path = path or repo_path(cfg['paths']['descriptive'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = render(*load_caches(), now=now)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    return path


if __name__ == '__main__':
    print(write())
