"""The fixed gross rule (spec fx-fibonacci/evaluation). GROSS of spread and slippage — not a
trading result, and no order, sizing or execution code.

On each out-of-sample row, in time order, the rule acts when the predicted class (argmax) is
1 (up) or 2 (down), holding at most one position per pair. A position opens at the bar's close
and exits where its label resolved: at the barrier touched first, or at the close of the
exit bar for a time-out or a same-bar tie. A signal while a position is open is skipped and
counted. Returns are in percent of the entry close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trades(rows: pd.DataFrame, pred: np.ndarray) -> tuple[pd.DataFrame, int]:
    """rows (time-ordered): time, pos, close, label, exit_offset, upper, lower, exit_close.
    Returns (trade table, skipped signals)."""
    out, skipped, busy_until = [], 0, -1
    r = rows.reset_index(drop=True)
    for i in range(len(r)):
        c = int(pred[i])
        if c not in (1, 2):
            continue
        pos = int(r.at[i, 'pos'])
        if pos < busy_until:
            skipped += 1
            continue
        d = 1.0 if c == 1 else -1.0
        lab = int(r.at[i, 'label'])
        exit_px = r.at[i, 'upper'] if lab == 1 else r.at[i, 'lower'] if lab == 2 else r.at[i, 'exit_close']
        off = int(r.at[i, 'exit_offset'])
        busy_until = pos + off
        out.append({'entry_time': r.at[i, 'time'], 'exit_pos': pos + off, 'direction': int(d),
                    'ret_pct': d * (exit_px - r.at[i, 'close']) / r.at[i, 'close'] * 100.0,
                    'exit_day': r.at[i, 'exit_day']})
    return pd.DataFrame(out, columns=['entry_time', 'exit_pos', 'direction', 'ret_pct', 'exit_day']), skipped


def stats(tr: pd.DataFrame, skipped: int, all_days) -> dict:
    """Profit factor, Sharpe on daily-summed returns (sqrt 252), max drawdown, win rate and
    expectancy, all gross, in percent."""
    if tr.empty:
        return {'trades': 0, 'skipped': skipped, 'win_rate': np.nan, 'expectancy_pct': np.nan,
                'profit_factor': np.nan, 'sharpe': np.nan, 'max_drawdown_pct': np.nan, 'total_pct': 0.0}
    r = tr['ret_pct']
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    daily = tr.groupby('exit_day')['ret_pct'].sum().reindex(pd.DatetimeIndex(sorted(set(all_days))), fill_value=0.0)
    sd = daily.std(ddof=1)
    equity = daily.cumsum()
    dd = (equity.cummax().clip(lower=0) - equity).max()
    return {'trades': int(len(r)), 'skipped': int(skipped), 'win_rate': float((r > 0).mean()),
            'expectancy_pct': float(r.mean()),
            'profit_factor': float(gains / losses) if losses > 0 else float('inf'),
            'sharpe': float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            'max_drawdown_pct': float(dd), 'total_pct': float(r.sum())}
