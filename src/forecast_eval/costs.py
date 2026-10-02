"""Cost model (design D6, spec backtesting "Cost model from measured spreads").

A sign strategy pays the spread when its position CHANGES -- entering from flat
or flipping long<->short. Holding an unchanged position overnight pays no new
spread (swap is separate, below). This is the convention of src/backtest.py.

Two cost levels are always reported:
  measured   the instrument's median M1 spread from results/curl/m1_coverage.csv
             (EURUSD: 5 points = 0.5 pip), the primary level;
  config     the round trip in config.json -> paper_trading.spread_pips (1.5 pip),
             the conservative sensitivity case.

A missing measured spread is an error, never a silent zero: a net figure built on
an assumed-free spread is the exact optimism this program exists to remove.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPREAD_TABLE = os.path.join(REPO, 'results', 'curl', 'm1_coverage.csv')
CONFIG_PATH = os.path.join(REPO, 'config.json')


class MissingSpreadError(LookupError):
    pass


def point_size(symbol: str) -> float:
    """MT5 point for 5-digit majors / 3-digit JPY crosses."""
    return 1e-3 if symbol.upper().endswith('JPY') else 1e-5


def pip_size(symbol: str) -> float:
    return 1e-2 if symbol.upper().endswith('JPY') else 1e-4


def measured_spread_price(symbol: str, table_path: str = SPREAD_TABLE) -> float:
    """Median measured spread in PRICE units."""
    if not os.path.exists(table_path):
        raise MissingSpreadError(f'spread table not found: {table_path}')
    table = pd.read_csv(table_path)
    row = table.loc[table['symbol'].str.upper() == symbol.upper()]
    if row.empty or pd.isna(row['median_spread_pts'].iloc[0]):
        raise MissingSpreadError(
            f'no measured spread for {symbol} in {os.path.basename(table_path)}; '
            'refusing to produce net figures without it')
    return float(row['median_spread_pts'].iloc[0]) * point_size(symbol)


def config_round_trip_price(symbol: str, config_path: str = CONFIG_PATH) -> float:
    with open(config_path, encoding='utf-8') as fh:
        pips = json.load(fh)['paper_trading']['spread_pips']
    return float(pips) * pip_size(symbol)


@dataclass(frozen=True)
class CostLevel:
    label: str
    price: float


def cost_levels(symbol: str, table_path: str = SPREAD_TABLE,
                config_path: str = CONFIG_PATH) -> list[CostLevel]:
    return [CostLevel('measured', measured_spread_price(symbol, table_path)),
            CostLevel('config_round_trip', config_round_trip_price(symbol, config_path))]


def position_changes(positions) -> np.ndarray:
    """Boolean per bar: True where a new spread is paid. The first bar is an
    entry from flat; a 0 (flat) position never pays to stay flat."""
    pos = np.asarray(positions, dtype=float)
    if pos.size == 0:
        return np.zeros(0, dtype=bool)
    prev = np.concatenate([[0.0], pos[:-1]])
    return (pos != prev) & (pos != 0)


def net_pnl(positions, price_moves, cost_price: float) -> dict:
    """Gross and net P&L of a position series, in price units.

    positions[t] in {-1, 0, +1} is held over the move price_moves[t]."""
    pos = np.asarray(positions, dtype=float)
    mv = np.asarray(price_moves, dtype=float)
    if pos.shape != mv.shape:
        raise ValueError('positions and price_moves must align')
    gross = pos * mv
    changes = position_changes(pos)
    charges = np.where(changes, cost_price, 0.0)
    net = gross - charges
    n_trades = int(changes.sum())
    return {
        'n_bars': int(pos.size), 'n_trades': n_trades,
        'gross_total': float(gross.sum()), 'net_total': float(net.sum()),
        'cost_total': float(charges.sum()),
        'gross_per_trade': float(gross.sum() / n_trades) if n_trades else float('nan'),
        'net_per_trade': float(net.sum() / n_trades) if n_trades else float('nan'),
    }


def rollovers_crossed(entry: pd.Timestamp, exit: pd.Timestamp, rollover_hour_utc: int) -> int:
    """How many daily rollovers (at rollover_hour_utc) fall in (entry, exit].
    Swap is charged once per rollover a position is held across."""
    if exit <= entry:
        return 0
    first = entry.normalize() + pd.Timedelta(hours=rollover_hour_utc)
    if first <= entry:
        first += pd.Timedelta(days=1)
    if first > exit:
        return 0
    return int((exit - first) // pd.Timedelta(days=1)) + 1


def swap_charge(direction: int, n_rollovers: int, swap_long: float, swap_short: float) -> float:
    """Swap in price units per unit position (MT5 quotes swap per lot in
    account currency or points; the caller converts to price units). A
    POSITIVE swap is a credit, so the charge is its negative."""
    if direction == 0 or n_rollovers == 0:
        return 0.0
    rate = swap_long if direction > 0 else swap_short
    return -rate * n_rollovers
