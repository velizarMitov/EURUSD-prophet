"""
Minimal long/flip backtest: does the direction signal's apparent edge survive
realistic transaction costs (spread), or is it an artifact of a near-chance
classifier? Deliberately simple -- no position sizing, no compounding, no
slippage beyond the flat per-trade cost -- since the point is a sanity check,
not a trading system.
"""
import numpy as np
import pandas as pd


def simulate_strategy(y_true_return_pct, y_pred_direction, cost_pct_per_trade: float = 0.0) -> dict:
    """
    Daily long/short strategy driven by a binary direction signal, scored on the
    SAME held-out test block the model is evaluated on (chronological order,
    matching src/features.py's target_direction convention: 1=UP, 0=DOWN).

    Position is +1 (long) when `y_pred_direction` is 1, else -1 (short). The
    day's gross return is `signal * y_true_return_pct` (both already in
    percent). `cost_pct_per_trade` is charged only on days the position
    actually CHANGES (a flat->long/short entry, or a long<->short flip) --
    not on every day -- since holding an unchanged position overnight incurs
    no fresh spread. The very first test-block day always incurs the cost
    (entering from flat).

    Returns a dict of gross/net cumulative and per-trade returns, hit rate, and
    trade count -- no look-ahead: each day's signal only ever multiplies that
    SAME day's already-realised return (identical to how every model in this
    project is scored), never a future one.
    """
    y_true_return_pct = np.asarray(y_true_return_pct, dtype=float)
    y_pred_direction = np.asarray(y_pred_direction)
    n = len(y_true_return_pct)
    if n == 0:
        return {
            "n_days": 0, "n_trades": 0, "hit_rate": float("nan"),
            "gross_return_pct_total": 0.0, "net_return_pct_total": 0.0,
            "gross_return_pct_per_trade": float("nan"), "net_return_pct_per_trade": float("nan"),
        }

    signal = np.where(y_pred_direction == 1, 1.0, -1.0)
    gross_daily = signal * y_true_return_pct

    prev_signal = np.concatenate(([0.0], signal[:-1]))  # flat before day 0
    position_changed = signal != prev_signal
    cost_daily = np.where(position_changed, cost_pct_per_trade, 0.0)
    net_daily = gross_daily - cost_daily

    n_trades = int(position_changed.sum())
    hit_rate = float((signal == np.where(y_true_return_pct > 0, 1.0, -1.0)).mean())

    return {
        "n_days": n,
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "gross_return_pct_total": float(gross_daily.sum()),
        "net_return_pct_total": float(net_daily.sum()),
        "gross_return_pct_per_trade": float(gross_daily.sum() / n_trades) if n_trades else float("nan"),
        "net_return_pct_per_trade": float(net_daily.sum() / n_trades) if n_trades else float("nan"),
    }


def backtest_table(y_true_return_pct, y_pred_direction, cost_scenarios_pips: dict) -> pd.DataFrame:
    """
    Run simulate_strategy across several spread scenarios (in pips, converted to
    a round-trip percent cost via EURUSD_PIP_TO_PCT) and return one comparison
    row per scenario, always including a zero-cost row first as the reference.
    """
    rows = []
    zero = simulate_strategy(y_true_return_pct, y_pred_direction, cost_pct_per_trade=0.0)
    rows.append({"scenario": "gross (no costs)", "cost_pips": 0.0, **zero})
    for name, pips in cost_scenarios_pips.items():
        cost_pct = pips * EURUSD_PIP_TO_PCT
        res = simulate_strategy(y_true_return_pct, y_pred_direction, cost_pct_per_trade=cost_pct)
        rows.append({"scenario": name, "cost_pips": pips, **res})
    return pd.DataFrame(rows)


# 1 pip = 0.0001 price move; at a representative EURUSD level of ~1.10,
# 0.0001 / 1.10 * 100 = ~0.0091% per pip of round-trip spread cost.
EURUSD_PIP_TO_PCT = 0.0001 / 1.10 * 100
