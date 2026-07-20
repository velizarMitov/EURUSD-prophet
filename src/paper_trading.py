"""
Forward paper-trading ledger — the real arbiter going forward.

Why this exists
---------------
The historical test block `[80%:100%]` has been touched repeatedly during
feature exploration, so it can no longer be treated as a clean out-of-sample
check (that's exactly what Steps 1-2 in IMPROVEMENT_LOG fix for feature search).
The only genuinely unbiased evaluation from here on is live paper trading on NEW
days as they arrive: each day's committee prediction is turned into a
hypothetical position, and once the forecast day's bar has closed it is scored
against the realised move — net of a realistic retail spread.

This is SIMULATED ONLY. There is no broker, no order execution, no position
sizing, no stop-loss anywhere here — just a hypothetical P&L ledger. Whether the
system is ever worth risking real capital should be decided by this ledger
accumulating a genuine, cost-net edge over a MEANINGFUL forward window (months,
not days), not by any further re-analysis of the spent historical test block.

Data flow
---------
Reconstructed on demand from `results/prediction_log.csv` (the same forecasts
`/history` scores) joined to the realised EUR/USD close once each forecast day
has settled — so the ledger grows by itself as the prediction log grows daily;
there is no separate write path to keep in sync. `MIXED / LOW CONFIDENCE` rows
carry no directional call and take no position (they are logged as flat, P&L 0).
"""
import os

import numpy as np
import pandas as pd

from .tracking import _actual_closes

# A pip is the 4th decimal of EUR/USD. One round-trip retail spread is ~1-2 pips;
# 1.5 is a representative default. Percent conversion mirrors src/backtest.py so
# the paper ledger and the historical backtest speak the same cost language.
PIP_SIZE = 0.0001
DEFAULT_SPREAD_PIPS = 1.5
TRADING_DAYS_PER_YEAR = 252

LEDGER_COLUMNS = [
    'as_of_date', 'forecasting_date', 'direction', 'entry', 'exit',
    'gross_pips', 'spread_pips', 'net_pips', 'net_return_pct',
    'cum_net_pips', 'cum_net_return_pct', 'outcome',
]


def build_ledger(log_path: str, data_cfg: dict, spread_pips: float = DEFAULT_SPREAD_PIPS,
                 now=None, direction_column: str = 'pred_direction') -> pd.DataFrame:
    """Resolve every logged forecast whose day has closed into a simulated,
    cost-net paper position and return the chronological ledger.

    `direction_column` selects WHICH committee's call drives the position —
    'pred_direction' (the with_macro variant, the lineage of every pre-dual
    row) or 'baseline_direction' (the price-only variant) — so the two model
    variants accumulate SEPARATE forward ledgers from the same prediction log.
    A row where that column is absent/NaN (e.g. pre-dual rows have no baseline
    forecast at all) is skipped entirely: no forecast was made, so no position
    — unlike an explicit MIXED call, which IS a forecast whose decision was
    "stand flat".

    Position is +1 (long) on an UP call, −1 (short) on a DOWN call; a
    MIXED/LOW-CONFIDENCE row takes no position. The day's gross move in pips is
    `(exit - entry) / PIP_SIZE * direction`; the round-trip `spread_pips` is
    charged once per taken position. No look-ahead: `entry` is the as-of close
    the forecast was made on and `exit` is the realised close of the forecast
    day itself — never a future bar. Rows still pending (forecast day not yet
    closed) are excluded, since their P&L is undefined.
    """
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    log = pd.read_csv(log_path).sort_values('as_of_date')
    closes = _actual_closes(data_cfg, now=now)

    rows = []
    for _, r in log.iterrows():
        exit_close = closes.get(str(r['forecasting_date']))
        entry = r.get('as_of_close')
        if exit_close is None or pd.isna(entry):
            continue  # forecast day not settled yet -> undefined P&L, skip

        raw_pred = r.get(direction_column)
        if raw_pred is None or (isinstance(raw_pred, float) and pd.isna(raw_pred)):
            continue  # this variant made no forecast that day (e.g. pre-dual row)

        pred = str(raw_pred)
        if pred.startswith('UP'):
            direction = 1
        elif pred.startswith('DOWN'):
            direction = -1
        else:
            direction = 0  # MIXED / LOW CONFIDENCE -> flat, no position taken

        gross_pips = (exit_close - entry) / PIP_SIZE * direction
        taken = direction != 0
        cost = spread_pips if taken else 0.0
        net_pips = gross_pips - cost
        # percent P&L on the entry price, direction-signed, minus the spread in pct.
        gross_pct = (exit_close - entry) / entry * 100 * direction
        cost_pct = (spread_pips * PIP_SIZE) / entry * 100 if taken else 0.0
        net_pct = gross_pct - cost_pct

        if not taken:
            outcome = 'flat'
        elif net_pips > 0:
            outcome = 'win'
        else:
            outcome = 'loss'

        rows.append({
            'as_of_date': r['as_of_date'], 'forecasting_date': r['forecasting_date'],
            'direction': {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}[direction],
            'entry': round(float(entry), 5), 'exit': round(float(exit_close), 5),
            'gross_pips': round(gross_pips, 2), 'spread_pips': cost,
            'net_pips': round(net_pips, 2), 'net_return_pct': round(net_pct, 4),
            'outcome': outcome,
        })

    ledger = pd.DataFrame(rows, columns=[c for c in LEDGER_COLUMNS
                                         if c not in ('cum_net_pips', 'cum_net_return_pct')])
    if ledger.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger['cum_net_pips'] = ledger['net_pips'].cumsum().round(2)
    ledger['cum_net_return_pct'] = ledger['net_return_pct'].cumsum().round(4)
    return ledger[LEDGER_COLUMNS]


def summarize(ledger: pd.DataFrame) -> dict:
    """Running scorecard over the TAKEN positions (flat days excluded from the
    win rate and the risk stats): trade count, win rate, cumulative net pips /
    percent, an annualised Sharpe-like ratio, and max drawdown on the cumulative
    net-return curve. Undefined statistics (empty ledger, <2 trades, zero
    variance) are None — JSON-safe (`null`) for the API, unlike NaN which the
    JSON encoder rejects outright."""
    if ledger is None or ledger.empty:
        return {"n_positions": 0, "n_wins": 0, "win_rate": None,
                "cum_net_pips": 0.0, "cum_net_return_pct": 0.0,
                "sharpe_like": None, "max_drawdown_pct": 0.0,
                "avg_net_pips": None}

    taken = ledger[ledger['direction'] != 'FLAT']
    n = len(taken)
    wins = int((taken['outcome'] == 'win').sum())
    daily = taken['net_return_pct'].to_numpy(dtype=float)

    # Sharpe-like: annualised mean/std of the per-position net return. Needs at
    # least 2 positions and non-zero dispersion, else it is undefined (None).
    if n >= 2 and daily.std(ddof=1) > 0:
        sharpe = round(float(daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)), 3)
    else:
        sharpe = None

    # Max drawdown on the cumulative net-return curve, seeded at 0 (pre-trade
    # flat), so a first-position loss is itself a drawdown.
    curve = np.concatenate(([0.0], np.cumsum(daily)))
    running_peak = np.maximum.accumulate(curve)
    max_dd = float((running_peak - curve).max())

    return {
        "n_positions": n,
        "n_wins": wins,
        "win_rate": round(wins / n, 4) if n else None,
        "cum_net_pips": round(float(taken['net_pips'].sum()), 2),
        "cum_net_return_pct": round(float(daily.sum()), 4),
        "avg_net_pips": round(float(taken['net_pips'].mean()), 3) if n else None,
        "sharpe_like": sharpe,
        "max_drawdown_pct": round(max_dd, 4),
    }


def build_and_save(log_path: str, data_cfg: dict, out_path: str,
                   spread_pips: float = DEFAULT_SPREAD_PIPS, now=None,
                   direction_column: str = 'pred_direction'):
    """Build one variant's ledger, persist it to `out_path`, and return
    (ledger, summary). Called by the `/paper-trading` endpoints so the CSV is
    refreshed on view."""
    ledger = build_ledger(log_path, data_cfg, spread_pips=spread_pips, now=now,
                          direction_column=direction_column)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    ledger.to_csv(out_path, index=False)
    return ledger, summarize(ledger)


# Human-readable panel titles for the configured ledgers.
VARIANT_LABELS = {
    'baseline': 'Price-Only Model (baseline)',
    'with_macro': 'With Macro Features (experimental)',
    'ti_h1': ('H1 TI-LSTM — ⚠ NOT VALIDATED, no demonstrated edge '
              '(owner-override ship, observational only)'),
}


def build_all_ledgers(log_path: str, data_cfg: dict, pt_cfg: dict,
                      base_dir: str = '', now=None) -> dict:
    """Build + persist every configured variant ledger (config.json →
    paper_trading.ledgers) and return {variant: {'ledger': df, 'summary': dict}}.
    One shared realised-closes fetch would be marginally cheaper, but each
    build_ledger call is already cache-friendly and this stays simple."""
    ledgers_cfg = pt_cfg.get('ledgers', {
        'with_macro': {'direction_column': 'pred_direction',
                       'log_path': 'results/paper_trading_log_macro.csv'},
    })
    spread_pips = pt_cfg.get('spread_pips', DEFAULT_SPREAD_PIPS)
    out = {}
    for variant, lcfg in ledgers_cfg.items():
        out_path = os.path.join(base_dir, lcfg['log_path']) if base_dir else lcfg['log_path']
        ledger, summary = build_and_save(
            log_path, data_cfg, out_path, spread_pips=spread_pips, now=now,
            direction_column=lcfg.get('direction_column', 'pred_direction'),
        )
        out[variant] = {'ledger': ledger, 'summary': summary}
    return out


def _scorecard_line(summary: dict) -> str:
    wr = summary['win_rate']
    sharpe = summary['sharpe_like']
    if not summary['n_positions']:
        return "no settled positions yet"
    line = (f"{summary['n_positions']} positions ({summary['n_wins']} wins) &nbsp;·&nbsp; "
            f"win rate <b>{wr:.0%}</b>"
            f" &nbsp;|&nbsp; cum. net <b>{summary['cum_net_pips']:+.1f} pips</b> "
            f"({summary['cum_net_return_pct']:+.3f}%) &nbsp;|&nbsp; "
            f"Sharpe-like {'—' if sharpe is None else f'{sharpe:+.2f}'} &nbsp;|&nbsp; "
            f"max DD {summary['max_drawdown_pct']:.3f}%")
    return line


def _ledger_section(label: str, ledger: pd.DataFrame, summary: dict) -> str:
    """One variant's <h2> section: scorecard line + ledger table."""
    def _cls(o):
        return {'win': 'hit', 'loss': 'miss', 'flat': 'pending'}.get(o, 'pending')

    if ledger is None or ledger.empty:
        body = ("<p class='empty'>No settled paper positions yet for this variant. "
                "Once a logged forecast's day closes, it will be scored here.</p>")
    else:
        trs = "".join(
            f"<tr class='{_cls(r['outcome'])}'>"
            f"<td>{r['as_of_date']}</td><td>{r['forecasting_date']}</td>"
            f"<td>{r['direction']}</td><td>{r['entry']:.5f}</td><td>{r['exit']:.5f}</td>"
            f"<td>{r['net_pips']:+.2f}</td><td>{r['net_return_pct']:+.4f}%</td>"
            f"<td>{r['cum_net_pips']:+.2f}</td><td>{r['outcome']}</td></tr>"
            for _, r in ledger.iloc[::-1].iterrows()
        )
        body = ("<table><thead><tr>"
                "<th>Data as of</th><th>Forecast for</th><th>Position</th>"
                "<th>Entry</th><th>Exit</th><th>Net pips</th><th>Net %</th>"
                "<th>Cum. net pips</th><th>Result</th></tr></thead><tbody>"
                + trs + "</tbody></table>")
    return (f"<h2>{label}</h2>"
            f"<div class='summary'>{_scorecard_line(summary)}</div>"
            f"{body}")


def render_html(variants: dict, spread_pips: float,
                title: str = "EUR/USD — Forward Paper-Trading Ledgers (simulated)") -> str:
    """Self-contained HTML view rendering EVERY variant's ledger + scorecard on
    one page (Price-Only vs With-Macro side by side, top to bottom), matching
    the /history page's look. Simulated only — no orders were placed.
    `variants` is build_all_ledgers()'s output dict."""
    sections = "".join(
        _ledger_section(VARIANT_LABELS.get(name, name), blk['ledger'], blk['summary'])
        for name, blk in variants.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #1a1a2e; background: #f7f7fb; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1.1rem; margin: 1.5rem 0 .25rem; }}
  .summary {{ color: #555; margin-bottom: 1.25rem; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th, td {{ padding: .55rem .8rem; text-align: left; border-bottom: 1px solid #eee; font-variant-numeric: tabular-nums; }}
  th {{ background: #2d6cdf; color: #fff; font-weight: 600; }}
  tr.hit {{ background: #eafaf0; }} tr.miss {{ background: #fdecec; }} tr.pending {{ background: #f4f4f7; color: #888; }}
  .empty {{ color: #888; }}
</style></head>
<body>
  <h1>{title}</h1>
  <div class="summary">Two independent simulated ledgers — one per model variant — driven by the
  same daily prediction log. The macro variant's features are statistically unproven
  (KEEP-provisional under the Bonferroni-corrected validation bar); whichever ledger nets
  better cost-adjusted P&amp;L over a meaningful forward window is the honest winner.</div>
  {sections}
  <p class="summary" style="margin-top:1rem;font-size:.85rem;">
    Simulated only — no broker orders are placed. Each taken position is charged a
    {spread_pips:g}-pip round-trip retail spread. These forward ledgers, accumulating from
    today, are the primary arbiter of production-worthiness — not the spent historical
    test block (see <code>ARCHITECTURE_DOCS.md</code> Production Methodology).
    Back to the <a href="/history">prediction-vs-actual history</a>.
  </p>
</body></html>"""


if __name__ == "__main__":
    import json
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base, "config.json")) as f:
        cfg = json.load(f)
    log = os.path.join(base, cfg["tracking"]["log_path"])
    all_ledgers = build_all_ledgers(log, cfg["data"], cfg.get("paper_trading", {}), base_dir=base)
    for name, blk in all_ledgers.items():
        print(f"\n===== {VARIANT_LABELS.get(name, name)} =====")
        print(blk['ledger'].to_string(index=False) if not blk['ledger'].empty
              else "No settled positions yet.")
        print("Scorecard:", json.dumps(blk['summary'], indent=2))
