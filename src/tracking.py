"""
Prediction-vs-actual tracking. Every call to PredictionService.predict() appends
(idempotently, one row per as_of_date) its forecast to a CSV log. Later, once the
forecast date's bar has actually closed, build_history_html() joins each logged
forecast against the realised market close, scores the directional call, and
renders an HTML table so the model can be compared with reality over time.
"""
import os

import pandas as pd

from .live_data import fetch_live_market_data, drop_incomplete_bars

LOG_COLUMNS = [
    'as_of_date', 'forecasting_date', 'as_of_close',
    'pred_direction', 'pred_return_pct', 'pred_confidence',
    'gbm_direction', 'lstm_direction',
    'h1_direction', 'h1_return_pct', 'h1_agreement', 'logged_at',
]


def log_prediction(result: dict, log_path: str) -> None:
    """
    Append the forecast in `result` (a PredictionService.predict() dict) to the
    CSV at `log_path`. Idempotent per as_of_date: re-predicting the same day
    overwrites that day's row rather than duplicating it. The predicted side is
    all that is stored here; the realised side is computed live at render time.
    """
    cons = result.get('consensus', {})
    bar = result.get('bar_used', {})
    h1_cons = result.get('h1', {}).get('consensus', {})   # auxiliary intraday ensemble
    row = {
        'as_of_date': result.get('as_of_date'),
        'forecasting_date': result.get('forecasting_date'),
        'as_of_close': bar.get('close'),
        'pred_direction': cons.get('direction'),
        'pred_return_pct': cons.get('predicted_return_pct'),
        'pred_confidence': cons.get('confidence'),
        'gbm_direction': result.get('gbm', {}).get('direction'),
        'lstm_direction': result.get('lstm', {}).get('direction'),
        'h1_direction': h1_cons.get('direction'),
        'h1_return_pct': h1_cons.get('predicted_return_pct'),
        'h1_agreement': h1_cons.get('confidence'),   # fraction of H1 models agreeing, NOT a probability
        'logged_at': pd.Timestamp.utcnow().isoformat(),
    }

    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        log = log[log['as_of_date'] != row['as_of_date']]                    # replace same-day forecast
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row], columns=LOG_COLUMNS)

    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log.sort_values('as_of_date').to_csv(log_path, index=False)


def _actual_closes(data_cfg: dict, bars: int = 400, now=None) -> dict:
    """
    date-iso -> realised close, from whichever live source answers first.
    Today's still-forming bar (and any stray MT5 weekend bar) is dropped via
    drop_incomplete_bars so a forecast for the current, not-yet-closed
    session is never scored against a mid-session price -- it must show as
    pending until the session has actually closed and the next live fetch
    sees the day's bar settle into the past. `now` is injectable for tests;
    it defaults to the local wall clock.
    """
    df, _ = fetch_live_market_data(
        data_cfg.get('symbol', 'EURUSD'),
        data_cfg.get('live_symbol', 'EURUSD=X'),
        bars=bars,
    )
    df = drop_incomplete_bars(df, now=now)
    if df is None or df.empty:
        return {}
    return {idx.date().isoformat(): float(close) for idx, close in df['close'].items()}


def build_history_html(log_path: str, data_cfg: dict,
                       title: str = "EUR/USD — Prediction vs. Actual Market", now=None) -> str:
    """
    Render the prediction log as a self-contained HTML page, scoring every row
    whose forecast date has already closed against the realised return/direction.
    Rows still in the future -- including today's still-open session -- show
    as 'pending'. `now` is injectable for tests; it defaults to the local wall
    clock. Returns the full HTML string.
    """
    if not os.path.exists(log_path):
        return _wrap(title, "<p class='empty'>No predictions logged yet. "
                            "Run a prediction first, then refresh.</p>", "—")

    log = pd.read_csv(log_path).sort_values('as_of_date', ascending=False)
    closes = _actual_closes(data_cfg, now=now)

    body_rows, n_resolved, n_correct = [], 0, 0
    n_h1_resolved, n_h1_correct = 0, 0
    for _, r in log.iterrows():
        actual_close = closes.get(str(r['forecasting_date']))
        resolved = actual_close is not None and pd.notna(r.get('as_of_close'))

        if resolved:
            actual_ret = (actual_close - r['as_of_close']) / r['as_of_close'] * 100
            actual_dir = 'UP' if actual_ret > 0 else 'DOWN'
            correct = str(r['pred_direction']).startswith(actual_dir)  # 'UP'/'DOWN'; 'MIXED...' never matches
            n_resolved += 1
            n_correct += int(correct)
            css = 'hit' if correct else 'miss'
            status = '✅ correct' if correct else '❌ wrong'
            actual_cell = f"{actual_dir} ({actual_ret:+.4f}%)"
        else:
            css, status, actual_cell = 'pending', '⏳ pending', '—'

        # H1 auxiliary ensemble, scored against the SAME realised session (it
        # forecasts the same next-day close). Absent on rows logged before H1
        # tracking existed -> shown as an em-dash rather than counted.
        h1_dir = r.get('h1_direction')
        has_h1 = isinstance(h1_dir, str) and h1_dir in ('UP', 'DOWN')
        if has_h1:
            h1_pred_cell = f"{h1_dir} ({_pct(r.get('h1_return_pct'))})"
            if resolved:
                h1_correct = h1_dir == actual_dir
                n_h1_resolved += 1
                n_h1_correct += int(h1_correct)
                h1_status = '✅' if h1_correct else '❌'
            else:
                h1_status = '⏳'
        else:
            h1_pred_cell, h1_status = '—', '—'

        body_rows.append(
            f"<tr class='{css}'>"
            f"<td>{r['as_of_date']}</td>"
            f"<td>{r['forecasting_date']}</td>"
            f"<td>{_fmt(r.get('pred_direction'))} ({_pct(r.get('pred_return_pct'))})</td>"
            f"<td>{_conf(r.get('pred_confidence'))}</td>"
            f"<td>{h1_pred_cell} {h1_status}</td>"
            f"<td>{actual_cell}</td>"
            f"<td>{status}</td>"
            f"</tr>"
        )

    daily_rate = (f"{n_correct}/{n_resolved} resolved &nbsp;·&nbsp; "
                  f"<b>{n_correct / n_resolved:.0%}</b> daily-committee hit-rate"
                  if n_resolved else "no resolved predictions yet")
    h1_rate = (f" &nbsp;|&nbsp; H1 ensemble: {n_h1_correct}/{n_h1_resolved} &nbsp;·&nbsp; "
               f"<b>{n_h1_correct / n_h1_resolved:.0%}</b>"
               if n_h1_resolved else "")
    hit_rate = daily_rate + h1_rate
    table = (
        "<table><thead><tr>"
        "<th>Data as of</th><th>Forecast for</th><th>Daily predicted</th>"
        "<th>Confidence</th><th>H1 ensemble</th><th>Actual market</th><th>Result</th>"
        "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"
    )
    return _wrap(title, table, hit_rate)


def _fmt(v):
    return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _pct(v):
    try:
        return f"{float(v):+.4f}%"
    except (TypeError, ValueError):
        return "—"


def _conf(v):
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return "—"


def _wrap(title: str, inner: str, summary: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #1a1a2e; background: #f7f7fb; }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  .summary {{ color: #555; margin-bottom: 1.25rem; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  th, td {{ padding: .55rem .8rem; text-align: left; border-bottom: 1px solid #eee; font-variant-numeric: tabular-nums; }}
  th {{ background: #2d6cdf; color: #fff; font-weight: 600; }}
  tr.hit    {{ background: #eafaf0; }}
  tr.miss   {{ background: #fdecec; }}
  tr.pending{{ background: #f4f4f7; color: #888; }}
  .empty {{ color: #888; }}
</style></head>
<body>
  <h1>{title}</h1>
  <div class="summary">{summary}</div>
  {inner}
  <p class="summary" style="margin-top:1rem;font-size:.85rem;">
    A directional hit-rate near 50% is the expected, theory-consistent result for daily EUR/USD
    (near random walk) — see <code>ARCHITECTURE_DOCS.md §4.2.1</code>.
  </p>
</body></html>"""
