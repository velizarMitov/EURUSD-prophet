"""
H_dir.1 OBSERVATIONAL SERVING — call log, forward ledger, and the /h1-direction view.

WHAT THIS IS FOR
----------------
The shipped H1 direction model is a full-history refit. It has NO out-of-sample
confirmation of its own -- the +3.10pp test-block result belongs to the [0:70%]
model, which is not what is served. So the forward ledger built here is the ONLY
evidence this model will ever have, and it has to be accounted for honestly:

  * Every call is logged, with its model_version, to results/h1_direction_log.csv
    -- a SEPARATE file from the daily prediction_log.csv. The cadences differ
    (hourly on demand vs one daily forecast) and mixing them would corrupt both
    scorers.
  * A prediction SETTLES once its forecast bar has CLOSED, scored against that
    bar's realised close from the H1 feed. Unsettled rows show as pending.
  * DEDUPLICATION: repeated calls inside one forecast bar are all LOGGED but only
    the FIRST settles into the ledger, so refreshing the page ten times in an
    hour cannot inflate the sample.
  * Results are grouped BY model_version. A single blended hit rate across a
    retrain is misleading -- after a retrain the older rows describe a model that
    is no longer served -- so the view never leads with one.

SIMULATED ONLY. This module computes a hypothetical pip result from a logged
direction and a realised close. It places no orders, sizes no positions, sets no
stops and applies no leverage. `src/paper_trading.py` is NOT modified; its
constants are imported so the two ledgers speak the same cost language.
"""

import json
import os

import numpy as np
import pandas as pd

# Imported, never redefined: one home for the pip size and the spread convention.
from .paper_trading import DEFAULT_SPREAD_PIPS, PIP_SIZE

LOG_PATH = 'results/h1_direction_log.csv'
LEDGER_PATH = 'results/paper_trading_log_h1_direction.csv'
H1_FALLBACK_CACHE = 'results/pooled_h1/EURUSD_h1.csv'

# At this effect size (~3pp over baseline) a model version needs on the order of
# this many settled observations before its ledger says anything at all.
TARGET_OBSERVATIONS = 1000

LOG_COLUMNS = [
    'called_at_utc', 'as_of_bar_close', 'as_of_close', 'forecast_bar_start',
    'forecast_bar_end', 'direction', 'probability', 'minutes_remaining_at_call',
    'forecast_bar_status', 'data_source', 'model_version',
]

LEDGER_COLUMNS = [
    'model_version', 'as_of_bar_close', 'forecast_bar_start', 'forecast_bar_end',
    'direction', 'entry', 'exit', 'gross_pips', 'spread_pips', 'net_pips',
    'net_return_pct', 'cum_net_pips', 'cum_net_return_pct', 'outcome',
]


def log_prediction(response: dict, log_path: str = LOG_PATH,
                   called_at=None) -> dict:
    """
    Append ONE row per call. Every row carries model_version so a retrain can
    never silently blend into the previous model's forward evidence.
    """
    called = (pd.Timestamp.utcnow() if called_at is None
              else pd.Timestamp(called_at))
    if called.tzinfo is not None:
        called = called.tz_localize(None)

    row = {
        'called_at_utc': called.isoformat(),
        'as_of_bar_close': response.get('as_of_bar_close'),
        'as_of_close': response.get('as_of_close'),
        'forecast_bar_start': response.get('forecast_bar_start'),
        'forecast_bar_end': response.get('forecast_bar_end'),
        'direction': response.get('direction'),
        'probability': response.get('probability'),
        'minutes_remaining_at_call': response.get('minutes_remaining'),
        'forecast_bar_status': response.get('forecast_bar_status'),
        'data_source': response.get('data_source'),
        'model_version': response.get('model_version'),
    }
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    frame = pd.DataFrame([row], columns=LOG_COLUMNS)
    frame.to_csv(log_path, mode='a', header=not os.path.exists(log_path),
                 index=False)
    return row


def read_log(log_path: str = LOG_PATH) -> pd.DataFrame:
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(log_path)
    for col in ('called_at_utc', 'as_of_bar_close', 'forecast_bar_start',
                'forecast_bar_end'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def realised_h1_closes(base_dir: str = '.', now=None, live: bool = True) -> pd.Series:
    """
    Realised closes of FULLY CLOSED H1 bars, tz-stripped for matching against the
    log. Live-first so a settlement is not stuck behind a stale cache, falling
    back to the on-disk H1 cache. NEVER writes results/eurusd_h1.csv (a protected
    file owned by the daily predictor) as a side effect of scoring.
    """
    from .live_data import drop_incomplete_h1_bars

    frame = None
    if live:
        try:
            from .live_data import fetch_h1_market_data
            frame, _src = fetch_h1_market_data(bars=2000, cache_path=None)
        except Exception:
            frame = None
    if frame is None or not len(frame):
        path = os.path.join(base_dir, H1_FALLBACK_CACHE)
        if not os.path.exists(path):
            return pd.Series(dtype=float)
        frame = pd.read_csv(path, index_col=0, parse_dates=True)

    frame = frame.sort_index()
    # `now` defaults to the FEED'S clock, not the wall clock: settling against a
    # wall-clock comparison would score a still-forming bar as if it had closed.
    frame = drop_incomplete_h1_bars(frame, now=now)
    if frame is None or not len(frame):
        return pd.Series(dtype=float)
    idx = frame.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(frame['close'].to_numpy(dtype=float), index=idx)


def build_ledger(log_path: str = LOG_PATH, base_dir: str = '.',
                 spread_pips: float = DEFAULT_SPREAD_PIPS, now=None,
                 closes: pd.Series = None) -> pd.DataFrame:
    """
    Resolve every logged prediction whose forecast bar has CLOSED into a
    simulated, cost-net position.

    DEDUPLICATION: rows are grouped by (model_version, forecast_bar_start) and
    only the FIRST call by wall-clock time settles. All calls stay in the log;
    only one becomes a position. Grouping INCLUDES model_version so two versions
    predicting the same bar each settle their own -- they are different models.

    Fixed size per position and the same 1.5-pip round-trip spread convention as
    src/paper_trading.py, imported rather than restated. Simulated only.
    """
    log = read_log(log_path)
    if not len(log):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    log = log.dropna(subset=['forecast_bar_start', 'direction', 'as_of_close'])
    # A prediction for a bar the market will never print cannot settle, and one
    # made while the feed clock was UNCONFIRMED may be anchored to the wrong base
    # bar entirely -- it must never reach the evidence this model is judged on.
    log = log[~log['forecast_bar_status'].isin(('market_closed', 'clock_unconfirmed'))]
    if not len(log):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    first = (log.sort_values('called_at_utc')
                .groupby(['model_version', 'forecast_bar_start'], as_index=False)
                .first())

    if closes is None:
        closes = realised_h1_closes(base_dir=base_dir, now=now)

    rows = []
    for _i, r in first.sort_values(['model_version', 'forecast_bar_start']).iterrows():
        bar = pd.Timestamp(r['forecast_bar_start'])
        if bar not in closes.index:
            continue                                   # not closed yet -> pending
        entry = float(r['as_of_close'])
        exit_px = float(closes.loc[bar])
        sign = 1 if str(r['direction']).upper() == 'UP' else -1
        gross = (exit_px - entry) / PIP_SIZE * sign
        net = gross - spread_pips
        rows.append({
            'model_version': r['model_version'],
            'as_of_bar_close': r['as_of_bar_close'],
            'forecast_bar_start': bar, 'forecast_bar_end': r['forecast_bar_end'],
            'direction': r['direction'], 'entry': entry, 'exit': exit_px,
            'gross_pips': gross, 'spread_pips': spread_pips, 'net_pips': net,
            'net_return_pct': net * PIP_SIZE / entry * 100.0,
            'outcome': 'WIN' if net > 0 else ('LOSS' if net < 0 else 'FLAT'),
        })

    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    if len(ledger):
        # Cumulative columns run WITHIN a model_version: carrying them across a
        # retrain would draw one equity path through two different models.
        ledger['cum_net_pips'] = ledger.groupby('model_version')['net_pips'].cumsum()
        ledger['cum_net_return_pct'] = (
            ledger.groupby('model_version')['net_return_pct'].cumsum())
    return ledger


def build_and_save(log_path: str = LOG_PATH, base_dir: str = '.',
                   out_path: str = LEDGER_PATH, **kw) -> pd.DataFrame:
    ledger = build_ledger(log_path=log_path, base_dir=base_dir, **kw)
    target = os.path.join(base_dir, out_path) if not os.path.isabs(out_path) else out_path
    os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
    ledger.to_csv(target, index=False)
    return ledger


def summarize_by_version(ledger: pd.DataFrame, log: pd.DataFrame = None) -> list:
    """
    Per model_version: settled observations, hit rate, net pips, and progress
    toward the ~1,000 needed for a meaningful forward answer at this effect size.
    Never a single blended rate across versions.
    """
    out = []
    if ledger is None or not len(ledger):
        return out
    for version, sub in ledger.groupby('model_version', sort=True):
        wins = int((sub['outcome'] == 'WIN').sum())
        n = int(len(sub))
        raw_calls = None
        if log is not None and len(log):
            raw_calls = int((log['model_version'] == version).sum())
        out.append({
            'model_version': version,
            'settled': n,
            'raw_calls': raw_calls,
            'hit_rate': (wins / n) if n else float('nan'),
            'wins': wins, 'losses': int((sub['outcome'] == 'LOSS').sum()),
            'net_pips': float(sub['net_pips'].sum()),
            'net_return_pct': float(sub['net_return_pct'].sum()),
            'first_bar': str(sub['forecast_bar_start'].min()),
            'last_bar': str(sub['forecast_bar_start'].max()),
            'progress_pct': 100.0 * min(1.0, n / TARGET_OBSERVATIONS),
            'remaining_to_target': max(0, TARGET_OBSERVATIONS - n),
        })
    return out


def _served_meta(base_dir: str = '.') -> dict:
    """The meta of the model CURRENTLY SERVED, so a reader can tell which of the
    listed versions is live."""
    path = os.path.join(base_dir, 'models/h1_direction/h1_direction_meta.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _meta_trained_through(base_dir: str = '.') -> dict:
    m = _served_meta(base_dir)
    if not m:
        return {}
    return {m.get('model_version'): str(m.get('train_end', ''))[:10]}


# ───────────────────────── the /h1-direction view ─────────────────────────────

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:2rem;
background:#0f1115;color:#e6e8eb}
h1{font-size:1.4rem;margin:0 0 .25rem}
.sub{color:#9aa4b2;font-size:.85rem;margin-bottom:1.5rem}
.warn{background:#3a2a12;border:1px solid #8a6d3b;color:#f0c674;padding:.75rem 1rem;
border-radius:6px;margin:1rem 0;font-size:.85rem}
.note{background:#16202b;border:1px solid #24405c;color:#9fc5e8;padding:.75rem 1rem;
border-radius:6px;margin:1rem 0;font-size:.85rem}
.card{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:1rem 1.25rem;
margin-bottom:1rem}
.vh{font-size:1.05rem;font-weight:600;margin-bottom:.35rem}
.nav{font-size:.85rem;margin-bottom:1.25rem}
.nav a{color:#7fb2ff;text-decoration:none}
.nav a:hover{text-decoration:underline}
code{background:#0f1115;padding:.1rem .35rem;border-radius:3px}
.meta{color:#9aa4b2;font-size:.8rem;margin-bottom:.75rem}
.bar{height:8px;background:#262b36;border-radius:4px;overflow:hidden;margin:.5rem 0}
.bar>span{display:block;height:100%;background:#3d7eff}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin-top:.5rem}
th,td{padding:.4rem .55rem;text-align:right;border-bottom:1px solid #262b36}
th:first-child,td:first-child{text-align:left}
th{color:#9aa4b2;font-weight:600}
.WIN{color:#4ade80}.LOSS{color:#f87171}.FLAT{color:#9aa4b2}
.pending{color:#9aa4b2;font-style:italic}
"""


def render_html(ledger: pd.DataFrame, log: pd.DataFrame,
                base_dir: str = '.', spread_pips: float = DEFAULT_SPREAD_PIPS) -> str:
    """The /h1-direction page: per-version hit rates, never a blended headline."""
    summaries = summarize_by_version(ledger, log)
    trained = _meta_trained_through(base_dir)
    served = _served_meta(base_dir)
    served_version = served.get('model_version')

    parts = [f'<style>{_CSS}</style>',
             '<h1>H1 next-bar direction — observational forward ledger</h1>',
             '<div class="sub">H_dir.1 architecture, refit on full history. '
             'Simulated only — no orders, no sizing, no stops.</div>',
             '<div class="nav"><a href="/">← dashboard</a> · '
             '<a href="/history">prediction-vs-actual history</a> · '
             '<a href="/paper-trading">paper-trading ledgers</a> · '
             '<a href="/kronos-volatility">Kronos volatility record</a></div>']

    # WHICH VERSION IS LIVE, and how far its evidence has actually got. These two
    # lines are the first thing on the page because they are the two questions a
    # reader actually has.
    served_settled = next((s['settled'] for s in summaries
                           if s['model_version'] == served_version), 0)
    parts.append(
        '<div class="card"><div class="vh">Currently served: '
        f'<code>{served_version or "— none loaded —"}</code></div>'
        f'<div class="meta">trained through {str(served.get("train_end", ""))[:10] or "—"}'
        f' · {served.get("n_train_rows", "—")} training rows</div>'
        f'<div class="bar"><span style="width:'
        f'{100.0 * min(1.0, served_settled / TARGET_OBSERVATIONS):.2f}%"></span></div>'
        f'<div class="meta"><b>{served_settled} of ~{TARGET_OBSERVATIONS} settled '
        'observations</b> — the number that decides whether this model has said '
        'anything yet.</div></div>')

    parts.append(
        '<div class="note"><b>What this model is worth.</b> The out-of-sample '
        'confirmation (+3.10pp on a reserved test block) belongs to the '
        '<b>[0:70%] model</b> and NOT to the full-history model served here, which '
        'has <b>seen that test block during training</b>. This model is '
        '<b>not validated out of sample</b> — the same distinction '
        '<code>h1_direction_meta.json</code> records. The ledger below is its only '
        'evidence.</div>')

    if len(summaries) > 1:
        parts.append(
            f'<div class="warn"><b>{len(summaries)} model versions in this '
            'ledger.</b> Rates are per version; they are not comparable across a '
            'retrain.</div>')

    if not summaries:
        n_calls = 0 if log is None or not len(log) else len(log)
        parts.append(
            f'<div class="card"><div class="vh">No settled observations yet</div>'
            f'<div class="meta">{n_calls} call(s) logged. A prediction settles '
            f'once its forecast bar has closed.</div>'
            f'<div class="meta">0 of ~{TARGET_OBSERVATIONS} settled observations '
            'needed for a meaningful forward answer at this effect size.</div>'
            '</div>')

    for s in summaries:
        through = trained.get(s['model_version'], '—')
        parts.append(
            f'<div class="card"><div class="vh">{s["model_version"]}</div>'
            f'<div class="meta">trained through {through} &middot; '
            f'{s["raw_calls"] if s["raw_calls"] is not None else "?"} raw calls '
            f'&rarr; <b>{s["settled"]}</b> settled (deduplicated to one per '
            f'forecast bar) &middot; {s["first_bar"]} .. {s["last_bar"]}</div>'
            f'<div class="meta">hit rate <b>{s["hit_rate"]:.4f}</b> '
            f'({s["wins"]}W / {s["losses"]}L) &middot; net '
            f'{s["net_pips"]:+.1f} pips at {spread_pips} pip round-trip spread</div>'
            f'<div class="bar"><span style="width:{s["progress_pct"]:.2f}%"></span></div>'
            f'<div class="meta">{s["settled"]} of ~{TARGET_OBSERVATIONS} settled '
            f'observations needed &mdash; {s["remaining_to_target"]} to go</div>'
            '</div>')

    if ledger is not None and len(ledger):
        parts.append('<div class="card"><div class="vh">Settled positions</div>'
                     '<table><tr><th>forecast bar</th><th>version</th>'
                     '<th>call</th><th>entry</th><th>realised close</th>'
                     '<th>net pips</th><th>cum pips</th><th>outcome</th></tr>')
        for _i, r in ledger.tail(200).iloc[::-1].iterrows():
            parts.append(
                f'<tr><td>{r["forecast_bar_start"]}</td>'
                f'<td>{r["model_version"]}</td><td>{r["direction"]}</td>'
                f'<td>{r["entry"]:.5f}</td><td>{r["exit"]:.5f}</td>'
                f'<td>{r["net_pips"]:+.1f}</td><td>{r["cum_net_pips"]:+.1f}</td>'
                f'<td class="{r["outcome"]}">{r["outcome"]}</td></tr>')
        parts.append('</table></div>')

    if log is not None and len(log):
        pending = log[~log['forecast_bar_start'].isin(
            ledger['forecast_bar_start'] if ledger is not None and len(ledger) else [])]
        if len(pending):
            parts.append(
                f'<div class="card"><div class="vh">Pending</div>'
                f'<div class="meta">{len(pending)} logged call(s) whose forecast '
                'bar has not closed yet (or was a duplicate inside an already-'
                'settled bar).</div></div>')
    return '\n'.join(parts)
