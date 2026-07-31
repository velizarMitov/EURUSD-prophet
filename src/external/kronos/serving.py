"""Forward log, simulated ledger and view for the Kronos observational model.

Mirrors src/h1_direction_serving.py deliberately -- same completed-bar
settlement, same group-by-model_version discipline, its OWN files so no existing
ledger can be affected. It reuses that module's `realised_h1_closes` rather than
re-deriving realised closes, because two implementations of "what did the bar
actually do" would eventually disagree.

SIMULATED ONLY. No order placement, no position sizing, no stop-loss. The one
thing worth watching here over time is CALIBRATION, not the hit rate: the
clean-window evaluation showed the model is sharp, so whether p_up means
anything is the open question.
"""

import os

import numpy as np
import pandas as pd

LOG_PATH = 'results/external_kronos/kronos_prediction_log.csv'
LEDGER_PATH = 'results/external_kronos/kronos_ledger.csv'

# At a claimed edge near chance, ~1,000 settled bars is the order of magnitude
# needed to separate a 2-3pp effect from noise. Shown as progress, not a promise.
TARGET_OBSERVATIONS = 1000

LOG_COLUMNS = [
    'called_at_utc', 'model_version', 'p_up', 'direction', 'n_paths',
    'gen_sd', 'gen_distinct_closes', 'as_of_bar_close', 'forecast_bar_start',
    'forecast_bar_end', 'minutes_remaining_at_call', 'forecast_bar_status',
    'spans_weekend_gap', 'as_of_close', 'data_source',
]
LEDGER_COLUMNS = [
    'forecast_bar_start', 'model_version', 'p_up', 'direction', 'as_of_close',
    'realised_close', 'realised_direction', 'correct', 'spans_weekend_gap',
]
# Statuses that must never reach the forward evidence: an already-closed or
# market-closed bar was never a live call, and clock_unconfirmed means the base
# bar itself may be an hour off (the cold-start boundary case).
EXCLUDED_STATUSES = ('already_closed', 'market_closed', 'clock_unconfirmed')


def log_prediction(response: dict, log_path: str = LOG_PATH, base_dir: str = '.') -> str:
    """Append one row per call. Every call is logged, including excluded
    statuses -- the log is the record of what was served; the LEDGER is what
    counts as evidence."""
    path = os.path.join(base_dir, log_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {
        'called_at_utc': str(pd.Timestamp.utcnow()),
        'model_version': response.get('model_version'),
        'p_up': response.get('p_up'),
        'direction': response.get('direction'),
        'n_paths': response.get('n_paths'),
        'gen_sd': response.get('gen_sd'),
        'gen_distinct_closes': response.get('gen_distinct_closes'),
        'as_of_bar_close': response.get('as_of_bar_close'),
        'forecast_bar_start': response.get('forecast_bar_start'),
        'forecast_bar_end': response.get('forecast_bar_end'),
        'minutes_remaining_at_call': response.get('minutes_remaining'),
        'forecast_bar_status': response.get('forecast_bar_status'),
        'spans_weekend_gap': response.get('spans_weekend_gap'),
        'as_of_close': response.get('as_of_close'),
        'data_source': response.get('data_source'),
    }
    pd.DataFrame([row])[LOG_COLUMNS].to_csv(
        path, mode='a', header=not os.path.exists(path), index=False)
    return path


def read_log(log_path: str = LOG_PATH, base_dir: str = '.') -> pd.DataFrame:
    path = os.path.join(base_dir, log_path)
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=LOG_COLUMNS)


def build_ledger(log_path: str = LOG_PATH, base_dir: str = '.',
                 now=None, live: bool = True) -> pd.DataFrame:
    """Settle each prediction once its forecast bar has closed.

    Multiple calls inside one forecast bar are all LOGGED; only the FIRST
    settles, so refreshing the page cannot inflate the sample."""
    from ...h1_direction_serving import realised_h1_closes

    log = read_log(log_path, base_dir)
    if not len(log):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    log = log[~log['forecast_bar_status'].isin(EXCLUDED_STATUSES)].copy()
    if not len(log):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    log['forecast_bar_start'] = pd.to_datetime(log['forecast_bar_start'], errors='coerce')
    log = log.dropna(subset=['forecast_bar_start'])
    log = log.sort_values('called_at_utc').drop_duplicates(
        subset=['model_version', 'forecast_bar_start'], keep='first')

    try:
        closes = realised_h1_closes(base_dir=base_dir, now=now, live=live)
    except Exception:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    if closes is None or not len(closes):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    ci = pd.DatetimeIndex(closes.index)
    if ci.tz is not None:
        closes = closes.copy()
        closes.index = ci.tz_localize(None)

    rows = []
    for _, r in log.iterrows():
        start = pd.Timestamp(r['forecast_bar_start'])
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        if start not in closes.index:
            continue                       # the bar has not printed yet
        realised = float(closes.loc[start])
        base = float(r['as_of_close'])
        if realised == base:
            continue                       # a flat bar has no direction to score
        realised_dir = 'UP' if realised > base else 'DOWN'
        rows.append({
            'forecast_bar_start': start,
            'model_version': r['model_version'],
            'p_up': float(r['p_up']),
            'direction': r['direction'],
            'as_of_close': base,
            'realised_close': realised,
            'realised_direction': realised_dir,
            'correct': bool(r['direction'] == realised_dir),
            'spans_weekend_gap': bool(r.get('spans_weekend_gap', False)),
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def build_and_save(log_path: str = LOG_PATH, base_dir: str = '.',
                   now=None, live: bool = True) -> pd.DataFrame:
    ledger = build_ledger(log_path, base_dir, now=now, live=live)
    path = os.path.join(base_dir, LEDGER_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        ledger.to_csv(path, index=False)
    except OSError:
        pass
    return ledger


def reliability_table(ledger: pd.DataFrame, n_bins: int = 5) -> list:
    """Realised up-rate against predicted p_up. This is the thing worth watching:
    a sharp model whose bins do not track the diagonal is confidently wrong."""
    if not len(ledger):
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    p = ledger['p_up'].to_numpy(float)
    up = (ledger['realised_direction'] == 'UP').to_numpy()
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        m = (p >= lo) & (p < hi) if k < n_bins - 1 else (p >= lo) & (p <= hi)
        out.append({'bin': '[%.1f, %.1f%s' % (lo, hi, ']' if k == n_bins - 1 else ')'),
                    'n': int(m.sum()),
                    'mean_p_up': float(p[m].mean()) if m.any() else None,
                    'realised_up_rate': float(up[m].mean()) if m.any() else None})
    return out


def summarize_by_version(ledger: pd.DataFrame) -> list:
    """Never blend across model_version: a blended rate describes a model that is
    no longer served."""
    if not len(ledger):
        return []
    out = []
    for ver, g in ledger.groupby('model_version', sort=False):
        wk = g[g['spans_weekend_gap'].astype(bool)]
        nw = g[~g['spans_weekend_gap'].astype(bool)]
        brier = float(np.mean((g['p_up'].to_numpy(float)
                               - (g['realised_direction'] == 'UP').to_numpy(float)) ** 2))
        out.append({
            'model_version': ver,
            'settled': int(len(g)),
            'hits': int(g['correct'].sum()),
            'hit_rate': float(g['correct'].mean()),
            'brier': brier,
            'mean_p_up': float(g['p_up'].mean()),
            'weekend_n': int(len(wk)),
            'weekend_hit_rate': float(wk['correct'].mean()) if len(wk) else None,
            'nonweekend_n': int(len(nw)),
            'nonweekend_hit_rate': float(nw['correct'].mean()) if len(nw) else None,
            'reliability': reliability_table(g),
            'progress_pct': min(100.0, 100.0 * len(g) / TARGET_OBSERVATIONS),
        })
    return out


_NAV = ('<p class="nav">Also: <a href="/">dashboard</a> &nbsp;·&nbsp; '
        '<a href="/history">prediction-vs-actual history</a> &nbsp;·&nbsp; '
        '<a href="/paper-trading">paper-trading ledgers</a> &nbsp;·&nbsp; '
        '<a href="/h1-direction">H1 direction ledger</a></p>')


def render_html(ledger: pd.DataFrame, log: pd.DataFrame, base_dir: str = '.') -> str:
    """Forward view, grouped BY model_version, leading with CALIBRATION.

    p_up is printed as a NUMBER everywhere and never drawn as a gauge or bar:
    the clean-window evaluation showed how narrow the usable range is, and a
    gauge would make 0.53 read as a strong call."""
    from .loader import (CLEAN_WINDOW_START, CONTAMINATION_NOTE, DISCLAIMER,
                         MC_NOISE_ESTIMATE, TOKENIZER_NOTE, TRAINING_CUTOFF,
                         WEEKEND_NOTE, model_version)

    blocks = summarize_by_version(ledger)
    served = model_version()
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f7f7f9;color:#222}
    .wrap{max-width:1100px;margin:0 auto;padding:1.5rem 1rem 3rem}
    h1{font-size:1.35rem;margin:.2rem 0 .1rem} h2{font-size:1.05rem;margin:1.6rem 0 .5rem}
    .sub{color:#666;font-size:.86rem;margin:0 0 1rem}
    .card{background:#fff;border:1px solid #e3e3e8;border-radius:10px;padding:1rem 1.1rem;margin:.8rem 0}
    table{border-collapse:collapse;width:100%;font-size:.85rem}
    th,td{padding:.4rem .55rem;border-bottom:1px solid #eee;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{background:#fafafc;font-weight:600}
    .mono{font-family:ui-monospace,Consolas,monospace}
    .warn{background:#fff8e6;border:1px solid #f0dca8;border-radius:8px;padding:.75rem .9rem;font-size:.83rem;margin:.8rem 0}
    .bar{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin-top:.35rem}
    .bar span{display:block;height:100%;background:#5b8def}
    .nav{font-size:.85rem;color:#555;margin-top:2rem}
    .muted{color:#777}
    """
    h = ['<style>%s</style><div class="wrap">' % css]
    h.append('<h1>Kronos — next-H1-bar forecast (observational)</h1>')
    h.append('<p class="sub">External foundation model, zero-shot. '
             'Served version <span class="mono">%s</span></p>' % served)

    h.append('<div class="warn"><b>What this is.</b> Kronos emits a distribution over '
             'sampled price paths; <b>p_up</b> is the share of paths closing above the '
             'last actual close, and it is the headline number — <i>direction</i> is a '
             'lossy reduction kept only so the ledger has something to settle. '
             'Run-to-run Monte-Carlo spread of p_up is <b>%.1f pp</b> at %d paths, so '
             'small differences in p_up are not meaningful.<br><br>'
             '<b>Contamination.</b> %s<br><br>'
             '<b>Tokenizer.</b> %s<br><br>'
             '<b>Weekends.</b> %s</div>'
             % (100 * MC_NOISE_ESTIMATE, 30, CONTAMINATION_NOTE, TOKENIZER_NOTE, WEEKEND_NOTE))

    if not blocks:
        h.append('<div class="card"><b>No settled observations yet.</b><br>'
                 '<span class="muted">Predictions settle once their forecast hour '
                 'closes. Calls made when the market was closed, when the bar had '
                 'already closed, or when the feed clock was unconfirmed are logged '
                 'but never counted.</span></div>')
    for b in blocks:
        h.append('<div class="card">')
        h.append('<h2 class="mono">%s</h2>' % b['model_version'])
        h.append('<p class="sub">%d of ~%d settled observations '
                 '(%.1f%%)</p><div class="bar"><span style="width:%.1f%%"></span></div>'
                 % (b['settled'], TARGET_OBSERVATIONS, b['progress_pct'], b['progress_pct']))
        h.append('<table><tr><th>metric</th><th>value</th><th>n</th></tr>')
        h.append('<tr><td>hit rate</td><td>%.4f</td><td>%d</td></tr>'
                 % (b['hit_rate'], b['settled']))
        h.append('<tr><td>Brier score</td><td>%.4f</td><td>%d</td></tr>'
                 % (b['brier'], b['settled']))
        h.append('<tr><td>mean p_up</td><td>%.4f</td><td>%d</td></tr>'
                 % (b['mean_p_up'], b['settled']))
        for lab, key_n, key_r in (('spans weekend gap', 'weekend_n', 'weekend_hit_rate'),
                                  ('no weekend gap', 'nonweekend_n', 'nonweekend_hit_rate')):
            r = b[key_r]
            h.append('<tr><td>%s</td><td>%s</td><td>%d</td></tr>'
                     % (lab, '%.4f' % r if r is not None else '—', b[key_n]))
        h.append('</table>')
        h.append('<h2>Calibration — realised up-rate vs predicted p_up</h2>')
        h.append('<table><tr><th>p_up bin</th><th>n</th><th>mean p_up</th>'
                 '<th>realised up-rate</th></tr>')
        for r in b['reliability']:
            h.append('<tr><td class="mono">%s</td><td>%d</td><td>%s</td><td>%s</td></tr>'
                     % (r['bin'], r['n'],
                        '%.4f' % r['mean_p_up'] if r['mean_p_up'] is not None else '—',
                        '%.4f' % r['realised_up_rate'] if r['realised_up_rate'] is not None else '—'))
        h.append('</table></div>')

    n_log = len(log) if log is not None else 0
    h.append('<p class="sub">%d calls logged in total; only the first call inside each '
             'forecast bar settles. Training cutoff %s — the only out-of-sample period '
             'is %s onward.</p>' % (n_log, TRAINING_CUTOFF, CLEAN_WINDOW_START))
    h.append('<div class="warn">%s</div>' % DISCLAIMER)
    h.append(_NAV)
    h.append('</div>')
    return '\n'.join(h)
