"""Forward record for the Kronos VOLATILITY channel: log, settlement, view.

WHY A SEPARATE MODULE FROM serving.py. That module records the retired DIRECTION
channel at pred_len=1. Its rows are settled against a different question, at a
different horizon, from a different model. Appending volatility rows to it would
silently mix two incomparable records in one file.

WHAT THE FORWARD RECORD IS FOR. The clean-window numbers compare Kronos against
persistence and GARCH -- NOT against this project's own volatility ensemble,
which has never been done on any row. Until that comparison exists this channel
is OBSERVATIONAL and the forward ledger is the only evidence. Calibration is the
thing worth watching, so the view leads with the reliability diagram.
"""

import os

import numpy as np
import pandas as pd

LOG_PATH = 'results/external_kronos/kronos_vol_log.csv'
LEDGER_PATH = 'results/external_kronos/kronos_vol_ledger.csv'

# A prediction whose window was already closed, or whose feed clock could not be
# confirmed, is not a forward observation and must never settle into the record.
EXCLUDED_STATUSES = ('already_closed', 'market_closed', 'clock_unconfirmed')

LOG_COLUMNS = [
    'logged_utc', 'as_of_bar_close', 'as_of_close',
    'forecast_window_start', 'forecast_window_end', 'minutes_remaining',
    'forecast_bar_status', 'spans_weekend_gap', 'context_bars', 'trailing_bars',
    'horizon_bars', 'sample_count', 'temperature', 'top_p',
    'pred_vol_pct_24h', 'pred_vol_pct_24h_scaled', 'pred_vol_scale_constant',
    'path_endpoint_sd_pct', 'p_vol_amp_raw', 'p_vol_amp_calibrated',
    'trailing_rv_24h_pct', 'n_paths', 'gen_distinct_endpoints',
    'calibration_fitted_utc', 'model', 'model_version', 'device', 'data_source',
]

LEDGER_COLUMNS = LOG_COLUMNS + [
    'realised_rv_24h_pct', 'abs_err_raw_pct', 'abs_err_scaled_pct',
    'amplified', 'brier_raw', 'brier_calibrated',
    'persistence_abs_err_pct', 'settled_bars',
]

# How many settled rows before the forward record can say anything.
#
# DERIVATION. For an AUC of A, SE(AUC) ~ sqrt(A(1-A)/(n/2)) -- the n/2 because a
# rank statistic is driven by the smaller of the two outcome classes, so a
# balanced sample of n rows contributes ~n/2 effective pairs. At the measured
# A = 0.6889 the distance to chance is A - 0.5 = 0.189, so
#     n =  50 -> SE 0.0926 -> 2.0 SE above 0.5  (borderline)
#     n = 100 -> SE 0.0655 -> 2.9 SE            (solid)
#     n = 150 -> SE 0.0535 -> 3.5 SE
# 100 is therefore the point where this channel's effect, IF it is real at the
# clean-window size, becomes distinguishable from chance.
#
# WHY IT WAS 500. That figure was carried over from the DIRECTION channel
# (serving.py, TARGET_OBSERVATIONS=1000 scaled down), where the measured effect
# was AUC 0.509 -- a distance to chance of 0.009, about 20x smaller than this
# one. An effect 20x larger needs far fewer rows to see; reusing the direction
# channel's n was importing another channel's power requirement.
TARGET_SETTLED = 100

# The clean-window amplification AUC these power numbers are computed at. It is
# an ASSUMPTION carried forward from a different (retrospective) sample, not a
# forward measurement -- se_above_half() answers "if the forward effect matched
# the clean window, how strong would the evidence be at this n", never "how
# strong is the forward effect".
CLEAN_WINDOW_AUC = 0.6889


def se_above_half(n_settled: int, auc: float = CLEAN_WINDOW_AUC):
    """How many standard errors above 0.5 an AUC of `auc` would sit at this n.

    Returns None below n=2, where the quantity is not defined. See the
    derivation beside TARGET_SETTLED.
    """
    if n_settled is None or n_settled < 2:
        return None
    se = np.sqrt(auc * (1.0 - auc) / (n_settled / 2.0))
    if se <= 0:
        return None
    return float((auc - 0.5) / se)


def log_prediction(response: dict, log_path: str = LOG_PATH, base_dir: str = '.') -> str:
    """Append one served response. Logging must NEVER break a response, so the
    caller wraps this; here we only guarantee a stable column set."""
    path = os.path.join(base_dir, log_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {c: response.get(c) for c in LOG_COLUMNS}
    row['logged_utc'] = pd.Timestamp.utcnow().tz_localize(None).isoformat()
    df = pd.DataFrame([row], columns=LOG_COLUMNS)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return path


def read_log(log_path: str = LOG_PATH, base_dir: str = '.') -> pd.DataFrame:
    path = os.path.join(base_dir, log_path)
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path)


def _realised(closes: pd.Series, anchor_ts, end_ts):
    """Paper eq.12 realised volatility over the actual bars in (anchor, end],
    anchored at the last actual close so it counts the same returns the forecast
    did. Returns (rv_pct, n_bars) or (None, n) when the window has not closed.

    RIGHT ENDPOINT IS INCLUSIVE. `forecast_window_end` (src/inference.py,
    predict_kronos_volatility) is built as `forward_bar_index(anchor,
    horizon_bars)[-1] + 1h` -- the CLOSE of the horizon_bars-th forecast bar,
    not one bar past it. So the horizon_bars closes the forecast actually
    covers are `end_ts` itself and the horizon_bars-1 before it; excluding
    `end_ts` (a strict `<`) drops one real bar and a 24-bar window can never
    reach n>=24. `realised_vol()` in predict.py (an array whose first element
    is the anchor close, producing exactly `pred_len` returns) and the
    clean-window act_rv both anchor the same way -- excluding end_ts here
    would settle against a different quantity than the one AUC 0.6889 was
    measured on.
    """
    idx = pd.DatetimeIndex(closes.index)
    anchor = idx[idx <= anchor_ts]
    if not len(anchor):
        return None, 0
    a = anchor[-1]
    window = closes[(idx > a) & (idx <= end_ts)]
    if not len(window):
        return None, 0
    series = np.concatenate([[float(closes.loc[a])], window.to_numpy(float)])
    lr = np.diff(np.log(np.maximum(series, 1e-12)))
    return float(np.sqrt((lr ** 2).sum()) * 100.0), int(len(window))


def build_ledger(log_path: str = LOG_PATH, base_dir: str = '.',
                 closes: pd.Series = None, horizon_bars: int = 24) -> pd.DataFrame:
    """Settle every logged prediction whose 24-bar window has fully closed.

    Records realised volatility, the absolute error of BOTH continuous forecasts
    (raw and our scaled one), whether amplification actually occurred, and the
    Brier contribution of BOTH probability fields -- so the raw model output and
    our correction are always visible side by side and neither can hide behind
    the other.
    """
    log = read_log(log_path, base_dir)
    if not len(log):
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    if closes is None:
        from ...h1_direction_serving import realised_h1_closes
        closes = realised_h1_closes(base_dir=base_dir)
    if closes is None or not len(closes):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    closes = closes.sort_index()

    log = log[~log['forecast_bar_status'].isin(EXCLUDED_STATUSES)]
    log = log.drop_duplicates(subset=['as_of_bar_close', 'model_version'], keep='last')

    rows = []
    for _, r in log.iterrows():
        try:
            anchor = pd.Timestamp(r['as_of_bar_close'])
            end = pd.Timestamp(r['forecast_window_end'])
        except Exception:
            continue
        if anchor.tzinfo is not None:
            anchor = anchor.tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_localize(None)
        rv, n = _realised(closes, anchor, end)
        # Only a FULLY closed window settles; a partial one would understate
        # realised volatility and flatter the forecast.
        if rv is None or n < horizon_bars:
            continue
        trail = float(r['trailing_rv_24h_pct'])
        amplified = int(rv > trail)
        d = dict(r)
        d.update({
            'realised_rv_24h_pct': rv,
            'abs_err_raw_pct': abs(float(r['pred_vol_pct_24h']) - rv),
            'abs_err_scaled_pct': abs(float(r['pred_vol_pct_24h_scaled']) - rv),
            'amplified': amplified,
            'brier_raw': (float(r['p_vol_amp_raw']) - amplified) ** 2,
            'brier_calibrated': (float(r['p_vol_amp_calibrated']) - amplified) ** 2,
            'persistence_abs_err_pct': abs(trail - rv),
            'settled_bars': n,
        })
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=LEDGER_COLUMNS)


def build_and_save(log_path: str = LOG_PATH, base_dir: str = '.',
                   ledger_path: str = LEDGER_PATH) -> pd.DataFrame:
    ledger = build_ledger(log_path, base_dir)
    path = os.path.join(base_dir, ledger_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ledger.to_csv(path, index=False)
    return ledger


def reliability_table(ledger: pd.DataFrame, column: str = 'p_vol_amp_calibrated',
                      n_bins: int = 5) -> list:
    """Running reliability of the calibrated amplification probability. This is
    the headline of the forward record: the clean window showed the RAW
    probability discriminates but is badly calibrated, so whether our frozen
    correction actually holds forward is the open question."""
    if not len(ledger):
        return []
    p = ledger[column].to_numpy(float)
    y = ledger['amplified'].to_numpy(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        out.append({'bin': '[%.1f, %.1f)' % (lo, hi), 'n': int(m.sum()),
                    'mean_p': float(p[m].mean()) if m.sum() else None,
                    'realised': float(y[m].mean()) if m.sum() else None})
    return out


def summarize_by_version(ledger: pd.DataFrame) -> list:
    """One row per model_version: MAE of our scaled forecast against persistence
    on the SAME settled rows, plus both Brier scores against a constant."""
    if not len(ledger):
        return []
    out = []
    for ver, g in ledger.groupby('model_version'):
        base = float(g['amplified'].mean())
        out.append({
            'model_version': ver,
            'n_settled': int(len(g)),
            'mae_scaled': float(g['abs_err_scaled_pct'].mean()),
            'mae_raw': float(g['abs_err_raw_pct'].mean()),
            'mae_persistence': float(g['persistence_abs_err_pct'].mean()),
            'brier_calibrated': float(g['brier_calibrated'].mean()),
            'brier_raw': float(g['brier_raw'].mean()),
            'brier_constant': float(base * (1 - base)),
            'amplified_rate': base,
        })
    return out


_NAV = ('<p class="nav">Also: <a href="/">dashboard</a> &nbsp;·&nbsp; '
        '<a href="/history">history</a> &nbsp;·&nbsp; '
        '<a href="/paper-trading">paper trading</a> &nbsp;·&nbsp; '
        '<a href="/h1-direction">H1 direction</a> &nbsp;·&nbsp; '
        '<a href="/kronos-direction">Kronos direction (retired)</a></p>')

_CSS = """<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;max-width:60rem;
padding:0 1rem;color:#1a1a1a;background:#fafafa}
h1{font-size:1.5rem;margin-bottom:.25rem}h2{font-size:1.1rem;margin-top:2rem}
table{border-collapse:collapse;width:100%;margin:.75rem 0;font-size:.9rem}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#f0f0f0}
.warn{background:#fff8e1;border-left:4px solid #ffa000;padding:.75rem 1rem;margin:1rem 0}
.gate{background:#eef4ff;border-left:4px solid #3d6fd0;padding:.75rem 1rem;margin:1rem 0}
.nav{font-size:.9rem;color:#555}.muted{color:#666;font-size:.85rem}
@media (prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}
th{background:#242424}th,td{border-color:#333}.warn{background:#2b2410}
.gate{background:#141c2b}.muted,.nav{color:#aaa}}
</style>"""


def render_html(ledger: pd.DataFrame, log: pd.DataFrame, base_dir: str = '.') -> str:
    n_log = len(log)
    n_settled = len(ledger)
    parts = ['<h1>Kronos volatility — forward record</h1>',
             '<p class="muted">External foundation model (Kronos-mini, 4.1M), zero-shot, '
             'pred_len=24. Observational.</p>', _NAV]

    parts.append(
        '<div class="gate"><strong>What still gates a real claim.</strong> This channel is '
        '<strong>OBSERVATIONAL</strong> until Kronos is tested against <em>this project\'s own '
        'volatility ensemble</em> (<code>models/volatility/</code>) on identical rows. '
        '<strong>That comparison has never been made, on any row.</strong> The clean-window '
        'numbers below describe a comparison against persistence and GARCH — not against what '
        'we already serve. Until then the forward record on this page is the only evidence, '
        'and the incremental-over-persistence result was found <strong>post-hoc</strong>.</div>')

    parts.append(
        '<div class="warn"><strong>Direction is not served.</strong> The Kronos direction '
        'channel was retired after being measured dead three ways: next-bar AUC 0.509, 24-bar '
        'AUC 0.517 (Brier skill −0.62), and cross-sectional RankIC fully attributable to a '
        'one-line reversal ranking. No field on this page is a direction call.</div>')

    # The caveats must be VISIBLE PAGE TEXT, not merely module constants a
    # reader would have to go looking for.
    parts.append(
        '<p class="muted">Kronos pre-training extends to <strong>June 2024</strong>; the '
        'published weights were uploaded 2025-06-30 and are unchanged since, so only '
        '2024-07-01 onward is out-of-sample. Anything earlier is in-corpus and no number '
        'measured there would mean anything. The model does not encode elapsed time, so a '
        '24-bar window that crosses the weekend break is flagged '
        '<code>spans_weekend_gap</code>.</p>')

    parts.append('<h2>Settlement progress</h2>')
    parts.append('<p>%d logged &middot; <strong>%d settled</strong> of the ~%d needed before '
                 'this record says anything.</p>'
                 % (n_log, n_settled, TARGET_SETTLED))
    # Show the evidence the CURRENT n would carry, so the reader watches evidence
    # accumulate rather than a distant fixed number.
    se_mult = se_above_half(n_settled)
    if se_mult is None:
        parts.append('<p class="muted">At n=%d there is not yet enough to state in standard '
                     'errors. Target n=%d is where an effect the size of the clean window\'s '
                     'would sit ~2.9 SE above chance.</p>' % (n_settled, TARGET_SETTLED))
    else:
        parts.append('<p class="muted">At the current n=%d, an effect the size of the '
                     'clean window\'s (AUC %.4f) would sit <strong>%.1f SE</strong> above '
                     '0.5. That is a statement about POWER, not a forward result: it says '
                     'what this sample could resolve if the effect held, not that it has '
                     'held. ~2 SE is borderline, ~2.9 SE (n=%d) solid.</p>'
                     % (n_settled, CLEAN_WINDOW_AUC, se_mult, TARGET_SETTLED))

    parts.append('<h2>Calibration of p_vol_amp_calibrated</h2>')
    rel = reliability_table(ledger)
    if rel:
        parts.append('<table><tr><th>predicted bin</th><th>n</th><th>mean p</th>'
                     '<th>realised rate</th><th>gap</th></tr>')
        for r in rel:
            if not r['n']:
                parts.append('<tr><td>%s</td><td>0</td><td>—</td><td>—</td><td>—</td></tr>'
                             % r['bin'])
            else:
                parts.append('<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td>'
                             '<td>%+.3f</td></tr>'
                             % (r['bin'], r['n'], r['mean_p'], r['realised'],
                                r['realised'] - r['mean_p']))
        parts.append('</table>')
    else:
        parts.append('<p class="muted">No settled rows yet. A 24-bar window takes a full '
                     'trading day to close.</p>')

    # A constant forecast is trivially perfect on a one-outcome sample, and its
    # Brier of 0.0000 in the table below would otherwise read as a constant
    # beating every model. Shown ONLY while the outcomes carry no variation.
    if n_settled and ledger['amplified'].nunique() == 1:
        only = int(ledger['amplified'].iloc[0])
        parts.append('<div class="warn"><strong>Every settled row has the same outcome '
                     '(%d of %d %s).</strong> On a sample with no variation the constant '
                     'forecast is trivially perfect, so <em>Brier const.</em> below is '
                     '0.0000 <strong>by construction</strong> — it does not mean a constant '
                     'beats these models, it means the sample cannot yet separate any of '
                     'them. Nothing in the table below is evidence until both outcomes '
                     'appear.</div>' % (n_settled, n_settled,
                                        'amplified' if only else 'not amplified'))

    parts.append('<h2>Running error, by model version</h2>')
    rows = summarize_by_version(ledger)
    if rows:
        parts.append('<table><tr><th>model_version</th><th>settled</th>'
                     '<th>MAE scaled</th><th>MAE raw</th><th>MAE persistence</th>'
                     '<th>Brier calib.</th><th>Brier raw</th><th>Brier const.</th></tr>')
        for r in rows:
            parts.append('<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td><td>%.4f</td>'
                         '<td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>'
                         % (r['model_version'], r['n_settled'], r['mae_scaled'],
                            r['mae_raw'], r['mae_persistence'], r['brier_calibrated'],
                            r['brier_raw'], r['brier_constant']))
        parts.append('</table>')
        parts.append('<p class="muted">MAE is in the volatility family\'s units '
                     '(|log return| × 100) over the 24-bar window. GARCH is reported in the '
                     'clean-window evaluation; it is not refit per request.</p>')
    else:
        parts.append('<p class="muted">Nothing settled yet.</p>')

    parts.append('<h2>What was measured once, on the clean window</h2>')
    parts.append(
        '<table><tr><th>quantity</th><th>value</th><th>95% CI</th><th>n</th></tr>'
        '<tr><td>amplification AUC (mini)</td><td>0.6889</td><td>[0.6449, 0.7309]</td>'
        '<td>519</td></tr>'
        '<tr><td>calibrated Brier, out-of-sample</td><td>0.23042</td>'
        '<td>vs constant 0.24999</td><td>260</td></tr>'
        '<tr><td>raw Brier / ECE</td><td>0.36294 / 0.36352</td>'
        '<td>worse than constant</td><td>519</td></tr>'
        '<tr><td>corr(Kronos RV, realised)</td><td>0.4585</td><td>[0.309, 0.575]</td>'
        '<td>519</td></tr>'
        '<tr><td>corr(persistence, realised)</td><td>0.4422</td><td>[0.297, 0.565]</td>'
        '<td>519</td></tr>'
        '<tr><td>corr(GARCH(1,1), realised)</td><td>0.3734</td><td>[0.247, 0.479]</td>'
        '<td>519</td></tr>'
        '<tr><td>Kronos − persistence (paired)</td><td>+0.0149</td>'
        '<td>[−0.0760, +0.0984] — <strong>not distinguishable</strong></td><td>519</td></tr>'
        '<tr><td>incremental over persistence</td><td>+0.2193</td>'
        '<td>[+0.1356, +0.2935] — <strong>post-hoc</strong></td><td>519</td></tr>'
        '</table>')
    parts.append('<p class="muted">Kronos does <strong>not</strong> beat a trailing window '
                 'alone. It adds information a trailing window does not contain.</p>')
    parts.append('<p class="muted"><em>External foundation model, zero-shot, observational. '
                 'Simulated only, <strong>not a trading instruction</strong> — no orders, '
                 'no position sizing, no stop-loss.</em></p>')
    parts.append(_NAV)
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Kronos volatility — forward record</title>' + _CSS +
            '</head><body>' + '\n'.join(parts) + '</body></html>')
