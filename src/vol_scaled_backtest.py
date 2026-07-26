"""
Volatility-scaled position-SIZING overlay — RESEARCH-ONLY RETROSPECTIVE
backtest on the already-settled paper-trading ledgers.

============================== HARD BOUNDARY ===============================
This module is a descriptive "what-if" report over HISTORICAL, ALREADY-
SETTLED ledger rows (results/paper_trading_log_baseline.csv,
results/paper_trading_log_macro.csv), produced by `src/paper_trading.py` and
left ENTIRELY UNCHANGED here — this module never imports `build_ledger`,
`summarize`, or `build_all_ledgers`, never edits `src/paper_trading.py`, and
never writes back to either ledger CSV. It does NOT:
  - change which direction was predicted — only re-weights the SIZE of a
    day's already-realized P&L in a parallel "what-if" column;
  - add any real position-sizing/execution/broker code path;
  - touch serving/API, or how future live positions get logged.
Real capital deployment remains a separate, explicit, FUTURE conversation
requiring the owner's direct approval — exactly as `src/paper_trading.py`'s
own module docstring already states. This script only ever produces a
descriptive report, logged to `results/vol_scaled_sizing_backtest.csv`.
==============================================================================

Why
---
Vol-targeting (size DOWN when expected volatility is high, size UP when it's
low) is a well-known real-world sizing overlay. This asks: on the SAME
already-realized daily P&L this project's live paper-trading ledgers have
accumulated, would sizing each day's position by the ALREADY-VALIDATED
volatility ensemble's own forecast (`models/volatility/`, the ONE neural
family in this project with a CI-confirmed edge over its honest GARCH
baseline) have changed the risk-adjusted outcome? Reuses
`load_frozen_volatility_ensemble` / `batch_predict_frozen_ensemble_vol_pct`
(`src/volatility.py`) UNCHANGED — pure batch inference on the frozen
artifacts, the SAME idiom hypothesis #9 already established; no retraining.

Sizing formula (PRE-REGISTERED, fixed before looking at any P&L result)
-------------------------------------------------------------------------
    trailing_ref_vol[t]  = pandas `.rolling(window=252, min_periods=1).median()`
                            of `predicted_vol_pct` — expanding median until
                            252 days of history exist, a genuine rolling
                            252-day median thereafter. CAUSAL by construction
                            (a rolling window only ever looks backward), so
                            the weight is computable in real time, not just
                            in hindsight.
    vol_weight[t]         = trailing_ref_vol[t] / predicted_vol_pct[t],
                            CLIPPED to [0.25, 4.0] — a fixed a priori
                            risk-management guardrail against a pathological
                            blow-up when predicted vol is near zero, not
                            tuned after seeing results.
    weighted_net_return_pct[t] = net_return_pct[t] * vol_weight[t] — ONLY the
                            SIZE of that day's already-realized P&L changes;
                            the direction call itself (and hence the
                            ledger's own win/loss outcome) is untouched.

`predicted_vol_pct` is computed over daily bars aggregated from
`results/eurusd_h1.csv` (H1 history extends well past
`results/eurusd_features.csv`'s tail, covering the ledgers' recent dates) —
this daily aggregation is used ONLY here, to extend price history far enough
for this retrospective report; it does not touch training, serving, or any
live feature path. Row t's forecast is made using data through day t, FOR
day t+1 — exactly `src.inference.PredictionService._predict_volatility`'s own
live alignment (the forecast surfaces at `as_of_date` for `forecasting_date`
= `as_of_date`'s next session) — so it joins onto a ledger row by matching
`as_of_date` directly, no extra date-shifting.

Statistical test
------------------
NOT a classification-accuracy hypothesis (no direction/return prediction is
being judged), so this does NOT belong in `feature_hypothesis_log.csv` or
`volatility_hypothesis_log.csv` — logged instead to its own
`results/vol_scaled_sizing_backtest.csv`. A MOVING-BLOCK (circular) bootstrap
(block length ~20 trading days, respecting the serial correlation daily P&L
carries — a plain iid bootstrap would understate the true uncertainty) on
the DIFFERENCE in the Sharpe-like ratio (vol-scaled minus original, 2000
resamples) gives the 95% CI. Vol-targeting's real-world value proposition is
typically LOWER DRAWDOWN at similar return, not necessarily a higher raw
return — the drawdown comparison is reported with EQUAL weight to the
Sharpe comparison, never buried under a Sharpe-only framing.

Sample size
-----------
Paper trading only recently started accumulating (see
`results/paper_trading_log_*.csv`) — almost certainly too few settled
positions for a meaningful block bootstrap. Fewer than ~40 matched positions
(either variant) is flagged PRELIMINARY/DIRECTIONAL ONLY, never a KEEP/DROP
decision, mirroring every other small-n caveat in this project (e.g. the
weekly-COT "~100 validation weeks" precedent).

Run:  python -m src.vol_scaled_backtest
"""
import os

import numpy as np
import pandas as pd

from src.features import compute_features, PRICE_FEATURE_COLUMNS
from src.volatility import load_frozen_volatility_ensemble, batch_predict_frozen_ensemble_vol_pct

BACKTEST_LOG = 'results/vol_scaled_sizing_backtest.csv'
BACKTEST_LOG_COLUMNS = [
    'n', 'date', 'variant', 'arbiter', 'n_settled_positions', 'n_matched_with_vol_forecast',
    'vol_weight_min', 'vol_weight_median', 'vol_weight_max',
    'cum_net_return_pct_original', 'cum_net_return_pct_weighted',
    'sharpe_like_original', 'sharpe_like_weighted', 'delta_sharpe',
    'delta_sharpe_ci_low', 'delta_sharpe_ci_high', 'sharpe_ci_excludes_zero_improving',
    'max_drawdown_pct_original', 'max_drawdown_pct_weighted', 'drawdown_improved',
    'sample_caveat', 'verdict', 'notes',
]
TRADING_DAYS_PER_YEAR = 252
TRAILING_WINDOW = 252
VOL_WEIGHT_MIN = 0.25
VOL_WEIGHT_MAX = 4.0
BLOCK_LEN = 20
BOOTSTRAP_RESAMPLES = 2000
MIN_POSITIONS_FOR_BOOTSTRAP = 40

DEFAULT_LEDGERS = {
    'baseline': 'results/paper_trading_log_baseline.csv',
    'with_macro': 'results/paper_trading_log_macro.csv',
}
DEFAULT_H1_CSV = 'results/eurusd_h1.csv'


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def build_daily_ohlcv_from_h1(h1: pd.DataFrame, min_hours: int = 12) -> pd.DataFrame:
    """Collapse hourly bars into daily OHLCV (open=first, high=max, low=min,
    close=last, tick_volume=sum), UTC-midnight indexed, dropping any session
    with fewer than `min_hours` bars — the same completeness rule
    `src.h1_features.aggregate_daily_features` already applies. Used ONLY to
    extend price history far enough to cover the paper-trading ledgers'
    recent dates, which postdate `results/eurusd_features.csv`'s own tail —
    this does not touch training, serving, or any live feature path."""
    h1 = h1.sort_index()
    idx = h1.index
    day = (idx.tz_convert('UTC') if idx.tz is not None else idx).normalize()
    g = h1.groupby(day)
    daily = pd.DataFrame({
        'open': g['open'].first(), 'high': g['high'].max(),
        'low': g['low'].min(), 'close': g['close'].last(),
        'tick_volume': g['tick_volume'].sum(), '_n_hours': g.size(),
    })
    return daily[daily['_n_hours'] >= min_hours].drop(columns='_n_hours')


def compute_predicted_vol_series(base_dir='', h1_csv=DEFAULT_H1_CSV, artifacts=None) -> pd.Series:
    """The frozen 5-seed volatility ensemble's own forecast (see
    `src.volatility.load_frozen_volatility_ensemble` /
    `batch_predict_frozen_ensemble_vol_pct`, reused UNCHANGED — pure
    `.transform()`/`.predict()` batch inference, no `.fit()` anywhere) over
    daily bars aggregated from `h1_csv`. Row t's value is the forecast MADE
    using data through day t, FOR day t+1 (see module docstring) — a
    ledger's `as_of_date` matches this series' index directly."""
    h1 = pd.read_csv(_p(base_dir, h1_csv), index_col=0, parse_dates=True)
    daily = build_daily_ohlcv_from_h1(h1)
    feat = compute_features(daily)
    # compute_features is inference-safe (no dropna, so the latest live bar
    # survives) but leaves the genuine SMA_200/BB/ATR warm-up NaNs at the
    # START of the series -- PCA.transform rejects NaN outright, and by the
    # time this series reaches the paper-trading ledgers' 2026 dates it is
    # ~9 years past warm-up anyway, so dropping those leading rows costs
    # nothing relevant here (unlike add_advanced_features, this does NOT
    # drop the final row -- there is no target-based drop at all).
    feat = feat.dropna(subset=PRICE_FEATURE_COLUMNS)
    art = artifacts or load_frozen_volatility_ensemble(base_dir=base_dir)
    pred = batch_predict_frozen_ensemble_vol_pct(feat, art)
    return pd.Series(pred, index=feat.index, name='predicted_vol_pct')


def compute_trailing_ref_vol(vol_series: pd.Series, window: int = TRAILING_WINDOW) -> pd.Series:
    """CAUSAL trailing median: expanding median until `window` observations
    exist, a genuine rolling `window`-period median thereafter. A pandas
    `.rolling(window, min_periods=1)` already IS exactly this — for any row
    with fewer than `window` prior observations it uses however many are
    available (expanding), and once `window` observations exist it uses
    exactly the trailing `window` (rolling) — never a future value."""
    return vol_series.rolling(window=window, min_periods=1).median()


def compute_vol_weight(predicted_vol_pct, trailing_ref_vol,
                       lo=VOL_WEIGHT_MIN, hi=VOL_WEIGHT_MAX):
    """vol_weight = trailing_ref_vol / predicted_vol_pct, clipped to [lo, hi]
    — a fixed a priori risk-management guardrail against a pathological
    blow-up when predicted vol is near zero. A non-positive
    `predicted_vol_pct` (never expected from the real ensemble output, but
    guarded defensively) saturates to `hi` via `np.where` rather than
    raising or propagating inf/NaN into the weighted return."""
    predicted_vol_pct = np.asarray(predicted_vol_pct, dtype=float)
    trailing_ref_vol = np.asarray(trailing_ref_vol, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = trailing_ref_vol / predicted_vol_pct
    raw = np.where(predicted_vol_pct > 0, raw, hi)
    return np.clip(raw, lo, hi)


def _sharpe_like(returns: np.ndarray):
    """Identical formula to `src.paper_trading.summarize`: annualised
    mean/std of daily net return. Undefined (None) with <2 observations or
    zero dispersion."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) >= 2 and np.std(returns, ddof=1) > 0:
        return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return None


def _max_drawdown_pct(returns: np.ndarray) -> float:
    """Identical construction to `src.paper_trading.summarize`: max drawdown
    on the cumulative net-return curve, seeded at 0 (pre-trade flat)."""
    returns = np.asarray(returns, dtype=float)
    curve = np.concatenate(([0.0], np.cumsum(returns)))
    running_peak = np.maximum.accumulate(curve)
    return float((running_peak - curve).max())


def _circular_block_bootstrap_indices(n, block_len, rng):
    """One resample's row indices for a circular (wrap-around) moving-block
    bootstrap — handles n < block_len gracefully (the block simply wraps)
    rather than requiring a full block to fit, which matters for this
    project's very short settled-position samples."""
    block_len = max(1, min(block_len, n))
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend((start + k) % n for k in range(block_len))
    return np.array(idx[:n])


def bootstrap_delta_sharpe(original: np.ndarray, weighted: np.ndarray,
                           block_len=BLOCK_LEN, n_boot=BOOTSTRAP_RESAMPLES,
                           random_state=42):
    """Moving-BLOCK (circular) paired bootstrap on the difference in the
    Sharpe-like ratio (weighted - original). Blocks of `block_len` trading
    days (wrapping around) respect the serial correlation daily P&L carries
    — a plain iid bootstrap would understate the true uncertainty. The SAME
    resampled indices are applied to both series each draw, so the pairing
    (same underlying days) is preserved. Draws where either resampled
    series has an undefined Sharpe (zero dispersion) are excluded (NaN),
    reported as the degenerate-draw fraction.

    CRITICAL degenerate case, refused rather than silently misreported: when
    `n <= block_len`, a "block" the length of (or longer than) the whole
    sample is just a cyclic ROTATION of every original value exactly once —
    mean and std (hence Sharpe) are invariant to order, so EVERY resample
    gives the identical scalar delta and the "CI" would collapse to a single
    point. That is not a real uncertainty estimate; it would look like a
    razor-thin, highly significant CI purely as an artifact of too few
    positions relative to the pre-registered block length. This is refused
    (nan, nan, nan) rather than clamping the block length down to `n` and
    reporting a falsely confident interval.

    Returns (ci_low, ci_high, degenerate_frac) — (nan, nan, nan) if the
    n<=block_len case applies OR fewer than 2 valid draws survive."""
    original = np.asarray(original, dtype=float)
    weighted = np.asarray(weighted, dtype=float)
    n = len(original)
    if n <= block_len:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(random_state)
    deltas = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_len, rng)
        s_orig = _sharpe_like(original[idx])
        s_wt = _sharpe_like(weighted[idx])
        if s_orig is not None and s_wt is not None:
            deltas[i] = s_wt - s_orig
    valid = deltas[~np.isnan(deltas)]
    degenerate = float(np.mean(np.isnan(deltas)))
    if len(valid) >= 2:
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return float(lo), float(hi), degenerate
    return float('nan'), float('nan'), degenerate


def analyze_variant(ledger_path: str, vol_series: pd.Series, base_dir='',
                    random_state=42) -> dict:
    """The full what-if comparison for ONE variant's ledger: join the
    volatility forecast by `as_of_date`, apply the pre-registered sizing
    formula, and compare original vs vol-scaled P&L (cumulative return,
    Sharpe-like, max drawdown, block-bootstrap CI on the Sharpe delta).
    Returns a result dict; `n_settled_positions` / `n_matched_with_vol_forecast`
    are reported even when the match is empty."""
    path = _p(base_dir, ledger_path)
    if not os.path.exists(path):
        return {'n_settled_positions': 0, 'n_matched_with_vol_forecast': 0}

    ledger = pd.read_csv(path)
    taken = ledger[ledger['direction'] != 'FLAT'].copy()
    n_settled = len(taken)
    if n_settled == 0:
        return {'n_settled_positions': 0, 'n_matched_with_vol_forecast': 0}

    taken['as_of_date'] = pd.to_datetime(taken['as_of_date'])
    vs = vol_series.copy()
    if vs.index.tz is not None:
        vs.index = vs.index.tz_convert('UTC').tz_localize(None)
    taken['predicted_vol_pct'] = vs.reindex(taken['as_of_date']).to_numpy()
    matched = taken.dropna(subset=['predicted_vol_pct']).copy()
    n_matched = len(matched)
    if n_matched == 0:
        return {'n_settled_positions': n_settled, 'n_matched_with_vol_forecast': 0}

    trailing = compute_trailing_ref_vol(vs)
    matched['trailing_ref_vol'] = trailing.reindex(matched['as_of_date']).to_numpy()
    matched['vol_weight'] = compute_vol_weight(matched['predicted_vol_pct'], matched['trailing_ref_vol'])
    matched['weighted_net_return_pct'] = matched['net_return_pct'] * matched['vol_weight']

    original = matched['net_return_pct'].to_numpy(float)
    weighted = matched['weighted_net_return_pct'].to_numpy(float)

    sharpe_orig = _sharpe_like(original)
    sharpe_wt = _sharpe_like(weighted)
    delta_sharpe = (sharpe_wt - sharpe_orig) if (sharpe_orig is not None and sharpe_wt is not None) else None

    ci_lo, ci_hi, degenerate = (float('nan'), float('nan'), float('nan'))
    if delta_sharpe is not None:
        ci_lo, ci_hi, degenerate = bootstrap_delta_sharpe(original, weighted, random_state=random_state)
    ci_excludes_zero_improving = bool(ci_lo == ci_lo and ci_lo > 0)   # NaN-safe: NaN != NaN

    dd_orig = _max_drawdown_pct(original)
    dd_wt = _max_drawdown_pct(weighted)

    return {
        'n_settled_positions': n_settled,
        'n_matched_with_vol_forecast': n_matched,
        'vol_weight_min': float(matched['vol_weight'].min()),
        'vol_weight_median': float(matched['vol_weight'].median()),
        'vol_weight_max': float(matched['vol_weight'].max()),
        'cum_net_return_pct_original': float(original.sum()),
        'cum_net_return_pct_weighted': float(weighted.sum()),
        'sharpe_like_original': sharpe_orig,
        'sharpe_like_weighted': sharpe_wt,
        'delta_sharpe': delta_sharpe,
        'delta_sharpe_ci_low': ci_lo, 'delta_sharpe_ci_high': ci_hi,
        'boot_degenerate_frac': degenerate,
        'sharpe_ci_excludes_zero_improving': ci_excludes_zero_improving,
        'max_drawdown_pct_original': dd_orig,
        'max_drawdown_pct_weighted': dd_wt,
        'drawdown_improved': bool(dd_wt < dd_orig),
    }


def _upsert_backtest_log(row: dict, out_path: str):
    """Upsert one variant's row into the sizing-backtest log by `variant`
    name (re-run refreshes that row; the other variant's row is preserved) —
    same convention as `src.cot_weekly_check._upsert_weekly_log`."""
    new = pd.DataFrame([row], columns=BACKTEST_LOG_COLUMNS)
    if os.path.exists(out_path):
        log = pd.read_csv(out_path)
        log = log[log['variant'] != row['variant']]
        out = pd.concat([log, new], ignore_index=True)
    else:
        out = new
    out['n'] = range(1, len(out) + 1)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def run(base_dir='', ledgers=None, h1_csv=DEFAULT_H1_CSV, out_log=BACKTEST_LOG,
       random_state=42, register=True):
    ledgers = ledgers or DEFAULT_LEDGERS
    vol_series = compute_predicted_vol_series(base_dir=base_dir, h1_csv=h1_csv)

    print("=" * 78)
    print("VOLATILITY-SCALED SIZING OVERLAY — RESEARCH-ONLY RETROSPECTIVE BACKTEST")
    print("  HARD BOUNDARY: descriptive what-if report only. No execution/sizing code,")
    print("  no change to src/paper_trading.py or live logging. Real capital deployment")
    print("  remains a separate, future, owner-approved conversation.")
    print("=" * 78)

    out_path = _p(base_dir, out_log)
    rows = []
    for variant, ledger_path in ledgers.items():
        res = analyze_variant(ledger_path, vol_series, base_dir=base_dir, random_state=random_state)
        n_settled = res['n_settled_positions']
        n_matched = res['n_matched_with_vol_forecast']
        print(f"\n--- {variant} --- settled positions={n_settled}  matched w/ vol forecast={n_matched}")

        if n_matched < 2 or res.get('sharpe_like_original') is None:
            print("  Too few matched positions for a meaningful comparison — skipped.")
            row = {
                'n': 0, 'date': pd.Timestamp.utcnow().date().isoformat(), 'variant': variant,
                'arbiter': 'paper_trading_settled_ledger', 'n_settled_positions': n_settled,
                'n_matched_with_vol_forecast': n_matched,
                'sample_caveat': f'INSUFFICIENT ({n_matched} matched positions) — no comparison computed',
                'verdict': 'PRELIMINARY/INSUFFICIENT — too few settled+matched positions to report',
                'notes': 'research-only what-if report; no model/feature/serving/execution change',
            }
            rows.append(row)
            if register:
                _upsert_backtest_log(row, out_path)
            continue

        thin = n_matched < MIN_POSITIONS_FOR_BOOTSTRAP
        caveat = (f"PRELIMINARY — only {n_matched} matched settled positions (< "
                 f"{MIN_POSITIONS_FOR_BOOTSTRAP}); paper trading has only recently started "
                 f"accumulating. Treat as directional only, NOT a KEEP/DROP decision."
                 if thin else f"{n_matched} matched settled positions.")
        print(f"  {caveat}")
        print(f"  vol_weight distribution: min={res['vol_weight_min']:.3f}  "
              f"median={res['vol_weight_median']:.3f}  max={res['vol_weight_max']:.3f}")
        print(f"  cum net return %: original={res['cum_net_return_pct_original']:+.4f}  "
              f"vol-scaled={res['cum_net_return_pct_weighted']:+.4f}")
        print(f"  Sharpe-like: original={res['sharpe_like_original']:+.3f}  "
              f"vol-scaled={res['sharpe_like_weighted']:+.3f}  "
              f"delta={res['delta_sharpe']:+.3f}  95% CI[{res['delta_sharpe_ci_low']:+.3f}, "
              f"{res['delta_sharpe_ci_high']:+.3f}]  "
              f"{'CI ENTIRELY > 0 (improving)' if res['sharpe_ci_excludes_zero_improving'] else 'CI does not confirm improvement'}")
        print(f"  max drawdown %: original={res['max_drawdown_pct_original']:.4f}  "
              f"vol-scaled={res['max_drawdown_pct_weighted']:.4f}  "
              f"{'IMPROVED (lower)' if res['drawdown_improved'] else 'NOT improved'}")

        if thin:
            verdict = 'PRELIMINARY/DIRECTIONAL ONLY — sample too thin for a KEEP/DROP decision'
        elif res['sharpe_ci_excludes_zero_improving'] and res['drawdown_improved']:
            verdict = 'DIRECTIONALLY POSITIVE on both Sharpe and drawdown (still descriptive, not a production change)'
        elif res['sharpe_ci_excludes_zero_improving'] and not res['drawdown_improved']:
            verdict = 'MIXED — Sharpe CI improving but drawdown WORSE (do not let Sharpe framing bury this)'
        elif not res['sharpe_ci_excludes_zero_improving'] and res['drawdown_improved']:
            verdict = 'MIXED — drawdown improved but Sharpe CI does not confirm an edge'
        else:
            verdict = 'NO CONFIRMED IMPROVEMENT on Sharpe or drawdown at this sample size'
        print(f"  VERDICT: {verdict}")

        row = {
            'n': 0, 'date': pd.Timestamp.utcnow().date().isoformat(), 'variant': variant,
            'arbiter': 'paper_trading_settled_ledger',
            'n_settled_positions': n_settled, 'n_matched_with_vol_forecast': n_matched,
            'vol_weight_min': round(res['vol_weight_min'], 4),
            'vol_weight_median': round(res['vol_weight_median'], 4),
            'vol_weight_max': round(res['vol_weight_max'], 4),
            'cum_net_return_pct_original': round(res['cum_net_return_pct_original'], 4),
            'cum_net_return_pct_weighted': round(res['cum_net_return_pct_weighted'], 4),
            'sharpe_like_original': round(res['sharpe_like_original'], 4),
            'sharpe_like_weighted': round(res['sharpe_like_weighted'], 4),
            'delta_sharpe': round(res['delta_sharpe'], 4),
            'delta_sharpe_ci_low': round(res['delta_sharpe_ci_low'], 4) if res['delta_sharpe_ci_low'] == res['delta_sharpe_ci_low'] else np.nan,
            'delta_sharpe_ci_high': round(res['delta_sharpe_ci_high'], 4) if res['delta_sharpe_ci_high'] == res['delta_sharpe_ci_high'] else np.nan,
            'sharpe_ci_excludes_zero_improving': res['sharpe_ci_excludes_zero_improving'],
            'max_drawdown_pct_original': round(res['max_drawdown_pct_original'], 4),
            'max_drawdown_pct_weighted': round(res['max_drawdown_pct_weighted'], 4),
            'drawdown_improved': res['drawdown_improved'],
            'sample_caveat': caveat,
            'verdict': verdict,
            'notes': (f'research-only retrospective what-if sizing overlay on the already-settled '
                     f'{variant} ledger; vol_weight=trailing_ref_vol/predicted_vol_pct clipped to '
                     f'[{VOL_WEIGHT_MIN},{VOL_WEIGHT_MAX}]; trailing_ref_vol=causal rolling(252,'
                     f'min_periods=1).median(); block-bootstrap block_len={BLOCK_LEN}, '
                     f'n_boot={BOOTSTRAP_RESAMPLES}, boot_degenerate_frac='
                     f'{res["boot_degenerate_frac"]:.3f}. Direction calls untouched; no model/'
                     f'feature/serving/execution change; real capital deployment is a separate '
                     f'future conversation.'),
        }
        rows.append(row)
        if register:
            _upsert_backtest_log(row, out_path)

    if register:
        print(f"\nLogged: {out_path}")
    return rows


if __name__ == '__main__':
    run()
