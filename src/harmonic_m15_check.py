"""
M15 harmonic-pattern event-conditional model — hypotheses #5/#6 (H1.5/H1.6),
SAME family as `results/harmonic_pattern_hypothesis_log.csv` (n=4 already,
from H1.1-H1.4 on H1 bars, `src/harmonic_event_check.py`). This does NOT
reset the family's Bonferroni budget — it is the IDENTICAL underlying
question (does XABCD harmonic-pattern completion + triple-barrier labeling
carry a real edge) asked on a FINER timeframe, so it shares the SAME log and
the family bar tightens to `0.05/6 ≈ 0.0083` each (computed dynamically at
run time from the log's current hypothesis count, same convention as
`harmonic_event_check.run()`).

STEP 0 — verified before building
-----------------------------------
M15 EURUSD history was fetched from MT5 ONLY
(`src.live_data.fetch_m15_market_data` → `mt5.copy_rates_from_pos(symbol,
mt5.TIMEFRAME_M15, 0, bars)` — the SAME bar-count OHLCV rates API as the
existing H1 fetch, just swapping the timeframe constant; NOT tick-level data
(`copy_ticks_from`/`copy_ticks_range`), and deliberately NO Yahoo Finance
fallback for this timeframe unlike the D1/H1 chains — an on-disk cache
remains the OFFLINE fallback, MT5 is the sole LIVE source). This broker's
MT5 terminal retains M15 history back to 1971 (the same synthetic-backfill
start date as this project's other bundled EURUSD series) — comfortably
deep. Rather than pull the full ~55-year depth (needless compute, and
pre-1999 EURUSD is a legacy DEM-converted backfill already treated elsewhere
in this project as usable-but-secondary), 350,000 M15 bars were fetched into
`results/eurusd_m15.csv`, spanning **2012-06-25 → 2026-07-24 (~14.1 years)**
— MORE real-world calendar span than the ~9.7 years `results/eurusd_h1.csv`
(H1.1-H1.4's own data source) covers. The row-span-comparability check this
step exists to make PASSES cleanly; there is no history-depth concern here.

Why M15 was tried, and the tradeoffs flagged BEFORE building
--------------------------------------------------------------
More raw bars → more raw swing/pattern completions to test on, which sounds
like more statistical power. Three real costs come with it, stated honestly
up front rather than discovered after the fact:

  1. AUTOCORRELATION / CLUSTERING — M15 events cluster far more in time than
     H1 events (a single volatile hour can produce several M15 swings that
     an H1 chart would show as ONE swing). A plain i.i.d. bootstrap treats
     each event as an independent draw, UNDERSTATING the true uncertainty
     when neighboring events are not independent. Mitigated below by
     switching the validation-arbiter bootstrap from i.i.d. (H1.1-H1.4's
     convention) to a MOVING-BLOCK (circular) bootstrap, block length = 20
     EVENTS (not bars) — see `bootstrap_delta_and_mcnemar_block`. The
     median/IQR time gap between consecutive validation events is reported
     directly as an honest clustering diagnostic (not just the point
     estimate + CI, as if the effective sample size equalled the row count).
  2. SIGNAL-TO-NOISE — a finer timeframe's price moves are dominated
     proportionally more by microstructure noise (spread bounce, momentary
     imbalances) than genuine multi-bar swing structure. ZigZag's ATR-scaled
     threshold is the best available mitigation (it adapts to the
     instrument's own local volatility at whatever timeframe it is fed) but
     does not eliminate this concern.
  3. TRANSACTION-COST DRAG — the SAME absolute 1.5-pip round-trip cost
     (0.00015) sits on top of a typical M15 bar range that is itself much
     SMALLER in absolute price terms than a typical H1 bar range (ATR(14
     M15 bars) spans 3.5 hours of price action; ATR(14 H1 bars) spans 14
     hours) — so the fixed cost is a proportionally BIGGER bite out of a
     "typical move" at M15. `run()` reports the actual mean ATR(14) in pips
     at both timeframes side by side so this claim is grounded in real
     numbers, not just assumed. The cost threshold is NOT relaxed to make
     it easier to clear — that would hide the real economics of trading
     this pattern at this timeframe, not fix a bug.

Swing basis: ZigZag ONLY (`src.zigzag_swings.zigzag_swings`, algorithm
UNCHANGED — it is already generic over whatever OHLC array it is fed), NOT
Williams fractals. A fractal's fixed 5-bar window is already the noisiest
swing basis on H1 (see `harmonic_event_check`'s H1.1/H1.2 → H1.3/H1.4
rationale); at M15 a 5-bar window spans only 75 minutes — almost certainly
too short to represent genuine swing structure, and not worth spending 2
more of this family's hypothesis slots re-confirming an even-worse-expected
result.

PRE-REGISTERED constants, RE-DERIVED (not H1's numbers copy-pasted) to
preserve the same real-world economic MEANING at M15's finer granularity
-------------------------------------------------------------------------
  ZigZag threshold: k=1.5 * ATR(14) UNCHANGED — the ATR period stays "14
    bars" (a scale-invariant, timeframe-agnostic convention already used
    project-wide for `ATR_14`; ATR shrinks in absolute price terms on a
    finer timeframe, which auto-scales the reversal threshold appropriately
    without needing a different k).
  Triple-barrier EWMA span: 96 M15 bars = 24h * 4 — preserves H1's span=24's
    "~1 day of memory" MEANING, not its literal bar count.
  Triple-barrier time horizon: 480 M15 bars = 120 H1 bars * 4 — preserves
    H1's horizon's "~5 trading days" holding-period MEANING.
  horizon_vol = EWMA_std(span=96) * sqrt(480) — identical sqrt-time
    principle (`src.triple_barrier.horizon_vol_from_ewma_std`, UNCHANGED
    function), M15-native units.
  Target/stop multipliers (1.5x / 1.0x): UNCHANGED — risk:reward ratios are
    timeframe-independent by construction (they scale WITH horizon_vol,
    whatever timeframe it was measured on).
  best_fit_score >= 0.5: UNCHANGED — a pattern-quality/geometry threshold,
    timeframe-independent.
  Transaction cost: SAME absolute 1.5 pips (0.00015) — see tradeoff #3 above;
    NOT relaxed.

Model pipeline: IDENTICAL to H1.3/H1.4 otherwise, reusing
`src.harmonic_event_check`'s `build_event_dataset` / `train_h1_1_logistic` /
`train_h1_2_mlp` / `predict_h1_2_mlp` / `_chronological_split` / `_upsert_log`
UNCHANGED — only the swing source (M15 OHLC, passed directly into
`build_event_dataset(h1=<M15 frame>, ...)`, which is already
timeframe-agnostic) and the pre-registered constants above differ. Same 8
features, same `class_weight='balanced'`, same `random_state=42`, same
chronological 70/15/15 split, same anti-cherry-pick primary/corroborating
structure (H1.6 MLP's PRIMARY test is vs H1.5 LogReg's own predictions on
identical validation rows; vs train-majority baseline is corroborating
only).

Statistical test: MOVING-BLOCK (circular) bootstrap, block length = 20
EVENTS, 2000 resamples, for ALL THREE comparisons this run needs (H1.5 vs
train-majority, H1.6 vs H1.5 PRIMARY, H1.6 vs train-majority corroborating)
— replacing `bootstrap_delta_and_mcnemar`'s plain i.i.d. resampling for THIS
run only. H1.1-H1.4 remain logged under the i.i.d. convention they were
already judged by; re-analyzing them under a different bootstrap after the
fact would itself be a form of post-hoc tuning, so they are left untouched.
McNemar (exact on the observed table, not a resampling test) is reported
unchanged alongside the block-bootstrap CI as corroboration, never as a
second, less conservative decision path.

Run:  python -m src.harmonic_m15_check
"""
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.h1_features import _normalize_h1, load_h1_frame, DEFAULT_H1_CACHE
from src.zigzag_swings import _atr14
from src.triple_barrier import ewma_log_return_std, horizon_vol_from_ewma_std
from src.paper_trading import PIP_SIZE
from src.harmonic_event_check import (
    HARMONIC_LOG, HARMONIC_LOG_COLUMNS, FAMILY_ALPHA, MIN_BEST_FIT_SCORE,
    TARGET_MULT, STOP_MULT, SPREAD_PIPS_DEFAULT, MODEL_FEATURE_COLUMNS,
    RANDOM_STATE, build_event_dataset, _chronological_split, _upsert_log,
    train_h1_1_logistic, train_h1_2_mlp, predict_h1_2_mlp,
)

DEFAULT_M15_CACHE = 'results/eurusd_m15.csv'
M15_EWMA_SPAN = 96          # 24h * 4 -- preserves H1's "~1 day of memory"
M15_HORIZON_BARS = 480      # 120 H1 bars * 4 -- preserves "~5 trading days"
BLOCK_LEN_EVENTS = 20       # block length in EVENTS, not bars
BOOTSTRAP_RESAMPLES = 2000

H1_5_NAME = 'harmonic_h1_5_logistic_vs_majority_m15'
H1_6_NAME = 'harmonic_h1_6_mlp_vs_h1_5_primary_m15'
LABEL1, LABEL2 = 'H1.5', 'H1.6'


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def load_m15_frame(cache_path=DEFAULT_M15_CACHE, allow_fetch=True):
    """M15 OHLCV as a UTC-indexed frame — cache-first, MT5-only live fetch
    (`src.live_data.fetch_m15_market_data`, no yfinance fallback for this
    timeframe). Reuses `src.h1_features._normalize_h1` UNCHANGED (already
    timeframe-agnostic despite its name: UTC-localize, sort, select the 5
    canonical OHLCV columns)."""
    df = None
    if cache_path and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    elif allow_fetch:
        from src.live_data import fetch_m15_market_data
        df, _ = fetch_m15_market_data(cache_path=cache_path)
    if df is None or len(df) == 0:
        raise RuntimeError("No M15 data available (cache missing and live fetch failed).")
    return _normalize_h1(df)


# ── clustering diagnostic ─────────────────────────────────────────────────

def event_gap_diagnostics(entry_idx) -> dict:
    """Median + IQR of the bar-gap between CONSECUTIVE (already chronological)
    event entry indices — a direct, honest measure of how clustered events
    are in time. A small median gap relative to BLOCK_LEN_EVENTS is exactly
    the clustering scenario the moving-block bootstrap exists to protect
    against."""
    entry_idx = np.asarray(entry_idx)
    if len(entry_idx) < 2:
        return {'median_gap_bars': float('nan'), 'iqr_low_bars': float('nan'),
                'iqr_high_bars': float('nan')}
    gaps = np.diff(np.sort(entry_idx))
    q1, med, q3 = np.percentile(gaps, [25, 50, 75])
    return {'median_gap_bars': float(med), 'iqr_low_bars': float(q1), 'iqr_high_bars': float(q3)}


# ── moving-block (circular) bootstrap over EVENTS ─────────────────────────

def _mcnemar_exact(correct_a, correct_b):
    """Two-sided exact-binomial McNemar on the discordant pairs of A vs B
    over identical rows. Timeframe-agnostic, tiny primitive duplicated here
    to keep this module self-contained -- same convention already used by
    src.ablation / src.cot_weekly_check / src.harmonic_event_check, each
    with their own copy."""
    from math import comb
    b = int(np.sum((~correct_a) & correct_b))
    c = int(np.sum(correct_a & (~correct_b)))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * p)


def _circular_block_bootstrap_indices(n, block_len, rng):
    """One resample's row indices for a circular (wrap-around) moving-block
    bootstrap over EVENT rows (`block_len` counts events, not underlying
    price bars) -- handles n <= block_len gracefully (the block wraps
    around) rather than requiring a full block to fit."""
    block_len = max(1, min(block_len, n))
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend((start + k) % n for k in range(block_len))
    return np.array(idx[:n])


def bootstrap_delta_and_mcnemar_block(y_val, pred_reference, pred_challenger, alpha,
                                      block_len=BLOCK_LEN_EVENTS, n_boot=BOOTSTRAP_RESAMPLES,
                                      random_state=RANDOM_STATE):
    """The M15 family's statistical test — a MOVING-BLOCK (circular) bootstrap
    of (acc_challenger - acc_reference) over blocks of `block_len`
    CONSECUTIVE validation EVENTS (not bars), replacing
    `src.harmonic_event_check.bootstrap_delta_and_mcnemar`'s plain i.i.d.
    resampling for H1.5/H1.6 only. M15 events cluster far more in time than
    H1 events (see module docstring); an i.i.d. bootstrap would silently
    UNDERSTATE the true uncertainty by treating clustered, non-independent
    events as independent draws. McNemar is reported unchanged (it is exact
    on the observed table, not a resampling test, so event autocorrelation
    does not bias it the way it biases a naive i.i.d. bootstrap CI) --
    reported as descriptive corroboration alongside the block-bootstrap CI,
    never as a second, less conservative decision path.

    Same alpha-scaled CI-width convention as
    `bootstrap_delta_and_mcnemar` (a stricter alpha only ever RAISES the bar
    a KEEP must clear)."""
    y_val = np.asarray(y_val)
    pred_reference = np.asarray(pred_reference)
    pred_challenger = np.asarray(pred_challenger)
    correct_ref = (pred_reference == y_val)
    correct_chg = (pred_challenger == y_val)
    acc_ref, acc_chg = float(correct_ref.mean()), float(correct_chg.mean())

    rng = np.random.default_rng(random_state)
    n = len(y_val)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_len, rng)
        deltas[i] = correct_chg[idx].mean() - correct_ref[idx].mean()
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    b, c, mcp = _mcnemar_exact(correct_ref, correct_chg)

    cleared = bool(lo > 0 and mcp < alpha)
    return {
        'acc_reference': round(acc_ref, 4), 'acc_challenger': round(acc_chg, 4),
        'delta_acc': round(acc_chg - acc_ref, 4),
        'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4),
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': round(mcp, 4),
        'cleared': cleared,
    }


# ── transaction-cost-drag diagnostic (M15 vs H1, real numbers) ────────────

def cost_drag_diagnostic(m15: pd.DataFrame, base_dir='') -> dict:
    """Mean ATR(14) in pips at M15 vs at H1 (same ATR primitive,
    `src.zigzag_swings._atr14`, UNCHANGED, just fed a different OHLC array)
    -- the direct empirical grounding for tradeoff #3 in the module
    docstring: the SAME fixed 1.5-pip round-trip cost is a proportionally
    BIGGER bite out of a "typical bar range" at the finer timeframe."""
    atr_m15 = _atr14(m15['high'].to_numpy(float), m15['low'].to_numpy(float),
                     m15['close'].to_numpy(float))
    mean_atr_m15_pips = float(np.nanmean(atr_m15)) / PIP_SIZE

    h1 = load_h1_frame(_p(base_dir, DEFAULT_H1_CACHE))
    atr_h1 = _atr14(h1['high'].to_numpy(float), h1['low'].to_numpy(float),
                    h1['close'].to_numpy(float))
    mean_atr_h1_pips = float(np.nanmean(atr_h1)) / PIP_SIZE

    cost_pips = SPREAD_PIPS_DEFAULT
    return {
        'mean_atr14_m15_pips': mean_atr_m15_pips,
        'mean_atr14_h1_pips': mean_atr_h1_pips,
        'cost_frac_of_atr_m15': cost_pips / mean_atr_m15_pips if mean_atr_m15_pips > 0 else float('nan'),
        'cost_frac_of_atr_h1': cost_pips / mean_atr_h1_pips if mean_atr_h1_pips > 0 else float('nan'),
    }


# ── orchestration ─────────────────────────────────────────────────────

def run(base_dir='', out_log=HARMONIC_LOG, random_state=RANDOM_STATE, register=True,
       m15_cache_path=DEFAULT_M15_CACHE):
    """Run the H1.5 (LogReg) / H1.6 (MLP) pair on M15 ZigZag-basis events --
    SAME family log as H1.1-H1.4, dynamically-computed Bonferroni bar."""
    m15 = load_m15_frame(cache_path=_p(base_dir, m15_cache_path))

    out_path = _p(base_dir, out_log)
    existing = set(pd.read_csv(out_path)['hypothesis']) if os.path.exists(out_path) else set()
    family_size = len(existing | {H1_5_NAME, H1_6_NAME})
    alpha = FAMILY_ALPHA / family_size

    drag = cost_drag_diagnostic(m15, base_dir=base_dir)

    print('=' * 78)
    print(f'M15 HARMONIC-PATTERN EVENT-CONDITIONAL MODEL — swing_source=zigzag ({LABEL1}/{LABEL2})')
    print(f'  M15 history: {len(m15):,} bars, {m15.index.min()} -> {m15.index.max()} '
          f'({(m15.index.max() - m15.index.min()).days / 365.25:.1f} years)')
    print(f'  TRANSACTION-COST DRAG diagnostic (real numbers, cost NOT relaxed):')
    print(f'    mean ATR(14) M15 = {drag["mean_atr14_m15_pips"]:.2f} pips  |  '
          f'{SPREAD_PIPS_DEFAULT} pip cost = {100 * drag["cost_frac_of_atr_m15"]:.1f}% of a typical M15 bar range')
    print(f'    mean ATR(14) H1  = {drag["mean_atr14_h1_pips"]:.2f} pips  |  '
          f'{SPREAD_PIPS_DEFAULT} pip cost = {100 * drag["cost_frac_of_atr_h1"]:.1f}% of a typical H1 bar range')
    print(f'  hypotheses already registered: {len(existing)}  ->  family size {family_size}  '
          f'->  BONFERRONI BAR alpha = {FAMILY_ALPHA}/{family_size} = {alpha:.4g}')
    print('=' * 78)

    built = build_event_dataset(base_dir=base_dir, h1=m15, random_state=random_state,
                                swing_source='zigzag', ewma_span=M15_EWMA_SPAN,
                                horizon_bars=M15_HORIZON_BARS)
    dataset = built['dataset']
    n_raw, n_filtered = len(built['events_raw']), len(built['events_filtered'])
    n_labeled = len(dataset)

    print(f'\n  raw XABCD events detected (M15, zigzag swings): {n_raw:,}')
    print(f'  filtered (best_fit_score >= {MIN_BEST_FIT_SCORE}): {n_filtered:,}')
    print(f'  excluded (insufficient forward history): {built["excluded_insufficient_history"]:,}')
    print(f'  FINAL labeled event dataset: {n_labeled:,}')
    print(f'  baseline (b) non-event random sample: n={built["baseline_b"]["n_sample"]:,}  '
          f'label==1 rate={built["baseline_b"]["label_1_rate"]:.4f}  (descriptive context only)')

    if n_labeled < 20:
        print('\n  TOO FEW LABELED EVENTS for a meaningful split/model — aborting run '
              '(nothing logged).')
        return None

    train_end, val_end = _chronological_split(n_labeled)
    train, val, test = dataset.iloc[:train_end], dataset.iloc[train_end:val_end], dataset.iloc[val_end:]
    print(f'\n  chronological split: train[0:{train_end}]={len(train)}  '
          f'val[{train_end}:{val_end}]={len(val)}  test[{val_end}:{n_labeled}]={len(test)} RESERVED')

    gap = event_gap_diagnostics(val['entry_idx'].to_numpy())
    print(f'  CLUSTERING DIAGNOSTIC (validation-slice event gaps, in M15 bars): '
          f'median={gap["median_gap_bars"]:.1f}  IQR=[{gap["iqr_low_bars"]:.1f}, {gap["iqr_high_bars"]:.1f}]')
    if gap['median_gap_bars'] == gap['median_gap_bars'] and gap['median_gap_bars'] < BLOCK_LEN_EVENTS:
        print(f'  -> events are CLUSTERED relative to the {BLOCK_LEN_EVENTS}-event block length '
              f'(median gap < block length); the block bootstrap below is doing real work, not '
              f'a formality.')

    scaler = StandardScaler().fit(train[MODEL_FEATURE_COLUMNS])
    X_train = scaler.transform(train[MODEL_FEATURE_COLUMNS]).astype('float32')
    X_val = scaler.transform(val[MODEL_FEATURE_COLUMNS]).astype('float32')
    y_train = train['label'].to_numpy()
    y_val = val['label'].to_numpy()

    maj_class = int(round(y_train.mean()))
    pred_majority_val = np.full(len(y_val), maj_class)

    # ---- H1.5: Logistic Regression vs train-majority baseline (BLOCK bootstrap) ----
    print(f'\n--- {LABEL1}: LOGISTIC REGRESSION (linear baseline, M15/zigzag) ---')
    clf = train_h1_1_logistic(X_train, y_train, random_state=random_state)
    pred_h15_val = clf.predict(X_val)
    r15 = bootstrap_delta_and_mcnemar_block(y_val, pred_majority_val, pred_h15_val,
                                            alpha=alpha, random_state=random_state)
    print(f'  train-majority baseline acc = {r15["acc_reference"]:.4f}  |  '
          f'{LABEL1} val acc = {r15["acc_challenger"]:.4f}  |  delta = {r15["delta_acc"]:+.4f}')
    print(f'  {100 * (1 - alpha):.2f}% BLOCK-bootstrap CI[{r15["ci_low"]:+.4f}, {r15["ci_high"]:+.4f}] '
          f'(block_len={BLOCK_LEN_EVENTS} events)  '
          f'McNemar b={r15["mcnemar_b"]} c={r15["mcnemar_c"]} p={r15["mcnemar_p"]:.4f}  alpha={alpha:.4g}')
    verdict_h15 = (
        f'KEEP — {LABEL1} beats the train-majority baseline at the pre-registered bar (block bootstrap)'
        if r15['cleared'] else
        f'DROP — {LABEL1} indistinguishable from the train-majority baseline at the pre-registered bar (block bootstrap)'
    )
    print(f'  VERDICT ({LABEL1}): {verdict_h15}')

    # ---- H1.6: MLP — PRIMARY vs H1.5, CORROBORATING vs majority (BLOCK bootstrap) ----
    print(f'\n--- {LABEL2}: MLP (non-linear interactions, M15/zigzag) ---')
    mlp = train_h1_2_mlp(X_train, y_train, X_val, y_val, random_state=random_state)
    pred_h16_val = predict_h1_2_mlp(mlp, X_val)

    r16_primary = bootstrap_delta_and_mcnemar_block(y_val, pred_h15_val, pred_h16_val,
                                                    alpha=alpha, random_state=random_state)
    r16_corroborating = bootstrap_delta_and_mcnemar_block(y_val, pred_majority_val, pred_h16_val,
                                                          alpha=alpha, random_state=random_state)
    print(f'  PRIMARY (MLP vs {LABEL1}, same val rows, BLOCK bootstrap): {LABEL1} acc = '
          f'{r16_primary["acc_reference"]:.4f}  |  MLP acc = {r16_primary["acc_challenger"]:.4f}  |  '
          f'delta = {r16_primary["delta_acc"]:+.4f}')
    print(f'    {100 * (1 - alpha):.2f}% CI[{r16_primary["ci_low"]:+.4f}, {r16_primary["ci_high"]:+.4f}]  '
          f'McNemar b={r16_primary["mcnemar_b"]} c={r16_primary["mcnemar_c"]} p={r16_primary["mcnemar_p"]:.4f}')
    print(f'  CORROBORATING ONLY (MLP vs train-majority, BLOCK bootstrap, context — not decision-bearing): '
          f'delta = {r16_corroborating["delta_acc"]:+.4f}  '
          f'CI[{r16_corroborating["ci_low"]:+.4f}, {r16_corroborating["ci_high"]:+.4f}]')

    if r16_primary['cleared']:
        verdict_h16 = (f'KEEP — MLP beats {LABEL1} at the pre-registered bar on the PRIMARY '
                       'comparison (block bootstrap; the corroborating vs-majority result is consistent context)')
    elif r16_corroborating['cleared']:
        verdict_h16 = (f'DROP — MLP beats the majority baseline (corroborating context) but does '
                       f'NOT beat {LABEL1} at the pre-registered bar on the PRIMARY comparison: the '
                       'extra non-linear capacity found nothing the linear model had not already '
                       'found (anti-cherry-pick rule)')
    else:
        verdict_h16 = (f'DROP — MLP beats neither {LABEL1} (primary) nor the train-majority baseline '
                       '(corroborating) at the pre-registered bar')
    print(f'  VERDICT ({LABEL2}): {verdict_h16}')

    print(f'\n  (power/clustering caveat: {len(val)} validation events, median gap '
          f'{gap["median_gap_bars"]:.1f} bars — the moving-block bootstrap above already accounts '
          f'for the clustering; treat any KEEP as preliminary and any DROP as correspondingly weak '
          f'evidence of absence, same as every small-n family in this project)')

    date = pd.Timestamp.utcnow().date().isoformat()
    swing_note = ('swing_source=\'zigzag\' (M15 bars, ATR(14 M15 bars)*1.5 causal ZigZag pivots, '
                 'variable reveal-bar lag). ')
    cost_note = (f'Transaction-cost drag: {SPREAD_PIPS_DEFAULT} pip cost = '
                f'{100 * drag["cost_frac_of_atr_m15"]:.1f}% of a typical M15 ATR(14) bar range '
                f'(vs {100 * drag["cost_frac_of_atr_h1"]:.1f}% at H1) -- NOT relaxed. ')
    cluster_note = (f'Event clustering: validation-slice event gap median='
                    f'{gap["median_gap_bars"]:.1f} bars, IQR=[{gap["iqr_low_bars"]:.1f}, '
                    f'{gap["iqr_high_bars"]:.1f}] -- MOVING-BLOCK (circular) bootstrap used '
                    f'(block_len={BLOCK_LEN_EVENTS} events, n_boot={BOOTSTRAP_RESAMPLES}), NOT the '
                    f'i.i.d. bootstrap H1.1-H1.4 used. ')

    row_h15 = {
        'date': date, 'hypothesis': H1_5_NAME,
        'arbiter': 'event_validation[70:85]_M15_block_bootstrap',
        'n_events_raw': n_raw, 'n_events_filtered': n_filtered, 'n_events_labeled': n_labeled,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'acc_challenger': r15['acc_challenger'], 'acc_reference': r15['acc_reference'],
        'delta_acc': r15['delta_acc'], 'delta_acc_ci_low': r15['ci_low'],
        'delta_acc_ci_high': r15['ci_high'], 'mcnemar_b': r15['mcnemar_b'],
        'mcnemar_c': r15['mcnemar_c'], 'mcnemar_p': r15['mcnemar_p'],
        'alpha': round(alpha, 4), 'cleared_bar': r15['cleared'], 'verdict': verdict_h15,
        'notes': (f'{swing_note}M15 XABCD triple-barrier events, best_fit_score>={MIN_BEST_FIT_SCORE}, '
                 f'EWMA(span={M15_EWMA_SPAN} M15 bars = ~1 day)*sqrt({M15_HORIZON_BARS} M15 bars = '
                 f'~5 trading days) horizon_vol, target={TARGET_MULT}x/stop={STOP_MULT}x, cost '
                 f'threshold={SPREAD_PIPS_DEFAULT}pips={SPREAD_PIPS_DEFAULT * PIP_SIZE}. '
                 f'LogisticRegression(class_weight=balanced). {cost_note}{cluster_note}'
                 f'baseline(b) non-event random-sample label==1 rate='
                 f'{built["baseline_b"]["label_1_rate"]:.4f} (n={built["baseline_b"]["n_sample"]}, '
                 f'descriptive context only, not decision-bearing).'),
    }
    row_h16 = {
        'date': date, 'hypothesis': H1_6_NAME,
        'arbiter': 'event_validation[70:85]_M15_block_bootstrap',
        'n_events_raw': n_raw, 'n_events_filtered': n_filtered, 'n_events_labeled': n_labeled,
        'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'acc_challenger': r16_primary['acc_challenger'], 'acc_reference': r16_primary['acc_reference'],
        'delta_acc': r16_primary['delta_acc'], 'delta_acc_ci_low': r16_primary['ci_low'],
        'delta_acc_ci_high': r16_primary['ci_high'], 'mcnemar_b': r16_primary['mcnemar_b'],
        'mcnemar_c': r16_primary['mcnemar_c'], 'mcnemar_p': r16_primary['mcnemar_p'],
        'alpha': round(alpha, 4), 'cleared_bar': r16_primary['cleared'], 'verdict': verdict_h16,
        'notes': (f'{swing_note}PRIMARY reference = {LABEL1} predictions on IDENTICAL val rows (not '
                 f'majority baseline). CORROBORATING (context only) MLP-vs-majority: '
                 f'delta_acc={r16_corroborating["delta_acc"]:+.4f} '
                 f'CI[{r16_corroborating["ci_low"]:+.4f}, {r16_corroborating["ci_high"]:+.4f}] '
                 f'McNemar p={r16_corroborating["mcnemar_p"]:.4f}, cleared='
                 f'{r16_corroborating["cleared"]}. MLP (raw PyTorch, not Keras, IDENTICAL '
                 f'implementation to H1.2/H1.4\'s train_h1_2_mlp, only the data source swapped): '
                 f'Linear(16,L2=1e-3 via weight_decay)-ReLU-Dropout(0.3)-Linear(8,L2=1e-3)-ReLU-'
                 f'Dropout(0.3)-Linear(1)-Sigmoid, Adam lr=0.001, epochs<=100, ES patience=10 on '
                 f'val_loss (explicit model.train()/eval()+no_grad() toggling), batch_size=32, '
                 f'per-sample balanced class weights via BCELoss(weight=...) each batch. {cluster_note}'),
    }

    if register:
        _upsert_log(row_h15, out_path)
        _upsert_log(row_h16, out_path)
        print(f'\nLogged both hypotheses: {out_path}')

    return {'h1_5': row_h15, 'h1_6': row_h16, 'built': built, 'gap': gap, 'drag': drag}


if __name__ == '__main__':
    run()
