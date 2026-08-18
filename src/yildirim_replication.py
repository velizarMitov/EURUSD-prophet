"""
REPLICATION STUDY — Yıldırım, Toroslu & Fiore (2021), *Forecasting directional
movement of Forex data using LSTM with technical and macroeconomic indicators*,
Financial Innovation 7:1.  https://doi.org/10.1186/s40854-020-00220-2

WHAT THIS IS NOT
================
This module makes **no claim about EUR/USD predictability**. It re-runs an
EXTERNAL published method under five selection protocols to measure how much of
that paper's reported improvement is attributable to three design choices its
80/20 split cannot make cleanly. Accordingly it:

  * costs ZERO alpha — it registers **no** hypothesis in **any** family and
    writes **no** row to any ``*_hypothesis_log.csv``. It does not tighten the
    Bonferroni bar in ``results/feature_hypothesis_log.csv`` (currently
    0.05/9) or in any other family. Nothing here is a candidate production
    feature, and no result from this module may be used to justify one.
  * is structurally separate — it is NEVER imported by ``api.py``,
    ``src/inference.py`` or ``_train_pipeline.py``, writes NOTHING under
    ``models/``, and touches no existing artifact. Everything it produces lands
    in ``results/yildirim_replication/``.

Same isolation contract as ``src/ti_lstm_h1_experimental.py`` (which is named
after this paper's TI-LSTM — the reason the paper is worth replicating at all).

THE QUESTION
============
Not "did they cheat". The paper reports single-model 1-day profit_accuracy of
50.69% / 52.18% / 53.05% (ME / TI / ME-TI, Tables 4-6) on n≈150 transactions —
a coin flip — and a hybrid that abstains from low-confidence trades at 77.32% /
77.76% (Table 7). There is NO validation set anywhere in the paper ("The data
set was split into the training and test sets, with ratios of 80% and 20%"),
yet three things had to be chosen:

  Leak 1  LABEL THRESHOLD SCOPE. The three-class threshold is picked by an
          entropy criterion over a histogram-derived upper bound. Entropy of the
          class balance is UNSUPERVISED, so this is clean if computed on
          training rows only. Computed over the full series it leaks the test
          period's volatility scale (not its direction). Table 3 reports one
          global threshold per horizon with train/test counts underneath it,
          i.e. the full-series reading is the faithful one.
  Leak 2  MODEL SELECTION. "For each experiment, we performed 50, 100, 150, and
          200 iterations in the training phases to properly compare different
          models." With no validation set that comparison can only have been
          made on test.
  Leak 3  HYBRID DECISION RULES. The confidence/abstention rules that produce
          the headline gain have tunable parameters and no clean block to tune
          them on.

FIVE ARMS (identical in every respect except the SELECTION protocol; the SCORING
rows are byte-identical across arms, and no arm ever TRAINS on val or test):

  A0  threshold train-only, iterations picked on val, hybrid tuned on val,
      test scored ONCE                                   <- house discipline
  A1  A0 + threshold over the full series                    (leak 1 only)
  A2  A0 + iteration count picked on test                    (leak 2 only)
  A3  A0 + hybrid rules tuned on test                        (leak 3 only)
  A4  all three                                    (approximates as-published)

TWO DATA SCALES: ``primary`` — the paper's own calendar window trimmed to its
own row count (1214 daily bars -> 243 test bars, matching Tables 4-15 exactly);
``secondary`` — our full euro-era daily set, to check whether any gap is a
small-sample artifact.

SEEDS: every arm is run over 5 seeds (42-46, the project's ensemble convention)
and the MEAN is reported. This project has already measured TF/oneDNN
run-to-run spread of the same magnitude as the effects under test, so a
single-seed A-vs-B difference is uninterpretable.

UNCERTAINTY: every arm-to-arm gap carries a moving-block (circular) bootstrap CI
built on ``src.walk_forward_validation._circular_block_bootstrap_indices`` — the
project's existing machinery, same clustering rationale (adjacent FX days are
not independent). profit_accuracy is a RATIO with a random denominator (the
transaction count), so the bootstrap resamples rows and recomputes
sum(wins)/sum(transactions), never a mean of per-row accuracies.

Entry point:  python -m src.yildirim_replication [--scale primary|secondary|both]
Outputs:      results/yildirim_replication/{arms.csv,summary.csv,REPLICATION.md}

Every place the paper is underspecified and a choice had to be made is recorded
in ``UNDERSPECIFIED`` below and reproduced verbatim in the report — that list is
part of the result, not an appendix to it.
"""
import os

# Process-local Keras backend selection — MUST precede any keras import.
# Same pattern as src/ti_lstm_h1_experimental.py (production imports tf.keras in
# its own processes and is unaffected, and no production module imports this
# one), but the OPPOSITE choice of backend: this study's LSTM is tiny (32 units,
# 20 steps, ~8 features) and therefore framework-overhead-bound, not
# compute-bound. Measured on this machine at the primary scale, 200 epochs takes
# 261s on the torch backend and 65s on tensorflow. The torch backend DOES reach
# the RTX 4070 and is still the slower of the two (1.307 s/epoch on CUDA vs
# 1.305 on CPU — a dead heat, because the model is too small to fill a GPU);
# tensorflow >= 2.11 has no GPU support on native Windows at all and runs on CPU
# regardless. Either way there is no GPU win available here, so the run is
# parallelised across CPU worker processes instead.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.h1_features import wilder_rsi
from src.ti_lstm_h1_experimental import bollinger_percent_b, cci

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'yildirim_replication')
PANEL_CACHE = os.path.join(OUT_DIR, 'panel_cache.csv')
ARMS_CSV = os.path.join(OUT_DIR, 'arms.csv')
SUMMARY_CSV = os.path.join(OUT_DIR, 'summary.csv')
REPORT_MD = os.path.join(OUT_DIR, 'REPLICATION.md')
RUN_META_JSON = os.path.join(OUT_DIR, 'run_meta.json')

HISTORY_CSV = os.path.join(REPO_ROOT, 'results', 'eurusd_features.csv')
YIELD_CACHE = os.path.join(REPO_ROOT, 'results', 'yield_differential.csv')
POLICY_CACHE = os.path.join(REPO_ROOT, 'results', 'policy_rate_differential.csv')
INFLATION_CACHE = os.path.join(REPO_ROOT, 'results', 'inflation_differential.csv')
EQUITY_CACHE = os.path.join(OUT_DIR, 'equity_indices.csv')

# Paper's own class encoding: "the number of increases (labeled as 2) and
# decreases (labeled as 1) above the threshold value are both determined, and
# the rest of the changes are assumed to be no_change (labeled 0)".
NOACT, DEC, INC = 0, 1, 2

HORIZONS = (1, 3, 5)
ITERATION_GRID = (50, 100, 150, 200)          # paper, "Experiments" section
SEEDS = (42, 43, 44, 45, 46)                  # project ensemble convention
ARM_NAMES = ('A0', 'A1', 'A2', 'A3', 'A4')
MODEL_TYPES = ('ME_LSTM', 'TI_LSTM', 'ME_TI_LSTM')
ALL_MODELS = MODEL_TYPES + ('HYBRID',)

# Paper's window: "values from the period January 2013-January 2018 ... 1214
# consecutive days ... first 971 days ... to train ... last 243 days to test".
PAPER_WINDOW_START = '2013-01-01'
PAPER_WINDOW_END = '2018-02-01'
PAPER_N_ROWS = 1214
EURO_ERA_START = '1999-01-04'

# House-discipline chronological split (config.json is 80/10 train/val for the
# production families; the replication's A0 uses the 70/10/20 stated in the
# brief so that a validation block exists at all — the paper has none).
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10

# Selection guard, see UNDERSPECIFIED[#8]: profit_accuracy over a
# self-selected transaction subset is degenerate without a floor on the
# denominator (the paper's own Table 7 reports 100.00% on 8 transactions and
# Table 11 on 2). A configuration must trade on at least this fraction of the
# SELECTION block's rows to be selectable. Applied identically in every arm, so
# it cannot contaminate the leak decomposition.
MIN_TXN_FRACTION = 0.10

# Hybrid tuning surface. min_confidence=0.0 reduces the combiner EXACTLY to the
# paper's three published rules, so the paper's own rule is inside every arm's
# grid rather than being competed against from outside it.
HYBRID_MIN_CONF_GRID = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
HYBRID_TIE_BREAK_GRID = ('ti', 'me')

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK_LEN = 20      # ~one trading month, == the LSTM lookback
BOOTSTRAP_ALPHA = 0.05

TIME_STEPS = 20               # config.json lstm.time_steps
LSTM_UNITS = 32
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_BATCH = 32
SCALE_CLIP = 5.0

# Paper feature sets. "the ME_LSTM model was trained using all of these
# macroeconomic factors together with the closing values of the EUR/USD pair" /
# "TI_LSTM ... trained using these seven technical indicators together with the
# closing values of the EUR/USD pair".
ME_COLUMNS = ['close', 'rate_de', 'rate_eu', 'rate_fed',
              'infl_eu', 'infl_us', 'sp500', 'dax']
TI_COLUMNS = ['close', 'ma_10', 'macd_12_26', 'roc_2', 'mom_4',
              'rsi_10', 'bb_pctb_20', 'cci_20']
ME_TI_COLUMNS = ME_COLUMNS + [c for c in TI_COLUMNS if c != 'close']

MODEL_FEATURES = {
    'ME_LSTM': ME_COLUMNS,
    'TI_LSTM': TI_COLUMNS,
    'ME_TI_LSTM': ME_TI_COLUMNS,
}

# ---------------------------------------------------------------------------
# The underspecification ledger — reproduced verbatim in REPLICATION.md.
# Each entry: (area, what the paper does not pin down, what we chose and why).
# ---------------------------------------------------------------------------

UNDERSPECIFIED = [
    (
        "Histogram upper bound",
        '"the maximum difference value of the last bin added was used as the '
        'upper bound of the threshold value" — ambiguous between the bin\'s '
        'upper EDGE and the largest difference OBSERVED inside that bin.',
        "We take the largest observed |difference| inside the last bin added. "
        "This is the only reading consistent with the paper's own worked "
        "example: with max difference 0.029 and 10 equal-width bins the edges "
        "are multiples of 0.0029, and the reported bound 0.00652 is not one of "
        "them. The bin-edge reading is available as "
        "histogram_threshold_upper_bound(..., use_bin_edge=True) and is "
        "unit-tested alongside the primary reading.",
    ),
    (
        "Histogram bin ties",
        "Bins are 'sorted in descending order' by count; ties are not resolved.",
        "Stable descending sort, so equal counts keep ascending bin order "
        "(the lower/denser bin is consumed first).",
    ),
    (
        "Entropy base",
        "'Entropy = - sum p_i * log p_i' does not state the logarithm base.",
        "Natural log. The base is a positive constant factor and cannot change "
        "the argmax, so this choice is inert for the threshold actually used.",
    ),
    (
        "Difference definition",
        "'the EUR/USD ratio differences between consecutive days' is stated for "
        "the 1-day case; the 3- and 5-day-ahead labels are never re-defined.",
        "diff_t = close_{t+n} - close_t (absolute price difference, not a "
        "return) for n in {1,3,5}. Consistent with Table 3, whose thresholds "
        "grow with the horizon (0.0023 / 0.0040 / 0.0055) exactly as an "
        "n-day price difference would.",
    ),
    (
        "Zero difference",
        "profit_accuracy converts a predicted inc/dec on a true no_act row into "
        "a correct prediction 'if the actual movement is in the same direction'. "
        "An exactly-zero move has no direction.",
        "diff == 0 counts as a LOSS on any transaction (it can never be "
        "'in the same direction'). Affects a handful of rows.",
    ),
    (
        "LSTM architecture",
        "No layer count, unit count, activation, optimizer, loss, batch size, "
        "lookback window or scaler is given anywhere in the paper.",
        f"One LSTM layer of {LSTM_UNITS} units -> Dropout({LSTM_DROPOUT}) -> "
        f"Dense(3, softmax); Adam(lr={LSTM_LR}); sparse categorical "
        f"cross-entropy; batch {LSTM_BATCH}; lookback {TIME_STEPS} bars "
        "(config.json lstm.time_steps). 'Iterations' is read as epochs. "
        "Held IDENTICAL across all five arms, so it cannot affect the leak "
        "decomposition — only the absolute level.",
    ),
    (
        "Feature scaling",
        "No scaling is described, yet the feature set is dominated by "
        "non-stationary LEVELS (close, SMA, S&P 500, DAX, policy rates).",
        f"StandardScaler fit on TRAIN rows only (never val/test — a full-series "
        f"scaler would be a fourth leak, and this study measures exactly three), "
        f"then clipped to +/-{SCALE_CLIP} sigma so a trending level cannot "
        "saturate the LSTM into a constant output. Identical in every arm. "
        "This penalises the level-heavy ME feature set on the long secondary "
        "scale, and that is an honest consequence of the paper's feature "
        "choice, not a bug.",
    ),
    (
        "Selection criterion floor",
        "The paper reports profit_accuracy over a self-selected transaction "
        "subset with no floor on the transaction count — Table 7 reports "
        "100.00% on 8/243 transactions, Table 11 on 2/243.",
        f"Any configuration is selectable only if it transacts on at least "
        f"{MIN_TXN_FRACTION:.0%} of the SELECTION block's rows; if none clears "
        "the floor we fall back to the configuration with the most "
        "transactions. Applied identically in all five arms. Without such a "
        "floor the metric is maximised by trading once, which is the mechanism "
        "behind the paper's own 100% cells.",
    ),
    (
        "Hybrid tunable surface",
        "The combiner's 'smart decision rules' are given as three fixed rules, "
        "but the paper also reports two variants ('modification based on "
        "ME_LSTM' / 'based on TI_LSTM') that are never defined, and describes "
        "the mechanism as 'eliminating transactions with weaker confidence' "
        "without naming a confidence level.",
        "The two reported variants are read as the tie-break model. The "
        "confidence mechanism is exposed as one parameter, min_confidence, "
        "over the grid " + repr(HYBRID_MIN_CONF_GRID) + "; min_confidence=0.0 "
        "reproduces the paper's three rules EXACTLY, so the published rule is "
        "a member of every arm's grid.",
    ),
    (
        "Iteration comparison scope",
        "'we performed 50, 100, 150, and 200 iterations ... to properly compare "
        "different models' — the block on which that comparison was made is "
        "never named, and no validation block exists.",
        "This is Leak 2 and is exactly what arms A0/A2 vary. For the hybrid, "
        "each constituent's iteration count is inherited from that "
        "constituent's own single-model choice, which keeps Leak 2 and Leak 3 "
        "separable.",
    ),
    (
        "Iteration grid inconsistency",
        "'Training classifiers and labeling the data' says iterations of "
        "(50, 100, 150); 'Experiments' says (50, 100, 150, 200) and every "
        "results table reports 200.",
        "We use (50, 100, 150, 200), the grid the tables actually report.",
    ),
    (
        "Macroeconomic sources",
        "The ME set is 'Interest Rate_GER', 'Interest Rate_EU', 'FED Funds "
        "Rate', 'Inflation Rate_EU', 'Inflation Rate_USA', 'Close(S&P 500)', "
        "'Close(DAX)', sourced from ECB SDW / BLS / FRED / Yahoo Finance. "
        "Exact series IDs are not given.",
        "Mapped onto this project's existing FRED framework: rate_de = "
        "IRLTLT01DEM156N (DE long-term rate, monthly), rate_eu = ECBDFR (ECB "
        "deposit facility), rate_fed = DFF (effective fed funds), infl_eu = DE "
        "HICP YoY (CP0000DEM086NEST, Germany standing in for the euro area, "
        "the project's existing choice), infl_us = US CPI YoY (CPIAUCSL). "
        "S&P 500 and DAX have no series in config.json's FRED set (FRED's "
        "SP500 carries only a rolling 10 years and FRED has no DAX at all), so "
        "they come from yfinance ^GSPC / ^GDAXI — already a project dependency "
        "— and are cached to results/yildirim_replication/equity_indices.csv.",
    ),
    (
        "Technical indicator outputs",
        "'MACD with short- and long-term periods of 12 and 26' names no signal "
        "line; 'BB with period of 20' names no band or output; 'MA with a "
        "period of 10' does not say whether the raw level or a ratio is fed.",
        "MACD = EMA12 - EMA26 (line only, no signal line specified). Bollinger "
        "is fed as %B, period 20, 2.0 sigma, reusing "
        "src.ti_lstm_h1_experimental.bollinger_percent_b. MA(10) is fed as the "
        "raw level, as written. RSI(10) is src.h1_features.wilder_rsi (the "
        "paper cites Wilder 1978). CCI(20) is "
        "src.ti_lstm_h1_experimental.cci (Lambert 1980, c=0.015). ROC(2) = "
        "(close/close_{t-2} - 1)*100, Momentum(4) = close - close_{t-4}.",
    ),
    (
        "Trading calendar",
        "'1234 data points in which the markets were open' (~247 rows/year) — "
        "the paper's bars are weekday bars.",
        "Our MT5-derived daily history carries a short Sunday session bar "
        "(~1421 of 8606 euro-era rows). Sunday bars are dropped so the cadence "
        "matches the paper's; the remaining Mon-Fri rows run ~260/year, and "
        "holidays survive as flat bars.",
    ),
    (
        "Sequence warm-up",
        "The lookback window is never stated, so neither is what happens to the "
        "first rows of the training block.",
        f"Sequences end-index at or after row {TIME_STEPS - 1}, so the training "
        "block loses its first 19 rows. The validation and test blocks are "
        "untouched, which keeps the test count at exactly 243/243/242 for "
        "horizons 1/3/5 — the paper's own counts.",
    ),
    (
        "Minibatch shuffling",
        "Not mentioned.",
        "Keras default shuffle=True. Each sample is a self-contained trailing "
        "window, so shuffling minibatches cannot leak the future; it only "
        "affects optimisation.",
    ),
    (
        "Macro publication lag (not a choice, an observation)",
        "'Monthly inflation rates were collected from the websites of central "
        "banks, and they were repeated for all days of the corresponding month "
        "to fill the fields in our daily records.' A month's CPI is not known "
        "until the following month, so a January CPI stamped on 2 January is a "
        "value the market did not have.",
        "We reproduce the paper's convention exactly (month-stamped level, "
        "forward-filled across the month) because deviating from it would stop "
        "being a replication. This is a fourth, built-in leak that the paper's "
        "design carries independently of the three under study; it is present "
        "identically in ALL FIVE arms, so it inflates the absolute level of "
        "every ME-containing arm without contaminating any arm-to-arm gap. "
        "Flagged rather than fixed.",
    ),
    (
        "Reported-figure inconsistency (not a choice, an observation)",
        "The paper's per-model tables and its summary table disagree. Tables "
        "4/5/6 give 1-day averages of 50.69 / 52.18 / 53.05, while Table 20 "
        "gives 50.16 / 51.43 / 49.89 for the same three models. Table 7 gives "
        "hybrid 1-day averages of 77.32 / 77.76, while Table 20 gives 73.09.",
        "We quote both in the report and compare against the per-model tables "
        "(Tables 4-7), which the brief cites and which are the more detailed "
        "of the two.",
    ),
]


# ---------------------------------------------------------------------------
# 1. Labelling — the paper's histogram upper bound + entropy threshold search.
#    Pure numpy/pandas: importable and testable without keras or torch.
# ---------------------------------------------------------------------------

def histogram_threshold_upper_bound(diffs, n_bins: int = 10, coverage: float = 0.85,
                                    use_bin_edge: bool = False) -> float:
    """Algorithm 1 of the paper.

    "We placed the EUR/USD ratio differences between consecutive days into 10
    bins (as number_of_bins value), which range equally between the minimum
    (which is 0) and maximum difference values. We determined the count of each
    bin and sorted them in descending order. After that, the counts of the bins
    were summed until the sum exceeded 85% of the whole count (the data set
    size). Then, the maximum difference value of the last bin added was used as
    the upper bound of the threshold value."

    The differences are ABSOLUTE ("the minimum (which is 0)"). ``use_bin_edge``
    switches between the two readings of "the maximum difference value of the
    last bin added" — see UNDERSPECIFIED[#1]; the default (largest difference
    actually observed in that bin) is the reading consistent with the paper's
    worked example (max 0.029 -> bound 0.00652, which is not a bin edge).
    """
    d = np.abs(np.asarray(diffs, dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("histogram_threshold_upper_bound: no finite differences")
    d_max = float(d.max())
    if d_max <= 0:
        return 0.0

    edges = np.linspace(0.0, d_max, n_bins + 1)
    # np.digitize with right=False puts the maximum value in a phantom bin
    # n_bins+1; fold it back into the last real bin.
    bin_idx = np.clip(np.digitize(d, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)

    # Descending by count, stable -> ties keep ascending bin order.
    order = np.argsort(-counts, kind='stable')
    need = coverage * d.size
    cumulative = 0
    last_bin = int(order[0])
    for b in order:
        cumulative += int(counts[b])
        last_bin = int(b)
        if cumulative > need:
            break

    if use_bin_edge or counts[last_bin] == 0:
        return float(edges[last_bin + 1])
    return float(d[bin_idx == last_bin].max())


def entropy_of_counts(counts) -> float:
    """Eq. 24: Entropy = - sum p_i * log p_i (natural log; see
    UNDERSPECIFIED[#3]). Empty classes contribute 0."""
    c = np.asarray(counts, dtype=float)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-(p * np.log(p)).sum())


def label_three_class(diffs, tau: float) -> np.ndarray:
    """Paper's labels: strictly above +tau -> INC(2), strictly below -tau ->
    DEC(1), everything else -> NOACT(0)."""
    d = np.asarray(diffs, dtype=float)
    out = np.full(d.shape, NOACT, dtype=np.int64)
    out[d > tau] = INC
    out[d < -tau] = DEC
    return out


def entropy_threshold_search(diffs, upper_bound: float, step: float = 1e-5):
    """Algorithm 2: sweep tau over [0, upper_bound] in increments of 0.00001
    and keep the tau whose three-class distribution maximises entropy.

    Vectorised via searchsorted rather than the paper's explicit while-loop —
    identical counts, but O(steps log n) instead of O(steps * n). Returns
    (tau, entropy). On an exact entropy tie the SMALLEST tau wins (more
    transactions, the conservative reading).
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        raise ValueError("entropy_threshold_search: no finite differences")
    if upper_bound <= 0:
        return 0.0, entropy_of_counts(np.bincount(label_three_class(d, 0.0), minlength=3))

    # Build the grid as k*step and round it back onto that grid: np.arange
    # accumulates binary error, and 10*1e-5 landing at 9.999...e-05 instead of
    # 1e-4 silently flips a strict `>` comparison for any difference sitting
    # exactly on a grid point.
    n_steps = int(math.floor(upper_bound / step)) + 1
    taus = np.round(np.arange(n_steps, dtype=float) * step, 10)
    ds = np.sort(d)
    # n_inc(tau) = #{x >  tau};  n_dec(tau) = #{x < -tau}
    n_inc = n - np.searchsorted(ds, taus, side='right')
    n_dec = np.searchsorted(ds, -taus, side='left')
    n_noact = n - n_inc - n_dec

    counts = np.stack([n_noact, n_dec, n_inc], axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = counts / n
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
        ent = -(p * logp).sum(axis=1)

    best = int(np.argmax(ent))            # argmax returns the FIRST maximum
    return float(taus[best]), float(ent[best])


def choose_threshold(diffs, n_bins: int = 10, coverage: float = 0.85,
                     step: float = 1e-5):
    """Algorithm 1 then Algorithm 2. Returns (tau, upper_bound, entropy)."""
    ub = histogram_threshold_upper_bound(diffs, n_bins=n_bins, coverage=coverage)
    tau, ent = entropy_threshold_search(diffs, ub, step=step)
    return tau, ub, ent


# ---------------------------------------------------------------------------
# 2. The paper's profit_accuracy (Eq. 25 / Table 2).
# ---------------------------------------------------------------------------

def profit_accuracy_counts(y_true_label, y_pred_label, raw_diff) -> dict:
    """The six cells of the paper's Table 2.

                     Pred(no_act)   Pred(dec)            Pred(inc)
        True(no_act)      -         False_dec_noact      False_inc_noact
        True(dec)         -         True_dec             False_inc_dec
        True(inc)         -         False_dec_inc        True_inc

    with the paper's conversion rule applied first: "our model predicts the
    class as 'increase' (or 'decrease'), but according to our three-class
    classification, it actually corresponds to a 'no_act' class. In that case,
    we check if the actual movement is in the same direction with the
    prediction ... If that is the case, then the prediction is correct, and we
    treat this test case as the correct classification." Hence there is no
    True_inc_noact / True_dec_noact cell — such rows land in True_inc /
    True_dec. A diff of exactly 0 has no direction and stays a false
    (UNDERSPECIFIED[#5]).

    Rows predicted no_act appear in no cell: they are not transactions.
    """
    y_true = np.asarray(y_true_label)
    y_pred = np.asarray(y_pred_label)
    diff = np.asarray(raw_diff, dtype=float)
    if not (y_true.shape == y_pred.shape == diff.shape):
        raise ValueError("profit_accuracy_counts: shape mismatch")

    c = dict(True_dec=0, True_inc=0, False_dec_noact=0, False_inc_noact=0,
             False_inc_dec=0, False_dec_inc=0)

    for t, p, dv in zip(y_true, y_pred, diff):
        if p == NOACT:
            continue                                   # no transaction
        if p == INC:
            if t == INC or (t == NOACT and dv > 0):
                c['True_inc'] += 1                     # incl. the conversion
            elif t == NOACT:
                c['False_inc_noact'] += 1
            else:                                      # t == DEC
                c['False_inc_dec'] += 1
        else:                                          # p == DEC
            if t == DEC or (t == NOACT and dv < 0):
                c['True_dec'] += 1                     # incl. the conversion
            elif t == NOACT:
                c['False_dec_noact'] += 1
            else:                                      # t == INC
                c['False_dec_inc'] += 1
    return c


def profit_accuracy_from_counts(counts: dict) -> float:
    """Eq. 25 verbatim:

        ProfitAccuracy = (True_dec + True_inc) / (False_dec_noact +
            False_inc_noact + True_dec + False_inc_dec + False_dec_inc +
            True_inc)

    i.e. profitable transactions over ALL transactions. NaN when no transaction
    was made (the paper prints 'Nan' in exactly this case — Table 15, 150/200).
    """
    num = counts['True_dec'] + counts['True_inc']
    den = (counts['False_dec_noact'] + counts['False_inc_noact'] + counts['True_dec']
           + counts['False_inc_dec'] + counts['False_dec_inc'] + counts['True_inc'])
    if den == 0:
        return float('nan')
    return num / den


def transaction_outcomes(y_pred_label, raw_diff):
    """Vector form of the same metric, for the bootstrap and the run loop.

    Returns (traded, won) boolean arrays. Equivalent to Eq. 25 by construction:
    a transaction is any predicted inc/dec, and it is profitable exactly when
    the RAW movement has the predicted sign — which is what the six cells above
    collapse to once the conversion rule has been applied.
    """
    pred = np.asarray(y_pred_label)
    diff = np.asarray(raw_diff, dtype=float)
    traded = pred != NOACT
    won = traded & np.where(pred == INC, diff > 0, diff < 0)
    return traded, won


def profit_accuracy(y_pred_label, raw_diff) -> float:
    traded, won = transaction_outcomes(y_pred_label, raw_diff)
    n = int(traded.sum())
    return float(won.sum()) / n if n else float('nan')


# ---------------------------------------------------------------------------
# 3. Data — the paper's ME and TI panels, from this project's own sources.
# ---------------------------------------------------------------------------

def _load_equity_indices(start, end) -> pd.DataFrame:
    """S&P 500 and DAX daily closes. yfinance -> on-disk cache, the project's
    standard degradation order (src/live_data.py, src/macro_data.py). These two
    are the only ME series with no FRED equivalent in config.json —
    UNDERSPECIFIED[#12]."""
    try:
        import yfinance as yf
        frames = []
        for col, symbol in (('sp500', '^GSPC'), ('dax', '^GDAXI')):
            h = yf.Ticker(symbol).history(start=start, end=end, interval='1d',
                                          auto_adjust=False)
            if h is None or h.empty:
                raise RuntimeError(f"empty history for {symbol}")
            s = h['Close'].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            frames.append(s.rename(col))
        out = pd.concat(frames, axis=1).sort_index()
        os.makedirs(OUT_DIR, exist_ok=True)
        out.to_csv(EQUITY_CACHE, index_label='date')
        return out
    except Exception as exc:                                  # noqa: BLE001
        if os.path.exists(EQUITY_CACHE):
            print(f"  [equity] live fetch failed ({exc}); using cache")
            out = pd.read_csv(EQUITY_CACHE, parse_dates=['date']).set_index('date')
            return out.sort_index()
        raise


def _load_macro_levels() -> pd.DataFrame:
    """The five FRED-sourced ME components, read from the caches that
    src/macro_data.py already maintains (they carry the raw legs, not just the
    differentials: us10y/de10y, fed/ecb, us_cpi_yoy/de_cpi_yoy)."""
    y = pd.read_csv(YIELD_CACHE, parse_dates=['DATE']).set_index('DATE')
    p = pd.read_csv(POLICY_CACHE, parse_dates=['DATE']).set_index('DATE')
    i = pd.read_csv(INFLATION_CACHE, parse_dates=['DATE']).set_index('DATE')
    out = pd.DataFrame(index=y.index.union(p.index).union(i.index))
    out['rate_de'] = y['de10y'].reindex(out.index)
    out['rate_eu'] = p['ecb'].reindex(out.index)
    out['rate_fed'] = p['fed'].reindex(out.index)
    out['infl_eu'] = i['de_cpi_yoy'].reindex(out.index)
    out['infl_us'] = i['us_cpi_yoy'].reindex(out.index)
    out.index = out.index.tz_localize(None).normalize()
    return out.sort_index()


def build_panel(refresh: bool = False) -> pd.DataFrame:
    """The full weekday EUR/USD panel with both the paper's ME columns and its
    TI columns. Indicators are computed on the CONTINUOUS series (trailing
    windows only) before any window is sliced out, so the primary window's
    first rows get proper burn-in rather than NaN. Macro levels are ffilled —
    a past value carried forward, never a future one carried back."""
    if not refresh and os.path.exists(PANEL_CACHE):
        return pd.read_csv(PANEL_CACHE, parse_dates=['time']).set_index('time')

    px = pd.read_csv(HISTORY_CSV, parse_dates=['time'])
    px = px[['time', 'open', 'high', 'low', 'close']].dropna()
    px = px[px['time'].dt.dayofweek <= 4]                 # UNDERSPECIFIED[#14]
    px = px.drop_duplicates(subset='time').set_index('time').sort_index()

    df = pd.DataFrame(index=px.index)
    df[['open', 'high', 'low', 'close']] = px[['open', 'high', 'low', 'close']]

    close = px['close']
    df['ma_10'] = close.rolling(10).mean()
    df['macd_12_26'] = (close.ewm(span=12, adjust=False).mean()
                        - close.ewm(span=26, adjust=False).mean())
    df['roc_2'] = (close / close.shift(2) - 1.0) * 100.0
    df['mom_4'] = close - close.shift(4)
    df['rsi_10'] = wilder_rsi(close, 10)
    df['bb_pctb_20'] = bollinger_percent_b(close, period=20, ndev=2.0)
    df['cci_20'] = cci(px['high'], px['low'], close, period=20)

    macro = _load_macro_levels()
    eq = _load_equity_indices(start='1970-01-01', end=str(px.index.max().date()))
    macro = macro.join(eq, how='outer').sort_index()
    aligned = macro.reindex(macro.index.union(df.index)).ffill().reindex(df.index)
    for c in ['rate_de', 'rate_eu', 'rate_fed', 'infl_eu', 'infl_us', 'sp500', 'dax']:
        df[c] = aligned[c]

    df = df.dropna()
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(PANEL_CACHE, index_label='time')
    return df


def slice_scale(panel: pd.DataFrame, scale: str) -> pd.DataFrame:
    """``primary`` — the paper's own calendar window trimmed to its own row
    count (1214), which reproduces its 243/243/242 test counts exactly.
    ``secondary`` — the full euro-era set."""
    if scale == 'primary':
        w = panel.loc[PAPER_WINDOW_START:PAPER_WINDOW_END]
        if len(w) > PAPER_N_ROWS:
            w = w.iloc[:PAPER_N_ROWS]
        return w
    if scale == 'secondary':
        return panel.loc[EURO_ERA_START:]
    raise ValueError(f"unknown scale {scale!r}")


# ---------------------------------------------------------------------------
# 4. Model — one LSTM, snapshotted at every iteration count in the grid.
# ---------------------------------------------------------------------------

def make_sequences(X: np.ndarray, time_steps: int):
    """Trailing windows. Sequence i ends at row ``time_steps - 1 + i`` and
    contains only rows at or before it. Returns (sequences, end_indices)."""
    n = len(X)
    if n < time_steps:
        return np.empty((0, time_steps, X.shape[1]), dtype=X.dtype), np.empty(0, dtype=int)
    idx = np.arange(time_steps - 1, n)
    seq = np.stack([X[e - time_steps + 1:e + 1] for e in idx])
    return seq, idx


def _layers(n_features: int):
    import keras
    return [
        keras.layers.Input(shape=(TIME_STEPS, n_features)),
        keras.layers.LSTM(LSTM_UNITS),
        keras.layers.Dropout(LSTM_DROPOUT),
        keras.layers.Dense(3, activation='softmax'),
    ]


_MODEL_CACHE = {}


def _compiled_model(n_features: int):
    """One COMPILED model per input width, reused across fits.

    TensorFlow traces the training graph on the first fit() of a model, and for
    an LSTM on this machine that trace costs ~68s — as much as an entire
    200-epoch training run. Building a fresh model per fit would therefore have
    spent more than half the study's compute on graph construction. Weights and
    optimizer state are reset explicitly between fits instead (see
    train_with_snapshots), which is exactly equivalent to a fresh model and
    keeps the trace.
    """
    import keras
    if n_features not in _MODEL_CACHE:
        model = keras.Sequential(_layers(n_features))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR),
                      loss='sparse_categorical_crossentropy')
        _MODEL_CACHE[n_features] = model
    return _MODEL_CACHE[n_features]


def _seeded_initial_weights(n_features: int):
    """Freshly initialised weights for the current RNG state. An uncompiled
    Sequential only allocates variables — no graph, no trace."""
    import keras
    return keras.Sequential(_layers(n_features)).get_weights()


def train_with_snapshots(x_train, y_train, x_val, x_test, seed: int,
                         iteration_grid=None):
    """Train ONCE to max(iteration_grid) epochs, capturing val/test class
    probabilities at every iteration count in the grid.

    This is what makes Leak 2 measurable at all: arms A0..A4 differ only in
    WHICH snapshot they select, so the underlying fitted weights are shared and
    the arm-to-arm gap cannot be contaminated by a second training run's noise.
    """
    import keras
    # Resolved at CALL time, not bound as a default: ITERATION_GRID is a module
    # global that a smoke run or a test may legitimately shrink, and a default
    # argument would silently keep training the full 200 epochs.
    iteration_grid = ITERATION_GRID if iteration_grid is None else iteration_grid
    n_features = x_train.shape[2]

    # Seed once, then draw this fit's initial weights from that stream; fit()'s
    # dropout mask and shuffle order continue from the same stream, so the whole
    # fit is reproducible from `seed` alone.
    keras.utils.set_random_seed(seed)
    init_weights = _seeded_initial_weights(n_features)

    model = _compiled_model(n_features)
    model.set_weights(init_weights)
    if getattr(model.optimizer, 'built', False):
        # Zero Adam's moment accumulators AND its iteration counter, so a reused
        # model starts each fit from the same state a brand-new one would.
        for v in model.optimizer.variables:
            v.assign(keras.ops.zeros_like(v))

    wanted = set(int(i) for i in iteration_grid)
    saved = {}

    class _Snapshot(keras.callbacks.Callback):
        # Snapshot WEIGHTS during training and run the forward passes after it,
        # rather than predicting inside the callback. Calling the model from
        # on_epoch_end while fit() owns a compiled train function makes TF trace
        # a fresh inference graph for every call — measured at ~8s per call
        # against ~0.3s for an entire training epoch, i.e. the instrumentation
        # would have cost 50x the experiment. The weights are ~5k parameters.
        def on_epoch_end(self, epoch, logs=None):
            e = epoch + 1
            if e in wanted:
                saved[e] = [w.copy() for w in model.get_weights()]

    model.fit(x_train, y_train, epochs=max(wanted), batch_size=LSTM_BATCH,
              shuffle=True, verbose=0, callbacks=[_Snapshot()])

    # Direct eager calls, not model.predict(): predict() traces its own
    # inference graph per input shape, and the forward pass on a few hundred
    # 20x8 windows is far cheaper eagerly than the trace that would wrap it.
    snaps = {}
    for e in sorted(saved):
        model.set_weights(saved[e])
        snaps[e] = (np.asarray(model(x_val, training=False)),
                    np.asarray(model(x_test, training=False)))
    model.set_weights(init_weights)
    return snaps


# ---------------------------------------------------------------------------
# 5. The hybrid combiner.
# ---------------------------------------------------------------------------

def hybrid_decide(p_me: np.ndarray, p_ti: np.ndarray, min_confidence: float = 0.0,
                  tie_break: str = 'ti') -> np.ndarray:
    """The paper's postprocessing rules, plus the one tunable knob its prose
    implies but never names.

    Paper, verbatim:
      * "If one model's prediction is class_noact, then the final decision will
        be class_noact."
      * "If both models agree on the labels, we set the final decision as this
        label."
      * "If the predictions of the two models are different, we choose for the
        final decision the one whose prediction has higher probability. If the
        probability is the same, we choose the prediction of the TI_LSTM model."

    plus, from "eliminating transactions with weaker confidence":
      * abstain (-> class_noact) when the surviving decision's confidence is
        below ``min_confidence``.

    ``min_confidence=0.0`` is a no-op, so this function with min_confidence=0
    IS the published combiner. ``tie_break`` reads the paper's two unexplained
    reported variants ("modification based on ME_LSTM" / "on TI_LSTM").
    """
    if tie_break not in ('me', 'ti'):
        raise ValueError("tie_break must be 'me' or 'ti'")
    c_me, c_ti = p_me.argmax(axis=1), p_ti.argmax(axis=1)
    conf_me, conf_ti = p_me.max(axis=1), p_ti.max(axis=1)

    prefer_me = conf_me > conf_ti
    if tie_break == 'me':
        prefer_me = conf_me >= conf_ti
    disagree_pick = np.where(prefer_me, c_me, c_ti)

    out = np.where(c_me == c_ti, c_me, disagree_pick)
    out = np.where((c_me == NOACT) | (c_ti == NOACT), NOACT, out)
    out = np.where(np.maximum(conf_me, conf_ti) < min_confidence, NOACT, out)
    return out.astype(np.int64)


# ---------------------------------------------------------------------------
# 6. Arms — selection protocol, then a single scoring pass on test.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmSpec:
    name: str
    threshold_scope: str        # 'train_only' | 'full_series'
    iter_block: str             # 'val' | 'test'
    hybrid_block: str           # 'val' | 'test'
    description: str


ARMS = {
    'A0': ArmSpec('A0', 'train_only', 'val', 'val',
                  'clean: threshold on train, iterations on val, hybrid on val, test scored once'),
    'A1': ArmSpec('A1', 'full_series', 'val', 'val', 'leak 1 only: threshold over the full series'),
    'A2': ArmSpec('A2', 'train_only', 'test', 'val', 'leak 2 only: iteration count selected on test'),
    'A3': ArmSpec('A3', 'train_only', 'val', 'test', 'leak 3 only: hybrid rules tuned on test'),
    'A4': ArmSpec('A4', 'full_series', 'test', 'test', 'all three leaks (approximates as-published)'),
}


@dataclass
class Block:
    """One scoring/selection block: labels + raw diffs for its rows. ``proba``
    is filled per (model, iteration) as snapshots arrive."""
    name: str
    y: np.ndarray
    diff: np.ndarray
    proba: dict = field(default_factory=dict)   # (model_type, iters) -> (n, 3)


def _guarded_best(candidates, n_rows: int):
    """Pick the candidate with the highest profit_accuracy among those trading
    on >= MIN_TXN_FRACTION of the block; if none qualifies, the one with the
    most transactions. ``candidates`` is a list of (key, pa, n_txn).
    See UNDERSPECIFIED[#8]."""
    floor = MIN_TXN_FRACTION * n_rows
    eligible = [c for c in candidates if c[2] >= floor and not math.isnan(c[1])]
    if eligible:
        return max(eligible, key=lambda c: (c[1], c[2]))[0]
    return max(candidates, key=lambda c: c[2])[0]


def select_iterations(spec: ArmSpec, blocks: dict, model_type: str):
    """Leak 2's decision point. Reads ONLY blocks[spec.iter_block]; for A0/A1/A3
    that is the validation block and the test block is never touched."""
    blk = blocks[spec.iter_block]
    cands = []
    for it in ITERATION_GRID:
        pred = blk.proba[(model_type, it)].argmax(axis=1)
        traded, won = transaction_outcomes(pred, blk.diff)
        n = int(traded.sum())
        cands.append((it, (float(won.sum()) / n if n else float('nan')), n))
    return _guarded_best(cands, len(blk.y))


def select_hybrid(spec: ArmSpec, blocks: dict, iters_me: int, iters_ti: int):
    """Leak 3's decision point. Reads ONLY blocks[spec.hybrid_block]."""
    blk = blocks[spec.hybrid_block]
    p_me = blk.proba[('ME_LSTM', iters_me)]
    p_ti = blk.proba[('TI_LSTM', iters_ti)]
    cands = []
    for mc in HYBRID_MIN_CONF_GRID:
        for tb in HYBRID_TIE_BREAK_GRID:
            pred = hybrid_decide(p_me, p_ti, min_confidence=mc, tie_break=tb)
            traded, won = transaction_outcomes(pred, blk.diff)
            n = int(traded.sum())
            cands.append(((mc, tb), (float(won.sum()) / n if n else float('nan')), n))
    return _guarded_best(cands, len(blk.y))


def run_arm(spec: ArmSpec, blocks_by_scope: dict):
    """Run one arm end to end for one seed and one horizon.

    ``blocks_by_scope`` maps 'train_only'/'full_series' -> {'val': Block,
    'test': Block}. Selection happens first and reads only the block its spec
    names; scoring then happens ONCE on test. Returns
    {model: (pred_on_test, meta)}.
    """
    blocks = blocks_by_scope[spec.threshold_scope]
    test = blocks['test']
    out = {}

    chosen_iters = {}
    for mt in MODEL_TYPES:
        it = select_iterations(spec, blocks, mt)
        chosen_iters[mt] = it
        pred = test.proba[(mt, it)].argmax(axis=1)
        out[mt] = (pred, {'iters_self': it, 'iters_me': '', 'iters_ti': '',
                          'min_conf': '', 'tie_break': ''})

    mc, tb = select_hybrid(spec, blocks, chosen_iters['ME_LSTM'], chosen_iters['TI_LSTM'])
    pred = hybrid_decide(test.proba[('ME_LSTM', chosen_iters['ME_LSTM'])],
                         test.proba[('TI_LSTM', chosen_iters['TI_LSTM'])],
                         min_confidence=mc, tie_break=tb)
    out['HYBRID'] = (pred, {'iters_self': '', 'iters_me': chosen_iters['ME_LSTM'],
                            'iters_ti': chosen_iters['TI_LSTM'],
                            'min_conf': mc, 'tie_break': tb})
    return out


# ---------------------------------------------------------------------------
# 7. Uncertainty — moving-block bootstrap on the project's existing machinery.
# ---------------------------------------------------------------------------

def gap_bootstrap(trade_a, win_a, trade_b, win_b, block_len: int = BOOTSTRAP_BLOCK_LEN,
                  n_boot: int = BOOTSTRAP_RESAMPLES, alpha: float = BOOTSTRAP_ALPHA,
                  random_state: int = 42):
    """Paired moving-block (circular) bootstrap of
    profit_accuracy(a) - profit_accuracy(b) over the shared test rows.

    Uses src.walk_forward_validation._circular_block_bootstrap_indices — the
    same resampler the walk-forward validation and the vol-scaled backtest use,
    for the same reason: adjacent FX days are not independent, so an i.i.d.
    bootstrap would understate the interval.

    profit_accuracy is a RATIO whose denominator is itself random (the model
    chooses when to trade), so each resample recomputes sum(wins)/sum(trades)
    rather than averaging per-row accuracies. Inputs are per-row seed MEANS
    (fractional in [0, 1]): the reported estimate is the seed-mean arm, so the
    interval describes row-sampling uncertainty around that mean.
    """
    from src.walk_forward_validation import _circular_block_bootstrap_indices

    trade_a, win_a = np.asarray(trade_a, float), np.asarray(win_a, float)
    trade_b, win_b = np.asarray(trade_b, float), np.asarray(win_b, float)
    n = len(trade_a)

    def _pa(tr, wn):
        d = tr.sum()
        return (wn.sum() / d) if d > 0 else np.nan

    point = _pa(trade_a, win_a) - _pa(trade_b, win_b)
    rng = np.random.default_rng(random_state)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _circular_block_bootstrap_indices(n, block_len, rng)
        deltas[i] = _pa(trade_a[idx], win_a[idx]) - _pa(trade_b[idx], win_b[idx])
    good = deltas[np.isfinite(deltas)]
    if good.size < n_boot * 0.5:
        return {'delta': point, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_boot_valid': int(good.size)}
    lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {'delta': point, 'ci_low': float(lo), 'ci_high': float(hi),
            'n_boot_valid': int(good.size)}


# ---------------------------------------------------------------------------
# 8. Driver
# ---------------------------------------------------------------------------

@dataclass
class HorizonGeometry:
    """Everything about one (scale, horizon) that is shared by every arm, model
    and seed: the row blocks, the raw diffs, and the labels under BOTH threshold
    scopes. Derived identically in the parent and in every worker process, so a
    worker can rebuild its own inputs from a 5-tuple instead of having tens of
    megabytes of sequence tensors pickled to it."""
    window: pd.DataFrame
    diff: np.ndarray
    n: int
    train_end: int
    val_end: int
    seq_end: np.ndarray
    is_train: np.ndarray
    is_val: np.ndarray
    is_test: np.ndarray
    scopes: dict


def prepare_horizon(win: pd.DataFrame, horizon: int) -> HorizonGeometry:
    """Labels, split points and sequence geometry for one horizon.

    The label is the sign of ``close_{t+n} - close_t`` against the threshold, so
    the last ``horizon`` rows carry no label and are dropped. Sequences end-index
    at or after ``TIME_STEPS - 1``, which costs the TRAIN block its first 19 rows
    and leaves val/test untouched — that is what keeps the test block at exactly
    the paper's 243/243/242 rows.
    """
    close = win['close'].to_numpy(float)
    diff_all = np.full(len(close), np.nan)
    diff_all[:len(close) - horizon] = close[horizon:] - close[:len(close) - horizon]
    labeled = np.isfinite(diff_all)

    n = int(labeled.sum())
    w = win.iloc[:n]
    diff = diff_all[labeled]
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))

    scopes = {}
    for scope, src in (('train_only', diff[:train_end]), ('full_series', diff)):
        tau, ub, ent = choose_threshold(src)
        scopes[scope] = {'tau': tau, 'ub': ub, 'entropy': ent,
                         'y': label_three_class(diff, tau)}

    seq_end = np.arange(TIME_STEPS - 1, n)
    return HorizonGeometry(
        window=w, diff=diff, n=n, train_end=train_end, val_end=val_end,
        seq_end=seq_end,
        is_train=seq_end < train_end,
        is_val=(seq_end >= train_end) & (seq_end < val_end),
        is_test=seq_end >= val_end,
        scopes=scopes,
    )


def model_inputs(geo: HorizonGeometry, model_type: str):
    """Standardised, clipped feature sequences for one model's column set.

    The scaler is fit on TRAIN rows only and the result clipped to
    +/-SCALE_CLIP sigma — see UNDERSPECIFIED[#7]. Identical in every arm.
    """
    raw = geo.window[MODEL_FEATURES[model_type]].to_numpy(float)
    mu = raw[:geo.train_end].mean(axis=0)
    sd = raw[:geo.train_end].std(axis=0)
    sd[sd == 0] = 1.0
    z = np.clip((raw - mu) / sd, -SCALE_CLIP, SCALE_CLIP)
    seq, _ = make_sequences(z, TIME_STEPS)
    return seq


# --- worker side: one fit per job, rebuilt from a 5-tuple -------------------

_WORKER_CACHE = {}


def _worker_geometry(scale: str, horizon: int) -> HorizonGeometry:
    key = (scale, horizon)
    if key not in _WORKER_CACHE:
        if 'panel' not in _WORKER_CACHE:
            _WORKER_CACHE['panel'] = build_panel()
        _WORKER_CACHE[key] = prepare_horizon(
            slice_scale(_WORKER_CACHE['panel'], scale), horizon)
    return _WORKER_CACHE[key]


def _fit_job(job):
    """One (scale, horizon, scope, model_type, seed) fit. Returns the job key
    and the val/test probability snapshots at every iteration count."""
    scale, horizon, scope, model_type, seed = job
    geo = _worker_geometry(scale, horizon)
    seq = model_inputs(geo, model_type)
    y_all = geo.scopes[scope]['y']
    snaps = train_with_snapshots(
        seq[geo.is_train], y_all[geo.seq_end[geo.is_train]],
        seq[geo.is_val], seq[geo.is_test], seed)
    return job, {it: (pv.astype(np.float32), pt.astype(np.float32))
                 for it, (pv, pt) in snaps.items()}


def _worker_init(threads: int):
    """Workers run on CPU: the benchmark above shows no GPU advantage for a
    model this small, and one process per core beats one process fighting for a
    GPU it cannot saturate. Threads are capped so N workers do not oversubscribe
    the machine."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    for var in ('OMP_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
        os.environ[var] = str(threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'


def run_scale(scale: str, panel: pd.DataFrame, seeds=SEEDS, workers: int = 1,
              verbose: bool = True):
    """Every arm x horizon x model x seed for one data scale.

    Returns (arm_rows, rowvecs) where rowvecs[(horizon, arm, model, seed)] =
    (traded, won) over the test rows — the raw material for the bootstrap.

    The fits are dispatched first and the arms are then pure post-processing
    over the resulting snapshots. That is deliberate: A0..A4 read the SAME
    fitted weights wherever their protocols agree, so an arm-to-arm gap can
    never be contaminated by a second training run's noise.
    """
    win = slice_scale(panel, scale)
    geos = {h: prepare_horizon(win, h) for h in HORIZONS}

    if verbose:
        for h, geo in geos.items():
            print(f"  [{scale} h={h}] n={geo.n} train={geo.train_end} "
                  f"val={geo.val_end - geo.train_end} test={geo.n - geo.val_end} | "
                  f"tau_train={geo.scopes['train_only']['tau']:.5f} "
                  f"(ub={geo.scopes['train_only']['ub']:.5f})  "
                  f"tau_full={geo.scopes['full_series']['tau']:.5f} "
                  f"(ub={geo.scopes['full_series']['ub']:.5f})")

    jobs = [(scale, h, scope, mt, seed)
            for h in HORIZONS for scope in ('train_only', 'full_series')
            for mt in MODEL_TYPES for seed in seeds]

    t0 = time.time()
    results = {}
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            key, snaps = _fit_job(job)
            results[key] = snaps
            if verbose:
                print(f"    fit {i}/{len(jobs)} {job[1:]} "
                      f"({time.time() - t0:.0f}s elapsed)")
    else:
        threads = max(1, (os.cpu_count() or 8) // workers)
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(threads,)) as pool:
            for i, (key, snaps) in enumerate(pool.map(_fit_job, jobs), 1):
                results[key] = snaps
                if verbose and (i % 5 == 0 or i == len(jobs)):
                    rate = (time.time() - t0) / i
                    print(f"    fit {i}/{len(jobs)} "
                          f"({time.time() - t0:.0f}s elapsed, "
                          f"~{rate * (len(jobs) - i):.0f}s left)", flush=True)
    if verbose:
        print(f"  [{scale}] {len(jobs)} fits in {time.time() - t0:.0f}s")

    # --- arms (pure post-processing over the snapshots) ---------------------
    arm_rows, rowvecs = [], {}
    for horizon in HORIZONS:
        geo = geos[horizon]
        end_val, end_test = geo.seq_end[geo.is_val], geo.seq_end[geo.is_test]
        diff = geo.diff

        blocks_by_scope = {}
        for scope, meta in geo.scopes.items():
            blocks_by_scope[scope] = {
                'val': Block('val', meta['y'][end_val], diff[end_val]),
                'test': Block('test', meta['y'][end_test], diff[end_test]),
            }
            for mt in MODEL_TYPES:
                for seed in seeds:
                    for it, (pv, pt) in results[(scale, horizon, scope, mt, seed)].items():
                        blocks_by_scope[scope]['val'].proba[(mt, it, seed)] = pv
                        blocks_by_scope[scope]['test'].proba[(mt, it, seed)] = pt

        scopes = geo.scopes
        for seed in seeds:
            seed_blocks = {}
            for scope, bl in blocks_by_scope.items():
                seed_blocks[scope] = {}
                for bname, b in bl.items():
                    sub = Block(b.name, b.y, b.diff)
                    sub.proba = {(mt, it): b.proba[(mt, it, seed)]
                                 for mt in MODEL_TYPES for it in ITERATION_GRID}
                    seed_blocks[scope][bname] = sub

            for arm_name in ARM_NAMES:
                spec = ARMS[arm_name]
                res = run_arm(spec, seed_blocks)
                test_blk = seed_blocks[spec.threshold_scope]['test']
                for model, (pred, meta) in res.items():
                    traded, won = transaction_outcomes(pred, test_blk.diff)
                    rowvecs[(horizon, arm_name, model, seed)] = (traded, won)
                    n_txn = int(traded.sum())
                    arm_rows.append({
                        'scale': scale, 'arm': arm_name, 'horizon': horizon,
                        'model': model, 'seed': seed,
                        'profit_accuracy': (float(won.sum()) / n_txn) if n_txn else np.nan,
                        'n_transactions': n_txn, 'n_test_rows': len(test_blk.y),
                        'transaction_rate': n_txn / len(test_blk.y),
                        'accuracy_3class': float((pred == test_blk.y).mean()),
                        'threshold_scope': spec.threshold_scope,
                        'threshold_tau': scopes[spec.threshold_scope]['tau'],
                        'threshold_upper_bound': scopes[spec.threshold_scope]['ub'],
                        'iter_selection_block': spec.iter_block,
                        'hybrid_selection_block': spec.hybrid_block,
                        **meta,
                    })

    return arm_rows, rowvecs


ABSTENTION_CSV = os.path.join(OUT_DIR, 'abstention_curve.csv')
ABSTENTION_GRID = (0.0, 0.34, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                   0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def abstention_curve(scale: str, panel: pd.DataFrame, horizon: int = 1,
                     seeds=SEEDS, workers: int = 1) -> pd.DataFrame:
    """profit_accuracy against transaction count as the hybrid's confidence
    floor is raised, with the selection floor REMOVED.

    This exists to put a number on UNDERSPECIFIED[#8] rather than assert it.
    profit_accuracy is a ratio whose denominator the model chooses, so raising
    the abstention threshold is expected to raise it mechanically; the question
    is how far, and on how few transactions. The iteration counts are the ones
    A0 picks on validation, so only the confidence floor varies.
    """
    geo = prepare_horizon(slice_scale(panel, scale), horizon)
    jobs = [(scale, horizon, 'train_only', mt, seed)
            for mt in ('ME_LSTM', 'TI_LSTM') for seed in seeds]
    if workers <= 1:
        results = dict(_fit_job(j) for j in jobs)
    else:
        threads = max(1, (os.cpu_count() or 8) // workers)
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                 initargs=(threads,)) as pool:
            results = dict(pool.map(_fit_job, jobs))

    end_val, end_test = geo.seq_end[geo.is_val], geo.seq_end[geo.is_test]
    y = geo.scopes['train_only']['y']
    val = Block('val', y[end_val], geo.diff[end_val])
    test = Block('test', y[end_test], geo.diff[end_test])

    rows = []
    for seed in seeds:
        for blk, sel in ((val, 0), (test, 1)):
            for mt in ('ME_LSTM', 'TI_LSTM'):
                for it in ITERATION_GRID:
                    blk.proba[(mt, it)] = results[(scale, horizon, 'train_only', mt, seed)][it][sel]
        # A0's own iteration choice — validation only.
        it_me = select_iterations(ARMS['A0'], {'val': val, 'test': test}, 'ME_LSTM')
        it_ti = select_iterations(ARMS['A0'], {'val': val, 'test': test}, 'TI_LSTM')
        for mc in ABSTENTION_GRID:
            pred = hybrid_decide(test.proba[('ME_LSTM', it_me)],
                                 test.proba[('TI_LSTM', it_ti)],
                                 min_confidence=mc, tie_break='ti')
            traded, won = transaction_outcomes(pred, test.diff)
            n = int(traded.sum())
            rows.append({'scale': scale, 'horizon': horizon, 'seed': seed,
                         'min_confidence': mc, 'n_transactions': n,
                         'n_test_rows': len(test.y),
                         'profit_accuracy': (float(won.sum()) / n) if n else np.nan})
    return pd.DataFrame(rows)


def summarise(arms: pd.DataFrame, rowvecs_by_scale: dict) -> pd.DataFrame:
    """Seed means per (scale, arm, horizon, model) + a bootstrap CI for every
    leak gap Ax - A0."""
    records = []

    g = arms.groupby(['scale', 'arm', 'horizon', 'model'], dropna=False)
    for (scale, arm, horizon, model), sub in g:
        records.append({
            'record': 'arm_mean', 'scale': scale, 'arm': arm, 'horizon': horizon,
            'model': model, 'comparison': '',
            'profit_accuracy_mean': sub['profit_accuracy'].mean(),
            'profit_accuracy_sd': sub['profit_accuracy'].std(ddof=1),
            'n_transactions_mean': sub['n_transactions'].mean(),
            'transaction_rate_mean': sub['transaction_rate'].mean(),
            'n_test_rows': int(sub['n_test_rows'].iloc[0]),
            'n_seeds': len(sub),
            'delta': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
            'distinguishable_from_zero': '',
        })

    for scale, rowvecs in rowvecs_by_scale.items():
        seeds = sorted({k[3] for k in rowvecs})
        for horizon in HORIZONS:
            for model in ALL_MODELS:
                def _mean_vecs(arm):
                    tr = np.mean([rowvecs[(horizon, arm, model, s)][0] for s in seeds], axis=0)
                    wn = np.mean([rowvecs[(horizon, arm, model, s)][1] for s in seeds], axis=0)
                    return tr, wn

                tr0, wn0 = _mean_vecs('A0')
                for arm in ('A1', 'A2', 'A3', 'A4'):
                    tra, wna = _mean_vecs(arm)
                    bs = gap_bootstrap(tra, wna, tr0, wn0)
                    lo, hi = bs['ci_low'], bs['ci_high']
                    dist = ''
                    if np.isfinite(lo) and np.isfinite(hi):
                        dist = 'yes' if (lo > 0 or hi < 0) else 'no'
                    records.append({
                        'record': 'gap', 'scale': scale, 'arm': arm, 'horizon': horizon,
                        'model': model, 'comparison': f'{arm}-A0',
                        'profit_accuracy_mean': np.nan, 'profit_accuracy_sd': np.nan,
                        'n_transactions_mean': np.nan, 'transaction_rate_mean': np.nan,
                        'n_test_rows': len(tr0), 'n_seeds': len(seeds),
                        'delta': bs['delta'], 'ci_low': lo, 'ci_high': hi,
                        'distinguishable_from_zero': dist,
                    })
    return pd.DataFrame.from_records(records)


# --- The paper's own reported figures, for the report's side-by-side --------
# Tables 4-6 (single models, per-model averages over the 4 iteration counts)
# and Table 7/11/15 (hybrid averages over the 4x4 iteration grid).
PAPER_REPORTED = {
    # horizon: {model: (profit_accuracy_pct, mean_transactions, n_test)}
    1: {'ME_LSTM': (50.69, 149.50, 243), 'TI_LSTM': (52.18, 155.25, 243),
        'ME_TI_LSTM': (53.05, 157.25, 243), 'HYBRID': (77.32, 64.75, 243)},
    3: {'ME_LSTM': (51.31, 174.50, 243), 'TI_LSTM': (48.58, 146.50, 243),
        'ME_TI_LSTM': (53.84, 158.50, 243), 'HYBRID': (78.98, 65.13, 243)},
    5: {'ME_LSTM': (47.31, 206.25, 242), 'TI_LSTM': (49.88, 151.50, 242),
        'ME_TI_LSTM': (48.73, 138.75, 242), 'HYBRID': (84.08, 69.31, 242)},
}
PAPER_SUMMARY_TABLE20 = {
    1: {'ME_LSTM': 50.16, 'TI_LSTM': 51.43, 'ME_TI_LSTM': 49.89, 'HYBRID': 73.09},
    3: {'ME_LSTM': 50.56, 'TI_LSTM': 47.49, 'ME_TI_LSTM': 43.22, 'HYBRID': 68.31},
    5: {'ME_LSTM': 45.08, 'TI_LSTM': 45.11, 'ME_TI_LSTM': 44.56, 'HYBRID': 79.42},
}


def write_report(arms: pd.DataFrame, summary: pd.DataFrame, scales, meta: dict):
    """REPLICATION.md — the paper's figures, our A0 next to them, the leak
    decomposition with CIs, the underspecification ledger, and a plain
    statement of what is and is not distinguishable from zero."""
    means = summary[summary['record'] == 'arm_mean']
    gaps = summary[summary['record'] == 'gap']
    L = []

    L.append("# Replication: Yıldırım, Toroslu & Fiore (2021)\n")
    L.append("*Forecasting directional movement of Forex data using LSTM with technical "
             "and macroeconomic indicators*, **Financial Innovation 7:1**, "
             "[doi:10.1186/s40854-020-00220-2]"
             "(https://doi.org/10.1186/s40854-020-00220-2) (open access, CC BY 4.0).\n")
    L.append(f"Generated {meta['generated']} — {meta['n_fits']} LSTM fits, "
             f"seeds {list(meta['seeds'])}, scales {list(scales)}.\n")
    L.append("> **This study costs zero alpha.** It registers no hypothesis in any "
             "family, adds no row to any `*_hypothesis_log.csv`, and does not tighten "
             "any Bonferroni bar. It is a sensitivity analysis of an *external* "
             "published result, not a claim about EUR/USD predictability. Nothing "
             "here may be used to justify a production feature.\n")

    # 1. What the paper reports -------------------------------------------
    L.append("## 1. What the paper reports\n")
    L.append("The split, quoted: *\"The data set was split into the training and test "
             "sets, with ratios of 80% and 20%, respectively. The training phase was "
             "carried out with different numbers of iterations\"*; and for the extended "
             "set, *\"split into training and test sets, with ratios of 90% and 10%\"*. "
             "**There is no validation set anywhere in the paper.** Every number below "
             "is therefore a test-block number, including the ones that were compared "
             "against each other to make a choice.\n")
    L.append("Single-model profit_accuracy, averaged over the four iteration counts "
             "(Tables 4-6, 8-10, 12-14) and the hybrid averaged over the 4x4 iteration "
             "grid (Tables 7, 11, 15):\n")
    L.append("| Horizon | ME_LSTM | TI_LSTM | ME_TI_LSTM | Hybrid | test rows |")
    L.append("|---|---|---|---|---|---|")
    for h in HORIZONS:
        r = PAPER_REPORTED[h]
        L.append(f"| {h}-day | {r['ME_LSTM'][0]:.2f}% | {r['TI_LSTM'][0]:.2f}% | "
                 f"{r['ME_TI_LSTM'][0]:.2f}% | **{r['HYBRID'][0]:.2f}%** | "
                 f"{r['ME_LSTM'][2]} |")
    L.append("")
    L.append("The paper's own summary table (Table 20) disagrees with its per-model "
             "tables — 1-day ME/TI/ME-TI/Hybrid are 50.16 / 51.43 / 49.89 / 73.09 there "
             "versus 50.69 / 52.18 / 53.05 / 77.32 in Tables 4-7. We compare against "
             "Tables 4-7 throughout and note the discrepancy as-is.\n")
    L.append("At n=243 test rows and ~150 transactions, the three single-model figures "
             "are indistinguishable from a coin flip: the 95% binomial interval around "
             "50% on 150 trials is roughly [42%, 58%]. The paper's headline is not the "
             "single models — it is the hybrid's ~25-point jump, bought by abstaining "
             "from about three quarters of the trades.\n")

    # 2. Our clean arm -----------------------------------------------------
    L.append("## 2. Our A0 (clean) next to the paper\n")
    L.append("A0 = threshold fitted on training rows only, iteration count chosen on "
             "the validation block, hybrid rules tuned on the validation block, test "
             "block scored exactly once. Mean over "
             f"{meta['n_seeds']} seeds; +/- is the seed standard deviation.\n")
    for scale in scales:
        span = meta.get('scale_spans', {}).get(scale, ('?', '?', 0))
        L.append(f"**Scale: `{scale}`** — "
                 + ("the paper's own window trimmed to its own row count"
                    if scale == 'primary' else "our full euro-era daily set")
                 + f": {span[2]} weekday bars, {span[0]} .. {span[1]}.")
        if scale == 'primary':
            L.append("\nThe paper covers January 2013 - January 2018 in 1214 bars "
                     "('days in which the markets were open'); our weekday history "
                     "carries ~260 bars/year against their ~241, so the same 1214-row "
                     "count runs out earlier in calendar time. Row count is what fixes "
                     "the 243-row test block and therefore the statistical power, so "
                     "the count is what we matched.")
        L.append("")
        def _cell(arm, h, model):
            row = means[(means.scale == scale) & (means.arm == arm)
                        & (means.horizon == h) & (means.model == model)]
            if row.empty:
                return None
            r = row.iloc[0]
            sd = r.profit_accuracy_sd if np.isfinite(r.profit_accuracy_sd) else 0.0
            txt = ("n/a" if not np.isfinite(r.profit_accuracy_mean)
                   else f"{100 * r.profit_accuracy_mean:.2f}% ± {100 * sd:.2f}")
            return r, txt

        L.append("| Horizon | Model | paper | our A0 (clean) | our A4 (all 3 leaks) "
                 "| A0 transactions | paper transactions |")
        L.append("|---|---|---|---|---|---|---|")
        for h in HORIZONS:
            for model in ALL_MODELS:
                got = _cell('A0', h, model)
                if got is None:
                    continue
                r, pa0 = got
                pa4 = (_cell('A4', h, model) or (None, 'n/a'))[1]
                pr = PAPER_REPORTED[h][model]
                L.append(f"| {h}-day | {model} | {pr[0]:.2f}% | {pa0} | {pa4} | "
                         f"{r.n_transactions_mean:.1f}/{r.n_test_rows} | "
                         f"{pr[1]:.2f}/{pr[2]} |")
        L.append("")

    # 2b. Fidelity check on the labelling algorithm ------------------------
    L.append("### Fidelity check: the labelling algorithm reproduces Table 3\n")
    L.append("The paper's Table 3 reports the thresholds its entropy search "
             "selected: **0.0023 / 0.0040 / 0.0055** for 1 / 3 / 5 days ahead. "
             "Our independent implementation of Algorithm 1 + Algorithm 2, run "
             "on our own EUR/USD history, lands here:\n")
    L.append("| Scale | Horizon | τ (train-only, A0/A2/A3) | τ (full series, A1/A4) | paper Table 3 |")
    L.append("|---|---|---|---|---|")
    paper_tau = {1: 0.0023, 3: 0.0040, 5: 0.0055}
    for scale in scales:
        for h in HORIZONS:
            sub = arms[(arms.scale == scale) & (arms.horizon == h)]
            if sub.empty:
                continue
            tr = sub[sub.threshold_scope == 'train_only']['threshold_tau'].iloc[0]
            fu = sub[sub.threshold_scope == 'full_series']['threshold_tau'].iloc[0]
            L.append(f"| {scale} | {h}-day | {tr:.5f} | {fu:.5f} | {paper_tau[h]:.4f} |")
    L.append("")
    L.append("On the paper-scale window the three train-only thresholds agree "
             "with Table 3 to within a fraction of a pip. That is a meaningful "
             "check: the histogram upper bound and the entropy sweep were "
             "re-derived from the prose alone, on different data, and landed on "
             "the paper's numbers. Whatever else this replication does or does "
             "not reproduce, the labelling stage is faithful.\n")
    L.append("It also shows how *small* Leak 1 is mechanically: moving the "
             "threshold's scope from train-only to the full series shifts τ by "
             "only a few 1e-5. Leak 1 is real but its lever is short, and the "
             "leak decomposition below should be read with that in mind.\n")

    # 3. Leak decomposition ------------------------------------------------
    L.append("## 3. Leak decomposition\n")
    L.append("Each arm changes **one** selection input and nothing else. The training "
             "rows, the scoring rows, the architecture, the seeds and the "
             "transaction-count floor are identical across arms; A1/A4 retrain because "
             "a different threshold changes the training labels, and every arm reads "
             "the same fitted weights everywhere else. A positive gap means the leak "
             "inflates the reported figure.\n")
    L.append("| Arm | what leaks |")
    L.append("|---|---|")
    for a in ARM_NAMES:
        L.append(f"| {a} | {ARMS[a].description} |")
    L.append("")
    L.append("CIs are 95% moving-block (circular, block length "
             f"{BOOTSTRAP_BLOCK_LEN}, {BOOTSTRAP_RESAMPLES} resamples) paired "
             "bootstraps over the test rows, on "
             "`src.walk_forward_validation._circular_block_bootstrap_indices`. "
             "profit_accuracy has a random denominator, so each resample recomputes "
             "sum(wins)/sum(transactions).\n")
    for scale in scales:
        L.append(f"**Scale: `{scale}`**\n")
        L.append("| Horizon | Model | gap | Δ profit_accuracy | 95% CI | ≠ 0? |")
        L.append("|---|---|---|---|---|---|")
        for h in HORIZONS:
            for model in ALL_MODELS:
                for arm in ('A1', 'A2', 'A3', 'A4'):
                    row = gaps[(gaps.scale == scale) & (gaps.horizon == h)
                               & (gaps.model == model) & (gaps.arm == arm)]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    ci = ("n/a" if not np.isfinite(r.ci_low)
                          else f"[{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]")
                    L.append(f"| {h}-day | {model} | {arm}−A0 | "
                             f"{100 * r.delta:+.2f} pp | {ci} | "
                             f"{r.distinguishable_from_zero or 'n/a'} |")
        L.append("")

    # 3b. What the leaks do NOT explain ------------------------------------
    L.append("### What the leak decomposition does not explain\n")
    L.append("The gaps above are differences between OUR arms. They are only "
             "half the picture, because our A4 — the arm that takes all three "
             "shortcuts and is the closest thing here to 'as published' — does "
             "not land anywhere near the paper's reported hybrid:\n")
    L.append("| Scale | Horizon | paper hybrid | our A4 hybrid | our A0 hybrid "
             "| residual (paper − A4) |")
    L.append("|---|---|---|---|---|---|")
    for scale in scales:
        for h in HORIZONS:
            a0 = means[(means.scale == scale) & (means.arm == 'A0')
                       & (means.horizon == h) & (means.model == 'HYBRID')]
            a4 = means[(means.scale == scale) & (means.arm == 'A4')
                       & (means.horizon == h) & (means.model == 'HYBRID')]
            if a0.empty or a4.empty:
                continue
            v0 = 100 * a0.iloc[0].profit_accuracy_mean
            v4 = 100 * a4.iloc[0].profit_accuracy_mean
            pr = PAPER_REPORTED[h]['HYBRID'][0]
            L.append(f"| {scale} | {h}-day | {pr:.2f}% | {v4:.2f}% | {v0:.2f}% "
                     f"| {pr - v4:+.1f} pp |")
    L.append("")
    L.append("So the three selection choices under study move the hybrid by a "
             "few percentage points, while the distance between our as-published "
             "arm and the published figure is an order of magnitude larger. "
             "**The leaks do not account for the reported improvement.** Stated "
             "carefully, this replication establishes:\n")
    L.append("- Leak 3 (tuning the hybrid's abstention rule on the test block) "
             "is the only choice with a consistently non-zero effect here, worth "
             "roughly +6 pp at the 3- and 5-day horizons on the paper-scale "
             "window, with intervals excluding zero. Leak 2 (iteration count on "
             "test) is worth ~+1.5 to +2 pp for the single models at 1 day. "
             "Leak 1 (threshold scope) is not separable from zero anywhere — "
             "consistent with the fidelity table above, where widening the "
             "threshold's scope moves τ by only a few 1e-5.")
    L.append("- None of that explains a hybrid at 77-84%. Our hybrid sits near "
             "50% on every arm, at every horizon, on both scales.\n")
    L.append("What this does **not** establish: that the paper is wrong. We "
             "reproduce one reading of an underspecified method (Section 4 lists "
             "16 places where a reading had to be chosen) on a different vendor's "
             "EUR/USD series, with a different LSTM configuration, and with "
             "macro inputs mapped onto FRED series rather than the paper's own "
             "sources. Any of those could carry the difference. The honest "
             "summary is that **the reported hybrid improvement did not "
             "reproduce here, and the three selection leaks we can measure are "
             "not large enough to be its explanation** — which leaves it "
             "unexplained rather than explained away.\n")
    # Abstention curve — measured, not asserted (abstention_curve.csv).
    L.append("### What abstention actually does to profit_accuracy\n")
    L.append("profit_accuracy is a ratio whose denominator the model chooses, so "
             "it is worth measuring directly what happens as the hybrid's "
             "confidence floor rises and the transaction count collapses. "
             "`abstention_curve.csv` does that at the 1-day horizon with the "
             f"{MIN_TXN_FRACTION:.0%} selection floor REMOVED, holding the "
             "iteration counts at A0's validation-chosen values so that only the "
             "confidence floor varies (mean over "
             f"{meta.get('n_seeds', len(SEEDS))} seeds):\n")
    if os.path.exists(ABSTENTION_CSV):
        curve = pd.read_csv(ABSTENTION_CSV)
        for scale in scales:
            sub = curve[curve.scale == scale]
            if sub.empty:
                continue
            agg = sub.groupby('min_confidence').agg(
                pa=('profit_accuracy', 'mean'), n=('n_transactions', 'mean'),
                rows=('n_test_rows', 'first')).reset_index()
            L.append(f"**`{scale}`**\n")
            L.append("| confidence floor | profit_accuracy | transactions | of rows |")
            L.append("|---|---|---|---|")
            for _, r in agg.iterrows():
                pa = "n/a (no trades)" if not np.isfinite(r.pa) else f"{100 * r.pa:.1f}%"
                L.append(f"| {r.min_confidence:.2f} | {pa} | {r.n:.1f} | "
                         f"{int(r.rows)} |")
            L.append("")
    L.append("The curve does **not** show abstention buying accuracy. On the "
             "paper-scale window profit_accuracy stays in a 48-51% band while "
             "transactions fall from 158 to single digits, and the individual "
             "cells become erratic rather than better (31.8% on ~26 trades, then "
             "51.3% on ~12). On the euro-era window the only cells that reach "
             "100% are the ones where the denominator has collapsed below one "
             "trade per seed.\n")
    L.append("That reframes the paper's most striking cells. Table 7's 100.00% "
             "on 8 transactions, Table 11's 100.00% on 2, and Table 15's `Nan` "
             "on 0 are what a vanishing denominator looks like, not evidence "
             "that the filter is finding better trades — and averaging them into "
             "a headline gives them the same weight as a full-coverage cell. "
             "Our arms impose the "
             f"{MIN_TXN_FRACTION:.0%} floor (UNDERSPECIFIED[#8]) precisely to "
             "keep *selection* out of that regime; the curve above is what the "
             "floor is protecting against.\n")

    # 4. Underspecification ledger ----------------------------------------
    L.append("## 4. Where the paper is underspecified, and what we chose\n")
    L.append("This list is part of the result. A replication of an underspecified "
             "method is a replication of one reading of it.\n")
    for i, (area, gap, choice) in enumerate(UNDERSPECIFIED, start=1):
        L.append(f"**{i}. {area}**")
        L.append(f"- *Paper:* {gap}")
        L.append(f"- *Our choice:* {choice}\n")

    # 5. Verdict -----------------------------------------------------------
    L.append("## 5. Which gaps are distinguishable from zero\n")
    any_sig = gaps[gaps.distinguishable_from_zero == 'yes']
    if any_sig.empty:
        L.append("**None.** Every arm-to-arm gap's 95% moving-block interval covers "
                 "zero, at every horizon, for every model, on every scale tested. "
                 "At these sample sizes the three selection choices are not "
                 "separable from run-to-run and row-sampling noise. That is the "
                 "finding: this design is **underpowered** to attribute the paper's "
                 "reported improvement to selection, and equally underpowered to "
                 "clear it. The point estimates below are reported for completeness "
                 "and should not be read as effects.\n")
    else:
        L.append(f"{len(any_sig)} of {len(gaps)} gaps have a 95% interval excluding "
                 "zero:\n")
        L.append("| Scale | Horizon | Model | gap | Δ | 95% CI |")
        L.append("|---|---|---|---|---|---|")
        for _, r in any_sig.iterrows():
            L.append(f"| {r.scale} | {r.horizon}-day | {r.model} | {r.arm}−A0 | "
                     f"{100 * r.delta:+.2f} pp | [{100 * r.ci_low:+.2f}, "
                     f"{100 * r.ci_high:+.2f}] |")
        L.append("")
        L.append(f"The remaining {len(gaps) - len(any_sig)} gaps have intervals "
                 "covering zero and are **not** distinguishable from noise.\n")

    L.append("### Power, stated plainly\n")
    L.append("The primary scale has 243 test rows and, after abstention, often fewer "
             "than 100 transactions. A 95% interval on a proportion from ~100 trials "
             "is about ±10 percentage points before any block-dependence widening. "
             "Any gap smaller than that is unresolvable here **by construction** — no "
             "amount of seeding fixes it, because the limit is the number of scoring "
             "rows the paper's design provides, not the number of runs. The secondary "
             "scale exists precisely to check whether a gap that is invisible at n=243 "
             "becomes visible at euro-era length.\n")
    L.append("### Files\n")
    L.append("- `arms.csv` — one row per (scale, arm, horizon, model, seed), with the "
             "selection actually made (iteration counts, hybrid parameters, threshold).")
    L.append("- `summary.csv` — `record=arm_mean` rows (seed means/sds) and "
             "`record=gap` rows (bootstrap CIs per leak gap).")
    L.append("- `abstention_curve.csv` — profit_accuracy vs transaction count as "
             "the hybrid's confidence floor rises, with the selection floor "
             "removed (`--abstention-curve`).")
    L.append("- `panel_cache.csv`, `equity_indices.csv` — the assembled input panel, "
             "cached so the run is reproducible offline.")
    L.append("- `run_meta.json` — run parameters.\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--scale', choices=['primary', 'secondary', 'both'], default='both')
    ap.add_argument('--seeds', type=int, default=len(SEEDS),
                    help='number of seeds from the 42.. sequence (>=5 required by the brief)')
    ap.add_argument('--refresh-panel', action='store_true',
                    help='rebuild the cached input panel from FRED caches + yfinance')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 4),
                    help='parallel fit processes (1 = inline, easiest to debug)')
    ap.add_argument('--abstention-curve', action='store_true',
                    help='measure profit_accuracy vs transaction count with the '
                         'selection floor removed (writes abstention_curve.csv)')
    ap.add_argument('--report-only', action='store_true',
                    help='rewrite REPLICATION.md from the existing arms.csv / '
                         'summary.csv / run_meta.json without refitting anything')
    args = ap.parse_args(argv)

    seeds = tuple(SEEDS[:args.seeds]) if args.seeds <= len(SEEDS) else \
        tuple(range(42, 42 + args.seeds))
    scales = ('primary', 'secondary') if args.scale == 'both' else (args.scale,)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.abstention_curve:
        panel = build_panel()
        curve = pd.concat([abstention_curve(s, panel, horizon=1, seeds=seeds,
                                            workers=args.workers) for s in scales])
        curve.to_csv(ABSTENTION_CSV, index=False)
        print(f"wrote {ABSTENTION_CSV} ({len(curve)} rows)")
        return

    if args.report_only:
        arms = pd.read_csv(ARMS_CSV)
        summary = pd.read_csv(SUMMARY_CSV)
        with open(RUN_META_JSON, encoding='utf-8') as f:
            meta = json.load(f)
        write_report(arms, summary, tuple(meta['scales']), meta)
        print(f"rewrote {REPORT_MD} from {ARMS_CSV} ({len(arms)} rows)")
        return

    print("Building panel ...")
    panel = build_panel(refresh=args.refresh_panel)
    print(f"  panel: {len(panel)} weekday rows, {panel.index.min().date()} .. "
          f"{panel.index.max().date()}")

    t0 = time.time()
    all_rows, rowvecs_by_scale = [], {}
    for scale in scales:
        print(f"\n=== scale: {scale} ===")
        rows, rowvecs = run_scale(scale, panel, seeds=seeds, workers=args.workers)
        all_rows.extend(rows)
        rowvecs_by_scale[scale] = rowvecs

    arms = pd.DataFrame(all_rows)
    arms.to_csv(ARMS_CSV, index=False)
    print(f"\nwrote {ARMS_CSV} ({len(arms)} rows)")

    summary = summarise(arms, rowvecs_by_scale)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"wrote {SUMMARY_CSV} ({len(summary)} rows)")

    meta = {
        'generated': pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        'seeds': list(seeds), 'n_seeds': len(seeds), 'scales': list(scales),
        'iteration_grid': list(ITERATION_GRID), 'horizons': list(HORIZONS),
        'n_fits': len(scales) * len(HORIZONS) * len(MODEL_TYPES) * 2 * len(seeds),
        'time_steps': TIME_STEPS, 'lstm_units': LSTM_UNITS,
        'min_txn_fraction': MIN_TXN_FRACTION, 'workers': args.workers,
        'keras_backend': os.environ.get('KERAS_BACKEND'),
        'lstm_batch': LSTM_BATCH, 'scale_clip': SCALE_CLIP,
        'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES,
                      'block_len': BOOTSTRAP_BLOCK_LEN, 'alpha': BOOTSTRAP_ALPHA},
        'scale_spans': {s: (str(slice_scale(panel, s).index.min().date()),
                            str(slice_scale(panel, s).index.max().date()),
                            len(slice_scale(panel, s))) for s in scales},
        'runtime_seconds': round(time.time() - t0, 1),
    }
    with open(RUN_META_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    write_report(arms, summary, scales, meta)
    print(f"wrote {REPORT_MD}")
    print(f"total {meta['runtime_seconds']:.0f}s")


if __name__ == '__main__':
    main()
