# Proposal

## Why

The direction of the New York session cannot be predicted. fx-session-ny tested the
direction from 16:00 to 23:00 Sofia on 2026-11-21: both LightGBM cells DROP, both MLP cells
NOT TRIGGERED.

Its size is another matter. Volatility clusters, and in this repo it is the only target
with a confirmed edge: the volatility family's LSTM beats GARCH on daily data.

A reliable forecast of the session's range has practical uses:
- **A session filter.** Trade breakouts only in "hot" sessions, and skip "dead" ones where
  the spread eats the result.
- **An input to sizing.** It can scale the position.

The honest question is not whether a range forecast beats a constant; it does, almost
trivially. The question is whether a model with the pre-session information of the day
beats a HAR baseline built on the range's own history.

## What Changes

- **A new research family, `fx_nyvol`, for EURUSD and GBPUSD.** It has its own package
  `src/fx_nyvol/`, config, registry and results directory. AUDUSD enters only the
  cross-pair volatility feature.
- **Data and clock are fx-session-ny's, read-only.**
  - The H1 caches from 2012-01-01 on the Berlin label clock.
  - Weeks with an unverified clock are excluded, as are incomplete sessions. GBPUSD
    2014-03 is the declared gap.
  - New: a bar whose spread exceeds 50 pips voids that day's mid prices. This is the
    corrupt AUDUSD spread of 2014-01-07, found by the fx-bb-reversion probe on
    2026-11-21.
- **The target is the New York session range.**
  - The bars labelled 15:00 to 21:00 Berlin, inclusive, are 16:00–23:00 Sofia, before the
    rollover.
  - `y = log(ny_range / close14)`. `ny_range` is the mid high minus the mid low of those
    bars, and `close14` is the mid close of the bar labelled 14:00. The ratio keeps the
    target scale-free.
- **Features: 14 columns, all known at the close of the bar labelled 14:00 Berlin
  (16:00 Sofia).**
  - The range's own history, HAR-style: the previous day, and the 5- and 20-day means.
  - Today's Asia, London and data-hour ranges, and the absolute data-hour return.
  - London and data-hour ranges relative to their 20-day averages.
  - A cross-pair pre-session range ratio over EURUSD, GBPUSD and AUDUSD.
  - Day of week, NFP, FOMC, and the DST-mismatch flag.
- **The reference is a HAR baseline.** It is an OLS fit of `y` on the previous day's
  value and its 5- and 20-day means, fitted on each fold's training slice only.
- **Walk-forward from 2013-01-01, expanding, in 6-month folds with a 1-day embargo.**
  Validation is the last 15 % of each training slice.
- **Four cells at alpha = 0.05/4, each scored once.** Each compares the per-day squared
  error, one-sided, with a circular block bootstrap in 5-day blocks.
  1. EURUSD: LightGBM against HAR, always scored.
  2. GBPUSD: LightGBM against HAR, always scored.
  3. EURUSD: a CUDA MLP against LightGBM, only after cell 1 is KEEP. Otherwise it is
     NOT TRIGGERED, and its alpha is never reassigned.
  4. GBPUSD: the same, after cell 2.
- **Descriptive results, not verdicts.**
  - Out-of-sample R², and the MSE ratio to HAR.
  - Spearman rank correlation.
  - Classification of the top 25 % sessions against a causal 60-day Q75.
  - Per-year and per-fold errors.
  - A HAR-X line: the same OLS with all 14 features. It shows how much of any gain a
    linear model already gets.
- **The power check comes before any model.** `labels.md` gives the day counts. It gives
  the detectable mean loss difference in units of its own standard deviation: about
  0.05 σ on about 3 470 out-of-sample days per pair.
- **Gross of cost.** No verdict uses a spread or a trade.

## Capabilities

### New Capabilities
- `fx-ny-volatility/range-target`:
  - the read-only data and the eligibility rules shared with fx-session-ny;
  - the corrupt-spread rule;
  - the mid-price New York range target;
  - the power report.
- `fx-ny-volatility/range-features`: the fixed 14-column feature list, known at the close
  of the bar labelled 14:00 Berlin, and scale-free.
- `fx-ny-volatility/evaluation`:
  - the folds;
  - the HAR reference, LightGBM and the conditional MLP regressors;
  - the four-cell registry, one-shot scoring and the config guard;
  - the descriptive report.

### Modified Capabilities
(none)

## Impact

- **New code and results.** The code lives in `src/fx_nyvol/` and the tests in
  `tests/test_fx_nyvol_*.py`. Results go to `results/fx_nyvol/`, and frozen artifacts to
  the gitignored `research_models/fx_nyvol/`.
- **Read-only inputs.** These are:
  - the fx_session H1 caches, `session_calendar`, the feature helpers and the calendar
    block;
  - `fx_fib.score.test_cell`;
  - the CUDA determinism pattern of fx_session's MLP.

  The regression training code is new, because fx_session's model helpers are binary.
  fx_fib, fx_session and fx_rates keep their configs and results unchanged, and a test pins
  their digests.
- **Separate from the production volatility family.** This family never reads or writes
  `results/volatility_hypothesis_log.csv`, `models/volatility/` or the `0.05/9` bar.
- **Nothing else changes.** `models/`, `config.json`, the served API and every other
  registry stay untouched.
- **Docs.** `ARCHITECTURE_DOCS.md §3.18` and a CLAUDE.md invariant entry.
- **Dependencies.** None new.
