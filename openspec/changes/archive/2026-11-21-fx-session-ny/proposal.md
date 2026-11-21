# Proposal

## Why

The fx-fibonacci family ended in four DROPs (2026-11-21). Price geometry on every H1 bar
carried no information, and 77 % of its rows were the neutral class. The owner's next
hypothesis changes the *information and the moment*, not the model.

The new family makes one forecast per day, after the US data release and the New York
open. It asks whether the move to the end of the owner's trading window (23:00 Sofia) is
up or down. It is tested first with LightGBM, and a neural network is trained only if
LightGBM earns a KEEP.

## What Changes

- **A new research family, `fx_session`, for EURUSD and GBPUSD.** It lives in its own
  package `src/fx_session/` with its own config, registry and results directory. AUDUSD
  is loaded only as an input to the dollar factor.
- **H1 history from 2012, with the bar spread.** The family has its own MT5 H1 caches
  from 2012-01-01. Each bar keeps its spread, so the target can use the mid price and be
  checked against the bid.
  - The owner chose 2010, but changed the start after the first refresh. The broker's
    GBPUSD bars from December 2010 to November 2011 are one hour off the other pairs.
  - Evaluation still starts in 2013, so the out-of-sample size is unchanged.
  - Days in weeks whose weekly open is off the Berlin clock are excluded.
  - GBPUSD March 2014, which holds one bar per day, is a declared broker gap.
- **One example per trading day, on a fixed Sofia/Berlin clock.**
  - Entry is the close of the H1 bar labelled 14:00 Berlin (16:00 Sofia).
  - Exit is the close of the bar labelled 21:00 Berlin (23:00 Sofia).
  - The clock stays the same in the US/EU DST-mismatch weeks; a feature flags those
    days.
- **Target A: the sign of the mid-price return from entry to exit.** It is binary, uses
  no barrier parameter and no class weights, and drops ties.
- **Pre-session features.** About 20 scale-free columns, all known at entry:
  - the Asia, London and data-hour moves;
  - the London range and the price's position in it;
  - yesterday's bar, and whether today has broken yesterday's high or low;
  - a synthetic dollar factor from EURUSD, GBPUSD and AUDUSD, plus the pair's residual
    against it;
  - the day of week, the NFP and FOMC days, and the DST-mismatch flag.
- **A purged walk-forward from 2013-01-01 in 6-month folds.** It is the only arbiter:
  there is no forward phase.
- **Four registered cells at alpha = 0.05/4, each scored once.**
  - Per pair, LightGBM against the class prior on log-loss.
  - Per pair, a CUDA MLP against LightGBM. These two cells are trained and scored only
    if that pair's LightGBM cell is KEEP. Untriggered alpha is never reassigned.
- **Gross of spread and slippage, by owner decision.** The fixed sign rule's P&L is
  descriptive only, and no verdict is a trading result.

## Capabilities

### New Capabilities
- `fx-session/market-data`:
  - the family's MT5 H1 caches with spread, history from 2012, declared broker gaps, and
    clock verification for the whole history and for each week;
  - the mid price;
  - the trading-day calendar with the fixed entry and exit bars.
- `fx-session/session-features`:
  - the fixed pre-session feature list and the dollar factor;
  - the calendar flags (NFP, FOMC, DST mismatch);
  - the rule that every feature is known at the entry bar's close.
- `fx-session/session-target`: the binary mid-price sign target from entry to exit,
  eligibility of days, the tie rule and the bid-versus-mid parity check.
- `fx-session/evaluation`:
  - the folds, the LightGBM and conditional MLP models, and the class-prior reference;
  - the four-cell registry, the one-shot scoring and the config guard;
  - the power report and the descriptive gross report.

### Modified Capabilities
(none)

## Impact

- **New code and results.** The code lives in `src/fx_session/` and the tests in
  `tests/test_fx_session_*.py`. Results go to `results/fx_session/`, and frozen
  artifacts to the gitignored `research_models/fx_session/`.
- **Read-only reuse of fx_fib helpers.** These are MT5 symbol resolution, `verify_clock`,
  D1 aggregation and ATR, the block bootstrap, metrics and the MLP builder. None of
  them changes, and the fx_fib digests are pinned by a test.
- **Read-only inputs.** `results/fomc_dates.csv` is read and never changed. The MT5
  terminal is read only.
- **Nothing else changes.** `models/`, `config.json`, the served API and every other
  family's registry stay untouched. The family never touches the `0.05/9` bar.
- **Docs.** A new `ARCHITECTURE_DOCS.md` section and a CLAUDE.md invariant entry.
- **Dependencies.** None new: torch (CUDA), lightgbm and sklearn are already in `.venv`.
