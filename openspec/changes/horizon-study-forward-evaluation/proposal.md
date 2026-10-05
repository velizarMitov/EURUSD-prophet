# Proposal

## Why

The owner trades real money and wants to know **at which forecast horizon, if any,
the project's models know the direction of EUR/USD**. They also want backtesting
rebuilt on modern walk-forward methods with a random-walk benchmark. The repository
has answered the horizon question for exactly two horizons, and never in the form
that matters for trading:

- **Next day (24 h): no edge.** A 24-window walk-forward over the fixed daily configs
  (`results/walk_forward_validation_summary.csv`) gives mean accuracy 0.500 and
  AUC 0.519. Multi-day 24-bar reversion (`h1_multiday_hypothesis_log.csv`): DROP.
- **Next hour (1 h): a real predictive edge that cannot pay the spread.** H_dir.1 GBM
  is the only direction model with a confirmed edge: 52.96 % on its one-shot test
  block, CI excludes zero, replicated on AUDUSD and CHFUSD. That program deliberately
  excluded costs. A design calculation on the EURUSD train slice (no model scored)
  puts the cost breakeven at **53.32 %** at the measured 0.5-pip ActivTrades spread,
  and at **59.97 %** at the 1.5-pip round trip in `config.json`.
- **2, 4, 6, 12, 48 and 120 h: never tested** with a direction model. The arithmetic
  feasibility scan (`h1_horizon_feasibility.csv`) shows that labels exist there.
- **Below one hour: never tested at all.** The owner trades from 15:30 Europe/Sofia,
  which is 08:30 America/New_York, and asked for the sub-hourly horizons they can
  actually act on. 15-minute history exists on disk and the horizon question has never
  been asked of it.

The breakeven falls toward 50 % as the horizon lengthens, because the cost shrinks
relative to the move: the mean |r| is about 7.5 pips at 1 h, 41 pips at 24 h and
93 pips at 120 h. At the same time, the number of independent observations falls as
1/h. The horizon question is therefore a trade-off curve between **cost** and
**sample size**, and nobody has drawn it yet. Measured inside the owner's session
(15:30–23:00 Europe/Sofia), the short end of that curve reads:

| horizon | mean &#124;move&#124; | accuracy needed to beat the spread | years to a verdict |
|---|---:|---:|---:|
| 15 min | 4.2 pips | 56.02 % | 0.51 |
| 30 min | 5.9 pips | 54.27 % | 1.05 |
| 1 h | 8.3 pips | 53.01 % | 2.10 |
| 2 h | 11.9 pips | 52.10 % | 4.91 — over the cap |

Shorter horizons buy a verdict sooner and demand a much higher accuracy to pay for
themselves. **Finer input data does not shorten the wait**: a one-hour forecast needs
the same number of non-overlapping hours however often it is issued, so 15-minute bars
speed up a decision only by making the *horizon* shorter, not by making the forecasts
more frequent. What finer data can do is make the one-hour forecast more accurate.

Two further facts shape the design:

1. **Every historical arbiter for EUR/USD is spent.** The daily `[80:100%]` and H1
   `[85:100%]` blocks were each read once and are permanently spent. The validation
   slices have been reused across families. The owner has ruled that **the arbiter
   for this program is forward data only.**
2. **No forward data is being collected systematically.** Every forward log is
   written only when someone calls the API. So far that is 63 daily rows in about
   100 trading days, and 89 H1 rows out of about 1,000 bars. No log records which
   price source served the prediction. With a forward-only arbiter, nothing can be
   decided until this changes.

**A null result is an acceptable outcome.** "It does not work at day 1, therefore it
works at some horizon" does not follow. An efficient market can be unforecastable net
of cost at every horizon. This program is built so that answer is cheap to reach, and
so that nobody can avoid it by choosing the horizon after looking.

## What Changes

- **New backtest harness** (development only, no verdicts):
  - Purged and embargoed walk-forward, with the purge and embargo scaled to the
    horizon.
  - Combinatorial Purged Cross-Validation (CPCV), producing a distribution of
    out-of-sample paths.
  - Probability of Backtest Overfitting (PBO, via CSCV).
  - Deflated Sharpe Ratio, charged for every model × horizon tried.
  - Label-uniqueness weights for overlapping targets.
  - A cost model built from the measured per-instrument spreads.
  - Three random-walk nulls: coin-flip direction, zero return (driftless random
    walk, Meese–Rogoff), and random-sign strategies with matched turnover.
  - Statistical tests: Pesaran–Timmermann for direction, Clark–West for nested return
    forecasts against the random walk, and block bootstrap throughout.
- **New horizon study, built as a new standalone system:** the owner chose to build
  something new rather than modify the existing training code. The study implements
  its own **challenger** model for every model type in the project: daily GBM and
  LSTM on the price-only and with-macro feature sets, the H1→daily ensemble,
  TI-LSTM, the H1 next-bar GBM and the volatility multi-task LSTM ensemble. Kronos,
  an external pinned model, is run as-is. Each challenger's configuration is declared
  once before the first fit and kept identical across horizons. Features come from the
  project's existing feature code, imported read-only, so both systems describe the
  market identically. M15-cadence models use {1, 2, 4, 8, 16, 26} bars; H1-cadence
  models use {1, 2, 4, 6, 12, 24, 48, 120} bars; daily-cadence models use {1, 2, 5}
  trading days. The output is a descriptive, cost-net horizon curve per model.
  Volatility models are compared with their strongest known baseline,
  GARCH(1,1) × day-of-week, not plain GARCH. Findings apply to the challengers.
  Promoting any of them into production is a separate, later decision.
- **New M15 session layer**, covering the horizons the owner can trade:
  - Two new session challengers, a gradient-boosting and a sequence model, on
    15-minute bars, with the bar's position in the session, its spread and its tick
    volume available as features.
  - Bars aggregated from the one-minute history, which carries a per-bar spread, and
    every target and cost computed on the **mid** price rather than the bid close.
    The pinned `results/eurusd_m15.csv` has no spread column and is not used for
    targets.
  - Forecasts made only inside the owner's declared session, 15:30 to 23:00
    Europe/Sofia on weekdays, with both the as-of bar and the target bar required to
    fall inside it. The bar labels turned out **not** to be UTC but Europe/Berlin wall
    clock, established from the weekly market boundary; since Sofia and Berlin share
    one daylight-saving rule, the session is a fixed label window needing no conversion.
  - A **bid-versus-mid parity check** on every sub-hourly cell. Measured on history, a
    bid-only series reaches a 72 % one-hour "accuracy" at the New York 17:00 rollover,
    where the spread quintuples — entirely an artefact of the quote, worth nothing at
    any cost level. A cell whose bid and mid results diverge beyond a declared
    tolerance of 1.0 pp is labelled a spread artifact and never reported as skill.
    Inside the declared session the two agree to within 0.25 pp.
- **New forward arbiter:**
  - A scheduled forward logger records every model's prediction at every horizon on
    every bar close, whether or not anyone opens the dashboard.
  - Arbiter prices come from MT5 only, and each row records the price source and the
    bar's own spread.
  - Models are refitted walk-forward during the forward window on a pre-registered
    cadence, and every prediction carries the hash of the model version that made it.
  - A pre-registration is written before the first forward outcome. It contains a
    power table, the time-to-decision per cell, and the verdict rules: KEEP only if
    the lower bound of the accuracy CI exceeds **50 %**, i.e. the model calls the
    direction better than a coin. The owner chose this criterion on 2026-10-02: the
    forecast is used for direction, not for high-frequency trading. The spread result is
    reported beside every verdict but does not decide it.
- **Honest scope statement, made before any data:** with a forward-only arbiter, only
  the short horizons can reach a decision within a few years. Admitting only cells that
  can decide within 3 years gives a direction family of **12 cells** out of 46
  candidates, at alpha = 0.05/12 = 0.004167 and 3,817 forecasts per cell: the two
  session challengers at 15 min, 30 min and 1 h (0.51, 1.05 and 2.10 years), and the
  H1 GBM and Kronos direction at 1, 2 and 4 h (0.62, 1.24 and 2.47 years). The M15
  cells at 2 h and beyond need 4.9–14.7 years, H1 at h ≥ 6 needs 3.7–74, and **every
  daily-cadence cell would need 13–65 years.** Those 34 cells are registered
  `UNDERPOWERED — NO DECISION` by design and spend zero alpha, but are still logged and
  reported descriptively. The family must be fixed before the registration is
  committed, which is why the session layer lands first: one family of 12 is stronger
  than two families of 6.
- **No production model change.** Nothing changes under `models/`, `src/inference.py`,
  `src/paper_trading.py` or `_train_pipeline.py`. No order, broker, sizing or
  stop-loss code is added. Research artifacts live outside `models/`.
- **One additive serving change, at the owner's request (2026-10-05).** `api.py`
  gains a single read-only route, `/forecasts`, and `static/index.html` gains a link
  to it, so the forecast view appears in the program the owner already starts.
  Both are additions only: no existing line is edited or removed, which the
  additive-only contract test enforces. The owner had earlier ruled `api.py` out of
  scope and chose a separate server; they reversed that once the separate server
  proved easy to miss.

## Capabilities

### New Capabilities

- `forecast-evaluation/backtesting`: purged/embargoed walk-forward and CPCV splitting
  for overlapping horizon targets; the cost model; the three random-walk benchmarks;
  the statistical tests (Pesaran–Timmermann, Clark–West, block bootstrap,
  Romano–Wolf step-down); and the overfitting diagnostics (PBO, Deflated Sharpe).
  It is model-agnostic and owns how any forecast is scored on historical data.
- `forecast-evaluation/horizon-study`: the challenger model for each model type, how
  each is fitted per horizon with its configuration declared once and frozen, the
  horizon grid, the cost breakeven per horizon, and the descriptive horizon-curve
  report. It owns what is evaluated, and on which grid. It also owns the sub-hourly
  price definition (mid from the per-bar spread), the session window, and the
  bid-versus-mid parity check.
- `forecast-evaluation/forward-arbiter`: scheduled forward prediction logging,
  price-source and spread provenance, the walk-forward refit cadence and version
  hashing during the forward window, the pre-registration with its power table and
  family accounting, and the verdict rules. It owns when and how a horizon claim is
  decided.

### Modified Capabilities

None. `openspec/specs/` is empty, so no existing capability's requirements change.

## Impact

- **New code:** a standalone `src/forecast_eval/` package (splitting, costs,
  benchmarks, tests, diagnostics, horizon targets, the challenger models, the forward
  logger, and the pre-registration and power arithmetic), with tests under `tests/`.
  It imports the existing feature code read-only. No checksum-pinned file is modified:
  `_train_pipeline.py`, `src/features.py`, `config.json`, `src/h1_direction_model.py`,
  `src/volatility.py` and `src/inference.py` stay byte-identical, and no fixture is
  re-baselined.
- **New research artifacts:** per-horizon fitted models outside `models/`. They are
  not served and not checksum-pinned. A tracked SHA-256 manifest makes every forward
  prediction traceable to the exact weights that produced it.
- **New results:** `results/horizon_study/` holds the descriptive curves and the
  CPCV/PBO/DSR output. `results/forward_eval/` holds the forward predictions, the
  pre-registration, the power table and the hypothesis registry for the new families.
- **Operations:** the forward logger runs unattended (Windows Task Scheduler) and needs
  the MT5 terminal running and logged in. It attaches read-only to whatever account
  the terminal holds. Today that is **real account 1007437 on ActivTradesEU-Server**.
  No order call exists anywhere in the change.
- **Compute:** fitting every challenger per horizon is the expensive step. That is
  roughly 6 daily-cadence families × 3 horizons plus 1 trainable H1-cadence family ×
  8 horizons, dominated by the LSTMs. The session layer adds 2 families × 6 horizons on
  up to 60,163 eligible M15 rows, scored twice (mid and bid) for the parity check — the
  largest row count in the study, and the session LSTM becomes the most expensive single
  cell. Kronos is inference-only. Refits repeat monthly during the forward window.
- **Sub-hourly data:** the session layer reads `results/curl/raw/EURUSD_M1.parquet`
  (2,980,060 bars, 2018-08-08 → 2026-08-07), which is excluded from git by `DATA.md` §7
  and regenerable with `src/curl_mt5_fetch.py`. Its SHA-256 and row count go into the
  study record so a rebuilt copy can be proven identical. `pyarrow` is required to read
  it and is not a declared dependency, so a missing install must fail with a message
  naming it. The pinned `results/eurusd_m15.csv` is neither read for targets nor
  written. History is 8 years here against 11 for H1, because the spread column only
  exists that far back.
- **Methodology:** two new self-contained hypothesis families, one for direction and
  one for volatility. Neither touches the standing `0.05/9` feature bar. Existing
  hypothesis logs, fixtures and spent blocks are not modified.
- **Time:** the first possible verdict comes about 6 months after the forward logger
  starts scoring — the 15-minute session cells at 0.49 years and the 1-hour H1 cells at
  0.62 — assuming every bar close in the session is logged. Downtime lengthens this in
  proportion. Of 46 candidate cells, 12 can ever be adjudicated; the other 34 will be
  reported descriptively for the life of the program.
