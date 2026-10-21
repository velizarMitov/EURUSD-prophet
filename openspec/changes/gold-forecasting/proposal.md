# Proposal

## Why

The owner wants machine-learning and neural-network models that forecast the
**price and direction of gold**, built from the broker's MT5 bars and examined the same
way the EUR/USD models are: chronological splits, a validation arbiter, Bonferroni bars,
a one-shot test block, and forward data as the final judge. The project has never
modelled gold.

The live terminal (ActivTrades, symbol `GOLD`) dictates what is possible. Measured
2026-10-21:

- **D1: 3,465 bars, from 2013-08-08.** That is 40 % of the EUR/USD daily row set. A
  70/10/20 split leaves about 347 validation days. A 3 pp direction edge needs about
  3,363 scored trades at α = 0.05/6, which a daily cadence (~260 per year) can never
  supply. **Daily direction is underpowered by design.**
- **H1: 70,507 bars over the same span**, about 6,000 scoreable bars per year. This is
  where a direction claim can be decided. It is also the cadence of the only
  direction edge this project has confirmed: the H1 GBM, which replicated on AUDUSD and
  CHFUSD.
- **Price level tripled, $1,312 to $4,131.** The close at the 70 % mark is $1,644 and
  the training maximum is $2,062. EUR/USD's production features include raw price
  levels (`open`…`close`, `SMA_*`, `ATR_14`), and on gold the test block would sit far
  outside every value the scaler saw. **Gold features must be scale-free.**
- Volatility clusters strongly in gold (mean |r| 0.70 %, sd 0.99 % daily). Volatility is
  the only family where this project has a confirmed neural edge (LSTM ensemble vs.
  GARCH). It is the most likely gold KEEP.

Published gold work rarely meets this standard. It often predicts price *levels*, where a
"tomorrow ≈ today" model scores R² ≈ 0.99 with no skill. It reports error metrics without
a random-walk benchmark, or tunes on the test set. Where honest benchmarks are used,
daily direction sits near 50 % and DM tests against a random walk are not significant.
This change builds gold models that are judged honestly from the start.

## What Changes

- New research package `src/gold/` with its own config file. It reuses the pure parts of
  `src/forecast_eval/` (splits, targets, stats, benchmarks, power and overfit maths)
  read-only, and the scale-free H1 feature set from `src/pooled_h1_model.py`. It never
  modifies `forecast_eval`, whose EUR/USD forward arbiter is mid-pre-registration.
- Gold data acquisition from MT5 only, **OHLC only**. `tick_volume`, `real_volume` and
  `spread` are dropped at the source and never stored, so no model can consume them. A
  clock guard refuses a run when the broker's weekly opens stop matching the measured
  `Europe/Berlin` pattern.
- Targets: next-bar **log return** (percent), its **direction**, and **|return|
  volatility**. "Price" is derived as `close × exp(r̂)`. The price level is never a
  model target or feature.
- Challengers at D1 and H1. Every one is scored against the baseline it must beat:
  - **ML:** XGBoost classifier and pseudo-Huber regressor.
  - **NN:** multi-task LSTM (return, direction and volatility heads) as a 5-seed
    all-or-nothing ensemble.
  - **Baselines:** zero-return / train-majority direction, and train-only GARCH(1,1)
    (D1) and GARCH with intraday seasonality (H1) for volatility.
- **Three gold hypothesis families** (direction, return, volatility), each with its own
  Bonferroni α and registry under `results/gold/`. A family's composition is fixed from
  power arithmetic, and underpowered cells are registered `UNDERPOWERED — NO DECISION`.
  None of them touches the `0.05/9` feature bar or any EUR/USD family.
- A pre-registration is committed before any validation number exists. Validation
  `[70:80]` is the only arbiter. The test block `[80:100]` is run once, behind a guard
  that refuses a second run.
- Forward logger `EURUSDProphet-GoldLogger` (hourly) records frozen-model forecasts as
  bars close and settles them. A forward pre-registration is committed before any
  forward row is scored.
- **Scoring is gross of cost (owner decision).** Spread is used nowhere, so verdicts
  measure statistical skill only. No pips, P&L or cost-net ledger are produced, and no
  trading claim can follow from this change. To keep bid-only prices from manufacturing
  skill around the daily break, an H1 forecast is eligible only when its as-of bar and
  target bar are contiguous and neither borders a market gap.
- Small read-only view: `GET /api/gold` and `/gold` show the newest logged gold forecast
  per cell: direction with probability, expected return and implied price, and expected
  move. Each verdict is shown as recorded, labelled "research — no proven edge" until a
  forward KEEP. The view runs no model and writes nothing.

## Capabilities

### New Capabilities
- `gold-forecasting/study`: gold data acquisition, clock guard, scale-free features,
  targets, challengers, the validation arbiter, the three registered families and the
  one-shot test report.
- `gold-forecasting/forward-arbiter`: hourly forward logging of frozen gold models,
  gap and failure records, settlement, the forward pre-registration, and gross-of-cost
  forward verdicts.
- `gold-forecasting/gold-view`: the read-only `/gold` page and `/api/gold` endpoint, and
  what they may present as a current forecast.

### Modified Capabilities
- (none; `forecast-view` and `app-launcher` are unchanged)

## Impact

- **New code:** `src/gold/` (data, clock, features, challengers, study, prereg,
  forward_logger, summary), `src/gold/gold_config.json`, `static/gold.html`,
  `scripts/gold/install_task.ps1` / `uninstall_task.ps1`, and `tests/test_gold_*.py`.
- **Touched:** `api.py` gains two read-only routes and `/advanced` gains a link.
  `ARCHITECTURE_DOCS.md` and `CLAUDE.md` gain a gold section.
- **Outputs:** `results/gold/` (OHLC caches, registries, pre-registrations, reports,
  forward CSVs, artifact manifest). Model weights go to gitignored `research_models/gold/`.
- **Untouched by contract:** `models/`, `config.json`, `_train_pipeline.py`,
  `src/inference.py`, `src/features.py`, `src/paper_trading.py`, every `forecast_eval`
  file and the EUR/USD forward logger. The pinned protected-set fixtures must stay
  green.
- **New runtime dependency:** none. `arch` (GARCH) and XGBoost/TensorFlow are already in
  use. The forward logger adds one Windows scheduled task, installed by an explicit
  script.
- **Out of scope:**
  - macro features for gold (real yields `DFII10`, USD index, VIX, breakevens), which
    become registered ADD-tests in a follow-up once the price-only baseline is on record;
  - M15 (history starts only in 2022-08, and bid-only sub-hourly scoring is the known
    spread-artifact trap);
  - Kronos or other foundation-model challengers;
  - any serving under `models/`, paper-trading P&L, or execution.
