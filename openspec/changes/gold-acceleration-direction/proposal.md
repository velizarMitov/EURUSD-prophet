# Proposal

## Why

The gold analytics page shows a strong time-of-day acceleration. In the hour from 15:00
Sofia, when New York opens and the 8:30 releases land, the mean move about doubles.
The owner wants to know whether acceleration helps the models call the next hour's
direction.

The current H1 models (h1_gbm, h1_lstm) see the hour of day, but not whether today is
accelerating more than usual. That is new information. Every gold test block is already
spent, so this can only be answered as a descriptive walk-forward backtest, never as a
verdict.

## What Changes

- **New acceleration features.** Each is scale-free, causal (bars up to t only) and
  needs no fit:
  - `accel_1h`: |last hour's move| − |the previous hour's move|, in %;
  - `activity_rel`: |last hour's move| / the trailing mean |move| of the same hour of
    day over the previous 60 trading days;
  - `accel_expected_next`: the trailing profile's mean |move| for the next hour minus
    this hour's;
  - `accel_m15`: inside the hour just closed, |last 15-minute move| − the mean |move| of
    its first three 15-minute bars, from MT5 M15. MT5 holds M15 only from 2022-08.
- **Five research models**, all predicting next-hour direction (`p_up`):
  - `h1_gbm_accel` and `h1_lstm_accel`: the registered h1_gbm and h1_lstm configurations
    with the three H1 acceleration features added to their 15 features. These are
    shadow copies built from `src.gold.challengers.make` with a column override. They
    never replace, register or promote a model;
  - `h1_accel`: a new GBM on the acceleration features plus hour and weekday encoding;
  - `h1_accel_m15` and `h1_gbm_accel_m15`: the same with `accel_m15`, predicting only
    the years that M15 covers (2024 on).
- **Like-for-like comparisons.** The same configurations without acceleration
  (`h1_gbm`, `h1_lstm`), a coin, and the training-majority class.
- **A yearly expanding walk-forward.** It follows `gold-window-strategy`: refit on the
  past only, predict the year, with a per-year checkpoint, and failures recorded, never
  patched.
- **One descriptive report**, `results/gold/accel/backtest.md`:
  - hit rate, AUC and Brier score, for all hours and for the owner's window
    (15:30–23:00 Sofia);
  - each model with acceleration against its own model without it, as a paired
    block-bootstrap interval;
  - per year;
  - one fixed net-of-cost check: hold the sign of p_up − 0.5 for one hour inside the
    window, paying the real spread.
- **Descriptive only.** No verdict, registry row, pre-registration or forward logger.
  Every year has been seen.

## Capabilities

### New Capabilities
- `gold-forecasting/acceleration-direction`: acceleration features for gold, the shadow
  and new direction models that use them, their walk-forward backtest, and the
  descriptive report.

### Modified Capabilities
None. The registered gold models, registries, loggers and pages are unchanged.

## Impact

- **New code:**
  - package `src/gold_accel/` (features, walkforward, backtest) with
    `accel_config.json`;
  - tests `tests/test_gold_accel_*.py`.
- **Writes only** to gitignored `research_models/gold/accel/` and to
  `results/gold/accel/` (`backtest.md`, `artifact_manifest.csv`).
- **Read-only inputs:**
  - `results/gold/gold_h1.csv`;
  - the M15 cache `research_models/gold/analytics/bars_M15.csv.gz`;
  - the H1 spread cache of `gold-window-strategy` (as a cost only);
  - `src.gold.challengers`, `src.gold.targets` and `src.gold.load_config`.
- **Unchanged:** `src/gold`, `gold_config.json`, `models/`, `config.json`, `api.py`, the
  loggers and every registry.
- **Compute:** CPU only. h1_lstm_accel refits five seeds per year, about 1 hour in all.
  The baseline h1_lstm predictions of `gold-window-strategy` are reused when their
  configuration digest matches.
- **Docs:** one paragraph in `ARCHITECTURE_DOCS.md` §3.12, one bullet in `CLAUDE.md`, and
  the result appended to `results/gold/RESULTS.md`.
