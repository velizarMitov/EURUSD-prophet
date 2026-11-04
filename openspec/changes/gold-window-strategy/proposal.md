# Proposal

## Why

The owner asked for a trading strategy built on the models that catches the period when
gold makes its biggest moves. The strategy must work inside their window, 16:00–23:00
Sofia.

The descriptive statistics run on 2026-11-04 (2015 to November 2026, H1, log % per year)
show where gold's gain comes from:

- **The gain is in the Asian session (01:00–09:00 Sofia):** +15.4 / +6.4 / +13.3 % a
  year in the three spans 2015–18, 2019 to October 2022, and November 2022–26.
- **The owner's window has no drift:** −3.6 / +1.1 / −0.4 % a year. It does have the
  largest moves of the day: the mean |H1 move| at 16:00–19:00 Sofia is about twice the
  Asian hours.
- **The gain is concentrated in a few days.** Dropping the best 50 days of 12 years turns
  +120 log % negative. The big days are up only 49 % of the time, but their size
  clusters.

So inside the window, the only lever is **when a big move comes, and which way it breaks
once it starts**. That is what the size models and conditional rules can try to exploit.

Every year from 2013 to 2026 in MT5 and in the repo has now been looked at. A backtest on
those years cannot confirm a rule. Four direction strategies were already backtested and
failed on 2026-11-03/04 (`results/gold/RESULTS.md`). An honest backtest needs years nobody
has looked at.

## What Changes

- **A new history source.** Free Dukascopy XAU/USD M1 bid and ask candles are
  downloaded into gitignored `research_data/dukascopy/`, with digests in
  `results/gold/window/data_manifest.csv`. The data is:
  - converted to the repo's Berlin label clock;
  - checked against MT5 on the overlap 2013-08-08 to 2014-12-31.

  Only timestamps of the earlier years are read before registration.
- **A reproducible descriptive report**, `results/gold/window/DESCRIPTIVE.md`. It covers
  return and |move| by Sofia hour, session, weekday and month, best-day concentration and
  size clustering, on the MT5 caches. It is labelled descriptive, never evidence.
- **Walk-forward model predictions** for the unseen years. Each model is refit once a year
  on past data only, with the frozen `gold_config.json` settings:
  - the gold `h1_lstm` and `d1_lstm` multitask ensembles (5 seeds);
  - a D1 GARCH(1,1).

  The artifacts go into gitignored `research_models/gold/window/` with digests. They are
  never a registered model or a shadow version, and they are never promoted.
- **A new research family, `gold_window`, at alpha = 0.05 / 4 = 0.0125.** Every cell
  enters and exits inside 15:30–23:00 Sofia and never holds overnight, so it pays no swap.

  | cell | rule |
  |---|---|
  | `h1_gated_direction` | trades the `h1_lstm` call for one hour, only when its own size forecast is in the top 20 % of its trailing year |
  | `continuation` | holds the sign of the 23:00 → 16:00 move from 16:00 to 23:00, only when GARCH forecasts an above-median day |
  | `us_open_breakout` | trades the first break of the 15:30–16:30 range until 20:00, exits at 23:00, under the same GARCH gate |
  | `d1_gated_direction` | holds the `d1_lstm` call from 16:00 to 23:00, only when its own size forecast is above its trailing median |

- **Net of cost, the first gold family to be.** Trades fill at Dukascopy's ask and bid.
  The verdict uses the larger of that cost and the broker's window median (0.0062 % of
  price per round trip).
- **Scored once.**
  - The confirmation span is the walk-forward years through 2013-07-31.
  - Each cell's verdict is CLEARED only if the lower bound of its two-sided (1 − 0.0125)
    block-bootstrap interval of mean daily net return is above 0. Otherwise it is DROP,
    or "UNDERPOWERED — NO DECISION" below the trade minimum.
  - The registration is written and committed by the owner before any confirmation-span
    price is read. Code refuses otherwise.
- **After registration**, a descriptive replay of the same four rules on the seen span
  (November 2022 – 2026) uses the existing honest predictions. It is labelled already seen.
- Out of scope:
  - holding overnight or acting outside the window;
  - order, broker, position-sizing and stop-loss code;
  - a paper ledger or live signal for a cleared cell (that is a separate change);
  - any change to the existing gold registries, registrations, `gold_config.json`, the MT5
    indicator, or anything under `models/`.

## Capabilities

### New Capabilities
- `gold-forecasting/archive-history`: obtaining, clock-aligning, verifying and pinning an
  independent pre-2013 gold history, and guarding which parts of it may be read before
  a registration.
- `gold-forecasting/window-strategy`: the owner-window strategy family. Covers the
  descriptive report, walk-forward predictions, the four rules, cost-net simulation, the
  one-shot confirmation and how its results are reported.

### Modified Capabilities
- None. The existing gold capabilities keep their requirements. In particular, the other
  gold families stay gross of cost. This family is a separate registry with its own
  cost rule.

## Impact

- New code:
  - `src/gold/window/`: `dukascopy.py`, `clock.py`, `describe.py`, `walkforward.py`,
    `rules.py`, `simulate.py`, `prereg.py`, `confirm.py`, `replay.py`;
  - `src/gold/window/window_config.json`;
  - tests `tests/test_gold_window_*.py`.
- New results:
  - `results/gold/window/`: `DESCRIPTIVE.md`, `data_manifest.csv`,
    `artifact_manifest.csv`, `PRE_REGISTRATION.md` and `.json`, `confirmation_report.md`,
    `replay_seen.md`;
  - the registry `results/gold/gold_window_hypothesis_log.csv`.
- New gitignored folders: `research_data/dukascopy/` and `research_models/gold/window/`.
- Network: a one-off download from `datafeed.dukascopy.com`, throttled and resumable.
  There are no new Python dependencies, because `lzma` and `struct` are standard library.
- GPU: the walk-forward refits take the shared `research_models/external_tsfm/gpu_job.lock`,
  so the hourly external pass falls back to CPU instead of contending.
- Documentation: `ARCHITECTURE_DOCS.md` gets a gold window-strategy subsection;
  `CLAUDE.md` gold item 7 gets one bullet; `results/gold/RESULTS.md` records the verdicts.
