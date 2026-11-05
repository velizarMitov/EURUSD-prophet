# Proposal

## Why

The owner asked for a trading strategy built on the models that catches the period when
gold makes its biggest moves, inside their window of 16:00–23:00 Sofia.

The descriptive statistics of 2026-11-04 (now `results/gold/window/DESCRIPTIVE.md`) show
where gold's gain comes from:

- **The gain is in the Asian session (01:00–09:00 Sofia).**
- **The owner's window has no drift but has the day's largest moves.**
- **The gain is concentrated in a few days.** Their size clusters, but their sign is a
  coin.

So inside the window the only lever is **when a big move comes, and which way it breaks
once it starts**.

On 2026-11-04 the owner decided against fetching an independent pre-2013 history.
Dukascopy blocks direct download, and its official route needs an AWS account. Every year
available (MT5 from 2013) has already been looked at, and the rules below were chosen
after seeing those statistics. **So this change produces a descriptive backtest, not a
verdict.** It shows how fixed, model-gated rules would have done net of cost, with the
models refit year by year on the past only. It cannot prove any rule has an edge.

## What Changes

- **A reproducible descriptive report**, `results/gold/window/DESCRIPTIVE.md`, built from
  the MT5 caches. (This part is done.)
- **Walk-forward model predictions on the MT5 history.** The gold `h1_lstm` and `d1_lstm`
  multitask ensembles (5 seeds) and a D1 GARCH(1,1) are refit once a year:
  - on an expanding window;
  - on past data only;
  - with the frozen `gold_config.json` settings.

  Artifacts go into gitignored `research_models/gold/window/`, with digests in
  `results/gold/window/artifact_manifest.csv`. They are never a registered model or a
  shadow version, and they are never promoted.
- **Four fixed rules.** Each enters and exits inside 15:30–23:00 Sofia and never holds
  overnight, so it pays no swap.

  | rule | what it does |
  |---|---|
  | `h1_gated_direction` | trades the `h1_lstm` call for one hour, only when its own size forecast is in the top 20 % of its trailing year |
  | `continuation` | holds the sign of the 23:00 → 16:00 move from 16:00 to 23:00, only when GARCH forecasts an above-median day |
  | `us_open_breakout` | trades the first break of the 15:30–16:30 range until 20:00, exits at 23:00, under the same GARCH gate (M15 bars, which MT5 keeps only from 2022-08-12) |
  | `d1_gated_direction` | holds the `d1_lstm` call from 16:00 to 23:00, only when its own size forecast is above its trailing median |

- **Simulated net of cost.**
  - Prices are MT5 bid closes, and a buy pays the bar's MT5 spread.
  - Every round trip is charged at least 0.0062 % of price, the broker's window median on
    2026-11-04.
  - A stress figure is shown at 0.0122 %.
- **One descriptive backtest report**, `results/gold/window/backtest.md`. For each rule
  it shows:
  - the trades, net and stress return, drawdown and hit rate;
  - a per-year table;
  - the same rule with its gate removed;
  - an always-long-in-the-window baseline;
  - a time-shift placebo of the gate;
  - a 95 % block-bootstrap interval.

  Every section is labelled "descriptive: rules chosen after seeing these years, simulated,
  net of the stated costs".
- Out of scope:
  - any verdict, registration or registry row;
  - an overnight hold or action outside the window;
  - order, broker, position-sizing and stop-loss code;
  - a live signal;
  - any change to the existing gold registries, registrations, `gold_config.json`, the MT5
    indicator, or anything under `models/`.

## Capabilities

### New Capabilities
- `gold-forecasting/window-strategy`: the owner-window rules on gold. Covers the
  descriptive statistics, walk-forward predictions, the four rules, cost-net simulation and
  the descriptive backtest report.

### Modified Capabilities
- None. The other gold families stay gross of cost. This one has its own package and
  never writes their files.

## Impact

- New code:
  - `src/gold_window/`: `describe.py` (done), `bars.py`, `walkforward.py`, `rules.py`,
    `simulate.py`, `backtest.py`;
  - `src/gold_window/window_config.json`;
  - tests `tests/test_gold_window_*.py`.
- New results in `results/gold/window/`: `DESCRIPTIVE.md`, `backtest.md`,
  `artifact_manifest.csv`.
- A new gitignored folder, `research_models/gold/window/`.
- MT5 is read only, for M15 bars and the per-bar H1/M15 spread.
- GPU: the walk-forward refits take the shared `research_models/external_tsfm/gpu_job.lock`
  only when TensorFlow sees a GPU. The `.venv` TensorFlow here is CPU-only.
- Documentation:
  - `ARCHITECTURE_DOCS.md` gets a gold window-strategy subsection;
  - `CLAUDE.md` gold item 7 gets one bullet;
  - `results/gold/RESULTS.md` gets a summary, labelled descriptive.
