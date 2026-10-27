# Proposal

## Why

The owner trades gold intraday from 13:00 Europe/Sofia and closes everything before
midnight. Before the session they need to know one thing: whether the afternoon is likely
to be quiet or wild. Direction has no proven skill in either gold round. The size of the
move is the only gold forecast with skill, through the hourly LSTM.

An exploration on 2026-10-27 compared the LSTM with GVZ. GVZ is the CBOE Gold ETF
Volatility Index: the volatility the options market prices into GLD over the next 30 days.
FRED publishes it for free as `GVZCLS`. On the 13:00–23:00 Sofia range it ranks afternoons
as well as the model does:

| Rank correlation with the afternoon range | validation (305 days) | test (604 days) |
|---|---|---|
| h1_lstm forecast known at 13:00 | 0.43 | 0.55 |
| GVZ, previous US close | 0.42 | 0.56 |
| GVZ, one trading day older (FRED at 13:00) | 0.40 | 0.54 |
| Correlation of the LSTM with GVZ | 0.74 | 0.83 |

These figures are **descriptive only**. They were computed after looking at both blocks,
were never registered, and are not a claim. They are the reason to show the market's own
number beside the model's. GVZ is not used anywhere in the project today.

## What Changes

- **GVZ is acquired daily for gold.**
  - FRED `GVZCLS` supplies the history and the cache, through the shared FRED fetcher the
    gold macro module already uses.
  - yfinance `^GVZ` tops up only the days FRED has not published yet. At 13:00 Sofia,
    FRED is usually two trading days behind.
  - Only completed US sessions are stored. Every value keeps its date and source.
- **The hourly gold logger refreshes GVZ.**
  - It fetches only while the newest stored value is older than the last completed US
    session.
  - A GVZ failure is recorded and never stops or changes the round-one pass.
- **`GET /api/gold` gains a `gvz` block.** It is built from files only, like the rest of
  the summary. It carries:
  - the value, with its date and source;
  - its percentile against the GVZ history since 2013;
  - the arithmetic implied daily move, GVZ / √252, in percent and in dollars at the
    latest gold close;
  - a stale flag.
- **`/gold` shows a "morning check" card** at the top of the "Now" tab.
  - It shows the GVZ level, the percentile and the implied move, beside the h1_lstm
    next-hour move.
  - It is bilingual (BG/EN), shows the value's date, and says the direction is unknown.
  - It has no direction and no trade, stop or cost wording.
- **A daily observation log, `results/gold/gvz_log.csv`.**
  - On each weekday, the first logger pass at or after 13:00 Sofia records GVZ beside the
    h1_lstm forecast.
  - After 23:00 Sofia, a settlement row in `results/gold/gvz_settlements.csv` records the
    realised 13:00–23:00 range, so the owner can compare the two over time. Both logs are
    append-only, like every gold forward log.
  - This is **not** a registered cell: there is no registry row, alpha, verdict or
    leaderboard entry.
- **Configuration** lives in a new top-level `gvz` block in `src/gold/gold_config_v2.json`.
  The digest-pinned `study` block and `src/gold/gold_config.json` are not touched.

Out of scope:
- GVZ as a model feature. That would be a new hypothesis in the gold volatility family
  and would need its own pre-registration.
- A table of historical quintiles. The owner chose to show only market data and
  arithmetic.
- Any comparison chart on the page.
- Any change to the EUR/USD side.

## Capabilities

### New Capabilities
- `gold-forecasting/gvz-check`: acquiring GVZ (FRED with a yfinance top-up), refreshing it
  from the gold logger, serving it read-only in `/api/gold`, showing it as the morning-check
  card on `/gold`, and the unscored daily GVZ-versus-LSTM observation log.

### Modified Capabilities
- None. The `/gold` view requirements still live in the unarchived `gold-forecasting`
  change, not in `openspec/specs/`. The card's requirements therefore sit in the new
  capability, and every existing gold-view requirement still holds unchanged: the view is
  read-only, it never fetches, and it shows no cost words.

## Impact

- **New code**
  - New module `src/gold/gvz.py`. It acquires and merges GVZ, computes the percentile and
    implied move, and keeps the observation log.
  - Network imports happen only inside its fetch functions, so `summary.py` stays free of
    network and model imports.
- **Changed code**
  - `src/gold/forward_logger.py`: `run_once` gains a guarded `log_gvz` step, and the
    health note mentions it.
  - `src/gold/summary.py`: `build()` adds the `gvz` block from files.
  - `static/gold.html`: the card and its BG/EN strings.
  - `src/gold/gold_config_v2.json`: the new `gvz` block.
- **New data files**
  - `results/gold/macro/gvzcls.csv` (FRED cache).
  - `results/gold/macro/gvz_yfinance.csv` (top-up cache).
  - `results/gold/gvz_log.csv` and `results/gold/gvz_settlements.csv`.
- **Dependencies:** none new. `yfinance` and the FRED fetcher (`src/macro_data.py`) are
  already in use.
- **Untouched**
  - `models/`, `config.json` and `results/forward_eval/`.
  - Every gold registry, registration and manifest.
  - The forward predictions and settlements schemas.
  - The EUR/USD app.
