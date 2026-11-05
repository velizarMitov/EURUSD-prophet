# Proposal

## Why

The owner wants one page that shows how gold behaves: price, sessions, time of day,
volatility, spread, news hours and history. They want it to read the market, not to
forecast it.

The 2026-11-04 research and backtests set its shape:
- The documented gold effects are timing effects:
  - overnight up and day down (Blose et al. 2018, and our `DESCRIPTIVE.md`);
  - volatility at the 8:30 ET releases, which is 15:30 Sofia, the start of the owner's
    window.
- Direction rules failed here.

So the page must make those patterns visible, and the real cost of trading (the spread)
visible hour by hour. It must never suggest a trade.

## What Changes

- **A new page, `/gold-analytics` ("Gold analytics" / "Анализ на златото").** It sits in
  the main navigation after Gold, and is bilingual and themed through the app shell. It
  is read-only. It runs no model, writes nothing outside its own cache, and contains no
  "buy" or "sell". It has these sections:
  - **Overview:**
    - the live XAU/USD price from MT5, the daily change in $ and %, the high, low, open
      and previous close;
    - the current session;
    - historical and intraday volatility;
    - a price chart in 1m, 5m, 15m, 30m, 1h, 4h, D1 and W1.
  - **Sessions:** Asia, London, New York and the London–New York overlap, each defined in
    its own exchange time zone, so daylight saving is right. For each it shows open,
    close, high, low, range, % return, realised volatility, directional bias, breakout
    frequency and trend persistence, compared across sessions and periods.
  - **Time of day:** return, absolute move, range, acceleration, momentum, where the
    day's high and low form, and the **real bid–ask spread** by time of day from MT5
    ticks. Intervals are 1, 5, 15, 30 and 60 minutes.
  - **News hours:** the move and the spread around NFP, CPI (8:30 ET) and FOMC (14:00 ET),
    against the same hour on other days. It states that without consensus forecasts the
    direction of a surprise is unknown.
  - **Volatility:** ATR, realised volatility, a regime class, and today against history.
  - **History:** 5, 20, 60, 120 and 250 trading days or a custom range. It gives average
    return and range, ATR, realised volatility, maximum drawdown, trend persistence, the
    return distribution, and the current session, volatility and range against the
    historical average.
  - **Benchmarks:** TWAP over a chosen interval, current price against TWAP, and arrival
    price with the move after it for a hypothetical start time.
- **No volume anywhere (owner decision, 2026-11-04).** The broker's real volume is 0 and
  tick volume is not volume, so VWAP, volume profile (POC, VAH, VAL, HVN, LVN), PWP and
  volume metrics are left out entirely, not approximated.
- **Data.**
  - MT5, read-only: live tick, M1/M15/H1/D1 bars, and ticks for the spread.
  - FRED release dates for NFP and CPI, plus `results/fomc_dates.csv`.
  - Heavy pulls (ticks) are aggregated into a gitignored per-day cache by a refresh
    command, so a page request never pulls months of ticks.
- **Out of scope:**
  - AI Gold Search (deferred);
  - the overnight-drift strategy test (deferred to its own change);
  - any order, sizing or stop-loss code;
  - any verdict or registry;
  - any change to `/gold`, `src/gold`, the gold registries, the loggers or `models/`.

## Capabilities

### New Capabilities
- `gold-analytics/market-data`: how the analytics obtains, caches and labels gold data:
  the MT5 read-only sources, the tick-to-spread aggregation, the clock and session
  definitions, the event calendar, and the no-volume rule.
- `gold-analytics/dashboard`: the `/gold-analytics` page and its read-only API, covering
  each section's metrics, parameters, errors, wording and layout.

### Modified Capabilities
- `app-shell`: "Navigation shows where the owner is" gains a Gold analytics item in the
  main navigation.

## Impact

- New code:
  - `src/gold_analytics/`: `data.py`, `clock.py`, `sessions.py`, `intraday.py`,
    `volatility.py`, `history.py`, `events.py`, `benchmarks.py`, `service.py`,
    `refresh.py`, `analytics_config.json`;
  - `static/gold_analytics.html`;
  - tests `tests/test_gold_analytics_*.py`.
- `api.py` gains lines only: `GET /gold-analytics`, `GET /api/gold-analytics/<section>`.
- `src/app_shell.py`: `PAGE_PATHS` and `NAV` gain the page; `tests/test_app_shell.py`
  follows.
- A new gitignored cache, `research_models/gold/analytics/`.
- Network: FRED release-dates calls with the existing `FRED_API_KEY`.
- Documentation: `ARCHITECTURE_DOCS.md` §5.4 and `CLAUDE.md`.
- `tsfm-second-opinion` (not archived) modifies the same navigation requirement. This
  delta includes its Second opinion item, so whichever is archived second must be
  rebased.
