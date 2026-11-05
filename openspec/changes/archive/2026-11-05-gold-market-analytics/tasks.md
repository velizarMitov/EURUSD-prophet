# Tasks

## 1. Package, config and data layer

- [x] 1.1 Create `src/gold_analytics/` with `__init__.py` (isolation contract docstring)
  and `analytics_config.json`. The config holds:
  - the sessions with their zones;
  - the intervals 1, 5, 15, 30 and 60;
  - the timeframes;
  - the periods 5, 20, 60, 120 and 250;
  - the regime terciles;
  - the event windows;
  - the cache paths under `research_models/gold/analytics/`.

  Verify: a config test asserts each value. A static test asserts that the package
  contains no `order_send`, `order_check`, `positions_get`, `orders_get`, `tick_volume`
  or `real_volume`, and imports nothing from `src.gold` except
  `src.gold.data.resolve_gold_symbol`.
- [x] 1.2 In `data.py`, implement `Source.tick()`, `bars()` (chunked by year) and
  `ticks()`. It keeps only time, OHLC, bid and ask. When MT5 is unreachable it falls back
  to the cache, with `stale`/`as_of`. It reads D1 and H1 from the repo caches, topped up
  live.

  Verify with tests using a fake MT5:
  - the fallback sets `stale: true`;
  - the volume fields never reach the frame;
  - a long range is chunked by year.

## 2. Clocks, sessions and the spread cache

- [x] 2.1 In `clock.py`, implement label → instant (Berlin, with daylight saving) and
  display in Sofia. In `sessions.py`, build per-day session intervals from each zone and
  the London–New York overlap.

  Verify with tests:
  - label 14:00 displays as 15:00 Sofia;
  - New York 08:20 is 15:20 Sofia in January and 14:20 Sofia in a March mismatch week;
  - a custom 07:00–10:00 London session works;
  - the overlap is the intersection on each day.
- [x] 2.2 In `refresh.py`, implement the per-day tick pull into 5-minute buckets, storing
  the median and 90th-percentile spread and the tick count. It skips cached days, writes
  `spread_5m.csv.gz`, and also refreshes the M1 and M15 bar caches.

  Verify with fake-MT5 tests:
  - 249 of 250 days cached pulls exactly one day;
  - the bucket median matches a hand-computed value;
  - nothing is written outside `research_models/gold/analytics/`.
- [x] 2.3 In `events.py`, implement the FRED release-dates fetch for release_id 50 and 10
  with a disk cache, and FOMC from `results/fomc_dates.csv`. Map them to instants
  (08:30 / 14:00 America/New_York).

  Verify with tests using a fake FRED client:
  - CPI in January and July maps to 15:30 Sofia, and in a March mismatch week to
    14:30 Sofia;
  - when FRED raises, the cache is used and its age is reported.

## 3. Metrics

- [x] 3.1 In `sessions.py`, implement the session statistics: open, close, high, low,
  range, % return, realised volatility, bias with the Wilson 95 % interval, breakout
  frequency, persistence, and the overnight-against-day summary.

  Verify with tests on synthetic bars, including 33 of 60 sessions giving 55 % with the
  correct Wilson bounds.
- [x] 3.2 In `intraday.py`, implement the time-of-day statistics: return, absolute move,
  range, acceleration, momentum (first against last window hour), the distribution of
  high and low times, and the spread by slot from the cache. Intervals are 1, 5, 15, 30
  and 60, with the history limit stated.

  Verify with tests:
  - the 60-minute spread aggregates the 5-minute rows;
  - a 1-minute request beyond M1 states the days used.
- [x] 3.3 In `volatility.py` and `history.py`, implement ATR(14), realised volatility
  (5/20/60), intraday volatility, the regime terciles, today against average, periods and
  custom ranges, maximum drawdown, persistence and the histogram.

  Verify with tests against hand-computed values on a fixed series, including a regime
  above the 67th percentile reading high.
- [x] 3.4 In `benchmarks.py`, implement TWAP, price against TWAP, and the arrival price
  with the moves at 15, 30, 60 and 120 minutes. In `events.py`, add the news-hour windows
  against non-event days.

  Verify with tests:
  - TWAP equals the mean of the closes;
  - the arrival price is the close of the bar ending at the start;
  - the event comparison uses the same weekday's slots.

## 4. API

- [x] 4.1 In `service.py`, build the section payloads, with parameter validation (400
  with the allowed values), `stale`/`as_of`, the BG/EN dictionary and the descriptive
  label, and caching (5 s for the overview, file-mtime for the rest).

  Verify with tests:
  - `timeframe=2h` gives 400 with the list;
  - every payload has the label in both languages;
  - no field name or text contains volume, vwap, poc, profile, buy, sell, купи or продай.
- [x] 4.2 Add `GET /gold-analytics` and `GET /api/gold-analytics/{section}` to `api.py`
  (lines only). Add the page to `src/app_shell.py` `PAGE_PATHS` and `NAV`, and the
  navigation scenario to `tests/test_app_shell.py`.

  Verify:
  - `pytest tests/test_app_shell.py` passes;
  - a TestClient GET of every section with fake data answers 200;
  - `git diff api.py` shows only added lines.

## 5. Page

- [x] 5.1 Write `static/gold_analytics.html`:
  - cards in order: Overview with the chart (8 timeframes), Sessions, Time of day (window
    shaded), News hours, Volatility, History, Benchmarks;
  - inline SVG;
  - BG/EN from the payload;
  - the overview refreshes every 10 s while visible;
  - a single column at 360 px.

  Re-validate the chart colours with the dataviz palette validator in both themes.

  Verify:
  - a static test finds no external URL;
  - a wording test finds no buy or sell words;
  - in the running app, the page renders in BG and EN, light and dark, at 360 px with no
    horizontal scroll, and every control works from the keyboard.
- [x] 5.2 Run `python -m src.gold_analytics.refresh --days 250` on this PC. Open
  `/gold-analytics` and show the owner each section.

  Verify:
  - the spread cache covers 250 trading days;
  - the chart states each timeframe's first bar;
  - the overview shows the live price, and is stale with MT5 closed.

## 6. Documentation and integration

- [x] 6.1 Add the page to `ARCHITECTURE_DOCS.md` §5.4 (route, package, data sources, the
  no-volume rule, the refresh command, and the note on MT5 "Max bars in chart"). Add one
  line to `CLAUDE.md` under the app shell.

  Verify:
  - `pytest tests/test_gold_analytics_*.py tests/test_app_shell.py tests/test_gold_*.py`
    passes;
  - `openspec validate gold-market-analytics --strict` is clean;
  - `git status` shows changes only in `src/gold_analytics/`, `static/gold_analytics.html`,
    `api.py`, `src/app_shell.py`, the tests and the docs.
