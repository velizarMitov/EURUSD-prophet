# Design

## Context

- **Data available on this PC**, all read-only and probed on 2026-11-04:
  - MT5 symbol `GOLD`. Tick history reaches at least 2021, with bid and ask. The median
    spread at 16:00–17:00 Sofia was $0.32 in 2023 and $0.48 in 2026.
  - M1 and M15 are capped by the terminal's "Max bars in chart" (100,000). That is about
    70 trading days of M1, and M15 from 2022-08-12.
  - H1 and D1 are also in the repo caches, from 2014 and 2013.
  - `real_volume` is 0, and `market_book_add` fails, so there is no order book.
- **Pages and the shell.**
  - `src/app_shell.py` defines `PAGE_PATHS`, `NAV` and `MORE`.
  - `api.py`'s `ShellLayer` injects the top bar into every path listed in `PAGE_PATHS`.
  - The pages live in `static/*.html`, styled with `--ps-*` tokens. `static/gold.html`
    is an example; its chart colours were validated with the dataviz validator.
- **Constraints.**
  - `api.py` may only gain lines, under the app-shell spec "Pinned page sources stay
    byte-identical".
  - `/gold` must show no spread, under "Redesigned pages keep their contracts".
  - `src/gold` must never read spread, under its isolation contract. So the analytics
    lives in its own package and its own page.
- **Reusable code.**
  - `src/gold/data.resolve_gold_symbol` resolves the symbol name.
  - `src/macro_data` reads `FRED_API_KEY` with the `fredapi` client.
  - `results/fomc_dates.csv` holds FOMC dates through 2027.

## Goals / Non-Goals

**Goals:**
- Every request is answered in under 2 s from caches. Heavy pulls happen only in
  `refresh`.
- Correct clocks: Berlin labels in storage, Sofia time on screen, sessions and releases
  in their own zones.
- One small pure function per metric, each tested alone.

**Non-Goals:**
- Volume of any kind, an order book, or "price impact versus trade size".
- AI Gold Search, the overnight-drift strategy test, and any forecast or signal.
- Changing `/gold`, `src/gold`, loggers, registries or `models/`.

## Decisions

### D1. Package and isolation

The code lives in `src/gold_analytics/`, with settings in `analytics_config.json`:
sessions, intervals, periods, the regime split and the cache paths.

The package:
- imports `src.gold.data.resolve_gold_symbol` read-only, and no other part of `src/gold`;
- runs no model and never calls an order or position function. A static test scans for
  `order_send`, `order_check`, `positions_get` and `orders_get`;
- is the only module that `api.py` imports for the page, through `service.py`.

### D2. Data layer (`data.py`)

`Source` wraps MT5 with one `initialize`/`shutdown` per call, as `src/live_data.py` does.
It exposes:
- `tick()`;
- `bars(tf, start, end)`, chunked by year to avoid MT5's "Invalid params";
- `ticks(start, end)`.

Only time, OHLC, bid and ask are kept. `tick_volume` and `real_volume` are dropped at the
boundary, as `src/gold/data.py` does with OHLC.

Bars are cached in gitignored `research_models/gold/analytics/bars_{M1,M15}.csv.gz`. The
D1 and H1 bars come from `results/gold/gold_{d1,h1}.csv`, topped up live.

When MT5 is unreachable, every function returns its cached frame with
`stale=True, as_of=<newest label>`.

- **Alternative rejected:** reading through the gold logger's caches only. They are H1
  and D1 only, with no ticks, and they refresh once an hour.

### D3. Spread cache (`refresh.py`)

`python -m src.gold_analytics.refresh --days 250` handles each trading day not yet
cached:
- it pulls that day's ticks with `copy_ticks_range` in hour chunks;
- it buckets them into 5-minute slots on the label clock;
- it stores the median and 90th-percentile spread and the tick count per slot, one row
  per slot, in `spread_5m.csv.gz`.

A day takes about 1–2 s, so the first build of 250 days takes about 5–8 minutes. After
that only new days are pulled. The 60-minute view aggregates the 5-minute rows with a
tick-count weighted median. The current day is computed live from today's ticks only.

- **Alternative rejected:** pulling ticks per request. That is about 100 M ticks for 250
  days.

### D4. Clocks and sessions (`clock.py`, `sessions.py`)

- A label is converted to an instant with
  `tz_localize('Europe/Berlin', ambiguous=..., nonexistent='shift_forward')`, using the
  fall-back hour's order to resolve ambiguity.
- Display is `tz_convert('Europe/Sofia')`.
- A session on day d is built from `datetime.combine(d, start, ZoneInfo(zone))` and its
  end, converted to Berlin labels. Bars with labels in [start, end) belong to it.
- The overlap is `max(starts)`–`min(ends)` of London and New York for that day.

Session metrics:
- open is the first bar's open, close the last bar's close;
- realised volatility is √Σr² of in-session bar log returns;
- breakout means high > previous session high or low < previous session low;
- persistence compares sign(return) with the previous session's sign;
- the bias interval is Wilson 95 %, implemented in the package, as `src/gold/history.py`
  does.

### D5. Timeframes

| timeframe | source | notes |
|---|---|---|
| 1m | M1 | |
| 5m | M1, resampled | |
| 15m | M15 | |
| 30m | M1, resampled | M15 resampled beyond M1's start |
| 1h | H1 | |
| 4h | H1 resampled, on Sofia-midnight bins | the label is shifted +1 h, binned, shifted back |
| D1 | D1 | |
| W1 | D1 resampled | `W-MON` anchored |

Each response states `first_bar`. Charts cap at 2,000 bars per request, with a `start`
parameter for paging.

### D6. Events (`events.py`)

FRED `release/dates` gives the dates for release_id 50 (Employment Situation) and 10
(CPI), cached in `events.csv`. A release is at 08:30 America/New_York, and FOMC at 14:00
America/New_York.

For each event and window (−2 h to +3 h in 30-minute steps), the section shows:
- the mean absolute move;
- the median spread from the cache;
- the same clock slots on the same weekday on non-event days, for comparison.

The text always says that the direction depends on the surprise, which needs consensus
data the page does not have.

### D7. Volatility, history and benchmarks

| metric | definition |
|---|---|
| ATR(14) | Wilder ATR on D1 |
| realised volatility | std of D1 log returns × √252 |
| intraday volatility | √Σr² of today's M5 log returns, scaled to a day |
| regime | 20-day realised volatility against the terciles of its trailing 250 values |
| maximum drawdown | on D1 closes |
| histogram | D1 returns in 0.25 % bins |
| TWAP | mean of bar closes in the interval, on M1 if available, else M15 |
| arrival | the close of the bar ending at the start, and the moves to +15/30/60/120 min |

### D8. API (`service.py`, routes in `api.py`)

`GET /gold-analytics` serves the page. The sections are
`GET /api/gold-analytics/{overview|chart|sessions|intraday|events|volatility|history|benchmarks}`.

- Parameters are validated against the config, and any other value answers 400 with the
  allowed values.
- Results are cached in memory for 5 s (overview and chart, which show live bars) or
  until the cache files change, at most 10 minutes (everything else).
- Every payload carries `stale`, `as_of`, the descriptive label in BG and EN, and the
  dictionary of section texts.

### D9. Page (`static/gold_analytics.html`)

- The page follows the `static/second_opinion.html` pattern: `--ps-*` tokens, inline SVG
  charts, and the BG/EN dictionary from the payload, following the shell's switch.
- It has one section per card, in this order: Overview with its chart, Sessions, Time of
  day, News hours, Volatility, History, Benchmarks.
- On a phone the cards stack in one column, and charts use `viewBox` and scale with the
  width.
- The owner's window is shaded in the time-of-day charts.
- The overview refreshes every 10 s while the tab is visible. Other sections update only
  when the owner changes a control.
- The chart colours come from the validated palette in `static/gold.html`, and are
  re-validated with the dataviz validator in both themes.

### D10. Shell

- `/gold-analytics` is added to `PAGE_PATHS`, and `NAV` gets
  `('/gold-analytics', {'bg': 'Анализ на златото', 'en': 'Gold analytics'})` after Gold.
- `tests/test_app_shell.py` gains the scenario.
- No other shell behaviour changes.

## Risks / Trade-offs

- **[MT5 bar caps hide history]** → Each response states `first_bar`. The docs say that
  raising "Max bars in chart" to Unlimited lengthens M1 and M15.
- **[Tick pulls are slow, or the terminal lags]** → They run only in `refresh`, one day at
  a time and resumable. A page request never pulls history.
- **[Concurrent MT5 use with the hourly loggers]** → Each call does a short
  `initialize`/`shutdown`, the same pattern the loggers and `live_data` already share.
- **[FRED is unavailable or the key is missing]** → The cached `events.csv` is used, and
  the section states its age. Without either, the section shows only FOMC.
- **[Statistics read as signals]** → Every section is labelled. Bias carries Wilson
  intervals, and no text names a trade. A wording test enforces this.
- **[A navigation conflict with `tsfm-second-opinion`]** → This delta already contains
  the Second opinion item. Whichever change is archived second must check its main spec.

## Migration Plan

1. Build the data, refresh, metrics and API with tests.
2. Run `refresh --days 250`.
3. Add the page and the navigation entry.
4. Rollback: remove the routes (`api.py` lines), the NAV and `PAGE_PATHS` entries, the
   package and the cache folder. No other state depends on them.

## Open Questions

- Should the overview's live refresh pause outside market hours? This can be decided
  during implementation without changing the specs, and the default is yes, while the
  market is closed.
