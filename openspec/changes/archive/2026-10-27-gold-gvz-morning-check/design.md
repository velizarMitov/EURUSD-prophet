# Design

## Context

See `proposal.md` for the motivation and `specs/gold-forecasting/gvz-check/spec.md` for
the requirements. The existing code sets these constraints:

- **`src/gold/summary.py::build()` is read-only by contract.**
  - It fetches no market data, writes nothing and imports no model library.
  - `tests/test_gold_summary.py` and `tests/test_gold_isolation.py` enforce this.
  - `api.py` may import only the read-only gold views.
- **The hourly logger is the only place gold touches the network.**
  - `src/gold/forward_logger.py::run_once` runs at :03.
  - Each optional step (`log_ranges`, `log_alerts`, `log_shadow`, `log_v2`) is wrapped,
    so its failure lands in `results/gold/forward/failures.csv` and never costs the
    round-one pass.
- **`src/gold/macro.py::fred_series` wraps the shared FRED chain.**
  - The chain is `src.macro_data.fetch_fred_feature`: API, then public CSV, then cache,
    with merge-not-overwrite cache writes.
  - The macro caches live in `results/gold/macro/`.
  - `macro.py` belongs to the round-two study, and its behaviour is part of what was
    registered.
- **Only `cfg2['study']` is digest-pinned** (`prereg2.study_digest`). The `alerts` and
  `retrain` blocks were added beside it after registration.
- **Gold forward logs are append-only**, with settlements in a sibling file:
  `predictions` + `settlements`, and `alerts` + `alert_settlements`.
- **Bar labels are the broker's Europe/Berlin wall clock**, stored as UTC as-is.
  Sofia is Berlin + 1 h all year.
- **The isolation tests forbid gold code from naming certain things:** `forward_eval`,
  the production `models/` folder, `config.json`, or a quoted spread or volume field.
- **`/gold` must contain none of the cost words** `spread`, `pips`, `p&l`, `profit`,
  `pnl` or `sharpe` (`tests/test_gold_view.py`).

## Goals / Non-Goals

**Goals:**
- One new module that owns GVZ end to end, with its network half and its file-only half
  clearly separated.
- No change to any existing gold output schema, registry or pinned setting.
- The card renders even if GVZ never loads.

**Non-Goals:**
- No intraday GVZ. The value is a daily close.
- No US holiday calendar.
- No backfill of observation rows for days before the change ships.

## Decisions

### D1. A new module `src/gold/gvz.py`, not an extension of `macro.py`

`gvz.py` has two halves.

**The acquire half is called only by the logger.**
- `refresh(cfg2, now)` is the network half. It imports `yfinance` lazily, inside the
  function.
- `observe(...)` and `settle(...)` write the observation log and the settlement log.

**The read half is called by both the summary and the logger.**
- `read_series(cfg2)` merges the two caches into one series that carries a source.
- `reading(cfg2, now, gold_close, close_bar)` returns the block: value, date, source,
  percentile, history_n, implied move in % and $, and stale.
- It touches files only.

`summary.py` imports only the read half's functions. With the lazy import, that keeps
`yfinance` out of the summary's import graph.

*Alternative:* put GVZ in `macro.py` next to VIXCLS. Rejected because `macro.py` is
study code covered by the round-two registration, and the morning check is a live
display. Mixing them would make a change to the display look like a change to the study.
For FRED, `gvz.py` still calls the shared `src.macro_data.fetch_fred_feature` directly, so
the FRED fallback chain stays the same everywhere.

### D2. Two caches, merged on read

- **`results/gold/macro/gvzcls.csv` is FRED only.** It is written by
  `fetch_fred_feature`, with its own merge-not-overwrite semantics.
- **`results/gold/macro/gvz_yfinance.csv` holds the top-up**, with columns `date,gvz`.
  It is merge-written by date.
- **`read_series` merges them:** FRED values for every FRED date, and yfinance values
  only for dates after the newest FRED date, each tagged with its source.

*Alternative:* one merged file with a source column. Rejected because
`fetch_fred_feature` merge-writes with keep-last semantics. A yfinance row in its cache
would be indistinguishable from FRED and could shadow a later FRED value.

### D3. Dates are US session dates, not bar labels

GVZ is a daily index, not a gold bar. It is stored as plain `YYYY-MM-DD` session dates:
- FRED's observation date as-is;
- yfinance's `America/New_York` index converted with `.date()`.

"Completed" means `date < today in America/New_York`. That one rule drops the in-progress
yfinance bar whenever the owner opens the page during US hours.

### D4. Refresh in the logger, only when stale

`log_gvz(cfg, now_iso)` runs in `run_once` after `log_alerts`, wrapped like the other
optional steps. It runs whether or not MT5 is live, because GVZ does not depend on MT5.

It calls `refresh` only if the newest stored date is before the last completed US session
(the US weekday before today's New York date). The fetch windows:

| Source | First run | Later runs |
|---|---|---|
| FRED | `history_start` (2013-01-01) to today, about 3,400 values in one request | the last 30 days |
| yfinance | `period='10d'` | `period='10d'` |

*Alternative:* fetch inside `/api/gold`. Rejected because it breaks the read-only summary
contract and its tests, and it would make page loads depend on the network.

### D5. The dollar move uses the gold close the logger recorded

- **The API** uses `as_of_close` and `as_of_bar` of the newest H1 row in
  `results/gold/forward/predictions.csv`. `build()` already reads that file, so there is
  no extra read and no scan of the 70k-row `gold_h1.csv` per request.
- **The observation row** uses the last bar of the in-memory `frames['H1']`.

Both record the bar the close came from, so the card can show it.

### D6. Observation and settlement scheduling

**Observation.**
- On a pass where Sofia local time is Monday to Friday and at or after `log_after_sofia`
  (13:00), the logger appends one row if `gvz_log.csv` has no row for today's Sofia date.
- The row records what is known at that moment: the GVZ reading from D1, the gold close
  and the newest `h1_lstm` row in `predictions.csv` (`vol_pct`, `as_of_bar`).
- An unknown field stays blank. Append-only means it is never filled later.
- Columns: `date, logged_at, gvz, gvz_date, gvz_source, percentile, implied_move_pct,
  implied_move_usd, gold_close, gold_close_bar, lstm_vol_pct, lstm_as_of_bar`.

**Settlement.**
- On every pass, each logged date without a row in `gvz_settlements.csv` is settled
  once `frames['H1']` holds a bar labelled after `date 21:00`.
- The logger refreshes the bar cache from MT5 over a long window. Once a later bar
  exists, the session bars present are therefore final, and any missing bar is a real
  market or data gap.
- Columns: `date, realised_range_pct, range_bars, first_open, settled_at`.
- With fewer than `min_session_bars` (9), `realised_range_pct` is blank.

### D7. API block shape

`build()` adds this block, computed inside `try/except` like the alert, so an error
yields `{"state": "none"}` and never breaks the payload:

```
"gvz": {"state": "ok" | "stale" | "none", "value": 24.34, "date": "2026-10-23",
        "source": "FRED" | "yfinance", "percentile": 91, "history_n": 3360,
        "history_start": "2013-01-01", "implied_move_pct": 1.53,
        "implied_move_usd": 62.86, "gold_close": 4100.0,
        "gold_close_bar": "2026-10-27 11:00:00+00:00",
        "lstm_vol_pct": 0.145, "lstm_as_of_bar": "..."}
```

The LSTM values are copied from the `h1_lstm` entry `build()` already computes, only when
that entry is current.

### D8. The card

- **Placement and data:** a new `<div id="gvz">` at the top of `#panel-now`, above
  `#body`. It is rendered by `renderGvz()` from `payload.gvz`, and re-rendered on the
  shell's `ps:lang` event like the other parts.
- **Strings:** BG/EN entries in the page's existing `I18N` table.
- **Styling:** the existing `ps-card` tokens, so light and dark themes come free.

Layout:

```
+------------------------------------------------------------+
| Сутрешна проверка · GVZ          24.34 · пт 23.10 · FRED   |
| Пазарът на опции очаква ~±1.53 % за цял ден (~±62.86 $)    |
| [########--] 91-ви перцентил от 2013 г.   тихо ... бурно   |
| Модел h1_lstm, следващ час: ±0.145 %                       |
| Посоката е неизвестна. GVZ е 30-дневното очакване на       |
| пазара на опции върху GLD, не прогноза на модел.           |
+------------------------------------------------------------+
```

- **The stale state** keeps the numbers, adds a warning line with the value's date and
  uses the existing `stall` notice colour.
- **The none state** is a single meta line.
- **Copy rules:** no direction arrows, and none of the forbidden cost words in either
  language. The existing page test already covers the English list.

### D9. Configuration

A new top-level block goes into `src/gold/gold_config_v2.json`:

```
"gvz": {"fred_id": "GVZCLS", "yahoo_symbol": "^GVZ", "history_start": "2013-01-01",
        "trading_days_per_year": 252, "stale_after_days": 5,
        "log_after_sofia": "13:00", "session_labels": [12, 21], "min_session_bars": 9,
        "fred_cache": "gvzcls.csv", "yahoo_cache": "gvz_yfinance.csv",
        "log": "results/gold/gvz_log.csv", "settlements": "results/gold/gvz_settlements.csv"}
```

- The cache names resolve under the existing `paths.macro_cache`.
- A test asserts that `prereg2.study_digest(cfg2)` still equals the digest recorded in
  `PRE_REGISTRATION_V2.json`.

### D10. The health note

`run_with_health` adds a short `gvz: <status>` segment to the note, for example
`gvz: fresh 2026-10-26 yfinance` or `gvz: failed ConnectionError`. The note is truncated
at 300 characters, so the segment stays under about 40 characters and sits right after
`alerts:`, ahead of the longer `shadow` and `v2` segments.

## Risks / Trade-offs

- **[yfinance changes or drops `^GVZ`]** → FRED alone still works, and the card shows the
  older date. The failure is visible in the health note.
- **[US market holiday: no new GVZ value, so the stale check fires every pass that day]**
  → about 24 cheap requests on a handful of days a year. Accepted instead of maintaining
  a holiday calendar.
- **[The owner reads the implied move as the afternoon range]** → the card says "for a
  full day, one standard deviation". The observation log records the realised afternoon
  range, so the owner can see the relation for themselves.
- **[GLD trades US hours on a 30-day horizon, while XAUUSD trades nearly 24 h]** → the
  card names GLD and the 30-day horizon. GVZ is presented as context, not a forecast of
  XAUUSD.
- **[The observation log is later treated as evidence]** → the spec forbids scoring it.
  Any future claim about GVZ must be pre-registered and scored only on rows logged after
  that registration.
- **[A FRED revision changes a past value]** → FRED wins on read, so the percentile uses
  the revised value. Observation rows keep what was served at the time.

## Migration Plan

- **Deploy:** ship the code and the config block. The next logger pass fills the FRED
  history (one request), tops up from yfinance, and the card appears. The first
  observation row is written at the first weekday pass at or after 13:00 Sofia.
- **Rollback:** remove the `log_gvz` call and the card. The caches and logs are inert
  files that nothing else reads.
