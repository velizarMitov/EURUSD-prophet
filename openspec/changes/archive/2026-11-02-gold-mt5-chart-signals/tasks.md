# Tasks

## 1. Signal export (Python)

- [x] 1.1 Create `src/gold/mt5_export.py` with row selection and scoring (design D6):
  - h1_lstm/H1 rows only, never `predictions_v2.csv`;
  - the mark hour is as-of + 1 h;
  - the 5-trading-day window by the Sofia date of the mark hour;
  - direction, probability and outcome (`open`/`right`/`wrong`/`void`);
  - alerts joined to their prediction row for close and size;
  - the right/wrong count.

  Verify with `python -m pytest -q tests/test_gold_mt5_export.py`. Use fixture CSVs in a
  temp `forward` folder covering:
  - h1_gbm and shadow rows excluded;
  - a weekend boundary with no phantom Sunday;
  - an ineligible settlement and an empty `realised_dir` both as `void`;
  - a missing alert prediction row shown in percent only.
- [x] 1.2 Add the texts (design D2) in BG and EN with per-language number formatting:
  - short text, hover description and panel line per mark;
  - the Sofia hour via `zoneinfo`, never a fixed +1;
  - the `#meta` lines: data time, count and research label.

  Verify with tests that check:
  - the spec examples: P(up) 0.5317 → "53 %"; 0.1446 % at 3897.45 → "~±5.64 $"; as-of
    2026-11-02 09:00 → "11:00–12:00"; 0.1954 % at 4063.36 → "±0.20 % (~±7.94 $)";
  - a mark hour across the 2026-10-25 DST change;
  - every alert text says the direction is unknown;
  - no text in either language matches the trade-word list (buy, sell, long, short,
    entry, stop, target, take profit, and the BG equivalents).
- [x] 1.3 Add a test that the export's direction and probability equal
  `summary.current_forecast` on the same rows. Verify it passes.
- [x] 1.4 Write the file (design D3, D4):
  - UTF-8 TSV `version=1`;
  - `MT5_COMMON_FILES` override, else `%APPDATA%\MetaQuotes\Terminal\Common\Files`;
  - `EURUSDProphet\gold_signals.tsv`;
  - a `.tmp` file, then `os.replace` with 5 retries on `PermissionError`;
  - no tab in any field, and `\n` escaped in descriptions.

  Verify with tests:
  - a round-trip parse of the written file;
  - a replace that fails twice, then succeeds;
  - a replace that always fails raises;
  - no `.tmp` is left after success.
- [x] 1.5 Add the CLI `python -m src.gold.mt5_export`. It rebuilds the file and prints a
  one-line summary. Verify by running it with `MT5_COMMON_FILES` pointed at the
  scratchpad: the file appears, and the summary names the row count and the window.

## 2. Logger hook and isolation

- [x] 2.1 Add `log_chart(cfg, now_iso)` to `src/gold/forward_logger.py` (design D5):
  - call it after `log_alerts` on every pass, as status key `chart`;
  - wrap it like `log_ranges`, with a `failures.csv` row for cell `chart`;
  - put `chart: …` in the health note right after `alerts: …`.

  Verify with tests in `tests/test_gold_forward_logger.py`:
  - an injected export failure still writes the pass's predictions, settlements and
    alerts, and records the failure;
  - a passing export appears in the health note.
- [x] 2.2 Amend the isolation contract in the `src/gold/__init__.py` docstring with the
  single named exception: the export file under `Common\Files\EURUSDProphet\`. Add a test
  to `tests/test_gold_isolation.py` that verifies:
  - `mt5_export.py` is the only gold module that names the `MetaQuotes`/`Common\Files`
    location;
  - no repo module reads `gold_signals.tsv`.

## 3. MT5 indicator (MQL5)

- [x] 3.1 Create `mql5/Indicators/ProphetGoldSignals.mq5` with the reading side:
  - `indicator_chart_window`, `indicator_plots 0`;
  - inputs for language, colours, panel corner and font size;
  - read the file with `FILE_COMMON` and `CP_UTF8`;
  - parse `#meta` and the rows, and refuse an unknown `version`;
  - a 30 s timer that re-reads on a `FILE_MODIFY_DATE` change.

  Verify it compiles with 0 errors via
  `metaeditor64.exe /compile:<path> /log:<log>` from one installed terminal.
- [x] 3.2 Add the drawing (design D8):
  - `PGS_`-prefixed objects;
  - up/down arrows anchored under the hour's H1 low or over its H1 high;
  - the alert marker stacked above;
  - `OBJ_TEXT` beside each mark with an outcome colour;
  - tooltips on mark and text;
  - deletion of only `PGS_` objects on redraw and deinit.

  Verify on a live XAUUSD H1 chart: arrows, alert marker and texts appear for the last
  5 trading days. Hovering shows the description, and removing the indicator leaves no
  `PGS_` object behind. Confirm that the alert glyph renders as a two-headed vertical
  marker; if code 244 does not, pick the code that does.
- [x] 3.3 Add the panel and the time-based states (design D7, D9):
  - current forecast or none, plus the last hour;
  - open alert;
  - count line, data time and research label;
  - the clock shift, named when non-zero;
  - the stale warning only while ticks arrive;
  - the missing-file message;
  - the not-gold chart and above-H1 messages with no marks.

  Verify:
  - on a EURUSD chart and an XAUUSD D1 chart, no marks are drawn and the reason is shown;
  - on M15, arrows sit on the first M15 bar of the hour;
  - renaming the data file shows the missing message without a crash, and restoring it
    redraws within 30 s;
  - in the terminal whose server clock is not Berlin, the shift is shown and the marks
    sit on the shifted hour.
- [x] 3.4 Add a source scan in `tests/test_gold_mt5_export.py` (design D10). It fails on
  `OrderSend`, `OrderSendAsync`, `CTrade`, `#include <Trade`, `PositionOpen`,
  `WebRequest`, `SocketCreate` or `#import` in the `.mq5`, and on any trade word in its
  string literals. Verify it passes on the indicator and fails on a fixture string that
  contains `OrderSend`.

## 4. Installer and docs

- [x] 4.1 Create `scripts/gold/install_mt5_indicator.ps1` (design D11):
  - enumerate `origin.txt` data folders;
  - copy the indicator to `MQL5\Indicators\EURUSDProphet\`;
  - compile with that terminal's `metaeditor64.exe`;
  - report each terminal as compiled when the `.ex5` exists and the log has 0 errors,
    otherwise as failed with the log;
  - run the export once.

  Verify by running it on this PC: three lines, one per terminal, and the `.tsv` exists
  in `Common\Files\EURUSDProphet\`.
- [x] 4.2 Create `scripts/gold/uninstall_mt5_indicator.ps1`. It removes only each
  terminal's `MQL5\Indicators\EURUSDProphet\` and `Common\Files\EURUSDProphet\`. Verify by
  running it: those folders are gone, and the rest of each `MQL5\Indicators` folder is
  unchanged. Then reinstall.
- [x] 4.3 Document the feature:
  - add install, attach, check and troubleshooting steps to
    `results/gold/forward/RUNBOOK.md`, including portable-mode terminals, the clock shift
    and the staleness warning;
  - add a chart-signals line to `ARCHITECTURE_DOCS.md` §3.12;
  - add the export exception to the gold bullet in `CLAUDE.md`.

  Verify that each documented command runs as written.

## 5. Daily signals and the probability filter (owner's request, 2026-11-02)

- [x] 5.1 Extend the export for the daily cells and the filter (design D3, D6):
  - d1_lstm and d1_gbm rows, with the mark day taken from `exit_bar` once settled and
    from `as_of + BDay(1)` before that;
  - the 20-day daily window;
  - texts carrying the cell tag, with day descriptions;
  - `cell` and `cadence` fields and `version=2`;
  - the `count_h1_*`, `count_d1_*` and `filter_*` templates.

  Verify with tests:
  - the spec's daily scenario (as-of 2026-10-29: d1_lstm down 52 % and d1_gbm up 78 % on
    2026-10-30);
  - `exit_bar` wins once settled;
  - the 20-day window;
  - d1_gbm has no move line;
  - direction parity with `summary.current_forecast` for D1;
  - every template is free of trade words.
- [x] 5.2 Draw the daily marks (design D13):
  - D1-only placement;
  - solid d1_lstm and hollow d1_gbm arrows, with the d1_gbm colour input;
  - same-side stacking with one joined text;
  - the panel's current daily forecast per cell;
  - the "H1 or lower, and D1" message on other timeframes.

  Verify that it compiles with 0 errors. Then verify on a live XAUUSD D1 chart:
  - two arrows per day;
  - stacked arrows when both cells point the same way;
  - H4 draws nothing and says why.
- [x] 5.3 Add the probability filter (design D12):
  - ≥50/52/54/56 % buttons in the panel;
  - the choice kept per chart in a terminal global variable, with a changed input
    winning;
  - direction arrows below the filter hidden, alerts kept;
  - the filter line;
  - the count filled from the arrows shown.

  Verify that it compiles with 0 errors and the source scan passes. Then verify on a
  live chart:
  - clicking 54 leaves only arrows ≥54 % and every alert marker;
  - the count changes;
  - H1 → D1 → H1 keeps 54.
- [x] 5.4 Update the RUNBOOK chart section, `ARCHITECTURE_DOCS.md` §3.12 and the
  `CLAUDE.md` gold bullet for the daily marks and the filter. Reinstall with
  `scripts/gold/install_mt5_indicator.ps1`. Verify that the documented commands run and
  all three terminals compile.

## 6. Integration check

- [x] 6.1 Let one scheduled `EURUSDProphet-GoldLogger` pass run with an XAUUSD H1 chart
  open and the indicator attached. Verify:
  - the new h1_lstm arrow appears within a minute of the pass, without reattaching;
  - the health row's note carries `chart: …`;
  - `failures.csv` has no `chart` row.

  Then run `python -m pytest -q` and confirm the full suite passes.

  Result 2026-11-02: the 12:03 Sofia pass wrote `chart: 132 mark(s), 5 day(s), 10 daily
  day(s)`, with no `chart` failure, and the new arrow appeared on the open chart. The full
  suite had 1283 passed and 8 failed. None of the 8 touch this change: 7 protected-set pins
  fail on `models/baseline/lstm_multitask_eurusd.keras` (last changed in `418fe94`,
  2026-10-21), and 1 is a stale DATA.md row count for `results/eurusd_h1.csv`.
