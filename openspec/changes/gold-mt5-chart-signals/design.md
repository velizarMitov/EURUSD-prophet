# Design

## Context

- The gold logger (`src/gold/forward_logger.py`, task `EURUSDProphet-GoldLogger`, hourly
  at :03) writes everything the chart needs to `results/gold/forward/`:
  - `predictions.csv`: one h1_lstm row per as-of bar, with `p_up`, `vol_pct`,
    `as_of_close` and the label-clock `as_of_bar`;
  - `settlements.csv`: `realised_dir` (1 up, 0 down, empty on a zero move),
    `realised_ret_pct` and `eligible`, joined by `key`;
  - `alerts.csv` and `alert_settlements.csv`: the alert rows and their realised size.
- `/gold` derives direction as `up if p_up > 0.5 else down` (`summary.current_forecast`).
  It names hours in Sofia time. A forecast for as-of bar *t* covers the bar *t + 1 h*.
- Bar labels are the logger's broker clock, `Europe/Berlin`, stored with a misleading
  `+00:00` suffix. Sofia is Berlin + 1 h all year.
- The PC has three MT5 terminals. All of them share
  `%APPDATA%\MetaQuotes\Terminal\Common\Files`, and each install folder has
  `metaeditor64.exe`.
- The Python `MetaTrader5` package reads data and trades. It cannot draw chart objects,
  so drawing needs MQL5 code that runs inside the terminal.
- The gold isolation contract (`src/gold/__init__.py`) allows writes only under
  `results/gold/` and `research_models/gold/`.

## Goals / Non-Goals

**Goals:**
- Python owns every word the chart shows, so the wording rules are tested in pytest. The
  MQL5 code only reads and draws.
- The chart works even when the FastAPI app is not running: the logger alone keeps it
  fresh.
- A failure on the chart side never costs a logger pass. A failed logger pass leaves the
  chart showing its last marks plus a warning.

**Non-Goals:**
- No expert advisor, trading call, order, alert-to-trade bridge, sound or push
  notification from MT5. The Windows toast already exists.
- No tested threshold. The probability filter (D12) is a display choice the owner sets.
  It is never registered, scored or reported as a rule.
- No support for portable-mode terminals in the installer. Those are installed by hand,
  as the RUNBOOK will describe.
- No change to what the logger forecasts, settles or alerts.

## Decisions

### D1. A file bridge through MT5 `Common\Files`
The logger writes one file to `Common\Files\EURUSDProphet\gold_signals.tsv`. The indicator
reads it with `FILE_COMMON`.

Alternatives considered:
- **Expert advisor + `WebRequest` to the API.** It needs the web app running, a URL
  whitelist in every terminal and the chart's single EA slot. EAs can also trade, which
  widens what this tool could do.
- **DLL import to read the repo directly.** It needs "Allow DLL imports", which is a
  security downgrade for a real-money terminal.
- **A junction from `Common\Files` into `results/`.** The MQL5 sandbox's handling of
  junctions is undocumented, and it would expose the repo to every MT5 program.

The shared folder serves all three terminals from one file. Its path is
`%APPDATA%\MetaQuotes\Terminal\Common\Files`, which can be overridden with the
`MT5_COMMON_FILES` environment variable for tests and odd setups. This is the one named
exception added to the isolation contract. Nothing in the repo reads the file back.

### D2. Python writes the texts, MQL5 only places them
`src/gold/mt5_export.py` puts the finished texts in the file, in both languages:
- each mark's short text;
- each mark's hover description;
- each forecast's panel line;
- the Sofia hour;
- the data time;
- the 5-day count line.

Numbers are formatted per language (BG `0,14 %`, EN `0.14 %`). The MQL5 code holds only
the fixed status strings, in BG and EN: data old, data missing, not a gold chart,
timeframe too high, no current forecast, clock shift and unknown format.

This keeps the wording rules ("no trade words", "direction unknown", the research label)
testable in pytest. A pytest scan covers the few MQL5 string literals.

### D3. File format, version 2
Version 1 had only h1_lstm and a precomputed count. Version 2 adds the daily cells and
the count templates the filter needs. An indicator built for version 1 shows "unknown
format, reinstall" instead of misreading the file.

The file is UTF-8, tab-separated and LF-terminated:
- **Line 1** is `#meta`, followed by `key=value` fields:
  - `version=2`;
  - `generated_utc`;
  - `label_tz=Europe/Berlin`;
  - `label_utc_offset_min`, the Berlin offset at export time;
  - `written_bg` and `written_en`;
  - `count_h1_*` and `count_d1_*`: count templates in BG and EN. The indicator fills
    `{min}`, `{r_<cell>}` and `{t_<cell>}` for the arrows it shows (D12);
  - `filter_*`: the "display choice, not a tested rule" line, with `{min}`;
  - `research_bg` and `research_en`.
- **Line 2** is the header.
- **Each row** has these fields:
  - `kind` (`forecast` or `alert`), `key`, `cell` and `cadence` (`H1` or `D1`);
  - `hour_label`, the bar the mark sits on, on the label clock. A daily row gives the
    forecast day at 00:00;
  - `hour_end_label`;
  - `direction` (`up`, `down` or empty), `prob_pct`, `vol_pct` and `close`;
  - `outcome` (`open`, `right`, `wrong`, `void` or `sized`);
  - `text_bg`, `text_en`, `tip_bg`, `tip_en`, `panel_bg`, `panel_en`;
  - `hour_sofia_bg` and `hour_sofia_en`.

Times are written as `YYYY.MM.DD HH:MM` so MQL5 `StringToTime` parses them. Field values
never contain a tab. Hover line breaks are written as the two characters `\n`, which the
indicator turns into real line breaks.

The indicator reads the whole file in one call with `FILE_READ|FILE_BIN|FILE_SHARE_READ|`
`FILE_SHARE_WRITE|FILE_COMMON` and decodes it with `CharArrayToString(…, CP_UTF8)`. It then
splits lines and fields with `StringSplit`. A binary read is used instead of `FILE_TXT`
because it does not depend on how text mode treats LF-only line ends, and it holds the
file open for the shortest time. An unknown `version`, or a missing column, draws nothing
and shows "unknown format, reinstall".

### D4. Atomic replace with retries
The export writes `gold_signals.tsv.tmp` in the same folder, then calls `os.replace`. On
`PermissionError` it retries 5 times, 200 ms apart, because the indicator may hold the file
open for a few milliseconds during a read. If the replace still fails, the export raises
and the logger records the failure.

### D5. Where the export runs
`forward_logger.run_once` gains `log_chart(cfg, now_iso)`. It runs after `log_alerts`, so
this pass's forecast, settlements and alerts are already on disk. It is wrapped like
`log_ranges`: an exception appends a `failures.csv` row with cell `chart` and returns
`failed: <Type>`.

The status key is `chart`, because `mt5` already means "MT5 reachable". The health note
gains `chart: …` right after `alerts: …`, so the 300-character truncation cuts the later
fields first. The CLI `python -m src.gold.mt5_export` rebuilds the file on demand. It
reads the logs, writes the file and prints a one-line summary.

The export runs on every pass, including a pass that forecast nothing new: settlements
and the passage of time still change what the chart shows.

### D6. Selecting and scoring the rows
- **Forecast rows:** `predictions.csv` rows with `cell == 'h1_lstm'` and
  `cadence == 'H1'`. Both phases are included, because `/gold` shows dry-run rows too.
  `predictions_v2.csv` is never read.
- **Mark hour:** as-of bar + 1 h.
- **5 trading days:** the 5 most recent distinct **Sofia dates of the mark hour**. Using
  Sofia dates stops the Sunday 23:00 weekly-open bar from creating a phantom Sunday.
- **Direction:** `up if p_up > 0.5 else down`. A test checks that this matches
  `summary.current_forecast` on the same rows, so the chart and `/gold` cannot drift
  apart.
- **Probability:** the larger of `p_up` and `1 − p_up`, as on `/gold`.
- **Outcome:**
  - no settlement row: `open`;
  - `eligible` false or `realised_dir` empty: `void` ("not counted");
  - otherwise `right` or `wrong`, carrying `realised_ret_pct`.
- **Alerts:** the `alerts.csv` rows inside the same window. Each takes its close and
  expected move from the h1_lstm prediction row with the same `key`. When that row is
  missing, the alert is shown in percent only. A settled alert shows its realised
  absolute move.
- **Dollar size:** `vol_pct / 100 × close`, the same formula as the `/gold` banner.
- **Daily rows:** `predictions.csv` rows of `d1_lstm` and `d1_gbm` with cadence `D1`.
  - The mark day is the settlement's `exit_bar` once settled, otherwise
    `as_of + BDay(1)`, as `/gold` dates a daily forecast.
  - The window is the last 20 distinct mark days.
  - The short text carries the cell tag (`lstm 53 %`, `gbm 78 % · wrong −0.45 %`),
    because a day has two arrows.
  - The hover description names the day, and the expected move where the cell has one.
    d1_gbm has none.
- **Count:** right versus wrong, per cell, over the arrows the chart shows. `void` and
  `open` are excluded. The count is labelled descriptive and not a test. Python writes
  the templates, and the indicator fills in the numbers because only it knows the
  filter (D12).

### D7. Placing marks on the chart's clock
- The rows carry label-clock times, and `#meta` carries Berlin's UTC offset at export.
- The indicator computes the chart's offset as `round((TimeTradeServer() − TimeGMT()) /
  900) × 900`.
- It places each mark at `hour_label + (chart_offset − label_offset)`.
- A non-zero shift is named in the panel, e.g. "clock shift +1 h".

Alternatives considered:
- **UTC plus the current chart offset.** Every mark from before a DST switch would sit an
  hour off for the next 5 days, even on the logger's own broker.
- **Matching the as-of close against the chart's bars.** Gold quotes differ between
  brokers by cents to dollars, so the match is ambiguous.

The label approach is exact whenever the chart broker's DST rule is Berlin's, which
includes the logger's own terminal.

### D8. Drawing
- **Indicator setup:** `#property indicator_chart_window` and `indicator_plots 0`. It
  has no buffers.
- **Object names:** every object starts with `PGS_`. The indicator deletes only its own
  objects, through `ObjectsDeleteAll(0, "PGS_")`, on each redraw and in `OnDeinit`.
- **Arrows:** `OBJ_ARROW` with Wingdings 233 (up) and 234 (down). An up arrow sits under
  the hour's H1 low with `ANCHOR_TOP`. A down arrow sits over the hour's H1 high with
  `ANCHOR_BOTTOM`. The low and high come from `iLow`/`iHigh` on `PERIOD_H1` at
  `iBarShift`, so the arrow lands on the first sub-bar of the hour on M1–M30. The gap is
  15 % of that hour's range, and at least a few points.
  - If the chart has no bar for that hour, the arrow is placed at the row's `close`.
- **Alert marker:** a two-headed vertical Wingdings glyph. The candidate is code 244; it
  is checked on the chart during implementation. The marker sits on the side of the
  candle that the hour's arrow leaves free: above the candle, or below it when the hour
  has a down arrow. The two marks never stack.
- **Text beside each mark:** an `OBJ_TEXT` in Segoe UI 8 pt, turned 90° so it runs away
  from the candle. Horizontal texts such as "52 % · right +0.11 %" are about 100 px wide,
  while H1 bars are about 10 px apart, so neighbouring hours would overlap. The outcome
  colour is green for right, red for wrong, grey for not counted, and the mark's own
  colour while the hour is open.
- **Hover description:** `OBJPROP_TOOLTIP` on both the mark and its text.
- **Colours:** arrows are coloured by direction. The defaults are DodgerBlue for up,
  OrangeRed for down and Gold for the alert. All colours are inputs.
- **Panel:** `OBJ_LABEL` lines in a corner chosen by an input. The default is the upper
  right, so the one-click trading panel is not covered.
- **Inputs:** `Language` (BG/EN, default BG), the colours, the panel corner and the
  font size. Also the starting probability filter (default 50) and the d1_gbm colour.

### D9. Refresh and time-based state
`EventSetTimer(30)`. Each timer tick:
- re-reads the file when `FILE_MODIFY_DATE` has changed;
- re-evaluates the time-based states (current forecast, open alert, staleness) even when
  the file has not changed;
- re-anchors the arrows on the live hour as its low and high move.

`OnCalculate` only triggers a redraw on a new bar.

- **Current forecast:** the newest forecast row whose shifted `hour_end_label` is later
  than `TimeTradeServer()`.
- **Open alert:** the same rule, applied to alert rows.
- **Stale data:** the file is more than 2 h old and the symbol's last tick
  (`SYMBOL_TIME`) is under 10 minutes old, which means gold is trading. A weekend
  therefore never warns.

### D10. The indicator cannot trade
MT5 already forbids trade functions in indicators. A pytest scan adds a second guard. It
fails on any of these in the `.mq5` source:
- `OrderSend`, `OrderSendAsync`, `CTrade`, `#include <Trade`, `PositionOpen`;
- `WebRequest`, `SocketCreate`;
- `#import`.

The scan also checks the MQL5 string literals for the trade-word list, the same list the
pytest export tests use.

### D11. Installer and uninstaller
`scripts/gold/install_mt5_indicator.ps1` works as follows:
1. It lists `%APPDATA%\MetaQuotes\Terminal\*\origin.txt`. Each entry maps a data folder
   to an install folder.
2. For each terminal, it copies `mql5/Indicators/ProphetGoldSignals.mq5` to
   `<data>\MQL5\Indicators\EURUSDProphet\`.
3. It runs `<install>\metaeditor64.exe /compile:"<mq5>" /log:"<log>"`.
4. It reports the terminal as compiled when the `.ex5` exists and the log reports
   0 errors. Otherwise it prints the log. MetaEditor's exit code is not reliable.
5. It then runs `.venv\Scripts\python.exe -m src.gold.mt5_export` once.

`uninstall_mt5_indicator.ps1` removes only the `MQL5\Indicators\EURUSDProphet\` folder from
each terminal and `Common\Files\EURUSDProphet\`. The scripts follow the style of
`scripts/gold/install_task.ps1`.

### D12. The probability filter
- **Buttons.** Four `OBJ_BUTTON`s show ≥50, ≥52, ≥54 and ≥56 % as the panel's first row.
  The active one is pressed. A click sets the filter and redraws.
- **Persistence.** The choice is kept in a terminal global variable named after the chart
  ID, so a timeframe change, which reinitialises the indicator, keeps it. A changed
  `InpMinProbability` setting wins over the stored choice.
- **What it hides.** Direction arrows, H1 and D1, whose `prob_pct` is below the filter.
  Alerts have no probability and always stay. The panel's current-forecast lines stay
  unfiltered, because they are the status.
- **Count.** The indicator fills the Python template from the rows it shows.
- **Not a tested rule.** A selective rule like this was registered once, as round two's
  `h1_selective`, and was DROP on validation. The panel says "display choice, not a
  tested rule" whenever the filter is above 50.

Alternative considered: only an indicator setting. Changing it means opening
Properties → Inputs each time, and the owner asked for a quick switch.

### D13. Daily marks
- **Where.** They are drawn on D1 only. Hourly marks and alerts are drawn at H1 and below.
  H2–H12, W1 and MN draw nothing and say so.
- **Placement.** A daily row is placed with
  `iBarShift(PERIOD_D1, day 00:00 + 12 h, exact=false)`, which gives the chart's candle
  for that calendar day. A broker clock a few hours off Berlin still lands on the same
  day, so no clock shift is applied.
- **Glyphs.** d1_lstm uses the solid arrows (233/234) in the up/down colours. d1_gbm uses
  the hollow arrows (241/242) in its own colour (default MediumOrchid).
- **Same side.** When both cells sit on the same side of a candle, their arrows stack
  next to each other. One vertical text then joins the shown cells' texts with " | ",
  and its hover joins both descriptions. Separate texts would put the second arrow
  about 120 px from the candle.

## Risks / Trade-offs

- **The owner reads the filter as an edge.** The ≥54 % arrows looked strong in the first
  ten dry-run days (20 right of 27), but that threshold was chosen after looking at
  those days. → The panel names the filter as a display choice and counts only the arrows
  shown, so the claim can be watched on new data. No threshold enters a registry.

- **Arrows on a live trading chart invite trading on a direction with no trading edge.**
  → Every hover description and the panel carry "research — no proven edge, gross of
  cost". The texts contain no trade words. There is no confidence filter that could look
  like a strategy. The indicator cannot trade. The per-arrow right/wrong marks and the
  5-day count let the owner see how often it is wrong.
- **The chart broker runs on a US-DST clock.** In the March and late-October mismatch
  weeks, the difference from Berlin changes, and marks from before the switch can sit
  one hour off within the 5-day window. → Each description names the Sofia hour, and the
  panel names the current shift.
- **The chart broker's prices differ from the logger's.** → Marks are anchored to the
  chart's own bars. The `$` sizes and closes come from the logger's broker, and the
  description says so ("at close 3897.45").
- **MT5 holds the file during a replace.** → D4 retries. A final failure is logged, and the
  indicator keeps its last marks.
- **The PC sleeps and the logger stops.** → The panel shows a staleness warning
  (D9). Logger health on `/gold` is unchanged.
- **`TimeGMT()` depends on the PC clock and time zone.** → The panel shows the computed
  shift, so a wrong PC clock is visible as an unexpected shift.
- **The glyph or font renders differently.** → Glyph codes and fonts are verified on a
  live chart during implementation (tasks).

## Migration Plan

1. Merge the code. The next logger pass starts writing the file, because the scheduled
   task runs the module fresh each hour.
2. Run `scripts/gold/install_mt5_indicator.ps1`, then attach "ProphetGoldSignals" to an
   XAUUSD/GOLD H1 chart in each terminal the owner uses.
3. **Rollback:** run `uninstall_mt5_indicator.ps1` and revert the `log_chart` call. The
   forward logs are untouched either way.
