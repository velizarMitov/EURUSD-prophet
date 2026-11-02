# Spec Delta

## Purpose

Shows the gold forward signals on the owner's MetaTrader 5 chart, where trades are watched,
with the same honesty labels as `/gold`. It never places or suggests a trade.

## ADDED Requirements

### Requirement: Only the chosen cells are drawn
The chart SHALL show:
- the registered h1_lstm hourly forecasts;
- the registered d1_lstm and d1_gbm daily forecasts;
- the large-move alerts.

It SHALL NOT draw h1_gbm, the GARCH baselines, round-two cells or shadow-version rows.

#### Scenario: Several cells forecast the same hour
- **WHEN** h1_gbm, h1_lstm and h1_garch_hod each logged a forecast for the as-of bar 2026-11-02 09:00
- **THEN** the chart shows one direction arrow for that hour, from h1_lstm

#### Scenario: Shadow row for the same hour
- **WHEN** predictions_v2 holds an h1_lstm shadow-version row for the same as-of bar
- **THEN** it does not appear on the chart

#### Scenario: Both daily cells forecast the same day
- **WHEN** d1_lstm and d1_gbm each logged a forecast for the as-of day 2026-10-29
- **THEN** the D1 chart shows two arrows for the next trading day, one per cell

### Requirement: A forecast is an arrow on the hour it forecasts
Each h1_lstm forecast SHALL be drawn as an up or down arrow on the hour it forecasts: the
bar after its as-of bar. The direction SHALL follow the `/gold` rule: up when P(up) is
above 0.5, otherwise down. An up arrow SHALL sit below that hour's candle and a down arrow
above it. Every forecast at or above the chart's probability filter SHALL be drawn.

#### Scenario: Up forecast
- **WHEN** h1_lstm logs P(up) 0.5317 for the as-of bar 2026-11-02 09:00 (label clock)
- **THEN** an up arrow appears under the 10:00 bar, with the text "53 %"

#### Scenario: Near-even forecast with the filter at 50
- **WHEN** h1_lstm logs P(up) 0.4994 and the filter is at 50
- **THEN** a down arrow is drawn with the probability 50 %

### Requirement: A daily forecast is an arrow on the day it forecasts
Each d1_lstm and d1_gbm forecast SHALL be drawn on a D1 chart as an up or down arrow on
the day it forecasts. That is the settled exit bar once settled, and the next business day
before that. The direction SHALL follow the same rule as the hourly arrows. d1_lstm SHALL be
drawn as a solid arrow and d1_gbm as a hollow arrow, each in its own colours. When both
arrows sit on the same side of a day's candle, both SHALL stay visible.

#### Scenario: Daily forecasts on Thursday
- **WHEN** d1_lstm logs P(up) 0.4777 and d1_gbm logs P(up) 0.7782 for the as-of day 2026-10-29
- **THEN** the D1 chart shows a solid down arrow (52 %) above the 2026-10-30 candle and a hollow up arrow (78 %) below it

#### Scenario: Same side
- **WHEN** both daily cells point up for the same day
- **THEN** both arrows are visible below that candle, and neither covers the other

### Requirement: The arrows can be filtered by probability
The chart SHALL offer buttons at 50, 52, 54 and 56 % that hide every direction arrow
below the chosen probability. 50 shows all. The starting value SHALL be an indicator
setting, and the chart SHALL keep its choice when the timeframe changes. Alert markers
SHALL never be filtered.

#### Scenario: Filter at 54
- **WHEN** the owner clicks the 54 % button on an XAUUSD H1 chart
- **THEN** only the arrows with probability 54 % or more stay on the chart, and every alert marker stays

#### Scenario: Timeframe change
- **WHEN** the filter is at 54 and the owner switches the chart from H1 to D1 and back
- **THEN** the filter is still at 54

### Requirement: The filter is shown as a display choice
While the filter is above 50, the panel SHALL name it and SHALL say it is a display choice,
not a tested rule. The right/wrong count SHALL cover only the arrows shown.

#### Scenario: Count under the filter
- **WHEN** the filter is at 54 and 27 settled arrows of the last 5 days are at 54 % or more, 20 of them right
- **THEN** the panel counts 20 right of 27 at ≥54 %, and names the filter as a display choice

### Requirement: An alert is a marker without direction
Each large-move alert SHALL be drawn on the hour it covers as a two-headed vertical marker
that never points one way, apart from the direction arrows, in its own colour. Its text
SHALL give the expected size in percent and, when the alert's close is known, in dollars
at that close.

#### Scenario: Alert on the 16:00 as-of bar
- **WHEN** an alert of ±0.1954 % is logged for the as-of bar 2026-10-27 16:00 (label clock) with close 4063.36
- **THEN** a two-headed marker appears on the 17:00 bar with the text "±0.20 % (~±7.94 $)" in EN

#### Scenario: Alert and arrow on one hour
- **WHEN** the same hour has an h1_lstm arrow and an alert
- **THEN** both are visible and neither covers the other

### Requirement: Every mark has a description on hover
Hovering a mark SHALL show its description:
- the hour in Sofia time, or the day for a daily arrow, never on the bar-label clock;
- the cell and the probability for an arrow, or the expected size for an alert;
- the expected move in % and $ where the cell forecasts one;
- the outcome once settled;
- the label "research — no proven edge, gross of cost".

#### Scenario: Hover an arrow
- **WHEN** the owner hovers the arrow for the as-of bar 2026-11-02 09:00 with expected move 0.1446 % at close 3897.45
- **THEN** the description names 11:00–12:00 Sofia time, up 53 %, about ±0.14 % (~±5.64 $), and the research label

### Requirement: Chart text never names a trade
No text the chart shows SHALL contain buy, sell, long, short, entry, stop, target or
take-profit wording, in BG or EN. This covers mark texts, descriptions, buttons and the
panel. Alert texts SHALL also say that the direction is unknown.

#### Scenario: Any chart text
- **WHEN** any mark text, description, button or panel line is produced in either language
- **THEN** it contains none of those words

#### Scenario: Alert description
- **WHEN** an alert's description is shown
- **THEN** it says the direction is unknown

### Requirement: Settled marks show their outcome
Once the forecast hour or day has settled, the mark SHALL show its outcome:
- an arrow is right when the realised direction matches it, otherwise wrong, with the
  realised move in percent;
- an alert shows the realised absolute move;
- a forecast the forward log marks ineligible, or with no realised direction, shows "not
  counted";
- an unsettled forecast shows that it is open.

#### Scenario: Wrong arrow
- **WHEN** an up arrow's hour settles with a realised move of −0.08 %
- **THEN** its text shows it was wrong and −0.08 %

#### Scenario: Ineligible hour
- **WHEN** the settlement row for an arrow's hour has `eligible` false
- **THEN** the mark shows "not counted" and is left out of the count

### Requirement: The chart keeps a fixed history
The chart SHALL keep:
- the hourly marks and alerts of the last 5 trading days that have an h1_lstm forecast;
- the daily marks of the last 20 forecast days.

It SHALL remove older marks. A day counts as a trading day when it has at least one such
forecast.

#### Scenario: Monday morning
- **WHEN** the chart is refreshed on Monday 2026-11-02 at 10:05 Sofia time
- **THEN** it shows Monday's hourly marks and those of the four previous trading days, and nothing from the weekend or earlier

### Requirement: A panel shows the current state
A corner panel SHALL show, for the chart's own cadence:
- the current forecast (hourly, or one per daily cell) with its hour or day, or that
  none is current;
- any open alert, on hourly charts;
- the right/wrong count over the arrows shown, marked descriptive and not a test;
- the time the data was written, in Sofia time;
- the research label.

#### Scenario: Open alert
- **WHEN** an alert's hour is still open
- **THEN** the panel shows the alert's size and that the direction is unknown

#### Scenario: No current forecast
- **WHEN** the newest h1_lstm forecast's hour has closed
- **THEN** the panel says there is no current forecast and names the last one's hour

### Requirement: Marks land on the chart's own clock
The hourly marks SHALL be placed on the chart's server clock. Forward-log times are on
the gold logger's broker clock (`Europe/Berlin`). When the chart's broker runs on another
clock, the hourly marks SHALL be shifted by the difference, and the panel SHALL name the
shift. Daily marks SHALL sit on the chart's candle for the forecast day.

#### Scenario: Same clock
- **WHEN** the chart's server clock is the logger's `Europe/Berlin` clock
- **THEN** the arrow for the as-of bar 2026-11-02 09:00 sits on the chart's 10:00 bar, with no shift shown

#### Scenario: Broker one hour ahead
- **WHEN** the chart's server clock is one hour ahead of `Europe/Berlin`
- **THEN** that arrow sits on the chart's 11:00 bar, and the panel shows a +1 h clock shift

### Requirement: Marks appear only where they mean something
Marks SHALL be drawn only on a gold chart, `GOLD` or `XAUUSD` with any broker suffix:
- hourly marks and alerts at H1 or a lower timeframe;
- daily marks on D1.

On any other chart or timeframe the indicator SHALL draw no marks and SHALL say why in
the panel.

#### Scenario: EUR/USD chart
- **WHEN** the indicator is attached to a EURUSD chart
- **THEN** no marks are drawn and the panel says the chart is not gold

#### Scenario: Daily chart
- **WHEN** the indicator is on an XAUUSD D1 chart
- **THEN** the d1_lstm and d1_gbm arrows are drawn, and no hourly mark or alert is

#### Scenario: H4 chart
- **WHEN** the indicator is on an XAUUSD H4 chart
- **THEN** no marks are drawn and the panel names H1 or lower, and D1

#### Scenario: M15 chart
- **WHEN** the indicator is on an XAUUSD.a M15 chart
- **THEN** each hourly arrow sits on the first M15 bar of its hour

### Requirement: The chart warns when its data is old or missing
The panel SHALL warn when the signal data is more than two hours old while gold is
trading. It SHALL say so when the data file is missing or unreadable. In both cases the
marks already drawn SHALL stay, and the chart SHALL NOT crash.

#### Scenario: Logger stopped
- **WHEN** the newest data was written at 07:03 Sofia time and gold is trading at 10:30 Sofia time
- **THEN** the panel warns that the data is old and names 07:03

#### Scenario: Weekend
- **WHEN** the market is closed for the weekend
- **THEN** no staleness warning is shown

#### Scenario: Not installed yet
- **WHEN** the data file does not exist
- **THEN** the panel says no gold data was found and how to create it

### Requirement: Each logger pass refreshes the chart data
Each gold logger pass SHALL rebuild the chart data from the forward logs after the alerts
step. A failure here SHALL be recorded in `failures.csv` and the health note, and SHALL NOT
cost the pass. A command SHALL rebuild the data on demand. A reader SHALL never see a
half-written file. An open chart SHALL pick up new data within a minute without being
reattached.

#### Scenario: Hourly pass
- **WHEN** the 10:03 Sofia time pass logs a new h1_lstm forecast
- **THEN** the arrow appears on an open chart by 10:04, without reattaching the indicator

#### Scenario: Export fails
- **WHEN** the export folder cannot be written
- **THEN** the forecasts, settlements and alerts of that pass are still written, and the failure is recorded

### Requirement: The chart data is the only gold write outside the repo
The chart data SHALL be one file in the MT5 shared files folder. It SHALL be the only gold
write outside `results/gold/` and `research_models/gold/`. Nothing in the repo SHALL read it
back.

#### Scenario: Export location
- **WHEN** the export runs
- **THEN** it writes one file under the MT5 shared `Common\Files\EURUSDProphet\` folder and nothing else outside the repo

### Requirement: The indicator never trades
The chart tool SHALL be an MT5 indicator, not an expert advisor. It SHALL contain no
trading, order, position, network or DLL calls. It SHALL only read its data file, draw,
and remember the chart's filter choice.

#### Scenario: Source check
- **WHEN** the indicator source is scanned
- **THEN** it contains no order, trade, position, web-request or DLL-import call

### Requirement: The indicator installs into every terminal
An installer SHALL copy the indicator into every MT5 terminal found on the PC, compile it
there and report the result per terminal. It SHALL then rebuild the chart data once. An
uninstaller SHALL remove what the installer added. The installer SHALL not change any other
file of a terminal.

#### Scenario: Three terminals
- **WHEN** the installer runs on a PC with three MT5 terminals
- **THEN** it reports one compiled-or-failed line per terminal, with the compiler log for a failure

#### Scenario: Uninstall
- **WHEN** the uninstaller runs
- **THEN** the indicator files and the chart data file are removed, and nothing else

### Requirement: The chart speaks the owner's language
Chart texts SHALL be available in Bulgarian and English. Bulgarian SHALL be the default,
and the language SHALL be an indicator setting.

#### Scenario: Default language
- **WHEN** the indicator is attached with default settings
- **THEN** the texts, descriptions and panel are in Bulgarian, with Bulgarian number formatting (0,14 %)
