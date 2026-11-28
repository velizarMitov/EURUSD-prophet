# Spec Delta

## Purpose

Defines how the gold NY-range forecast is shown in MT5: a TSV file rebuilt by the passes, and a
read-only indicator that draws the forecast size and the realised range, never a direction or a
trade.

## ADDED Requirements

### Requirement: Chart file
After each pass, the family SHALL rebuild `Common\Files\EURUSDProphet\gold_ny_range.tsv` from its
logs. The file SHALL hold the last 60 forecast days. Each row SHALL hold:
- the date and the status;
- the 14:00 close and the forecast range in price units;
- once settled, the realised session high and low.

Python SHALL build all wording in the file.

#### Scenario: Rebuilt after settlement
- **WHEN** the settlement pass settles a day
- **THEN** the file's row for that day carries the realised high and low

### Requirement: The only write outside the repo
The chart file SHALL be the family's only write outside the repo. Nothing in the family SHALL read
it back. A failed write SHALL not stop a pass or change a log.

#### Scenario: Folder missing
- **WHEN** the MT5 Common Files folder cannot be written
- **THEN** the pass still logs its rows and reports `export failed`

### Requirement: Read-only indicator
`mql5/Indicators/ProphetGoldNYRange.mq5` SHALL only read the chart file and draw objects. It SHALL
never place, modify or close an order. It SHALL draw only on the `GOLD` symbol, at H1 and below.

#### Scenario: Other symbol
- **WHEN** the indicator is attached to EURUSD
- **THEN** it draws nothing and its corner says it is for GOLD only

### Requirement: Size only, no direction
Each forecast day SHALL be drawn from 15:00 to 22:00 Berlin as a band centred on the 14:00 close,
with a total height equal to the forecast range. After settlement, a dotted box SHALL show the
realised high and low. Nothing SHALL show a direction, an entry, a stop, a target, a lot or a
promise.

#### Scenario: Wording check
- **WHEN** the chart file and the string literals in the indicator source are searched for trade words
- **THEN** no word such as buy, sell, long, short, entry, stop, target, lot or profit is found

### Requirement: No verdict on the chart
The indicator's corner SHALL show the forward day count out of 60 and SHALL say that no verdict
exists until the forward test is scored. After scoring, it SHALL show the registry verdict as
written.

#### Scenario: Before scoring
- **WHEN** the forward test has 25 forward days and no score
- **THEN** the corner shows 25 / 60 and states that there is no verdict yet

### Requirement: Install and remove
An install script SHALL copy the indicator into the MT5 data folder and compile it. An uninstall
script SHALL remove only this indicator's files and `gold_ny_range.tsv`.

#### Scenario: Uninstall
- **WHEN** the uninstall script runs
- **THEN** ProphetNYOrb, ProphetGoldSignals and their files are untouched
