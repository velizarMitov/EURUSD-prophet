# Spec Delta

## Purpose

Defines how RTS recommendations appear on the owner's MT5 charts: a TSV file rebuilt after each pass, and a
read-only indicator. It shows today's levels and past outcomes, always attributed to RTS and never presented
as this project's signal.

## ADDED Requirements

### Requirement: Chart file
After each pass the family SHALL rebuild `Common\Files\EURUSDProphet\rts_signals.tsv`. The file SHALL hold:
- the latest spot recommendation per pair;
- the last 20 report days of base trades with their gross and net result;
- the latest Morning report positions.

Python SHALL build every visible text in Bulgarian and English. Every time SHALL be a Berlin bar label.

#### Scenario: Report arrives
- **WHEN** the 2026-11-25 spot report is ingested
- **THEN** the file holds USDJPY buy 158.98, stop 158.58, target 159.42, starting at the session open

### Requirement: Today's levels shown at once
By the owner's decision of 2026-11-28, today's recommendation SHALL be drawn as soon as it is ingested:
- an entry line, a stop line and a target line;
- a direction arrow at the entry.

Its label SHALL say that it is RTS's suggestion and unverified. Before the forward cell is scored, the corner
SHALL say that no verdict exists.

#### Scenario: Before scoring
- **WHEN** the chart shows a new recommendation and the cell has 40 of 300 forward trades
- **THEN** the label reads like `RTS: SELL 1.1009 · stop 1.1044 · target 1.0968 · непроверено` and the corner shows `40 / 300 · няма присъда`

### Requirement: Past outcomes
Each past base trade SHALL be drawn with:
- its entry and exit points joined by a line;
- the stop and target lines up to the exit;
- a label with the exit reason and the net pips and R.

It SHALL be coloured by the net result. A recommendation without a trade SHALL show its reason.

#### Scenario: Entry not reached
- **WHEN** a guerilla entry was never touched
- **THEN** the chart shows the dashed entry level with `entry not reached`

### Requirement: Read-only indicator
`mql5/Indicators/ProphetRTS.mq5` SHALL only read the chart file and draw objects. It SHALL draw only on charts
of the six spot pairs, at H1 and below. It SHALL never place, modify or close an order. It SHALL contain no
lot size and no promise of profit.

#### Scenario: Order functions
- **WHEN** the indicator source is searched for order or trade functions
- **THEN** none is found

#### Scenario: Other symbol
- **WHEN** the indicator is attached to GOLD
- **THEN** it draws nothing and its corner says which pairs it covers

### Requirement: The only write outside the repo
The chart file SHALL be the family's only write outside the repo, and nothing SHALL read it back. A failed
write SHALL not stop a pass or change any stored row.

#### Scenario: Folder not writable
- **WHEN** the MT5 Common Files folder cannot be written
- **THEN** the pass still stores its rows and reports `export failed`

### Requirement: Install and remove
An install script SHALL copy the indicator into every MT5 data folder and compile it. An uninstall script SHALL
remove only this indicator's files and `rts_signals.tsv`.

#### Scenario: Uninstall
- **WHEN** the uninstall script runs
- **THEN** ProphetNYOrb, ProphetGoldSignals, ProphetGoldNYRange and their files are untouched
