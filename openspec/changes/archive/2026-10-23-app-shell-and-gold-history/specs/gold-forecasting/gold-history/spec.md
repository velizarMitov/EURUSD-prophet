# Spec Delta

## Purpose

Lets the owner compare what each gold model predicted with what gold actually did, live
and in the past research, without the view ever running a model, writing a file or
presenting a description as a verdict.

## ADDED Requirements

### Requirement: The history is read-only
`GET /api/gold/history` and the History view on `/gold` SHALL only read recorded files:
- the gold logger's predictions, settlements and gaps;
- the study's saved validation and test predictions.

They SHALL run no model, fetch no market data and write no file.

#### Scenario: History opened
- **WHEN** `/api/gold/history` is requested for every source and cell
- **THEN** no file under the repository changes and no model library is imported

### Requirement: Live history pairs each forecast with its own outcome
Each live forecast SHALL be paired with the settlement that has the same key and with no
other row. The row SHALL show:
- the bar it was made from, its close and the time it was logged;
- the predicted direction, probability, return, implied price and expected move;
- for a settled forecast, the realised direction, return, exit price and absolute move.

A blank field SHALL stay blank.

#### Scenario: Settled hourly forecast
- **WHEN** h1_lstm predicted from the 16:00 bar at close 4,120.00 with p_up 0.47, and its settlement exits at 4,131.48 with realised direction up
- **THEN** the row shows predicted "down, 53 %", actual "up", the exit price 4,131.48 and a miss

#### Scenario: Blank volatility
- **WHEN** an h1_gbm forecast has no volatility value
- **THEN** its expected move is blank, and no other model's value appears in its place

### Requirement: Pending, ineligible and missed bars are visible
A forecast with no settlement yet SHALL be shown as pending. A settled forecast that was
ineligible SHALL be shown as not scored, with its outcome. A gap record SHALL be shown as
a missed bar.

#### Scenario: Newest hour
- **WHEN** the newest H1 forecast's target bar has not closed
- **THEN** it is listed as pending with no actual values and is excluded from every summary

#### Scenario: Gap
- **WHEN** the gaps log records a missed H1 bar
- **THEN** the history lists that bar as missed rather than leaving a silent hole

### Requirement: A refit is never blended
History rows and summaries SHALL be grouped by the model digest that produced them. A
summary SHALL never combine forecasts from two digests of the same cell.

#### Scenario: Two digests
- **WHEN** h1_lstm has settled rows under digest A and later under digest B
- **THEN** the view shows two separate h1_lstm summaries, labelled with each digest

### Requirement: Past research is labelled as past research
The saved validation and test predictions SHALL be offered as separate sources. Each
SHALL say:
- which block it covers;
- which fit block produced it;
- that it was scored once by the study, and that it is not live forecasting.

The test source SHALL never be offered as a basis for choosing a model.

#### Scenario: Test source selected
- **WHEN** the owner switches the History view to the test block
- **THEN** a notice says these are the one-shot test predictions from models fit on the first 80 %, recorded for reference and not used to choose anything

#### Scenario: Study row
- **WHEN** a study row has close 2,306.03, a predicted return of -0.0117 % and a realised return of -0.0217 %
- **THEN** it shows the predicted price 2,305.76 and the actual next close 2,305.53

### Requirement: Models can be compared bar by bar
For a chosen cadence and source, the view SHALL show every model's forecast for the same
as-of bar side by side, next to the one realised outcome.

#### Scenario: Same hour, three models
- **WHEN** h1_gbm, h1_lstm and h1_garch_hod all forecast from the 17:00 bar
- **THEN** one line shows all three forecasts and the single realised move

### Requirement: Summaries are descriptive and say so
Each summary SHALL describe one model under one digest and one source, over settled,
eligible rows. It SHALL show:
- the number of rows;
- the direction hit rate with a Wilson 95 % interval, compared with a coin;
- the mean absolute volatility error next to its GARCH baseline on the same rows.

A summary SHALL label itself as descriptive, not a verdict, and SHALL never label a
model KEEP or DROP.

#### Scenario: Few rows
- **WHEN** a summary has fewer than 30 settled rows
- **THEN** it shows the count and "too few to say", with no hit rate

#### Scenario: Verdict wording
- **WHEN** any summary is rendered
- **THEN** it says "descriptive — not a verdict", and the recorded verdicts appear only where `/api/gold` already shows them

### Requirement: The history can be filtered, paged and downloaded
The endpoint SHALL accept a source, a cadence, an optional cell and a date range, and it
SHALL page its rows. The view SHALL offer a CSV download of the rows it shows.

#### Scenario: Large study block
- **WHEN** the H1 test block (about 14,000 rows) is requested with a page size of 200
- **THEN** 200 rows come back with the total count, and the next page continues where it stopped

#### Scenario: Unknown source
- **WHEN** an unknown source or cell is requested
- **THEN** the endpoint answers 400 with the allowed values, not 500

### Requirement: The chart shows predicted against actual
The History view SHALL draw, without any external library:
- predicted price and actual price over time for the chosen cell;
- expected move and realised move over time for a volatility cell.

#### Scenario: Volatility cell
- **WHEN** the owner picks h1_lstm on the live source
- **THEN** the chart shows the expected move and the realised absolute move for each settled hour

### Requirement: No cost figures
The gold history SHALL show no spread, pips, P&L, profit or equity.

#### Scenario: Cost words
- **WHEN** the history payload and the History view are searched for spread, pips, P&L, profit and equity
- **THEN** none is found
