# Spec Delta

## Purpose

Shows the external models' forecasts next to the in-house models' as a second opinion, with
their agreement and their forward record. It is labelled descriptive, and it never suggests a
trade.

## ADDED Requirements

### Requirement: The view is read-only
`GET /api/second-opinion` and the `/second-opinion` page SHALL only read recorded files.
They SHALL run no model, fetch no market data, write no file, and import no model library.

#### Scenario: Page load
- **WHEN** `/second-opinion` is opened
- **THEN** no file changes and no model or market-data call is made

### Requirement: Each instrument shows its current external forecasts
For gold and for EUR/USD, the view SHALL show each model's current forecast:
- the hour in Sofia time;
- the size in percent, and in dollars at the as-of close;
- P(up), marked "not registered, not judged".

A forecast whose hour has closed SHALL NOT be shown as current. The view SHALL say when the
last forecast was made.

#### Scenario: Current gold forecast
- **WHEN** Chronos-2 logged a size of 0.18 % at close 3897.67 for the gold 11:00 bar, and that hour is open
- **THEN** the view shows 12:00–13:00 Sofia, ±0.18 % (~±7.02 $) and its P(up) with the "not registered" mark

#### Scenario: Logger stopped
- **WHEN** the newest external forecast's hour closed two hours ago
- **THEN** no forecast is shown as current, and the time of the last one is named

### Requirement: Agreement is stated by fixed rules
Agreement SHALL be computed per instrument as follows:
- **Gold size:** a model agrees with h1_lstm when its size is on the same side of the
  large-move alert threshold.
- **Gold direction:** a model agrees with h1_lstm when its P(up) is on the same side of 0.5.
- **EUR/USD size:** shown as above or below the frozen GARCH baseline for that hour.
- **EUR/USD direction:** a model agrees with h1_gbm's one-hour forecast when both are on the
  same side of 0.5.

Agreement SHALL be labelled descriptive.

#### Scenario: Gold size disagreement
- **WHEN** the alert threshold is 0.151 %, h1_lstm expects 0.14 % and Chronos-2 expects 0.18 %
- **THEN** the view says Chronos-2 disagrees with h1_lstm on size, and shows both numbers

#### Scenario: No in-house forecast for the hour
- **WHEN** h1_lstm has no current forecast
- **THEN** the gold agreement reads "nothing to compare"

### Requirement: The forward record is descriptive until scored
For each cell the view SHALL show:
- the settled-row count;
- the mean absolute error of the size forecast;
- the paired difference against the baseline, and for gold also against h1_lstm.

Each row SHALL be labelled "descriptive — not a verdict". Below 30 settled rows a cell
SHALL show only its count and "too few to say". The view SHALL show KEEP or DROP only after
the registered score is written, and then exactly as written.

#### Scenario: Early days
- **WHEN** a cell has 12 settled rows
- **THEN** it shows "12 — too few to say"

#### Scenario: After scoring
- **WHEN** the gold Chronos-2 cell has been scored
- **THEN** the view shows its recorded verdict, count and interval, unchanged

### Requirement: `/gold` carries a second-opinion card
`/gold` SHALL show a card with each model's current gold size and its agreement with
h1_lstm, linking to `/second-opinion`. It SHALL read the second-opinion endpoint. The
`/api/gold` payload SHALL NOT change.

#### Scenario: Gold page
- **WHEN** `/gold` is open and both models have a current gold forecast
- **THEN** the card shows both sizes and whether each agrees with h1_lstm on size and on direction

### Requirement: Every view carries the honesty label and no trade words
Every second-opinion text SHALL carry "external model — research, no proven
edge" until a registered verdict exists. No text SHALL contain buy, sell, long, short,
entry, stop, target or take-profit wording, in BG or EN.

#### Scenario: Any text
- **WHEN** any second-opinion text is produced in either language
- **THEN** it carries the label and none of those words

### Requirement: The view speaks the owner's language
The page and the card SHALL be available in Bulgarian and English and SHALL follow the app
shell's language switch. Bulgarian SHALL be the default.

#### Scenario: Default language
- **WHEN** `/second-opinion` is opened for the first time
- **THEN** it is in Bulgarian, with Bulgarian number formatting
