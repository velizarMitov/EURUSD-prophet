# fx-session/session-features Specification

## Purpose

Defines the fx-session family's pre-session features: a fixed, scale-free list describing
the Asia, London and data-hour moves, yesterday's bar, a synthetic dollar factor and the
calendar, every value known when the entry bar closes.

## Requirements

### Requirement: Fixed feature list
The feature columns SHALL be exactly the ordered list in the family config, and the config
SHALL fix that list before any model is trained. Building features with a missing or an
extra column SHALL fail.

#### Scenario: Extra column
- **WHEN** the feature builder produces a column that is not in the config list
- **THEN** the build fails and names the column

### Requirement: Known at entry
Every feature for day D SHALL be computed only from bars whose close is at or before the
close of the entry bar (the bar labelled 14:00 Berlin on D) and from earlier days. Removing
every bar after the entry bar SHALL leave all of day D's features unchanged.

#### Scenario: Truncated history
- **WHEN** the history is cut right after D's entry bar and the features are rebuilt
- **THEN** D's feature row is identical to the one built from the full history

### Requirement: Session windows on Berlin labels
The pre-session windows for day D SHALL be defined on Berlin bar labels:
- Asia: bars labelled 00:00–08:00;
- London: bars 09:00–13:00;
- the data hour: the bar labelled 14:00, the entry bar.

A window move SHALL be its last bar's close minus its first bar's open.

#### Scenario: London move
- **WHEN** the 09:00 bar opens at 1.0800 and the 13:00 bar closes at 1.0830
- **THEN** the raw London move is +0.0030 before normalisation

### Requirement: Scale-free values
Every price-based feature SHALL be divided by a volatility measure from completed data:
- moves by the ATR(14) of the last completed D1 bar;
- ranges by the mean of the same range over the previous 20 eligible days.

No feature SHALL be a raw price level.

#### Scenario: Price scaling
- **WHEN** every price in a pair's history is multiplied by 2
- **THEN** every feature value is unchanged

### Requirement: Price-structure features
The list SHALL contain, for the scored pair:
- the Asia, London and data-hour moves;
- the London range ratio, the data-hour range ratio, and the entry close's position in the London range (0 at the low, 1 at the high);
- yesterday's move, range and close location in its range;
- the move from yesterday's close to the entry close;
- whether today's high so far broke yesterday's high, and whether today's low broke yesterday's low;
- ATR(14) of D1 as a fraction of the entry close.

#### Scenario: Broken high
- **WHEN** a bar labelled 03:00 on D trades above the previous trading day's high
- **THEN** the broke-previous-high flag for D is 1

### Requirement: Synthetic dollar factor
The dollar factor for day D SHALL be the mean, over EURUSD, GBPUSD and AUDUSD, of each
pair's normalised move from D's 00:00 bar open to the entry close. A second dollar factor
SHALL do the same for the data-hour move. The pair's residual SHALL be its own normalised
move minus the dollar factor.

#### Scenario: Common dollar move
- **WHEN** all three pairs rise by 0.5 ATR before entry
- **THEN** the dollar factor is 0.5 and each pair's residual is 0

### Requirement: Calendar flags
The list SHALL contain the day of week (0 for Monday to 4 for Friday) and three flags:
- NFP: 1 on the first Friday of the month, a declared approximation of the US jobs-report
  schedule;
- FOMC: 1 on a scheduled statement day in the project's existing FOMC date file;
- DST mismatch: 1 when the New York and Berlin UTC offsets differ from their usual
  6-hour gap.

#### Scenario: DST mismatch flag
- **WHEN** D is 2026-03-10, after the US clock change and before the European one
- **THEN** the DST-mismatch flag is 1

#### Scenario: Unscheduled FOMC action
- **WHEN** a Fed action on D was unscheduled
- **THEN** the FOMC flag for D is 0

### Requirement: Rows with incomplete inputs
A day SHALL get no feature row when any input is missing: a pair's pre-session bar, the
previous trading day, or the 20-day range history. Such days SHALL be excluded from
training and scoring and counted by reason.

#### Scenario: Warm-up days
- **WHEN** fewer than 20 eligible days precede D
- **THEN** D has no feature row and is counted as warm-up
