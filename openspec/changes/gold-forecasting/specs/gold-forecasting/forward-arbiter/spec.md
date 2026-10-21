# Spec Delta

## Purpose

Decides on forward data whether a gold model has statistical skill, using only forecasts
recorded before their outcomes existed, scored gross of cost because spread is
deliberately not used.

## ADDED Requirements

### Requirement: Forecasts are logged as each bar closes
An hourly scheduled job SHALL record, for every admitted and descriptive gold cell, the
frozen model's forecast for the newest closed H1 bar and the newest closed day. Each row
SHALL carry the as-of bar, model version digest, direction probability, return forecast,
volatility forecast where produced, and the logging time.

#### Scenario: New H1 bar
- **WHEN** a new H1 gold bar has closed since the last run
- **THEN** one row per H1 cell is appended for that as-of bar, stamped with the model digest

#### Scenario: Nothing new
- **WHEN** no new bar has closed since the last run
- **THEN** nothing is appended and the run reports "nothing new"

### Requirement: A forecast is written before its outcome exists
A forecast row SHALL be appended only when its target bar has not yet closed at logging
time. A row whose target bar had already closed SHALL never be written.

#### Scenario: Late run
- **WHEN** a run starts after a forecast's target bar has already closed
- **THEN** no forecast row is written for it and a gap record is written instead

### Requirement: Missed bars and failures are recorded, not hidden
Every bar the logger should have forecast but did not SHALL produce a gap record with its
reason. Every model that failed SHALL produce a failure record. Neither SHALL be filled
in later.

#### Scenario: Terminal closed for a day
- **WHEN** the MT5 terminal was unreachable for several H1 bars
- **THEN** each missed bar has one gap record and no forecast is back-filled

### Requirement: Settlement uses the same targets and eligibility as the study
Matured forecasts SHALL be settled with the realised log return, direction and absolute
return, computed exactly as the study's targets. The H1 gap-eligibility rule SHALL be
applied unchanged.

#### Scenario: Settled H1 row borders the daily break
- **WHEN** a logged H1 forecast's target bar is the first bar after the daily break
- **THEN** it is settled for the record but marked ineligible and never scored

### Requirement: Models stay frozen for the registered window
Forward forecasts SHALL come from the artifacts whose digests are recorded in the gold
artifact manifest. They SHALL not be refit during the registered forward window. A
digest mismatch SHALL stop logging for that cell with a failure record.

#### Scenario: Artifact changed on disk
- **WHEN** a model file's digest differs from the manifest
- **THEN** that cell logs a failure and no forecast, and the other cells continue

### Requirement: Forward pre-registration precedes forward scoring
A forward pre-registration SHALL be committed before any forward row is scored. It fixes:
- the forward cells and families;
- the power table and the decision cap in years;
- the alpha and verdict rules;
- the registered window length;
- the statement that scoring is gross of cost.

A dry-run phase SHALL log without scoring until it is committed.

#### Scenario: Scoring without registration
- **WHEN** forward scoring is started without a committed, unmodified forward pre-registration
- **THEN** scoring refuses to run

#### Scenario: Dry-run rows
- **WHEN** rows were logged during the dry run
- **THEN** they are reported for coverage only and are never scored

### Requirement: Forward verdicts measure skill, not profit
Forward direction cells SHALL be judged on accuracy against 50 % and the training
majority. Return cells SHALL be judged by Clark–West against zero, and volatility cells by
MAE against the GARCH baseline, each at its family alpha. No pips, P&L, Sharpe or cost-net
figure SHALL be produced.

#### Scenario: A direction cell clears
- **WHEN** a forward direction cell's accuracy CI excludes 50 % at its family alpha
- **THEN** the verdict reads "KEEP — statistical skill, gross of cost; not a trading result"

#### Scenario: No P&L anywhere
- **WHEN** any forward report is produced
- **THEN** it contains no profit, pip, Sharpe or ledger figure

### Requirement: The gold logger is independent of the EUR/USD logger
The gold logger SHALL run as its own scheduled task with its own lock, outputs and
manifest. It SHALL never read or write the EUR/USD forward-evaluation files.

#### Scenario: Both loggers run in the same minute
- **WHEN** the EUR/USD and gold loggers run concurrently
- **THEN** each completes on its own files and neither blocks or alters the other
