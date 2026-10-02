# Spec Delta

## Purpose

Decides whether a model has a cost-viable edge at a given horizon, using only forecasts
recorded before their outcomes existed. It covers continuous forward logging, the
pre-registration that fixes the rules in advance, and the verdicts.

## ADDED Requirements

### Requirement: Pre-registration before the first forward outcome
An immutable pre-registration SHALL be committed before the first forward row is
scored. It fixes the grid, models, estimand, cost assumption, power table, family
composition, alpha, verdict rules, refit cadence and stopping rules. Later corrections
SHALL be recorded in a separate results document, never edited into it.

#### Scenario: Registration precedes data
- **WHEN** the first forward prediction is scored
- **THEN** the pre-registration's commit is an ancestor of every commit containing scored forward rows

#### Scenario: The registration is not edited
- **WHEN** a correction to the registered protocol is needed
- **THEN** the registration file is unchanged and the correction appears in the results document with its reason

### Requirement: Family composition fixed from arithmetic before outcomes
Cells SHALL be admitted to a tested family only if their time-to-decision is at or
below the registered cap, at the family's own Bonferroni alpha, for an edge of 3 pp
over 50 %. Other cells SHALL be registered `UNDERPOWERED — NO DECISION` and spend
zero alpha. Family size SHALL never shrink after registration.

#### Scenario: Daily-cadence cells are not adjudicated
- **WHEN** the power table shows a daily-cadence cell needs more years than the cap
- **THEN** that cell is registered `UNDERPOWERED — NO DECISION`, is still logged and reported descriptively, and is excluded from the alpha count

#### Scenario: Longer sub-hourly cells are not adjudicated either
- **WHEN** the power table shows a sub-hourly cell at two hours or longer needs more years than the cap
- **THEN** that cell is registered `UNDERPOWERED — NO DECISION`, is still logged and reported descriptively, and is excluded from the alpha count

#### Scenario: A failing cell does not loosen the bar
- **WHEN** an admitted cell later becomes unrunnable
- **THEN** the family size and alpha stay as registered and the cell is recorded as unspent

### Requirement: Two self-contained families
Direction cells and volatility cells SHALL form two separate families, each with its
own Bonferroni alpha. Neither SHALL change the standing feature bar or any existing
hypothesis log. Each family SHALL have its own new registry file.

#### Scenario: Existing bars and logs untouched
- **WHEN** either family records a row
- **THEN** `feature_hypothesis_log.csv` and every other existing registry are byte-identical, and the feature bar is still 0.05/9

### Requirement: Scheduled forward logging independent of the API
A scheduled process SHALL record a prediction from every model at every horizon at
each bar close of that model's cadence, whether or not any user calls the API.
Sub-hourly models SHALL be recorded only at bar closes inside their declared session.
Each row SHALL carry: model, horizon, model-version hash, as-of bar, target bar, the
prediction, price source, the bar's spread and the UTC log time.

#### Scenario: Logging happens without a user
- **WHEN** no one opens the dashboard for a full trading day
- **THEN** that day still has one row per model per horizon per bar close

#### Scenario: Sub-hourly logging follows the session
- **WHEN** a sub-hourly bar closes outside the declared session
- **THEN** no prediction row is written for it, and no gap record is written for it either

#### Scenario: Every row is traceable
- **WHEN** any forward row is read
- **THEN** its model-version hash resolves to exactly one artifact in the study manifest

### Requirement: Sub-hourly forward rows are scored on the mid price
A sub-hourly forward row SHALL record the bar's bid close and the bar's own spread, and
SHALL be scored on the mid price built from them, matching the price definition the
study used. A row whose spread is missing SHALL be recorded and excluded from scoring.

#### Scenario: Entry and exit both use mid
- **WHEN** a sub-hourly forecast is settled
- **THEN** both the entry and the exit price are the mid built from that bar's recorded bid close and spread

#### Scenario: A missing spread is never assumed
- **WHEN** the terminal returns a sub-hourly bar with no spread
- **THEN** the row records the missing spread and is excluded from every accuracy figure and verdict

### Requirement: MT5 is the only arbiter price source
Prices used to score forward forecasts SHALL come from MT5 only. A row produced from
any other source SHALL be logged with that source and excluded from scoring. The
broker server name SHALL be recorded; the account login SHALL NOT be.

#### Scenario: Fallback prices never reach a verdict
- **WHEN** MT5 is unavailable and a row is built from Yahoo Finance
- **THEN** the row records source `yfinance` and does not enter any accuracy or verdict computation

#### Scenario: Account identity is not written
- **WHEN** forward rows are written
- **THEN** they contain the server name and no account login number

### Requirement: No retroactive predictions
A missed bar close SHALL produce a gap record, never a prediction generated later for a
past bar. Logs SHALL be append-only: a written row SHALL never be changed or deleted.

#### Scenario: Downtime leaves a gap
- **WHEN** the MT5 terminal is down for three bar closes
- **THEN** three gap records exist for those bars, and no prediction for them is ever added afterwards

#### Scenario: Settlement does not rewrite
- **WHEN** a forecast's target bar closes and the outcome is scored
- **THEN** the outcome is appended as a new settlement record, and the original prediction row is unchanged

### Requirement: Walk-forward refits during the forward window
Models SHALL be refitted on an expanding window at the registered cadence, with the
configuration frozen. Each forward prediction SHALL be made by a version fitted
strictly before its as-of bar. A refit SHALL never use data later than its own fit
time.

#### Scenario: Version precedes prediction
- **WHEN** any forward prediction is checked
- **THEN** the training-data end of the model version that made it is earlier than the prediction's as-of bar

#### Scenario: Refit cadence holds
- **WHEN** the registered cadence is monthly
- **THEN** a new version per model per horizon appears once per calendar month, and none in between

### Requirement: Verdict rule on directional accuracy
The owner's question is whether a model calls the direction (+ or -) better than a
coin. A cell SHALL be KEEP only if the one-sided lower confidence bound on accuracy, at
the family alpha, exceeds 50 %. At the registered sample size a cell that is not KEEP
SHALL be DROP. Verdicts SHALL be issued only at the registered sample size.

#### Scenario: Better than a coin is KEEP
- **WHEN** at the registered sample size the lower bound on accuracy is above 50 %
- **THEN** the verdict is KEEP

#### Scenario: Not shown better than a coin is DROP
- **WHEN** at the registered sample size the lower bound on accuracy is at or below 50 %
- **THEN** the verdict is DROP, labelled "not shown better than a coin"

### Requirement: Cost result reported beside every verdict
Every verdict SHALL be reported with the cell's spread breakeven and net result per
trade. These figures are informational and SHALL NOT change the verdict.

#### Scenario: Predictive but not cost-viable is labelled, not dropped
- **WHEN** a cell is KEEP but its accuracy is below its spread breakeven
- **THEN** the verdict stays KEEP and the cost label reads "predictive, not cost-viable"

#### Scenario: No early verdict
- **WHEN** running results look decisive before the registered sample size is reached
- **THEN** no verdict is written, and running figures are shown labelled "interim, not adjudicating"

### Requirement: Operator view of the forward log
A self-contained HTML view SHALL be regenerated from the forward log on every
logging run, showing the latest forecast per cell with the window it covers, the
running accuracy against its registered sample size, and the coverage problems.
It SHALL read the logs only and SHALL NOT be part of the serving application.

#### Scenario: The view is current without anyone opening the dashboard
- **WHEN** a logging run finishes
- **THEN** the view is rewritten from the logs, and it states the time it was generated

#### Scenario: Serving is untouched
- **WHEN** the view is generated
- **THEN** no route, template or module of the serving application has changed

#### Scenario: Times are shown in the owner's clock
- **WHEN** a forecast is displayed
- **THEN** its as-of bar and the window it covers are labelled in Europe/Sofia, not in bar-label time

### Requirement: The operator view shows direction, never a cost verdict
The view SHALL present the direction call and the running accuracy. It SHALL NOT
display spread, breakeven or net-profit figures, because the owner accounts for
trading cost themselves. Those figures SHALL remain in the logs and in the
pre-registration, where the methodology requires them.

#### Scenario: No cost arithmetic reaches the screen
- **WHEN** the view is rendered
- **THEN** it contains no spread, breakeven or net-profit figure

#### Scenario: The cost record survives
- **WHEN** the view is generated
- **THEN** every cost column in the forward log and the pre-registration is unchanged

#### Scenario: A running figure is labelled as not deciding
- **WHEN** the running accuracy is shown before the registered sample size
- **THEN** it is labelled "interim, not adjudicating" and the remaining count is shown

### Requirement: Read-only market access
The forward process SHALL only read prices and symbol metadata from the broker
terminal. No code in the change SHALL be able to send, modify or cancel an order.

#### Scenario: No order capability exists
- **WHEN** the source of the change is inspected
- **THEN** it contains no call capable of sending, modifying or cancelling a broker order, and a test asserts this

### Requirement: Per-model failure isolation in logging
If one model fails to load or predict, the logger SHALL record the failure for that
model and continue with the others. A failure SHALL never stop logging for the
remaining models.

#### Scenario: One broken model
- **WHEN** one model's artifact is missing at a bar close
- **THEN** a failure record is written for that model, and every other model's row for that bar is still written
