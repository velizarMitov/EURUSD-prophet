# Spec Delta

## Purpose

Defines the two daily passes of the gold NY-range forward test: the forecast made at 16:00 Sofia
from bars labelled 14:00 Berlin or earlier, and the settlement after the session. It also covers
their append-only logs and their scheduled tasks.

## ADDED Requirements

### Requirement: Forecast pass
After the bar labelled 14:00 Berlin has closed, the forecast pass SHALL:
- build gold_bbnr's fourteen NY-range features for the day from the live caches;
- apply the pinned HAR-X and HAR coefficients;
- log one row with `p_harx`, `p_har`, the forecast range in price units, the 14:00 close and the
  dollar-index contract used.

#### Scenario: Normal day
- **WHEN** the pass runs at 16:08 Sofia on a weekday with all inputs present
- **THEN** one forecast row with status `ok` is logged for the day

### Requirement: Causality of the forecast
The forecast SHALL use only gold and dollar-index bars labelled 14:00 Berlin or earlier on the day,
and targets of earlier days only. A pass that starts before the bar labelled 14:00 has closed SHALL
log nothing.

#### Scenario: Early start
- **WHEN** the pass starts at 14:50 Berlin
- **THEN** it logs nothing and reports `too early`

### Requirement: Days without a forecast
A day whose inputs are incomplete SHALL be logged once with its reason and no forecast. The reasons
are `missing pre-session bar`, `warm-up`, `no dollar data` and `unverified clock week`.

#### Scenario: Missing 13:00 bar
- **WHEN** the gold bar labelled 13:00 is missing on the day
- **THEN** the day is logged `missing pre-session bar` without a forecast

### Requirement: One forecast per day
The forecast log SHALL hold at most one row per day. A later run on the same day SHALL change
nothing.

#### Scenario: Second run
- **WHEN** the forecast pass runs twice on the same day
- **THEN** the log holds one row for the day, unchanged by the second run

### Requirement: Settlement pass
After the bar labelled 21:00 Berlin has closed, the settlement pass SHALL settle every logged
forecast not yet settled whose session has ended. For each it SHALL record:
- the realised target `y`;
- the HAR-X and HAR squared errors;
- the realised session high and low.

The realised target and eligibility SHALL be computed exactly as in gold_bbnr.

#### Scenario: Settled day
- **WHEN** the settlement pass runs at 23:08 Sofia after an `ok` forecast and a complete session
- **THEN** one settlement row with status `ok`, `y` and both squared errors is logged

### Requirement: Ineligible sessions
A forecast day whose session gold_bbnr would call ineligible SHALL be settled with that reason
(`invalid day` or `incomplete session`), no squared errors and no score.

#### Scenario: Holiday close
- **WHEN** the bars labelled 19:00–21:00 are missing because of an early close
- **THEN** the day is settled `incomplete session` with no squared errors

### Requirement: Append-only logs, never backfilled
The forecast and settlement logs SHALL be append-only. A settled row SHALL never change. A day with
no forecast row because the PC was off SHALL never get one later, and it SHALL never be settled.

#### Scenario: PC off at 16:00
- **WHEN** the PC is off from 15:00 to 23:30 Sofia on a weekday
- **THEN** that day has no forecast row and no settlement row, now or later

### Requirement: Scheduled tasks
Two tasks SHALL run Mon–Fri as the current user: `EURUSDProphet-GoldNYForecast` at 16:08 Sofia and
`EURUSDProphet-GoldNYSettle` at 23:08 Sofia. Each pass SHALL check the label clock itself, so a
wrong local time zone gives a refusal, never a bad row. An uninstall script SHALL remove only these
two tasks.

#### Scenario: Install and uninstall
- **WHEN** the install script runs and then the uninstall script
- **THEN** the two tasks exist after the first and neither exists after the second; no other task changed
