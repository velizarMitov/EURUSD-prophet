# gold-forecasting/move-alerts Specification

## Purpose

Tells the owner when the next gold hour is expected to move unusually much, using the
one forecast with proven skill, without ever suggesting a direction or a trade.

## Requirements

### Requirement: The threshold is fixed from training data
An alert SHALL fire when the frozen h1_lstm expected move for the next hour is at or
above a threshold. The threshold SHALL be fixed as a percentile of that model's
training-block forecasts, recorded in `src/gold/gold_config_v2.json` with its percentile and value,
and SHALL never be tuned on live outcomes.

#### Scenario: Configured percentile
- **WHEN** the percentile is 90
- **THEN** the threshold is the 90th percentile of the model's forecasts on its training rows, and both numbers are stored

### Requirement: An alert is logged, shown and notified once
Each alert SHALL be appended to `results/gold/forward/alerts.csv`, with the as-of bar,
the expected move and the threshold. The logger SHALL raise one Windows notification
per alert, and `/gold` SHALL show a banner while the alert's hour is open. If the
notification fails, the alert SHALL still be logged.

#### Scenario: Two passes on the same bar
- **WHEN** a second pass sees the same as-of bar
- **THEN** no second alert or notification is produced

#### Scenario: Notifications unavailable
- **WHEN** the Windows notification call fails
- **THEN** the alert row is still written and the failure is recorded in the health row

### Requirement: Alerts name size, never direction
Alert text SHALL state the expected size of the move and that its direction is unknown.
It SHALL contain no buy, sell, entry, stop or target wording.

#### Scenario: Alert text
- **WHEN** an alert for ±0.35 % is raised
- **THEN** the text reads as an expected large move of about ±0.35 % with direction unknown, in BG and EN

### Requirement: Alerts are scored
Each settled alert SHALL record the realised absolute move. The leaderboard SHALL show
how often alerted hours moved more than non-alerted hours, as a descriptive comparison
with its counts.

#### Scenario: Alert settled
- **WHEN** an alerted hour settles with an actual move of 0.42 %
- **THEN** the alert row gains the realised move, and the comparison counts it
