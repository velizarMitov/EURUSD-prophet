# Spec Delta

## Purpose

Turns the proven gold volatility forecast into calibrated "between X and Y" price
ranges for the next hour and the next day, and checks whether they hold their promised
coverage.

## ADDED Requirements

### Requirement: Ranges at two levels for two horizons
For each new H1 and D1 forecast, the system SHALL produce a price interval at 80 % and
95 %. The interval SHALL be centred on the as-of close and scaled by that cadence's
volatility forecast. Its width SHALL use only information available at the as-of bar.

#### Scenario: Hourly range
- **WHEN** the H1 forecast is made from a bar closing at 4,100.00 with an expected move of ±0.11 %
- **THEN** the 80 % and 95 % ranges are logged with that bar's key, with the 95 % range strictly wider than the 80 % range

### Requirement: Calibration adapts without looking ahead
The quantile that turns a volatility forecast into a range SHALL come from past
standardised errors only. It SHALL be updated online from past coverage misses, in the
style of adaptive conformal inference. An update at bar t SHALL use only outcomes
settled before t.

#### Scenario: Update order
- **WHEN** the range for bar t is computed
- **THEN** no settlement with an exit bar at or after t has entered its calibration

### Requirement: Ranges are scored on coverage and sharpness against a baseline
A range cell SHALL be scored on validation by:
- its empirical coverage at each level;
- its mean interval score (Winkler) against a GARCH-based range at the same levels on
  the same rows.

It SHALL clear only if the interval score is better with a confidence interval
excluding 0 at the family alpha, and its 80 % coverage is within the registered
tolerance.

#### Scenario: Sharp but under-covering
- **WHEN** a range has a better interval score but 71 % coverage at the 80 % level, with a tolerance of ±3 points
- **THEN** it does not clear, and the report names the coverage failure

### Requirement: Live coverage is tracked
Each settled live range SHALL record whether the exit price fell inside it, at each
level. The leaderboard and the gold page SHALL show the running coverage with its
count.

#### Scenario: Range missed
- **WHEN** the exit price is outside the 80 % range but inside the 95 % range
- **THEN** the settlement records "outside" at 80 % and "inside" at 95 %

### Requirement: Ranges are not trading levels
The page SHALL present ranges as "where gold is likely to be". It SHALL never present
them as targets, stops or entries.

#### Scenario: Page wording
- **WHEN** the range card is rendered
- **THEN** it contains no take-profit, stop-loss, entry or exit wording
