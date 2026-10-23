# Spec Delta

## Purpose

Gives gold direction a fair second attempt with methods suited to a nearly efficient
market (abstaining, barrier labels, a longer horizon), while keeping the first round's
power and baselines.

## ADDED Requirements

### Requirement: Selective direction abstains by a rule fixed in advance
The selective cell SHALL forecast a direction only when its confidence exceeds a
threshold chosen on the training block alone. The threshold SHALL meet a coverage floor
recorded in the registration, and SHALL never be tuned on validation. Abstained bars
SHALL be logged as abstained and SHALL never be scored as hits or misses.

#### Scenario: Threshold choice
- **WHEN** the selective cell is fitted
- **THEN** its threshold comes from inner walk-forward folds of the training block, and the registration records the expected coverage

#### Scenario: Abstained hour
- **WHEN** the model's confidence is below the threshold
- **THEN** the forward row says "abstain" and the hour is excluded from hit counts

### Requirement: Selective scoring counts only covered bars, at the registered power
A selective cell SHALL be scored on its covered validation bars against both the
training-majority baseline and a coin. It SHALL be admitted only if its expected
covered count meets the power rule.

#### Scenario: Too few covered bars
- **WHEN** the coverage implies fewer covered validation bars than the power rule requires
- **THEN** the cell is registered "UNDERPOWERED — NO DECISION" before any score

### Requirement: Barrier labels are defined before fitting
The triple-barrier cell SHALL label each H1 bar by which barrier is hit first within k
hours:
- an upper barrier at plus m times the bar's volatility forecast;
- a lower barrier at minus m times it;
- a time barrier at k hours.

k and m SHALL be fixed in the registration. A label SHALL use only bars after the as-of
bar, and the training block SHALL be purged of labels that reach into validation.

#### Scenario: Purge
- **WHEN** a training label's k-hour window extends past the training block's end
- **THEN** that row is dropped from training

### Requirement: The 4-hour horizon is registered by power
The 4-hour cell SHALL use non-overlapping 4-hour windows built from H1 bars on the label
clock. Its admission SHALL be decided by the power rule.

#### Scenario: Power table
- **WHEN** the registration is generated
- **THEN** the 4-hour cell's validation count and required count are printed, and the cell is admitted or UNDERPOWERED accordingly
