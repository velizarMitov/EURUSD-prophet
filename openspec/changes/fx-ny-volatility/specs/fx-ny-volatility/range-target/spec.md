# Spec Delta

## Purpose

Defines the fx-ny-volatility family's target: the log mid-price range of the New York
session, 16:00–23:00 Sofia. It also defines which days are eligible, the data it reads,
and the power report written before any model.

## ADDED Requirements

### Requirement: Read-only fx-session-ny data
The family SHALL read the EURUSD, GBPUSD and AUDUSD H1 caches of fx-session-ny read-only,
from 2012-01-01, on the Berlin label clock. It SHALL never refresh or write those caches.

#### Scenario: Caches untouched
- **WHEN** any stage of the family runs
- **THEN** the fx-session-ny H1 caches are byte-identical before and after

### Requirement: Eligible days
A day SHALL be eligible only when all of these hold:
- fx-session-ny's session calendar marks it eligible;
- its Asia, London and data-hour windows are complete;
- no bar it uses carries a corrupt spread;
- its HAR inputs and 20-day ratios have full windows.

Every other day SHALL be excluded with one reason: `incomplete session`,
`unverified clock week`, `missing pre-session bar`, `corrupt spread` or `warm-up`.

#### Scenario: Unverified clock week
- **WHEN** a 2015 day falls in a week whose weekly open is off the Berlin clock
- **THEN** it is excluded as `unverified clock week`

#### Scenario: Missing session bar
- **WHEN** the bar labelled 18:00 Berlin is missing for GBPUSD
- **THEN** the day is excluded as `incomplete session` for both pairs

### Requirement: Corrupt spread rule
A bar whose spread exceeds 50 pips SHALL be treated as carrying a corrupt spread. A day
that needs such a bar for its target or any feature SHALL be excluded as `corrupt spread`.

#### Scenario: AUDUSD 2014-01-07
- **WHEN** AUDUSD's bars on 2014-01-07 report a spread of 300 000 points
- **THEN** that day is excluded as `corrupt spread` for both scored pairs

### Requirement: New York range target
For each eligible day and pair, the target SHALL be `y = log((mid_high − mid_low) / close14)`.
`mid_high` and `mid_low` SHALL be the highest and lowest bid high and low plus half their
own bar's spread over the bars labelled 15:00 to 21:00 Berlin inclusive. `close14` SHALL be
the mid close of the bar labelled 14:00 Berlin.

#### Scenario: Hand-computed target
- **WHEN** the mid high is 1.1060, the mid low is 1.1005 and close14 is 1.1000
- **THEN** the target is log(0.0055 / 1.1000) = log(0.005)

#### Scenario: Rollover hour excluded
- **WHEN** the bar labelled 22:00 Berlin has a wider range than every session bar
- **THEN** the target is unchanged, because that bar is outside the window

### Requirement: Bid twin for parity
The family SHALL also compute the target from bid highs and lows. It SHALL report that
target's correlation with the mid target and the cells' MSE ratio on it. It SHALL never
register a verdict on it.

#### Scenario: Parity reported
- **WHEN** the report is generated
- **THEN** it contains the correlation between the mid and bid targets for each pair

### Requirement: Gross of cost
No verdict SHALL use a spread, a trade or a position. Every report and registry note SHALL
state that the result is gross of spread and slippage and not a trading result.

#### Scenario: Report header
- **WHEN** any fx-ny-volatility report is generated
- **THEN** its first content line states that it is gross of spread and slippage and not a trading result

### Requirement: Power report before any model
Before any model trains, the label report SHALL give, per pair:
- the weekdays, the eligible days, and the out-of-sample days from 2013-01-01;
- the exclusions by reason, and the eligible days per year;
- the detectable mean squared-error difference at alpha 0.0125 one-sided and 80 % power,
  in units of that difference's standard deviation.

It SHALL use no model output.

#### Scenario: Probe-sized input
- **WHEN** a pair has 3 470 out-of-sample days
- **THEN** the detectable difference reads about 0.05 standard deviations
