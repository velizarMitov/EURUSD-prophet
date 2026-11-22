# Spec Delta

## Purpose

Defines the fx-bb-reversion family's events and outcomes: the first daily close outside the
20-day, 2-SD Bollinger Bands, its counter-trend direction, the 3-day outcome in ATR units,
the binary target, the exclusions, and the power report written before any model.

## ADDED Requirements

### Requirement: Bands and ATR from the past only
For day `t`, the bands SHALL be `SMA20 ± 2·SD20` of the mid closes of the pair's valid days
`t−19 … t`, with the population standard deviation. `ATR20[t]` SHALL be the simple mean of
the bid true range over the same days. No value from after `t` SHALL enter either.

#### Scenario: Truncation leaves the bands unchanged
- **WHEN** every bar after day `t` is removed
- **THEN** the bands, ATR20 and the event flag of day `t` are unchanged

### Requirement: First-breach events with a counter-trend direction
Day `t` SHALL be an event when its mid close is outside its bands and the previous valid
day's mid close was inside that day's bands. The direction SHALL be −1 for a close above
the upper band and +1 for a close below the lower band.

#### Scenario: Close above the upper band
- **WHEN** the close of `t−1` is inside its bands and the close of `t` is above the upper band
- **THEN** day `t` is an event with direction −1

#### Scenario: Second day outside
- **WHEN** the closes of both `t−1` and `t` are above their upper bands
- **THEN** day `t` is not an event

### Requirement: Three-day cooldown per pair
An event SHALL be accepted only when at least 3 valid days of the same pair have passed
since that pair's previous accepted event, so that the outcome windows of one pair never
overlap.

#### Scenario: Re-breach after two days
- **WHEN** a pair has an accepted event on `t` and a new first breach on `t+2`
- **THEN** the breach on `t+2` is not an event

### Requirement: Outcome in ATR units
Each event SHALL have the outcome `r = dir · (close[t+3] − close[t]) / ATR20[t]`, using mid
closes, where `t+3` is the pair's third valid day after `t`. An event with no third valid
day SHALL have no outcome.

#### Scenario: Reversion after an upper breach
- **WHEN** an upper-band event closes at 1.1000 with ATR20 0.0080, and the close 3 valid days later is 1.0960
- **THEN** its outcome is +0.5

### Requirement: Binary target and ties
The Cell 2 target SHALL be 1 when `r > 0` and 0 when `r < 0`. An event with `r = 0` SHALL
get no target and SHALL be counted as a tie. No class weight, stop or profit target SHALL
be used.

#### Scenario: Losing event
- **WHEN** an event's outcome is −0.3
- **THEN** its target is 0

### Requirement: Pair-gap exclusion
The reference calendar SHALL be every date on which at least 4 of the started pairs have a
valid D1 bar. An event SHALL be excluded, with the reason "pair gap", when its pair lacks a
reference date between the date of `t−19` and the date of `t+3`.

#### Scenario: GBPUSD in March 2014
- **WHEN** a GBPUSD event's 20-day window includes days of 2014-03 that have no D1 bar while the other pairs do
- **THEN** the event is excluded as a pair gap and counted in the label report

#### Scenario: Common holiday
- **WHEN** no pair has a D1 bar on 25 December
- **THEN** that date is not a reference date and excludes no event

### Requirement: Gross of cost
Every outcome, target and verdict SHALL be gross of spread and slippage. Every report and
registry note SHALL state that the result is gross and not a trading result.

#### Scenario: Report header
- **WHEN** any fx-bb-reversion report is generated
- **THEN** its first content line states that it is gross of spread and slippage and not a trading result

### Requirement: Power report before any model
Before any model trains, the label report SHALL give, per pair and pooled: the events, the
first breaches removed by the cooldown, the exclusions by reason, the ties and the distinct
event dates, overall, from 2015 and per year. It SHALL use no event outcome and no model
output.

#### Scenario: Report generated before training
- **WHEN** the label report is generated on a checkout with no predictions
- **THEN** it succeeds and contains the event counts and the power table

### Requirement: Minimum detectable effects
The label report SHALL give, at alpha 0.025 one-sided and 80 % power:
- for Cell 1, the minimum detectable mean outcome from the standard deviation of
  `(close[t+3] − close[t]) / ATR20[t]` over all valid days, never over events;
- for Cell 2, the minimum detectable accuracy.

Both SHALL be computed on the event count and on the distinct-date count.

#### Scenario: Probe-sized inputs
- **WHEN** there are 1 054 out-of-sample events on 751 dates
- **THEN** the minimum detectable accuracy reads about 54.3 % on events and about 55.1 % on dates
