# Spec Delta

## Purpose

Defines the fx-session family's target: one binary label per eligible trading day, the
sign of the mid-price move from the 16:00 Sofia entry close to the 23:00 Sofia exit close,
gross of spread and slippage, with a bid-versus-mid check.

## ADDED Requirements

### Requirement: Binary sign target on mid prices
For each eligible day D, the target SHALL be 1 when the exit bar's mid close is above the
entry bar's mid close, and 0 when it is below. The exit bar is the one labelled 21:00
Berlin on D, and the entry bar is the one labelled 14:00 Berlin on D.

#### Scenario: Up day
- **WHEN** the entry mid close is 1.08003 and the exit mid close is 1.08250
- **THEN** the target for D is 1

### Requirement: Ties dropped
A day whose exit mid close equals its entry mid close SHALL get no target. It SHALL be
excluded from training and scoring and counted as a tie.

#### Scenario: Flat session
- **WHEN** the entry and exit mid closes are equal
- **THEN** D has no target and the tie count increases by one

### Requirement: No barrier and no overlap
The target SHALL use no barrier, no profit-target or stop parameter and no class weights.
Each day's target SHALL span only the bars labelled 15:00 to 21:00 Berlin of that same day,
so no two days' targets share a bar.

#### Scenario: Consecutive days
- **WHEN** the targets of D and the next trading day are built
- **THEN** their spans have no bar in common

### Requirement: Bid-versus-mid parity check
The family SHALL also compute the target on bid closes. For every pair and fold, it SHALL
report the up-rate on mid and on bid, and the gap between them. A gap above 1.0
percentage point for any pair SHALL be flagged in the report as a possible spread artifact.
The verdicts SHALL be decided on the mid target only.

#### Scenario: Spread artifact flagged
- **WHEN** a pair's up-rate is 51.0 % on mid and 49.6 % on bid
- **THEN** the report flags that pair's gap of 1.4 points

### Requirement: Gross of cost
The target and every result derived from it SHALL be gross of spread and slippage. Every
report and every registry note SHALL carry the statement that the result is gross and not
a trading result.

#### Scenario: Report header
- **WHEN** any fx-session report is generated
- **THEN** its first content line states that it is gross of spread and slippage and not a trading result

### Requirement: Label report before any model
Before any model is trained, the family SHALL write a label report. It SHALL include each
pair's eligible days, the up-rate overall and per year, the tie count, the excluded-day
counts by reason, and the number of DST-mismatch days.

#### Scenario: Label report content
- **WHEN** the label report is generated
- **THEN** it lists, per pair, the eligible days, ties, the up-rate per year and the DST-mismatch day count
