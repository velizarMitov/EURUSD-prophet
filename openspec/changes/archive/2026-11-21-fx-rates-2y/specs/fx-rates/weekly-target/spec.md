# Spec Delta

## Purpose

Defines the fx-rates family's target: the sign of the 5-day mid-price move from the close
of the decision day, gross of cost. It also defines the label and power report that states,
before any model, how large an edge the overlapping weekly targets can reveal.

## ADDED Requirements

### Requirement: Five-day sign target
For day `t`, the target SHALL be 1 when the D1 mid close 5 FX trading days later is above
day `t`'s mid close, and 0 when it is below. Days without a D1 bar 5 FX trading days ahead
SHALL get no target.

#### Scenario: Up week
- **WHEN** the mid close on `t` is 1.0800 and on `t+5` is 1.0850
- **THEN** the target for `t` is 1

### Requirement: Ties dropped, no weights
A day whose two mid closes are equal SHALL get no target, and SHALL be counted as a tie. No
barrier, profit-target, stop or class weight SHALL be used.

#### Scenario: Flat week
- **WHEN** the mid closes on `t` and `t+5` are equal
- **THEN** day `t` has no target and the tie count increases by one

### Requirement: Gross of cost
Every target and every result derived from it SHALL be gross of spread and slippage. Every
report and registry note SHALL state that the result is gross and not a trading result.

#### Scenario: Report header
- **WHEN** any fx-rates report is generated
- **THEN** its first content line states that it is gross of spread and slippage and not a trading result

### Requirement: Overlap-aware power report
Before any model trains, the label report SHALL give each pair's usable days, out-of-sample
days from the first fold, ties, and up-rate overall and per year. It SHALL also give the
minimum detectable accuracy at alpha 0.0125 one-sided and 80 % power, computed two ways:
- from the effective size `n / VIF`, where `VIF = 1 + 2·Σ ρ_k` over the target's own sign
  autocorrelation at lags 1–4 on the training-era days;
- from the number of non-overlapping weeks `n / 5`.

It SHALL use no model output.

#### Scenario: Probe-sized sample
- **WHEN** a pair has about 3 080 out-of-sample days and a VIF of about 3.7
- **THEN** the report gives a minimum detectable accuracy of about 55 % on the effective size and about 56 % on weeks

### Requirement: Source and staleness summary
The label report SHALL list each yield series' first and last date and its refresh time.
For each pair it SHALL give the number of usable days with `yield_stale_flag = 1`, and
the number excluded as stale yields.

#### Scenario: Stale counts
- **WHEN** the label report is generated
- **THEN** it lists, per pair, the stale-filled days and the days excluded as stale yields
