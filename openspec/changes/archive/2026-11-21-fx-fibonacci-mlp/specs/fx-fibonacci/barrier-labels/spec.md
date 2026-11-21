# Spec Delta

## Purpose

Defines the fx-fibonacci family's hourly target, a triple-barrier label scaled by
volatility, measured gross of cost and fixed before any model is scored.

## ADDED Requirements

### Requirement: Three classes from the first barrier touched
Each labelled H1 bar `t` SHALL get one class:
- `1`, the upper barrier was touched first;
- `2`, the lower barrier was touched first;
- `0`, neither was touched within the horizon.

The owner reads `1` as long and `2` as short. Reports SHALL call them up and down.

#### Scenario: Upper first
- **WHEN** a later bar's high reaches the upper barrier before any bar's low reaches the lower one
- **THEN** the label is 1

#### Scenario: Time runs out
- **WHEN** neither barrier is touched in the `H` bars after `t`
- **THEN** the label is 0

### Requirement: Barriers are set from the bar's close and daily ATR
The barriers SHALL be `close_t ± m × ATR_D1`. ATR_D1 SHALL come from the newest D1 bar that
bar `t` may see under the alignment rule. Touches SHALL be checked on bars `t+1` to `t+H`,
using their high and low. `m` and `H` SHALL be fixed in the config; defaults are
`m = 1.0` and `H = 24` H1 bars.

#### Scenario: No same-bar touch
- **WHEN** bar `t`'s own high exceeds the upper barrier
- **THEN** that does not count, because touches are checked from bar `t+1`

### Requirement: Both barriers in one bar is neutral
When a single H1 bar's range touches both barriers, and neither was touched earlier, the
label SHALL be `0`. Every such tie SHALL be counted and reported per pair.

#### Scenario: Wide bar
- **WHEN** the first bar touching any barrier touches both
- **THEN** the label is 0 and the tie count rises by one

### Requirement: Labels are gross of cost
Labels SHALL use the cached OHLC prices as they are, with no spread, slippage, swap or
commission. This is the owner's decision of 2026-11-21. Every report that shows labels,
or anything derived from them, SHALL state that it is gross of cost.

#### Scenario: Report header
- **WHEN** any label or performance report is generated
- **THEN** its first section states "gross of spread and slippage — not a trading result"

### Requirement: No partial labels
A bar whose full horizon `t+1` to `t+H` is not yet in the cache SHALL get no label. It
SHALL be excluded from training and scoring until its horizon is complete. Horizons SHALL
be counted in H1 bars, so weekends and missing hours are skipped, not counted.

#### Scenario: Newest bars
- **WHEN** labels are built on a cache whose last bar is `T`
- **THEN** bars after `T − H` have no label

#### Scenario: Weekend inside the horizon
- **WHEN** a Friday bar's horizon runs into the next week
- **THEN** the horizon covers `H` bars of trading data, and the weekend adds no bars

### Requirement: Label parameters are fixed before scoring
`m`, `H` and the tie rule SHALL be read from the family config. They SHALL never be
searched, and SHALL never be chosen by model performance. The class balance on train rows
SHALL be reported. Once a verdict is recorded, a run whose label parameters differ from the
recorded config digest SHALL be refused.

#### Scenario: Parameter changed after a verdict
- **WHEN** `m` in the config differs from the digest recorded with the verdicts
- **THEN** every label, training and scoring stage refuses to run

### Requirement: Label overlap is measured
The report SHALL give each pair's mean label uniqueness (the average, over labelled bars,
of one over the number of labels whose horizons cover each bar). This is descriptive. It
SHALL inform the bootstrap block length and the power statement.

#### Scenario: Uniqueness reported
- **WHEN** labels are built with `H = 24`
- **THEN** the report shows each pair's mean uniqueness, and it is at least 1/24
