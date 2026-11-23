# Spec Delta

## Purpose

Defines the gold input of the gold-bb-nyrange family: read-only broker H1 bars on the Berlin
label clock, the D1 bars that close at 17:00 New York, which days are valid, and the gross,
bid-only price convention.

## ADDED Requirements

### Requirement: Read-only gold H1 cache
The family SHALL read gold H1 bars only from the gold family's cache `results/gold/gold_h1.csv`.
It SHALL use the open, high, low and close columns. It SHALL never connect to MT5, refresh
that cache or write any file under `results/gold/`.

#### Scenario: Cache untouched
- **WHEN** any stage of the family runs
- **THEN** `results/gold/gold_h1.csv` and its meta file are byte-identical before and after

### Requirement: Fixed data window
The family SHALL use bars from 2015-01-02 through 2026-11-20 inclusive. That end date is fixed
in the config and the pre-registration. Bars after it SHALL be ignored, even when the cache
has grown.

#### Scenario: Cache grew after registration
- **WHEN** the cache holds bars labelled after 2026-11-20
- **THEN** no feature, event, target or outcome uses them

### Requirement: Berlin label clock check
Every stage SHALL first check the weekly-open labels of the H1 bars with the gold family's
clock check, at its threshold of 95 %. It SHALL refuse to run when the check fails.

#### Scenario: Clock drift
- **WHEN** fewer than 95 % of weekly opens land on the predicted Berlin hour
- **THEN** the stage stops with an error and writes nothing

### Requirement: D1 bars closing at 17:00 New York
A D1 bar SHALL hold the H1 bars whose label, read as Berlin time and converted to New York
time, falls between 17:00 of the previous New York day and 17:00 of that day. Its open is the
first bar's open, its high and low are the extremes, and its close is the last bar's close.
A D1 bar SHALL be valid only when it has at least 12 H1 bars and a weekday date.

#### Scenario: DST-mismatch week
- **WHEN** an H1 bar is labelled 23:00 Berlin in a week when Europe and the US are on different summer time
- **THEN** it belongs to the next New York day's D1 bar, not the current one

#### Scenario: Short day
- **WHEN** a day has only 8 H1 bars
- **THEN** that D1 bar is not valid and is skipped by every rolling window

### Requirement: Gross bid prices
Every price, outcome and target SHALL use the bid prices from the cache. No spread, slippage,
swap or commission SHALL be modelled. Every report and registry note SHALL state that the
results are gross and are not a trading result.

#### Scenario: Report header
- **WHEN** any report of the family is written
- **THEN** its first note says the results are gross of cost on bid prices and are not a trading result
