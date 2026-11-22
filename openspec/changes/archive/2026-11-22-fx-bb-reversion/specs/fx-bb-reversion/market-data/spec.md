# Spec Delta

## Purpose

Defines the fx-bb-reversion family's market data: six-pair H1 bars on the broker's verified
clock, the per-pair start dates and defect rules, and the daily bars closing at 17:00 New
York that every event and outcome is computed from.

## ADDED Requirements

### Requirement: Six-pair H1 sources
The family SHALL read EURUSD, GBPUSD and AUDUSD read-only from the fx_session H1 caches. It
SHALL keep USDJPY, USDCAD and USDCHF in its own append-only H1 caches, holding the bid OHLC
and the bar spread. A refresh SHALL write nothing for any pair when any one of them is
unreachable or fails a check.

#### Scenario: One new pair fails its checks
- **WHEN** a refresh finds an undeclared interior hole in USDCAD
- **THEN** none of the three family caches changes and the refresh log records the refusal for every pair

#### Scenario: fx_session caches are never written
- **WHEN** the family refreshes or runs any stage
- **THEN** the fx_session H1 caches are byte-identical before and after

### Requirement: Per-pair start dates
Every pair SHALL start on 2012-01-01, except USDCHF, which SHALL start on 2015-02-01. The
USDCHF date excludes its proven one-hour clock shift (2012-01 to 2014-11) and the SNB-floor
month of January 2015. No bar SHALL be shifted to repair a clock.

#### Scenario: USDCHF before its start
- **WHEN** the USDCHF cache holds bars dated before 2015-02-01
- **THEN** no band, ATR, event or feature uses them

### Requirement: Clock and lag checks
Every pair SHALL pass two checks: the project's weekly-open clock check at a match rate of
at least 90 % on the stretches between declared gaps, and a lag check. In the lag check,
for every calendar year from the pair's start, the absolute correlation of the pair's
hourly log returns with EURUSD's SHALL peak at lag 0 among the lags −1, 0 and +1. A failure
SHALL refuse the refresh or the run.

#### Scenario: One-hour shift that passes the weekly-open check
- **WHEN** a pair's hourly returns correlate with EURUSD's most strongly at lag +1 in one year, while 96 % of its weekly opens match
- **THEN** the refresh or run is refused and the reason names the pair and the year

### Requirement: Declared gaps
A month the config declares as a broker gap for a pair SHALL be accepted without filling.
USDCAD 2013-05 SHALL be declared. Any other interior hole or thin month SHALL refuse the
refresh.

#### Scenario: Declared USDCAD gap
- **WHEN** the USDCAD history lacks bars from 2013-05-03 to 2013-05-13
- **THEN** the refresh succeeds, nothing is filled, and the missing days have no D1 bar

### Requirement: Corrupt spread rule
A bar whose spread exceeds 50 pips SHALL be treated as carrying a corrupt spread. Any mid
price that needs such a bar SHALL be missing. Bid prices SHALL be unaffected.

#### Scenario: AUDUSD 2014-01-07
- **WHEN** every bar on 2014-01-07 reports a spread of 300 000 points
- **THEN** that day has no mid close and counts as a missing day for AUDUSD

### Requirement: Daily bars closing at 17:00 New York
Each D1 bar SHALL cover 17:00 New York on the previous day to 17:00 New York, found by
converting the Berlin labels to New York time. It SHALL hold the bid open, high and low,
the bid close, the mid close (the last bar's bid close plus half its spread) and its H1
bar count. Bars of a day with fewer than 12 H1 bars SHALL not be used.

#### Scenario: Sunday reopen
- **WHEN** the weekly reopen bar is labelled 23:00 Berlin on a Sunday
- **THEN** it belongs to Monday's D1 bar and no Sunday D1 bar exists

#### Scenario: Last bar of the day
- **WHEN** the US and EU clocks are aligned on a Wednesday
- **THEN** that day's close is the close of the bar labelled 22:00 Berlin, and the 23:00 Berlin rollover bar belongs to Thursday

#### Scenario: DST-mismatch week
- **WHEN** New York and Berlin are 5 hours apart in a March mismatch week
- **THEN** the day's last bar is the one labelled 21:00 Berlin

#### Scenario: Short day
- **WHEN** a weekday has 9 H1 bars because of a holiday
- **THEN** it has no D1 bar
