# Spec Delta

## Purpose

Defines how RTS recommendations are replayed on the owner's MT5 prices. Each recommendation becomes a base
trade, and the full rule chain is replayed for reconciliation. Both are reported gross and net of the recorded
spread.

## ADDED Requirements

### Requirement: Read-only MT5 prices
The simulation SHALL use M1 bid bars and their recorded spread for EURUSD, GBPUSD, AUDUSD, USDJPY, USDCHF and
USDCAD. The bars SHALL be read market-data-only from the MT5 terminal into the family's own append-only caches.
It SHALL never call an order or position function.

#### Scenario: No trading call
- **WHEN** the family's source is searched for order or position calls
- **THEN** none is found

### Requirement: Base trade
Each recommendation with an entry SHALL yield at most one base trade. The trade:
- enters per Rule 0 with proportionality, or per the guerilla rule, during the session after the report;
- exits at the first touch of the stop or the target, scanned minute by minute;
- otherwise exits at the next 16:00 New York close, or at 12:00 New York on a Friday in spot.

A minute bar that touches both the stop and the target SHALL count as the stop.

#### Scenario: Target first
- **WHEN** a sell at 1.0900 with stop 1.0930 and target 1.0866 touches 1.0866 before 1.0930
- **THEN** the base trade exits at the target

#### Scenario: Both in one minute
- **WHEN** one M1 bar's range covers both the stop and the target
- **THEN** the trade exits at the stop

### Requirement: Gross and net results
Every base trade SHALL be reported in pips and in R, where R is the entry-to-stop distance. Results SHALL be
given gross on bid prices and net of the spread:
- a buy enters at bid + spread and exits at bid;
- a sell enters at bid and exits at bid + spread;
- touches of the stop and the target use the same side.

No other cost, such as slippage, swap or commission, SHALL be implied. The report SHALL say so.

#### Scenario: Net is worse
- **WHEN** a winning buy trade is replayed with a spread of 1.2 pips at both entry and exit
- **THEN** the net result is lower than the gross result by the recorded spreads

### Requirement: Missing data
A recommendation whose session has missing M1 bars across its entry or exit SHALL be reported `no price data`
and not simulated. A guerilla entry never touched SHALL be reported `entry not reached`.

#### Scenario: Gap in bars
- **WHEN** the M1 cache has no bars for the hour that holds the entry
- **THEN** the recommendation is reported `no price data` and has no result

### Requirement: Full-rules replay
The full-rules replay SHALL apply the rule book's contingency rules to each pair's sessions in date order. It
SHALL list every trade it opens, with the rule number. It is descriptive and SHALL never be scored.

#### Scenario: Rule 2 after a target
- **WHEN** a sell hits its target and the price falls 40 more points the same session, on a weekday other than Friday
- **THEN** the replay opens a Rule 2 sell with stop 40 and target 60 points

### Requirement: Reconciliation with RTS's track record
Each trade in RTS's own spot track record SHALL be matched to our replay by pair, date, side and entry. Each
match SHALL be labelled:
- `confirmed`: our prices reach the same exit;
- `different exit`;
- `not reproducible`: our prices never reach the entry.

#### Scenario: Confirmed fill
- **WHEN** RTS lists `11/23/26 GBP/USD Sold 1.3081 ... 1.3024 57` and our M1 bars touch 1.3024 after a sell at 1.3081 before the stop
- **THEN** that trade is labelled `confirmed`
