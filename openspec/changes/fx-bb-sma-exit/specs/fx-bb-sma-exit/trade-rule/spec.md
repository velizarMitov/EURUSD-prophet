# Spec Delta

## Purpose

Defines the fx-bb-sma-exit trade: entry on the first D1 close outside the Bollinger Bands,
exit at the 20-day mean, at a symmetric stop or after 10 trading days, checked hour by
hour, and its result in units of risk net of the recorded spread.

## ADDED Requirements

### Requirement: Read-only reuse of the fx-bb-reversion data
The family SHALL read the six pairs' H1 caches and build the D1 bars closing at 17:00 New
York exactly as fx-bb-reversion does. That includes its clock and lag checks, per-pair
starts (USDCHF from 2015-02-01, the others from 2012-01-01), declared gaps and
corrupt-spread rule. It SHALL never refresh or write those caches.

#### Scenario: Caches untouched
- **WHEN** any stage of the family runs
- **THEN** every H1 cache and symbols file it reads is byte-identical before and after

### Requirement: Entry on a first band breach
A trade SHALL open at the mid close of day `t` when that close is outside the bands
`SMA20 ± 2·SD20` (population SD) of the mid closes of `t−19 … t`, and the previous valid
day's close was inside its own bands. The direction SHALL be −1 above the upper band and
+1 below the lower band.

#### Scenario: Upper breach
- **WHEN** the close of `t−1` is inside its bands and the close of `t` is above the upper band
- **THEN** a short trade (direction −1) opens at the mid close of `t`

### Requirement: One open trade per pair
A first breach SHALL open no trade while the same pair's previous trade is still open. A
trade that exits at or before the entry close of a new first breach SHALL not block it.
An excluded trade SHALL block its pair until the D1 close of its day `t+10`.

#### Scenario: Breach during an open trade
- **WHEN** a pair's trade opened on Monday is still open on Wednesday, when a new first breach occurs
- **THEN** no trade opens on Wednesday

#### Scenario: Time-stop exit and new entry at the same close
- **WHEN** a trade exits by time stop at the close of day `t+10` and day `t+10` is itself a first breach
- **THEN** a new trade opens at that close

### Requirement: Fixed take profit and stop loss
The take profit SHALL be `SMA20[t]`, fixed at entry. The stop loss SHALL be at
`dist = |Close[t] − SMA20[t]|` beyond the entry: `Close[t] − dist` for a long and
`Close[t] + dist` for a short. Neither level SHALL move after entry.

#### Scenario: Long levels
- **WHEN** a long opens at 1.0950 with `SMA20[t]` at 1.1000
- **THEN** the take profit is 1.1000, the stop is 1.0900 and `dist` is 0.0050

### Requirement: Hourly exit scan
The family SHALL scan the pair's H1 bars after the entry bar in time order, through the
closing bar of valid day `t+10`. A long's take profit SHALL be touched when the bar's mid
high reaches it, and its stop when the mid low reaches it; a short is the mirror image. A
bar's mid high and low SHALL be its bid high and low plus half its own spread.

#### Scenario: Take profit reached on the third day
- **WHEN** a long's take profit is first reached by an H1 bar of day `t+3` and no earlier bar reached the stop
- **THEN** the trade exits at the take profit on that bar, with the reason `take profit`

### Requirement: Stop wins a same-bar conflict
When one H1 bar reaches both the take profit and the stop, the trade SHALL exit at the
stop, with the reason `stop loss`.

#### Scenario: Both levels in one bar
- **WHEN** a short's first bar to reach either level has a low below the take profit and a high above the stop
- **THEN** the trade is recorded as a stop loss

### Requirement: Exit prices without optimism
A take-profit exit SHALL fill at the take-profit level. A stop exit SHALL fill at the stop
level, or at the bar's mid open when the bar opens beyond the stop.

#### Scenario: Gap through the stop
- **WHEN** a long's stop is 1.0900 and the next bar opens at a mid price of 1.0880
- **THEN** the trade exits at 1.0880

### Requirement: Ten-day time stop
When no H1 bar reaches either level by the closing bar of valid day `t+10`, the trade SHALL
exit at the mid close of day `t+10`, with the reason `time stop`.

#### Scenario: Neither level reached
- **WHEN** the price stays between the stop and the take profit for 10 valid days
- **THEN** the trade exits at the mid close of day `t+10` as a time stop

### Requirement: Result net of the recorded spread
Each trade SHALL record its gross mid result and its net result. The net result SHALL
subtract half of the entry bar's spread and half of the exit bar's spread. The result SHALL
be given in pips and in R (the net price result divided by `dist`). Slippage, swap and
commission SHALL not be modelled.

#### Scenario: Net R of a take profit
- **WHEN** a long with `dist` 0.0050 exits at its take profit and both bars' spreads are 1.0 pip
- **THEN** the gross result is +50 pips and +1.0 R, and the net result is +49 pips and +0.98 R

### Requirement: Bid/ask control fill
The family SHALL also simulate every trade on executable prices. A long enters at the ask
and exits at the bid; a short enters at the bid and exits at the ask. The ask SHALL be the
bid plus the bar's spread, with the same levels, rules and order of checks. This result
SHALL be reported for parity only and SHALL never change a verdict.

#### Scenario: Parity reported
- **WHEN** Cell 1 is scored
- **THEN** its registry row records the mean R of the bid/ask control beside the registered net mid result

### Requirement: Exclusions
A trade SHALL be excluded, keeping its slot in the no-overlap rule:
- as `pair gap`, when its pair lacks a reference date between the dates of `t−19` and `t+10`;
- as `corrupt spread`, when an H1 bar it scans carries a spread above 50 pips;
- as `open at data end`, when day `t+10` does not exist yet and no level was reached.

#### Scenario: AUDUSD on 2014-01-07
- **WHEN** an AUDUSD trade's scan includes the bars of 2014-01-07, which carry a 300 000-point spread
- **THEN** the trade is excluded as `corrupt spread` and its pair stays blocked until day `t+10`

### Requirement: No look-ahead at entry
The entry decision, the levels and `dist` SHALL use only data up to the close of `t`. No
H1 bar before the bar after the entry bar SHALL be scanned for an exit.

#### Scenario: Entry bar not scanned
- **WHEN** the entry bar's own high is above the take profit of a long
- **THEN** that bar does not trigger an exit
