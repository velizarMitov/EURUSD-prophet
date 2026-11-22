# fx-ny-orb-hot/orb-rule Specification

## Purpose

Defines the fx-ny-orb-hot trade: a New York opening-range breakout on M15 mid bars, taken
only on days the range forecast calls hot, with stops and targets scaled to that
forecast, results in R net of cost.

## Requirements

### Requirement: M15 mid bars on the Berlin label clock
The family SHALL use M15 bars of EURUSD and GBPUSD on the broker's Berlin label clock.
Historically they SHALL be aggregated from the M1 parquets, read-only, and live they SHALL
come from MT5. Each bar's mid open, high, low and close SHALL be its bid value plus half
its own spread. A bar with a spread above 50 pips SHALL void its day.

#### Scenario: Mid high
- **WHEN** an M15 bar's bid high is 1.10500 and its spread is 0.00010
- **THEN** its mid high is 1.10505

### Requirement: The forecast and the hot day
The forecast `F`, in price, SHALL be `exp(p) × close14`. Here `p` is the HAR-X predicted
log range for the day, and `close14` is the mid close of the bar labelled 14:00 Berlin. A
day SHALL be hot when `p` is above the causal Q75 of the realised log range over the
previous 60 eligible days. Historically `p` and the threshold SHALL be fx_nyvol's saved
walk-forward values.

#### Scenario: Hot day
- **WHEN** `p` is −5.30 and the threshold is −5.35
- **THEN** the day is hot

### Requirement: New York opening range
The opening range SHALL be the highest mid high and the lowest mid low of the four M15 bars
labelled 14:00, 14:15, 14:30 and 14:45 Berlin. A day missing any of them SHALL have no
trade.

#### Scenario: Missing 14:30 bar
- **WHEN** the bar labelled 14:30 is missing
- **THEN** the day has no opening range and no trade

### Requirement: First breakout and entry conditions
The breakout SHALL be the first M15 bar, labelled 15:00 or later, whose mid close is above
the range high (long) or below the range low (short). The trade SHALL enter at the next
bar's mid open only when all of these hold:
- the range width is at most 0.50 × `F`;
- the breakout close is at most 0.25 × `F` beyond the broken level;
- the entry bar is labelled before 19:00.

Otherwise the day SHALL have no trade.

#### Scenario: Breakout too far
- **WHEN** the first close outside the range is 0.30 × `F` above the range high
- **THEN** the day has no trade, and no later breakout is considered

#### Scenario: Late breakout
- **WHEN** the first close outside the range is the bar labelled 18:45
- **THEN** the entry bar would be labelled 19:00, and the day has no trade

### Requirement: Stop, target and forced exit
The stop SHALL be 0.25 × `F` from the entry and the target 0.60 × `F`, against and with the
direction. The M15 bars SHALL be scanned from the entry bar through the bar labelled 20:45:
- a bar that reaches both levels SHALL count as a stop;
- the target SHALL fill at its level;
- the stop SHALL fill at its level, or at the bar's mid open when the bar opens beyond it;
- with no touch, the trade SHALL exit at the mid close of the bar labelled 20:45.

#### Scenario: Both levels in one bar
- **WHEN** one M15 bar's mid low is below a long's stop and its mid high above its target
- **THEN** the trade is a stop loss

#### Scenario: Forced exit
- **WHEN** neither level is reached by the bar labelled 20:45
- **THEN** the trade exits at that bar's mid close

### Requirement: One trade per day per pair
A pair SHALL take at most one trade per day. A stopped trade SHALL never re-enter, in either
direction.

#### Scenario: Stop then opposite breakout
- **WHEN** a long is stopped at 16:30 and the price later closes below the range low
- **THEN** no second trade is opened

### Requirement: Result in R net of cost
Each trade SHALL record its result gross and net, in pips and in R. R is the stop distance
0.25 × `F`. The net result SHALL subtract:
- half the entry bar's spread;
- half the exit bar's spread;
- a fixed slippage of 0.4 bps of the entry price per trade.

#### Scenario: Target hit
- **WHEN** `F` is 0.0040, the target is hit, both spreads are 0.00005, and the entry price is 1.1000
- **THEN** the gross result is +2.4 R, and the net result is (0.0024 − 0.00005 − 0.000044) / 0.0010 = +2.306 R

### Requirement: Same rule on non-hot days
The rule SHALL be applied identically on non-hot days, using the same `F`. Those trades
SHALL serve only as the control in the comparison.

#### Scenario: Control trade
- **WHEN** a non-hot day has a qualifying breakout
- **THEN** its trade is simulated and labelled non-hot
