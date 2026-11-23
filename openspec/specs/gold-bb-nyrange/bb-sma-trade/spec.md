# gold-bb-nyrange/bb-sma-trade Specification
## Purpose

Defines Part 1b of gold-bb-nyrange: a trade opened on a gold Bollinger first breach, closed at
the 20-day mean, at a stop the same distance away, or after 10 trading days. Exits are checked
hour by hour. The result is in units of risk, gross of cost, and a walk-forward LightGBM
estimates the probability that the trade wins.

## Requirements
### Requirement: Entry on a first breach
A trade SHALL open at the D1 close of a first breach, as defined in Part 1a, but without the
3-day cooldown. The direction SHALL be −1 above the upper band and +1 below the lower band.

#### Scenario: Upper breach
- **WHEN** the close of `t` is the first close above the upper band
- **THEN** a short trade opens at that close

### Requirement: Take profit, stop and time stop
The take profit SHALL be `SMA20[t]`. The stop SHALL be at the same distance on the other side
of the entry. The trade SHALL close at the close of the D1 bar `t+10` (the tenth valid day
after `t`) if neither level is hit before.

#### Scenario: Levels
- **WHEN** a long trade enters at 1 900 and `SMA20[t]` is 1 940
- **THEN** the take profit is 1 940 and the stop is 1 860

### Requirement: Hourly exit scan
Exits SHALL be checked on the H1 bars after the entry close, in time order, up to the last H1
bar of `t+10`. A bar whose range touches both levels SHALL count as a stop. A bar that opens
beyond the stop SHALL fill at its open. A bar that opens beyond the take profit SHALL fill at
the take-profit level.

#### Scenario: One bar touches both
- **WHEN** an H1 bar's high reaches the take profit and its low reaches the stop
- **THEN** the trade exits at the stop

#### Scenario: Gap through the stop
- **WHEN** a long trade's stop is 1 860 and the next H1 bar opens at 1 850
- **THEN** the trade exits at 1 850

### Requirement: Result in R
The result SHALL be `R = dir × (exit − entry) / |entry − stop|`, gross of cost. A trade SHALL
be a win when `R > 0`.

#### Scenario: Stop hit
- **WHEN** a trade exits exactly at its stop
- **THEN** its R is −1.0

### Requirement: One open trade at a time
A first breach SHALL open no trade while the previous trade is still open. A trade that exits
at or before the entry close of a new breach SHALL not block it. A trade excluded for missing
data SHALL block new trades until the close of its `t+10`.

#### Scenario: Breach during an open trade
- **WHEN** a trade opened on Monday is still open when a new first breach closes on Wednesday
- **THEN** no trade opens on Wednesday

### Requirement: Excluded trades
A trade SHALL be excluded as `no exit data` when its H1 bars up to `t+10` are not all in the
data window, and as `warm-up` when any of the eleven Part 1a features is missing. Exclusions
SHALL be counted in the label report.

#### Scenario: Trade near the window end
- **WHEN** a breach occurs 4 valid days before 2026-11-20
- **THEN** the trade is excluded as `no exit data`

### Requirement: Walk-forward LightGBM on the trade
A binary LightGBM SHALL predict `R > 0` from the eleven Part 1a features, with fx-bb-sma-exit's
fixed hyperparameters. It SHALL train in 12-month folds from 2018-01-01, on trades whose exit
is at least 5 days before the fold starts. A fold with fewer than 60 training trades SHALL
predict nothing, and its trades SHALL be excluded from the model cell as `thin fold`.

#### Scenario: Purge by exit date
- **WHEN** a trade enters on 2020-12-20 and exits on 2021-01-04
- **THEN** it is not in the training set of the fold that starts on 2021-01-01
