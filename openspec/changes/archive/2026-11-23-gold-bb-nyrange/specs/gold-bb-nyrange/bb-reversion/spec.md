# Spec Delta

## Purpose

Defines Part 1a of gold-bb-nyrange: Bollinger Band (20, 2 SD) first breaches on gold D1 bars,
their 3-day mean-reversion outcome in ATR units, and a walk-forward LightGBM that estimates
the probability that a breach reverts.

## ADDED Requirements

### Requirement: Bands, ATR and events
On valid D1 bars, the bands SHALL be `SMA20 ± 2·SD20` (population SD) of the closes of
`t−19 … t`. ATR20 SHALL be the simple mean of the true range of `t−19 … t`. An event SHALL be
a close of `t` outside its bands when the close of `t−1` was inside its own bands. The
direction SHALL be −1 above the upper band and +1 below the lower band.

#### Scenario: Lower breach
- **WHEN** the close of `t−1` is inside its bands and the close of `t` is below the lower band
- **THEN** an event with direction +1 is recorded on `t`

### Requirement: Cooldown
An event SHALL be accepted only when at least 3 valid days have passed since the last
accepted event. The cooldown SHALL be applied to the signal before any exclusion.

#### Scenario: Second breach two days later
- **WHEN** an event is accepted on Monday and a new first breach occurs on Wednesday
- **THEN** the Wednesday breach is removed by the cooldown

### Requirement: The 3-day outcome
The outcome of an event SHALL be `r = dir × (close[t+3] − close[t]) / ATR20[t]`, where `t+3`
is the third valid day after `t`. An event with no `t+3` in the data window SHALL be excluded
as `no outcome`. Event detection SHALL read no close after `t`.

#### Scenario: Reversion after an upper breach
- **WHEN** an upper breach closes at 2 000, ATR20 is 20 and the close of `t+3` is 1 980
- **THEN** `r` is +1.0

#### Scenario: Truncation leaves detection unchanged
- **WHEN** the events up to `t` are detected on the full data and on data cut after `t`
- **THEN** both give the same events with the same directions

### Requirement: Nine gold features
Each event SHALL carry nine gold features computed from data up to the close of `t`:
`bb_zscore_abs`, `bb_width_rank_60d`, `rsi_14_dist`, `bar_range_vs_atr20`,
`close_pos_in_bar`, `ret_3d_atr`, `trend_ma50_alignment`, `atr_ratio_20_60` and
`breach_side`, defined as in fx-bb-reversion. The cross-pair feature `simultaneous_frac`
SHALL NOT exist, because gold is one instrument.

#### Scenario: No cross-pair feature
- **WHEN** the feature table is built
- **THEN** it has no `simultaneous_frac` column

### Requirement: Two dollar features
Each event SHALL also carry two features from the daily DXY closes dated before `t`:
`dxy_zscore_20d`, which is `dir_gold × (close − SMA20) / SD20` (population SD) of the latest
20 closes, and `dxy_ret_5d`, which is `dir_gold × log(close / close five closes earlier)`.
`dir_gold` is the event direction. That gives eleven features in config order. An event with
any missing feature SHALL be excluded as `warm-up`.

#### Scenario: Feature list fixed
- **WHEN** the feature table is built
- **THEN** its columns are exactly the eleven names in config order

### Requirement: Walk-forward LightGBM on reversion
A binary LightGBM SHALL predict `r > 0` with fx-bb-reversion's fixed hyperparameters. It
SHALL be trained in 12-month folds from 2018-01-01. Each fold SHALL train only on events
whose `t+3` is at least 5 days before the fold starts, with the last 15 % of those events
used for early stopping. A fold with fewer than 60 training events SHALL predict nothing.
Those events SHALL be excluded from the model cell as `thin fold`, and the label report SHALL
count them before any outcome is read.

#### Scenario: Thin first fold
- **WHEN** the 2018 fold has only 45 training events
- **THEN** no 2018 event gets a model probability, and the label report lists them as `thin fold`

#### Scenario: Purge
- **WHEN** an event on 2020-12-30 has its `t+3` in January 2021
- **THEN** it is not in the training set of the fold that starts on 2021-01-01
