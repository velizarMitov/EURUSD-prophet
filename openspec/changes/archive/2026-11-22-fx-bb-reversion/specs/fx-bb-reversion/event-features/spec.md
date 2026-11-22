# Spec Delta

## Purpose

Defines the ten fixed, scale-free features that describe each Bollinger event at the close
of its day. They are oriented by the breach side, so long and short events can be pooled
in one model.

## ADDED Requirements

### Requirement: Fixed feature list
Cell 2 SHALL use exactly these columns, in this order: `bb_zscore_abs`,
`bb_width_rank_60d`, `rsi_14_dist`, `bar_range_vs_atr20`, `close_pos_in_bar`,
`ret_3d_atr`, `trend_ma50_alignment`, `atr_ratio_20_60`, `simultaneous_frac`,
`breach_side`. Assembly SHALL raise an error on any extra or missing column. No pair
identity SHALL be a feature.

#### Scenario: Extra column
- **WHEN** a feature frame holds a column that is not in the config's list
- **THEN** assembly raises an error naming that column

### Requirement: Orientation by breach side
The breach side `s` SHALL be +1 for an upper-band event and −1 for a lower-band event.
`rsi_14_dist`, `close_pos_in_bar`, `ret_3d_atr` and `trend_ma50_alignment` SHALL be
expressed toward the breach side, so that a mirrored price path gives the same values.

#### Scenario: Mirrored event
- **WHEN** an upper-band event's price path is mirrored around a fixed level into a lower-band event
- **THEN** `rsi_14_dist`, `close_pos_in_bar`, `ret_3d_atr` and `trend_ma50_alignment` keep their values and `breach_side` changes sign

### Requirement: Feature definitions
`bb_zscore_abs` SHALL be `|close − SMA20| / SD20`. `bb_width_rank_60d` SHALL be the share of
the days `t−59 … t` whose band width `4·SD20 / SMA20` is at most day `t`'s. `rsi_14_dist`
SHALL be `s · (RSI14 − 50)`, with Wilder RSI on mid closes. `bar_range_vs_atr20` SHALL be the
bid range of `t` divided by ATR20. `atr_ratio_20_60` SHALL be ATR20 divided by ATR60.

#### Scenario: Hand-computed z-score
- **WHEN** the close is 1.1050, SMA20 is 1.1000 and SD20 is 0.0020
- **THEN** `bb_zscore_abs` is 2.5

### Requirement: Bar shape and momentum
`close_pos_in_bar` SHALL be the bid close's position within the day's bid range, toward
the breach side: 1 means it closed at the extreme. `ret_3d_atr` SHALL be
`s · (close[t] − close[t−3]) / ATR20`. `trend_ma50_alignment` SHALL be
`s · (close − SMA50) / ATR20`.

#### Scenario: Close at the low of a lower-band breach
- **WHEN** a lower-band event's bid close equals the day's bid low
- **THEN** `close_pos_in_bar` is 1

### Requirement: Simultaneous breaches
`simultaneous_frac` SHALL be the share of the other started pairs whose mid close on `t`
is outside their own bands. It SHALL use only closes of day `t`, which all pairs share at
17:00 New York.

#### Scenario: Dollar shock
- **WHEN** five pairs have started and 3 of the other 4 close outside their bands on `t`
- **THEN** `simultaneous_frac` is 0.75

### Requirement: Causal and scale-free
Every feature of day `t` SHALL be unchanged when every bar after `t` is removed, and when
every price of every pair is multiplied by a constant. An event without every feature
SHALL be excluded from Cell 2 only, with the reason "warm-up".

#### Scenario: Truncation test
- **WHEN** for 20 random events every bar after `t` is removed and the features are rebuilt
- **THEN** every feature value is unchanged

#### Scenario: Scale test
- **WHEN** every price is multiplied by 100
- **THEN** every feature value is unchanged
