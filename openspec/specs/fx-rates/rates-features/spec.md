# fx-rates/rates-features Specification

## Purpose

Defines the fx-rates family's twelve features: the 2-year yield spread and its momentum,
the US curve slope, scale-free FX momentum and volatility, and the divergence between the
exchange rate and the rate spread. Each is known at the close of the decision day.

## Requirements

### Requirement: Fixed feature list
The feature columns SHALL be exactly the ordered list in the family config, fixed before
any model trains. Building features with a missing or an extra column SHALL fail and name
the column.

#### Scenario: Missing column
- **WHEN** the builder does not produce `us_2s10s_slope`
- **THEN** the build fails and names `us_2s10s_slope`

### Requirement: Spread definition
The 2Y spread SHALL be the foreign 2Y yield minus the US 2Y yield, in percentage points:
DE2Y − US2Y for EURUSD and UK2Y − US2Y for GBPUSD. It SHALL be taken from the lagged,
forward-filled yields of the rates-data capability.

#### Scenario: EURUSD spread
- **WHEN** the lagged DE2Y is 2.10 and the lagged US2Y is 4.30
- **THEN** EURUSD's spread level is −2.20

### Requirement: Spread features
The list SHALL contain:
- the spread level;
- its change over 5, 20 and 60 FX trading days;
- its 60-day z-score: the level minus its 60-day mean, divided by its 60-day standard
  deviation.

Each window SHALL end at day `t`'s lagged value.

#### Scenario: Five-day change
- **WHEN** the spread on day `t` is −2.20 and 5 FX trading days earlier it was −2.35
- **THEN** `spread_2y_chg_5d` is +0.15

### Requirement: US curve slope
The list SHALL contain `us_2s10s_slope`, the lagged US 10Y yield minus the lagged US 2Y
yield.

#### Scenario: Slope value
- **WHEN** the lagged DGS10 is 4.10 and DGS2 is 4.30
- **THEN** `us_2s10s_slope` is −0.20

### Requirement: Scale-free FX features
The FX features SHALL use D1 mid closes up to and including day `t`. Let `vol20` be the
20-day standard deviation of daily log returns. The list SHALL contain:
- the 5- and 20-day log returns, each divided by `vol20 · √h`;
- the log of MA20 over MA50, divided by `vol20`;
- `vol20` divided by the 60-day standard deviation.

#### Scenario: Price scaling
- **WHEN** every FX price is multiplied by 2
- **THEN** every FX feature value is unchanged

### Requirement: FX-versus-rates divergence
`fx_vs_spread_div_20d` SHALL be the z-score of the 20-day FX log return minus the z-score
of the 20-day spread change. Each z-score SHALL use the mean and standard deviation of the
same quantity over the previous 252 FX trading days.

#### Scenario: FX lags the rates impulse
- **WHEN** the spread change is 2 standard deviations above its norm and the FX return is at its norm
- **THEN** the divergence is about −2

### Requirement: Known at the decision close
Every feature for day `t` SHALL be unchanged when all FX bars after `t` and all yield
values dated `t` or later are removed.

#### Scenario: Truncated inputs
- **WHEN** the inputs are cut after day `t` and the features are rebuilt
- **THEN** day `t`'s row is identical

### Requirement: Rows with incomplete inputs
A day SHALL have no feature row when any window lacks history, or its yields are stale
beyond the fill limit. Such days SHALL be counted by reason.

#### Scenario: Warm-up
- **WHEN** fewer than 252 FX trading days of history precede `t`
- **THEN** day `t` has no feature row and is counted as warm-up
