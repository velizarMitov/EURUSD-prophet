# Spec Delta

## Purpose

Defines the fourteen fixed, scale-free features that describe each day's volatility state
at the close of the bar labelled 14:00 Berlin (16:00 Sofia), before the New York session
range is known.

## ADDED Requirements

### Requirement: Fixed feature list
The models SHALL use exactly these columns, in this order:
- `log_ny_range_lag1`, `log_ny_range_ma5`, `log_ny_range_ma20`;
- `log_asia_range`, `log_london_range`, `log_data_hour_range`, `abs_data_hour_return`;
- `london_range_ratio`, `data_hour_range_ratio`, `cross_pre_range_ratio`;
- `day_of_week`, `is_nfp_day`, `is_fomc_day`, `dst_mismatch`.

Assembly SHALL raise an error on any extra or missing column.

#### Scenario: Missing column
- **WHEN** a feature frame lacks `cross_pre_range_ratio`
- **THEN** assembly raises an error naming that column

### Requirement: HAR history of the range
`log_ny_range_lag1`, `log_ny_range_ma5` and `log_ny_range_ma20` SHALL be the target of the
previous eligible day and the means of the targets of the previous 5 and 20 eligible days.
None SHALL include the current day.

#### Scenario: Today's range is excluded
- **WHEN** the current day's session range is doubled
- **THEN** the three HAR features of the current day are unchanged

### Requirement: Pre-session ranges
On day `d`'s bid bars, the windows SHALL be Asia (labelled 00:00–08:00 Berlin), London
(09:00–13:00) and the data hour (14:00). Each `log_*_range` SHALL be the log of the window's
range divided by the bid close of the 14:00 bar. `abs_data_hour_return` SHALL be the
absolute bid move of the 14:00 bar divided by its close.

#### Scenario: Hand-computed London range
- **WHEN** the London bars' highest high is 1.1030, the lowest low is 1.0975 and the 14:00 bid close is 1.1000
- **THEN** `log_london_range` is log(0.005)

### Requirement: Relative and cross-pair ranges
`london_range_ratio` and `data_hour_range_ratio` SHALL divide the window's range by its
mean over the previous 20 days that had one. `cross_pre_range_ratio` SHALL be the mean over
EURUSD, GBPUSD and AUDUSD of each pair's range of the bars labelled 00:00–14:00, divided by
that pair's mean over the previous 20 days that had one.

#### Scenario: Quiet morning everywhere
- **WHEN** each of the three pairs has a 00:00–14:00 range equal to half its previous 20-day mean
- **THEN** `cross_pre_range_ratio` is 0.5

### Requirement: Calendar features
`day_of_week` (0 = Monday to 4 = Friday), `is_nfp_day`, `is_fomc_day` and `dst_mismatch`
SHALL be computed exactly as in fx-session-ny.

#### Scenario: DST-mismatch day
- **WHEN** New York and Berlin are 5 hours apart at noon UTC on the day
- **THEN** `dst_mismatch` is 1

### Requirement: Causal and scale-free
Every feature of day `d` SHALL be unchanged when every bar labelled after 14:00 on `d`, and
every later bar, is removed. Every feature SHALL also be unchanged when every price of
every pair is multiplied by a constant.

#### Scenario: Truncation test
- **WHEN** for 20 random days every bar after the bar labelled 14:00 is removed and the features are rebuilt
- **THEN** every feature value is unchanged

#### Scenario: Scale test
- **WHEN** every price is multiplied by 100
- **THEN** every feature value is unchanged
