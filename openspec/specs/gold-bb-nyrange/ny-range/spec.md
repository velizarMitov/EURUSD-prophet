# gold-bb-nyrange/ny-range Specification
## Purpose

Defines Part 2 of gold-bb-nyrange: the size of gold's New York session range and how it is
forecast. Fourteen fixed features known at the close of the bar labelled 14:00 Berlin feed
HAR, HAR-X, LightGBM and an MLP in one walk-forward.

## Requirements
### Requirement: Session-range target
For each New York day, the target SHALL be
`y = log((max high − min low of the H1 bars labelled 15:00 … 21:00 Berlin) / close of the 14:00 bar)`.
A day SHALL be eligible only when its D1 bar is valid and all eight bars, 14:00 through 21:00,
exist.

#### Scenario: Early close
- **WHEN** the 21:00 bar is missing on a holiday
- **THEN** that day is not eligible and has no target

#### Scenario: Scale-free
- **WHEN** every price of a day is multiplied by 2
- **THEN** its target and all its features are unchanged

### Requirement: Gold range features at the 14:00 close
Each eligible day SHALL carry nine gold features from bars labelled 14:00 Berlin or earlier
and earlier days' targets: `log_ny_range_lag1`, `log_ny_range_ma5`, `log_ny_range_ma20`
(mean target over the previous 1, 5 and 20 eligible days); `log_asia_range` (00:00–08:00) and
`log_london_range` (09:00–13:00) over the 14:00 close; `log_data_hour_range` and
`abs_data_hour_return` of the 14:00 bar; `london_range_ratio` and `data_hour_range_ratio`
over their previous 20-day means.

#### Scenario: No look-ahead
- **WHEN** the features of a day are computed on the full data and on data cut after the 14:00 bar
- **THEN** both give identical values

### Requirement: Dollar and calendar features
Each eligible day SHALL also carry `dollar_pre_range_ratio`, the hourly DXY range over the
bars labelled 00:00–14:00 over its previous 20-day mean, and `day_of_week`, `is_nfp_day`,
`is_fomc_day` and `dst_mismatch` from fx-session's calendar, read-only, for fourteen features
in all. `dollar_pre_range_ratio` SHALL count as missing when that DXY day has fewer than 10 of
those bars.

#### Scenario: DXY hole
- **WHEN** DXY has no bars on a day, such as during the June 2024 gap
- **THEN** that gold day is excluded as `warm-up`

### Requirement: Missing features
A day with any missing feature SHALL be excluded as `warm-up` and counted in the label report.

#### Scenario: First weeks
- **WHEN** a day has fewer than 20 earlier eligible days
- **THEN** it is excluded as `warm-up`

### Requirement: Four models
The walk-forward SHALL produce four forecasts of `y` for each scored day:
- HAR: least squares on the three `log_ny_range` columns;
- HAR-X: least squares on all fourteen features;
- LightGBM: a regressor with fx-ny-volatility's fixed hyperparameters;
- MLP: the mean of a 5-seed network with fx-ny-volatility's fixed architecture.

The MLP SHALL refuse to run without CUDA.

#### Scenario: No GPU
- **WHEN** CUDA is not available
- **THEN** the MLP stage stops with an error instead of training on the CPU

### Requirement: Walk-forward folds
Scoring SHALL start on 2018-01-01, in 6-month folds. Each fold SHALL train on days with all
fourteen features that end at least 1 day before it starts. LightGBM and the MLP SHALL use the
last 15 % of those days for early stopping, and HAR and HAR-X SHALL use all of them. All four
models SHALL see the same rows.

#### Scenario: Fold boundary
- **WHEN** the fold starting 2020-01-01 is trained
- **THEN** no training day is on or after 2019-12-31

#### Scenario: Same rows for every model
- **WHEN** a day lacks `dollar_pre_range_ratio`
- **THEN** none of the four models trains or is scored on it

### Requirement: Frozen final models
After scoring, all four models SHALL be refit on every eligible day up to 2026-11-20. They SHALL
be stored under `research_models/gold_bbnr/`, with SHA-256 digests in
`results/gold_bbnr/frozen_manifest.csv`. Loading a frozen model SHALL refuse a file whose
digest differs.

#### Scenario: Altered frozen file
- **WHEN** a frozen model file is changed after its digest was recorded
- **THEN** loading it raises an error
