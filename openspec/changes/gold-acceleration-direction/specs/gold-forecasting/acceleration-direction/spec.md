# Spec Delta

## Purpose

Tests, descriptively, whether time-of-day acceleration helps gold models call the next
hour's direction. It covers the acceleration features, the shadow and new models that use
them, their walk-forward backtest and the report. None of it is ever a verdict.

## ADDED Requirements

### Requirement: Acceleration features are causal and scale-free
Each acceleration feature at bar t SHALL be computed from bars labelled at or before t
only. No feature SHALL be a raw price level, a spread or a volume. A feature that needs a
time-of-day profile SHALL use the trailing 60 trading days before t, never a profile
fitted on later data.

#### Scenario: No look-ahead
- **WHEN** every bar after t is changed
- **THEN** every acceleration feature at t is unchanged

#### Scenario: Scale-free
- **WHEN** every price in the history is multiplied by 3
- **THEN** every acceleration feature is unchanged

#### Scenario: Activity against the same hour
- **WHEN** the last hour moved 0.40 % and the same hour of day averaged 0.20 % over the previous 60 trading days
- **THEN** `activity_rel` is 2.0

### Requirement: Shadow models never touch the registered models
The models with acceleration SHALL be new research copies. They SHALL never:
- replace, register, promote or version a registered gold model;
- write under `models/` or into the gold registries;
- write the gold logger files;
- change `gold_config.json`.

They SHALL write only under the gitignored `research_models/gold/accel/` and under
`results/gold/accel/`.

#### Scenario: Registered models unchanged
- **WHEN** the walk-forward and the report have both run
- **THEN** the registered gold artifacts, `gold_config.json` and every registry are byte-identical to before

### Requirement: Each model with acceleration has its own like-for-like comparison
`h1_gbm_accel` and `h1_lstm_accel` SHALL be the registered h1_gbm and h1_lstm
configurations with only the acceleration columns added. Each SHALL be compared with the
same configuration without them, on the same rows. Every model SHALL also be compared with
a coin and with the training-majority class.

#### Scenario: Same rows
- **WHEN** a model with acceleration and its model without it are compared for a year
- **THEN** both are scored on exactly the same eligible rows, and the report gives that row count

### Requirement: Predictions are walk-forward and complete
For each model and prediction year, the fit SHALL use only rows whose bar and next-bar
target are both labelled before that year. Each year SHALL then predict every row
labelled in it. A neural year SHALL exist only as the full five-seed ensemble. A failed
year SHALL be recorded as failed and never filled in.

#### Scenario: Fit ends before the year
- **WHEN** the 2023 predictions are made
- **THEN** the latest target used in the fit belongs to a bar labelled before 2023-01-01

#### Scenario: Short M15 history
- **WHEN** a model uses `accel_m15`, which MT5 holds only from 2022-08
- **THEN** it predicts only the years after its warm-up, and the report compares it with the others on those years only

### Requirement: The report is descriptive
The report SHALL give, for each model, for all hours and for the owner's window
(15:30–23:00 Sofia):
- the hit rate, AUC and Brier score;
- the hit rate minus the coin and minus the training-majority class;
- each model with acceleration minus its model without it, with a paired
  moving-block-bootstrap 95 % interval;
- the same per year.

It SHALL carry the label "descriptive: every year has been seen; not evidence" in its
header and in every section. It SHALL write no verdict and no registry row, and use no
buy or sell wording.

#### Scenario: Interval that holds zero
- **WHEN** the interval of a model with acceleration minus its model without it contains 0
- **THEN** the report says the difference is not distinguishable from zero, and claims no improvement

### Requirement: One fixed net-of-cost check
The report SHALL include exactly one trading check per model, fixed before any result is
seen:
- inside the owner's window, take the sign of p_up − 0.5 at the close of a bar;
- hold for one hour;
- pay the real spread on entry or exit, with the cost floor of `gold-window-strategy`.

No threshold, holding time or filter SHALL be searched.

#### Scenario: Net of cost
- **WHEN** the check is reported
- **THEN** it gives the mean net return per trade, the share of winning trades and the cumulative net return, each with a block-bootstrap 95 % interval, beside the same check for the model without acceleration
