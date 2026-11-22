# Spec Delta

## Purpose

Defines how the fx-bb-reversion family is decided: the rule cell and the pooled LightGBM
cell, their day-block bootstrap tests, the two-cell registry scored once, the config
guard, and the descriptive report gross and after spread.

## ADDED Requirements

### Requirement: Two registered cells
The family SHALL register exactly two cells, each at `alpha = 0.025` (0.05/2), in its own
registry: `bb_rule_mean_reversion` (Cell 1) and `bb_lightgbm_vs_base_rate` (Cell 2). It
SHALL never touch the project's `0.05/9` bar or any other registry.

#### Scenario: Other registries unchanged
- **WHEN** both cells are scored
- **THEN** every other registry file in the repository is byte-identical before and after

### Requirement: Rule cell on every eligible event
Cell 1 SHALL be scored on every eligible event, from the pair's first complete bands to
the last event that has an outcome, with no fitted parameter. Its statistic SHALL be the
mean outcome `r` over the events.

#### Scenario: Window fixed in advance
- **WHEN** Cell 1 is scored
- **THEN** its first and last event dates follow from the data rules alone and are recorded in the registry row

### Requirement: Day-block bootstrap
Both cells SHALL be tested with a one-sided circular block bootstrap over business days
(Monday to Friday), in 5-day blocks, with 10 000 resamples and a fixed seed. Each
resample SHALL draw the same blocks for the daily sums and the daily event counts, and
SHALL recompute the event-weighted mean.

#### Scenario: Equivalence with the project bootstrap
- **WHEN** every day carries exactly one event
- **THEN** the bootstrap distribution equals the project's circular block bootstrap of the mean for the same seed

### Requirement: Rule verdict
Cell 1 SHALL be KEEP when the alpha-quantile of its bootstrap means is above zero, and
DROP otherwise.

#### Scenario: No reversion
- **WHEN** the event outcomes average +0.02 ATR with a bootstrap alpha-quantile of −0.05
- **THEN** Cell 1 is DROP

### Requirement: Bid parity flag
Cell 1 SHALL also be computed with bands, ATR and outcomes from bid closes. The registry
SHALL record that statistic and a parity flag when it differs from the mid statistic by
more than 0.02 ATR. The flag SHALL not change the verdict.

#### Scenario: Parity gap
- **WHEN** the mid statistic is 0.12 and the bid statistic is 0.08
- **THEN** the registry row carries the parity flag and the verdict follows the mid test alone

### Requirement: Walk-forward for the model cell
Cell 2 SHALL use expanding 12-month folds from 2015-01-01. Each fold SHALL train only on
events whose `t+3` date is more than 10 business days before the fold's first date. Early
stopping SHALL use the last 15 % of those events by date, with the same 10-day gap before
it. Every event SHALL be predicted out of sample once.

#### Scenario: No training outcome reaches the test fold
- **WHEN** a fold starts on 2020-01-01
- **THEN** every training event's `t+3` date is more than 10 business days before 2020-01-01

### Requirement: Pooled LightGBM against the base rate
Cell 2 SHALL compare a single LightGBM model pooled over the six pairs, with fixed settings
and no class weights, against the fit slice's up-rate as a constant probability. It SHALL
use log-loss per event, summed per day for the bootstrap. Cell 2 SHALL be KEEP when the
(1 − alpha)-quantile of the bootstrap mean difference is below zero.

#### Scenario: No filter value
- **WHEN** the model's mean log-loss is above the base rate's
- **THEN** Cell 2 is DROP

### Requirement: One-shot scoring and config guard
Each cell SHALL be scored once: a second scoring SHALL be refused and leave the registry
byte-identical. Every stage SHALL refuse to run when the config digest differs from the one
recorded with earlier stages or verdicts. A changed config is a new hypothesis.

#### Scenario: Second scoring
- **WHEN** Cell 1 has a registry row and scoring is run again
- **THEN** it is refused and the registry file is unchanged

#### Scenario: Changed band width
- **WHEN** the band width in the config is changed from 2 to 2.5 after the label report
- **THEN** the walk-forward and scoring stages refuse to run

### Requirement: Registry content
Each registry row SHALL record the hypothesis, alpha, block length, event and distinct-date
counts, first and last event dates, the statistic, its bound, the verdict, the config and
feature-list digests and a note. The note SHALL state that the result is gross of cost and
not a trading result, and SHALL name the earlier fx families as facts.

#### Scenario: Note text
- **WHEN** a row is written
- **THEN** its note says "gross of spread and slippage, not a trading edge" and names fx-fibonacci, fx-session and fx-rates

### Requirement: Descriptive report
The report SHALL give, for every event and for Cell 2's out-of-sample events above and
below P(win) 0.5: win rate with a Wilson interval on distinct dates, mean `r`, and mean
pips and profit factor, gross and after the pair's mean closing-bar spread charged once
per event. It SHALL break these down per pair, per year and per breach side. Only the
registry rows are verdicts.

#### Scenario: Pip size
- **WHEN** a USDJPY event's mid closes move 0.45 in its favour
- **THEN** it is reported as +45 pips gross

#### Scenario: Two identical builds
- **WHEN** the report is built twice from the same saved inputs
- **THEN** the two files differ only in the generation line

### Requirement: Protected paths
The family SHALL never write under `models/`, `config.json`, the fx_session, fx_fib or
fx_rates results or configs, or any other family's registry. Its frozen artifacts SHALL be
kept in a gitignored research directory with their digests recorded.

#### Scenario: Earlier families pinned
- **WHEN** the family's tests run
- **THEN** the fx_fib, fx_session and fx_rates configs match the digests recorded with their verdicts
