# fx-ny-volatility/evaluation Specification

## Purpose

Defines how the fx-ny-volatility family is decided:
- the walk-forward folds;
- the HAR reference, the LightGBM regressor and the conditional MLP;
- the four-cell registry, scored once against squared error;
- the config guard;
- the descriptive report.

## Requirements

### Requirement: Walk-forward folds
The evaluation SHALL use expanding 6-month folds from 2013-01-01, with the same boundaries
for both pairs. Each fold SHALL train only on eligible days before its first day, minus a
1-day embargo, and SHALL hold out the last 15 % of those days for early stopping. Every
out-of-sample day SHALL be predicted exactly once.

#### Scenario: No training day inside the test fold
- **WHEN** a fold starts on 2019-07-01
- **THEN** the last eligible day before it, Friday 2019-06-28, is embargoed and every training and validation day is on or before 2019-06-27

### Requirement: HAR reference
For each pair and fold, the reference SHALL be an ordinary least-squares fit of the target
on an intercept, `log_ny_range_lag1`, `log_ny_range_ma5` and `log_ny_range_ma20`. It SHALL
use only that fold's training days.

#### Scenario: Reference uses training days only
- **WHEN** the fold's test days are altered
- **THEN** the fold's HAR coefficients are unchanged

### Requirement: LightGBM regressor
For each pair, a LightGBM regressor with the squared-error objective and fixed settings
SHALL be trained per fold on all fourteen features, with early stopping on the validation
days. A same-seed run SHALL be identical.

#### Scenario: Reproducible run
- **WHEN** the stage-1 walk-forward is run twice with the same config
- **THEN** the out-of-sample LightGBM predictions are identical

### Requirement: Conditional MLP
An MLP regressor, the mean of 5 seeds trained on CUDA only, SHALL be trained for a pair
only after that pair's LightGBM cell is KEEP. Without that KEEP, the pair's MLP cell SHALL
be recorded as NOT TRIGGERED without training, and its alpha SHALL never be reassigned.

#### Scenario: Both LightGBM cells DROP
- **WHEN** both LightGBM cells are DROP
- **THEN** no MLP is trained and both MLP cells are recorded as NOT TRIGGERED

#### Scenario: No CUDA
- **WHEN** a KEEP pair's MLP stage starts on a machine without CUDA
- **THEN** it refuses to run

### Requirement: Four registered cells
The family SHALL register four cells at `alpha = 0.0125` (0.05/4) in its own registry:
- `EURUSD_lightgbm_vs_har` and `GBPUSD_lightgbm_vs_har`;
- `EURUSD_mlp_vs_lightgbm` and `GBPUSD_mlp_vs_lightgbm`.

It SHALL never touch the `0.05/9` bar, the production volatility registry or any other
registry.

#### Scenario: Other registries unchanged
- **WHEN** all four cells are scored or recorded
- **THEN** every other registry file in the repository is byte-identical before and after

### Requirement: Squared-error test
Each cell SHALL be tested on the per-day difference in squared error: the model minus the
reference. The test SHALL be a one-sided circular block bootstrap with 5-day blocks,
10 000 resamples and a fixed seed. A cell SHALL be KEEP only when the (1 − alpha) upper
bound of the mean difference is below zero.

#### Scenario: Better on average but not significant
- **WHEN** the mean difference is −0.001 and the upper bound is +0.0004
- **THEN** the cell is DROP

### Requirement: One-shot scoring and config guard
Each cell SHALL be scored once: a second scoring SHALL be refused and leave the registry
byte-identical. Every stage SHALL refuse to run when the config digest differs from the one
recorded with earlier stages or verdicts.

#### Scenario: Second scoring
- **WHEN** the LightGBM cells have registry rows and scoring is run again
- **THEN** it is refused and the registry file is unchanged

#### Scenario: Changed window
- **WHEN** the session window in the config is changed after stage 1
- **THEN** the scoring and stage-2 commands refuse to run

### Requirement: Registry content
Each scored row SHALL record:
- the day count, both MSEs and the relative MSE reduction;
- the mean difference, its upper bound and `mde_80`;
- the verdict, the config and feature-list digests, and a note.

The note SHALL state that the result is gross and not a trading result, and SHALL name the
earlier fx families, including fx-session-ny's direction DROP, as facts.

#### Scenario: Relative reduction recorded
- **WHEN** LightGBM's MSE is 0.180 and HAR's is 0.200
- **THEN** the row records a relative MSE reduction of 10 %

### Requirement: Descriptive report
The report SHALL give, per pair and model:
- MSE, MAE, out-of-sample R² and the MSE ratio to HAR;
- Spearman's ρ;
- precision, recall and accuracy for days above a causal Q75 (the previous 60 eligible
  days);
- MSE per year, per fold and on NFP, FOMC and DST-mismatch days;
- a HAR-X line fitted with all fourteen features.

Only the registry rows are verdicts.

#### Scenario: Causal hot-day threshold
- **WHEN** the hot-day threshold for a day is computed
- **THEN** it uses only the targets of the previous 60 eligible days

#### Scenario: Two identical builds
- **WHEN** the report is built twice from the same saved inputs
- **THEN** the two files differ only in the generation line

### Requirement: Protected paths
The family SHALL never write under `models/`, `config.json`, the production volatility
results, the fx_session, fx_fib or fx_rates results or configs, or any other family's
registry. Its frozen artifacts SHALL be kept in a gitignored research directory with their
digests recorded.

#### Scenario: Earlier families pinned
- **WHEN** the family's tests run
- **THEN** the fx_fib, fx_session and fx_rates configs match the digests recorded with their verdicts
