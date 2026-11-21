# Spec Delta

## Purpose

Defines how the fx-rates family is decided: a purged and embargoed walk-forward from 2015
trains LightGBM, and a conditional CUDA MLP, per pair. The out-of-sample predictions decide
four pre-registered cells once, at alpha 0.05/4, gross of cost.

## ADDED Requirements

### Requirement: Expanding walk-forward folds
Evaluation folds SHALL be consecutive 6-month spans from 2015-01-01, with the same calendar
boundaries for both pairs. Each fold's training set SHALL be the usable days before the
fold. Its validation slice, used only for early stopping, SHALL be the last 15 % of those
days in time order.

#### Scenario: Shared boundaries
- **WHEN** folds are built for EURUSD and GBPUSD
- **THEN** both pairs have identical fold start and end dates, the first starting 2015-01-01

### Requirement: Five-day purge plus five-day embargo
Between fit and validation, and between validation and evaluation, the last 5 FX trading
days before the later slice SHALL be purged. A further 5 days SHALL be embargoed, so no
used target window reaches the next slice. Rows SHALL never be shuffled. Any scaler SHALL be
fit on the fit slice only.

#### Scenario: Gap between slices
- **WHEN** the slices of a fold are inspected
- **THEN** at least 10 FX trading days separate the last fit day from the first validation day, and the last validation day from the first evaluation day

### Requirement: Class-prior reference
For each fold, the reference forecast SHALL be the up-rate of that fold's fit slice,
applied to every evaluation day.

#### Scenario: Prior value
- **WHEN** a fold's fit slice has 51 % up days
- **THEN** the reference predicts 0.51 for every day of that fold

### Requirement: LightGBM model
LightGBM SHALL be trained per pair and fold as a binary classifier on log-loss. Its
hyper-parameters SHALL be fixed in the config and never searched. It SHALL use no class
weights, a fixed seed, deterministic training, and early stopping on the validation slice.

#### Scenario: Same seed
- **WHEN** a fold is trained twice with the same config
- **THEN** the predicted probabilities are identical

### Requirement: Conditional MLP
The MLP architecture, optimiser, schedule, early stopping and 5 seeds SHALL be fixed in the
config before any run. It SHALL train on CUDA with no CPU fallback and no class weights. It
SHALL train for a pair only after that pair's LightGBM cell is KEEP.

#### Scenario: LightGBM dropped
- **WHEN** GBPUSD's LightGBM cell is DROP
- **THEN** no GBPUSD MLP is trained and its MLP cell is recorded NOT TRIGGERED

### Requirement: Four registered cells
The family SHALL register exactly four cells at alpha = 0.05/4 = 0.0125:
- EURUSD LightGBM against the prior;
- GBPUSD LightGBM against the prior;
- EURUSD MLP against LightGBM;
- GBPUSD MLP against LightGBM.

Alpha unspent by an untriggered cell SHALL NOT be reassigned.

#### Scenario: No trigger
- **WHEN** both LightGBM cells are DROP
- **THEN** both MLP cells are NOT TRIGGERED and every alpha stays 0.0125

### Requirement: Cell test on overlapping days
Each cell SHALL be tested on the per-day log-loss difference, model minus reference, over
the pair's pooled out-of-sample days in date order. The test SHALL be a one-sided circular
block bootstrap with a block of 10 FX trading days, 10 000 resamples and a fixed seed. A
cell SHALL be KEEP only when the upper bound at its alpha is below zero.

#### Scenario: Interval reaches zero
- **WHEN** a cell's one-sided upper bound is +0.0002
- **THEN** the cell is DROP

### Requirement: One-shot scoring and registry
Verdicts SHALL be written once per cell to `fx_rates_hypothesis_log.csv`. A second scoring
of a recorded cell SHALL be refused, and the registry left byte-identical. Each note SHALL
state the decision basis, and that the result is gross and not a trading edge. Notes SHALL
record these earlier results as facts, not blockers:
- hypotheses #1 and #3 of the feature registry;
- the horizon study's daily cells;
- the fx-fibonacci and fx-session verdicts.

#### Scenario: Second scoring
- **WHEN** the LightGBM cells are scored a second time
- **THEN** scoring is refused and the registry is unchanged

### Requirement: Config guard
Every label, training and scoring stage SHALL refuse to run when the config digest differs
from the digest recorded with the verdicts or with the predictions being scored.

#### Scenario: Changed horizon
- **WHEN** the horizon in the config is changed after the LightGBM cells are scored
- **THEN** the label, training and scoring stages each refuse to run

### Requirement: Descriptive weekly rule
The report SHALL show, per pair and model, a non-overlapping weekly rule, gross of cost and
labelled descriptive. The rule goes long when P(up) > 0.5, else short, on every fifth
out-of-sample day, and holds 5 days. The report SHALL give:
- the rule's accuracy with a Wilson 95 % interval;
- its mean and total pips;
- its Sharpe at √52 and its profit factor;
- the mean over the five start offsets.

#### Scenario: Offsets
- **WHEN** the weekly rule is computed
- **THEN** the report shows offset 0 and the mean over offsets 0–4 for each pair and model

### Requirement: Reproducible report and family isolation
Two report builds from the same inputs SHALL differ only in the generation line. The family
SHALL never write under `models/`, change `config.json`, or write another family's caches,
results or registry. Its artifacts SHALL live under `research_models/fx_rates/`, with a
SHA-256 manifest.

#### Scenario: Other families untouched
- **WHEN** a full fx-rates run and scoring complete
- **THEN** every other registry and the fx_session caches are byte-identical to before
