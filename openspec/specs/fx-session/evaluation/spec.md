# fx-session/evaluation Specification

## Purpose

Defines how the fx-session family is decided. A purged walk-forward from 2013 trains a
LightGBM model, and a conditional CUDA MLP, on each pair. The out-of-sample predictions
decide four pre-registered cells once, at alpha 0.05/4, gross of cost.

## Requirements

### Requirement: Expanding walk-forward folds
Evaluation folds SHALL be consecutive 6-month spans starting 2013-01-01, with the same
calendar boundaries for both pairs. Each fold's training set SHALL be all eligible days
before the fold. Its validation slice, used only for early stopping, SHALL be the last 15 %
of those days in time order.

#### Scenario: Shared boundaries
- **WHEN** folds are built for EURUSD and GBPUSD
- **THEN** both pairs have identical fold start and end dates

### Requirement: Purge and embargo
At least one eligible trading day SHALL separate the fit slice from the validation slice,
and the validation slice from the evaluation fold. Rows SHALL never be shuffled across
time. Any scaler SHALL be fit on the fit slice only.

#### Scenario: Day gap
- **WHEN** the slices of a fold are inspected
- **THEN** the last fit day, the first validation day, the last validation day and the first evaluation day are each at least one eligible day apart

### Requirement: Class-prior reference
For each fold, the reference forecast SHALL be the up-rate of that fold's fit slice,
applied as a constant probability to every evaluation day.

#### Scenario: Prior value
- **WHEN** a fold's fit slice has 52 % up days
- **THEN** the reference predicts 0.52 for every day in that fold

### Requirement: LightGBM model
LightGBM SHALL be trained per pair and fold as a binary classifier on log-loss. Its
hyper-parameters SHALL be fixed in the config, never searched. It SHALL use no class
weights, a fixed seed, deterministic training, and early stopping on the validation slice.

#### Scenario: Same seed
- **WHEN** a fold is trained twice with the same config
- **THEN** the predicted probabilities are identical

### Requirement: Conditional MLP
The MLP architecture SHALL be fixed in the config before any run: hidden sizes, dropout,
optimiser, learning-rate schedule, early stopping and a 5-seed ensemble. It SHALL train on
CUDA with no CPU fallback and no class weights. It SHALL train for a pair only after that
pair's LightGBM cell is scored KEEP.

#### Scenario: LightGBM dropped
- **WHEN** EURUSD's LightGBM cell is DROP
- **THEN** no EURUSD MLP is trained and the EURUSD MLP cell is recorded NOT TRIGGERED

#### Scenario: No GPU
- **WHEN** the MLP stage starts and CUDA is unavailable
- **THEN** the stage stops with an error and trains nothing

### Requirement: Four registered cells
The family SHALL register exactly four cells at alpha = 0.05/4 = 0.0125:
- EURUSD LightGBM against the class prior;
- GBPUSD LightGBM against the class prior;
- EURUSD MLP against LightGBM;
- GBPUSD MLP against LightGBM.

Alpha unspent by an untriggered cell SHALL NOT be reassigned to any other cell.

#### Scenario: One cell triggered
- **WHEN** only GBPUSD's LightGBM cell is KEEP
- **THEN** the GBPUSD MLP cell is tested at 0.0125 and the EURUSD MLP cell is NOT TRIGGERED, with no change to any alpha

### Requirement: Cell test
Each cell SHALL be tested on the per-day log-loss difference, model minus reference, over
all pooled out-of-sample days of that pair. The test SHALL be a one-sided circular block
bootstrap with a block of 5 days, 10 000 resamples and a fixed seed. A cell SHALL be KEEP
only when the upper bound of the one-sided interval at its alpha is below zero.

#### Scenario: Interval reaches zero
- **WHEN** a cell's one-sided upper bound is +0.0004
- **THEN** the cell is DROP

### Requirement: One-shot scoring and registry
Verdicts SHALL be written to the family's own registry, `fx_session_hypothesis_log.csv`,
once per cell. A second scoring of a cell already in the registry SHALL be refused, and
the registry left byte-identical. Each note SHALL state its decision basis and that the
result is gross and not a trading edge. Notes SHALL also record these earlier results as
facts, not blockers:
- the horizon study's session cells at 4 h and 6.5 h;
- the H1 direction family;
- the fx-fibonacci DROPs.

#### Scenario: Second scoring
- **WHEN** the LightGBM cells are scored a second time
- **THEN** scoring is refused and the registry is unchanged

### Requirement: Config guard
Every label, training and scoring stage SHALL refuse to run when the config digest differs
from the digest recorded with the verdicts or with the predictions being scored. A changed
config SHALL be treated as a new hypothesis.

#### Scenario: Changed parameter
- **WHEN** the entry hour in the config is changed after the LightGBM cells are scored
- **THEN** the label, training and scoring stages each refuse to run

### Requirement: Power report before scoring
Before any cell is scored, the family SHALL report, from the eligible-day counts alone, the
minimum detectable accuracy at alpha 0.0125 one-sided and 80 % power for each pair. The
report SHALL also give the matching minimum log-loss improvement. It SHALL NOT use any
model output.

#### Scenario: Power from counts
- **WHEN** a pair has 3 450 out-of-sample days
- **THEN** the report gives a minimum detectable accuracy of about 52.6 %

### Requirement: Descriptive report
The walk-forward report SHALL show, per pair, for the prior, LightGBM and any trained MLP:
- pooled and per-fold log-loss, Brier score and accuracy with a Wilson 95 % interval;
- accuracy per year and on DST-mismatch days;
- a reliability table and the bid-versus-mid parity check.

It SHALL also show the sign rule's gross result in pips and percent. All of it SHALL be
labelled descriptive. Two builds from the same inputs SHALL differ only in the generation
timestamp.

#### Scenario: Reproducible report
- **WHEN** the report is built twice from the same saved predictions and registry
- **THEN** the two files differ only in the generation line

### Requirement: Family isolation
The family SHALL never write under `models/`, never change `config.json`, and never write
another family's registry or results. Its trained artifacts SHALL be saved under
`research_models/fx_session/`, with a SHA-256 manifest in `results/fx_session/`. A
manifest digest mismatch SHALL refuse loading.

#### Scenario: Other registries
- **WHEN** a full fx-session run and scoring complete
- **THEN** every other family's registry file is byte-identical to before
