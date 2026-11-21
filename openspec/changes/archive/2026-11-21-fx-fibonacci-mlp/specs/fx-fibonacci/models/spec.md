# Spec Delta

## Purpose

Defines the fx-fibonacci family's models: the compact neural network ensemble trained on
the GPU, the gradient-boosting baseline it must be compared with, and the training
controls that guard against overfitting and leakage.

## ADDED Requirements

### Requirement: The registered network architecture
The registered model SHALL be an MLP with these layers in order:
- Linear(n_features, 128), LayerNorm, ReLU, Dropout 0.3;
- Linear(128, 64), LayerNorm, ReLU, Dropout 0.2;
- Linear(64, 32), ReLU;
- Linear(32, 3).

Its output SHALL be one softmax probability per class.

#### Scenario: Output shape
- **WHEN** the model scores a batch of N rows
- **THEN** it returns N × 3 probabilities, and each row sums to 1

### Requirement: Training controls
Training SHALL use float32 and class-weighted cross-entropy, with the weights computed from
the train fold only. The optimiser SHALL be AdamW with the configured `weight_decay`.
Training SHALL include ReduceLROnPlateau and early stopping on validation log-loss. The best
validation epoch's weights SHALL be kept.

#### Scenario: Early stop
- **WHEN** validation log-loss has not improved for the configured patience
- **THEN** training stops, and the model holds the weights of its best validation epoch

### Requirement: The validation slice is inside the train fold
Each fold's early-stopping slice SHALL be the chronological tail of that fold's training
period. It SHALL be separated from the fit rows by the purge and the embargo. Rows being
evaluated SHALL never steer early stopping, learning-rate changes or scaler fitting.

#### Scenario: Evaluation rows untouched
- **WHEN** a fold trains
- **THEN** no row of that fold's evaluation slice is read before training ends

### Requirement: Scalers are fit on train rows only
Feature scaling SHALL be fit on each fold's fit rows only. It SHALL then be applied
unchanged to that fold's validation and evaluation rows, and saved with the fold's models.

#### Scenario: Scaler statistics
- **WHEN** a fold's scaler is inspected
- **THEN** its statistics equal those computed from that fold's fit rows alone

### Requirement: Time order is never shuffled across splits
Train, validation and evaluation splits SHALL be chronological. Shuffling SHALL never be
used to form them. When minibatch order within the fit rows is shuffled, the shuffle SHALL
use a seeded generator.

#### Scenario: Split order
- **WHEN** splits are built
- **THEN** every fit row is earlier than every validation row, and every validation row is earlier than every evaluation row

### Requirement: The seed ensemble is all-or-nothing
The registered prediction SHALL be the mean of the class probabilities of exactly five
models with the configured seeds. A prediction with fewer than five members SHALL never be
logged or scored.

#### Scenario: One member missing
- **WHEN** one of the five frozen members fails to load or fails its digest check
- **THEN** no ensemble prediction is produced, and the failure is logged

### Requirement: Training runs on CUDA and records its device
Training SHALL run on the CUDA device when it is available. Each run SHALL record:
- the device name;
- the torch and CUDA versions;
- the seed;
- the epochs trained.

A run on CPU SHALL happen only when the config explicitly allows it.

#### Scenario: CUDA missing
- **WHEN** CUDA is unavailable and the config does not allow CPU
- **THEN** training refuses to start and says why

### Requirement: Runs are reproducible
Two training runs with the same data, config, seed and device SHALL produce predicted
probabilities that differ by no more than 1e-5.

#### Scenario: Rerun
- **WHEN** the same fold and seed are trained twice on the same device
- **THEN** the largest absolute probability difference is at most 1e-5

### Requirement: The LightGBM baseline
A LightGBM multiclass model SHALL be trained on exactly the same rows, features, folds,
class weights and early-stopping slice as the MLP. Its hyperparameters SHALL be fixed in
the config. They SHALL never be searched or tuned.

#### Scenario: Same inputs
- **WHEN** a fold is trained
- **THEN** the MLP and LightGBM receive identical row indices and feature columns

### Requirement: Descriptive variants are never registered
A 64-32 MLP and an L2 logistic regression SHALL be trained on the same folds and reported.
They SHALL be labelled descriptive. Neither SHALL be a model under test in the registry
or the scoring.

#### Scenario: Registry content
- **WHEN** the registry is read
- **THEN** no row names the 64-32 MLP or the logistic regression as the model under test

### Requirement: Artifacts never touch production
Frozen models, scalers and fold metadata SHALL be written only under
`research_models/fx_fib/`, with their SHA-256 digests in
`results/fx_fib/artifact_manifest.csv`. Nothing SHALL be written under `models/`. The family
SHALL never read `config.json`.

#### Scenario: Digest mismatch
- **WHEN** a frozen artifact's digest differs from the manifest
- **THEN** loading refuses that artifact
