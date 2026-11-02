# Spec Delta

## Purpose

Runs two external, pretrained time-series foundation models (Chronos-2 and TimesFM 2.5) as
pinned, train-free challengers on gold and EUR/USD hourly bars. Their size-of-move forecasts
are judged only on forward data, in their own pre-registered family.

## ADDED Requirements

### Requirement: Weights are pinned and verified
Each model SHALL be loaded only from weights that match a pinned Hugging Face revision and
the SHA-256 digests in `results/external_tsfm/artifact_manifest.csv`. The same SHALL hold
for its fine-tuned adapters. A model whose files are missing or differ SHALL forecast
nothing and SHALL record a failure. The weights SHALL never be updated as a side effect of
a run.

#### Scenario: Altered weight file
- **WHEN** a Chronos-2 weight file differs from its manifest digest
- **THEN** Chronos-2 logs a failure row for that pass and no forecast, and TimesFM still forecasts

#### Scenario: Newer upstream release
- **WHEN** a newer revision of either model is published
- **THEN** the logger keeps using the pinned revision until a new pin is recorded on purpose

### Requirement: The models run apart from the application
The models SHALL run in a separate Python environment with its own pinned requirements. The
application's environment and `requirements.txt` SHALL NOT change. No module on the
production path (`src/inference.py`, `_train_pipeline.py`, `src/paper_trading.py`,
`src/features.py`) SHALL import the external-forecast package. A model call SHALL time out
instead of hanging a pass.

#### Scenario: Side environment missing
- **WHEN** the separate environment is not installed
- **THEN** the pass records "external environment unavailable" in its health row and exits cleanly

#### Scenario: Worker hangs
- **WHEN** a model call runs past its time limit
- **THEN** it is stopped, a failure is recorded, and the pass continues with the other model

### Requirement: Nothing is scored inside the training window
No forecast for an as-of bar before 2025-11-01 SHALL be scored, replayed or reported. Both
models were published in autumn 2025, so earlier bars may be in their pretraining, and
they are in the fine-tuning by design.

#### Scenario: Old bar requested
- **WHEN** the replay is asked to include October 2025
- **THEN** it refuses and names the clean-window start

### Requirement: Fine-tuning uses the whole history before the clean window
Each model SHALL be fine-tuned once per instrument on that instrument's whole H1 history
before 2025-11-01:
- the number of steps SHALL be chosen on the year 2024-11-01 to 2025-10-31, with the
  earlier bars as training;
- the model SHALL then be refit on every bar before 2025-11-01 with that step count.

No bar on or after 2025-11-01 SHALL reach the fine-tuning. The fine-tuned adapters SHALL
be frozen with their digests and never refit while the forward window is open. If
fine-tuning fails, the zero-shot model SHALL remain available.

#### Scenario: Data boundary
- **WHEN** the fine-tuning data is assembled for gold
- **THEN** it holds every cached gold H1 bar before 2025-11-01, and none from that date on

#### Scenario: Fine-tuning failed
- **WHEN** TimesFM fine-tuning for EUR/USD fails
- **THEN** the failure is recorded, and the replay offers only TimesFM's zero-shot configurations for EUR/USD

### Requirement: The replay fixes one configuration before forward scoring
Before the forward registration, a replay on the clean window SHALL choose one
configuration per model and instrument from a grid declared in advance:
- weights: zero-shot or fine-tuned;
- input: hourly log returns or closes;
- context: 2,048 bars, or the model's long context (8,192 for Chronos-2, 16,256 for
  TimesFM, whose 16,384 limit includes the 128-step output patch).

The rule SHALL be the lowest mean absolute error of the size forecast on the first half of
the window. The second half SHALL be reported as a descriptive check. The replay SHALL be
labelled descriptive and SHALL never yield a verdict. The chosen configuration SHALL be
frozen in the pre-registration.

#### Scenario: Configuration chosen
- **WHEN** the replay finishes for gold
- **THEN** it records, per model, all eight first-half errors, the chosen configuration and its second-half error, and labels the run descriptive

#### Scenario: Replay rerun after registration
- **WHEN** the replay is run again after the pre-registration is committed
- **THEN** the frozen configuration does not change

### Requirement: A forecast is a size and a probability read from quantiles
Each model SHALL produce quantiles of the next bar's log return in percent. From them:
- the size forecast SHALL be the median of the absolute return implied by those
  quantiles, which is the error-minimising point for mean absolute error;
- P(up) SHALL be the share of the predictive distribution above zero, interpolated
  between quantile levels.

#### Scenario: Symmetric forecast
- **WHEN** the quantiles are symmetric around 0 with the 25th and 75th percentiles at −0.10 % and +0.10 %
- **THEN** the size forecast is 0.10 % and P(up) is 0.50

#### Scenario: Shifted forecast
- **WHEN** zero lies at the 40th percentile of the predictive distribution
- **THEN** P(up) is 0.60

### Requirement: The logger forecasts the newest closed bar every hour
The logger SHALL run hourly as its own scheduled task with its own lock. Each pass SHALL
forecast the newest closed H1 bar of gold and of EUR/USD with each model, from live MT5
data only. Bars missed since the last pass SHALL be recorded as gaps and never forecast
late. A pass SHALL refuse an instrument whose weekly opens stop matching that instrument's
registered bar-label clock.

#### Scenario: Normal hour
- **WHEN** the 11:01 Sofia pass runs and the newest closed gold bar is labelled 10:00
- **THEN** each model logs one gold forecast for the 11:00 bar, and likewise for EUR/USD

#### Scenario: PC asleep for three hours
- **WHEN** the next pass runs after three missed hours
- **THEN** it forecasts only the newest closed bar and records the missed bars as gaps

#### Scenario: MT5 closed
- **WHEN** MT5 is not reachable
- **THEN** nothing is forecast, and the health row records MT5 unreachable

### Requirement: Settlement follows each instrument's own rules
A forecast SHALL be settled once its target bar exists, with that instrument's own targets
and eligibility rule: the gold family's for gold, and the EUR/USD forward arbiter's for
EUR/USD. A row bordering a market break SHALL settle as not eligible.

#### Scenario: Same realised move as the in-house logger
- **WHEN** a gold forecast for the 11:00 bar settles
- **THEN** its realised absolute return equals the one the gold logger settled for the same as-of bar

### Requirement: A failure here costs nothing elsewhere
The logger SHALL write only under `results/external_tsfm/` and `research_models/external_tsfm/`.
It SHALL only read the gold and EUR/USD loggers' files. It SHALL write one health row per
pass. One model's failure SHALL NOT stop the other model or the other instrument.

#### Scenario: Gold data unavailable
- **WHEN** the gold symbol cannot be read
- **THEN** EUR/USD is still forecast, and the gold failure is recorded

#### Scenario: Write check
- **WHEN** any pass runs
- **THEN** no file under `models/`, `results/gold/`, `results/forward_eval/` or `results/horizon_study/` changes

### Requirement: The family is registered before it is scored
`results/external_tsfm/PRE_REGISTRATION.md` SHALL fix the following before any forward row
is scored:
- the 4 cells: {Chronos-2, TimesFM} × {gold, EUR/USD}, next-hour size;
- the configurations;
- the baselines: gold's frozen `h1_garch_hod`, and a frozen GARCH(1,1) × hour-of-day for
  EUR/USD fitted only on bars before the clean window;
- α = 0.05/4;
- the scoring sample: 1,500 settled eligible rows per cell, or 2027-04-30 if that comes
  first.

Rows SHALL be marked `dry_run` until that file is committed unmodified. Each cell SHALL be
scored once. It SHALL clear only when the block-bootstrap confidence interval of
(baseline error − model error) excludes 0 at that α. A cell with fewer than 500 rows at
the cap SHALL read "UNDERPOWERED — NO DECISION". Direction SHALL NOT be registered or
judged. The family SHALL never change the EUR/USD `0.05/9` bar or any gold registry.

#### Scenario: Before registration
- **WHEN** forward rows exist but the pre-registration is not committed
- **THEN** every row is `dry_run` and the scoring command refuses

#### Scenario: Scored once
- **WHEN** a cell reaches 1,500 settled eligible rows
- **THEN** its verdict is written once, with its count, interval and α, and later rows never change it

#### Scenario: Too few at the cap
- **WHEN** the EUR/USD TimesFM cell has 430 settled eligible rows on 2027-04-30
- **THEN** it reads "UNDERPOWERED — NO DECISION"
