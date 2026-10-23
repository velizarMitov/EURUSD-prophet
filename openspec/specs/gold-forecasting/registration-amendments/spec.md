# gold-forecasting/registration-amendments Specification

## Purpose

Lets a later gold research round add hypotheses without loosening the first round's
statistical bar or silently reusing data that has already been looked at.

## Requirements

### Requirement: A new round registers its families before any score
A new round SHALL record each of its hypothesis families before the first validation
score of that round, in a committed `results/gold/PRE_REGISTRATION_V2.{md,json}`. Each
family SHALL have its own registry, with:
- its cells and its alpha (0.05 divided by the family size);
- its metric and its baseline;
- its expected outcome.

Scoring SHALL refuse to run while that file is missing, uncommitted or modified.

#### Scenario: Scoring before registration
- **WHEN** a second-round validation stage is started and `PRE_REGISTRATION_V2.json` is not committed
- **THEN** the run stops before fitting anything and names the missing registration

#### Scenario: Registration edited after scoring
- **WHEN** the committed registration differs from the working copy at scoring time
- **THEN** the run stops and reports that the registration was modified

### Requirement: First-round families are untouched
A new round SHALL NOT change the first round's families, alphas, registries or verdicts.
New families SHALL never count toward the first round's family sizes.

#### Scenario: Registries after round two
- **WHEN** round two finishes
- **THEN** `gold_{direction,return,volatility}_hypothesis_log.csv` are byte-identical to their state before round two

### Requirement: The power rule is not loosened
A direction cell SHALL be admitted only if its eligible validation count reaches the
count the first round's power rule requires (a 3 % edge at 80 % power, at the family's
alpha). Otherwise it SHALL be recorded as "UNDERPOWERED — NO DECISION". Removing an
infeasible cell SHALL NOT increase the alpha of the cells that remain.

#### Scenario: 4-hour direction
- **WHEN** the 4-hour direction cell has fewer eligible validation rows than required
- **THEN** it is registered UNDERPOWERED and the family's alpha is computed from the registered family size

### Requirement: Reuse of seen data is disclosed and the test block stays spent
The registration SHALL state that the validation block was already seen in round one.
No round-two hypothesis SHALL be scored on the test block `[80:100]`. Round-two verdicts
SHALL be labelled as needing forward confirmation.

#### Scenario: Test block requested
- **WHEN** a round-two stage asks for the test block
- **THEN** it refuses and says the test block was spent in round one
