# Spec Delta

## Purpose

Defines how the reliability of RTS spot recommendations is judged. It covers the descriptive look at the
history already seen, the forward ledger, the pre-registration and the single scored cell.

## ADDED Requirements

### Requirement: Historical check is descriptive
The historical report SHALL show, per pair and in total:
- RTS's stated points;
- our gross and net base-trade results;
- the reconciliation counts.

It covers RTS's November 2026 track record and the reports already received. It SHALL state that the result
was seen first and SHALL carry no verdict.

#### Scenario: Historical report
- **WHEN** the historical report is built from the uploaded week and the November track record
- **THEN** it shows the counts and both results and says `descriptive — no verdict`

### Requirement: Forward ledger
Every spot recommendation dated after the pre-registration commit SHALL be added to an append-only forward
ledger once its session has ended. The entry SHALL hold the base-trade outcome or the reason there is none. A
settled row SHALL never change, and a missed report SHALL never be backfilled from a later email.

#### Scenario: Late email
- **WHEN** a report dated 2026-12-03 is first fetched on 2026-12-10
- **THEN** it is ledgered with `received late` and excluded from the scored cell

### Requirement: Pre-registration gate
`results/rts_signals/PRE_REGISTRATION.md` and `.json` SHALL be committed before any forward trade counts. Every
scoring run SHALL refuse unless:
- both files are tracked by git and unchanged;
- the family config digest matches;
- the rule-definition digest matches.

#### Scenario: Uncommitted
- **WHEN** scoring runs while `PRE_REGISTRATION.json` is untracked
- **THEN** it refuses and writes nothing

### Requirement: The cell
One cell, `rts_spot_net_r`, SHALL be registered at `alpha = 0.05` in
`results/rts_signals/rts_signals_hypothesis_log.csv`:
- it tests that the mean net R of forward base trades is above 0;
- it uses a one-sided bootstrap over 5-day blocks of report dates;
- it is KEEP only if the lower bound is above 0.

#### Scenario: No edge
- **WHEN** synthetic forward trades have zero mean net R
- **THEN** the cell is DROP

#### Scenario: Planted edge
- **WHEN** synthetic forward trades have a clearly positive mean net R
- **THEN** the cell is KEEP

### Requirement: Scored once at 300 trades or 2027-06-30
Scoring SHALL run once, on the first 300 forward base trades or on 2027-06-30, whichever comes first:
- with fewer than 300 trades on that date, the cell SHALL be recorded `UNDERPOWERED — NO DECISION`;
- earlier scoring SHALL refuse and write nothing;
- a second scoring SHALL refuse and leave the registry unchanged.

#### Scenario: Too early
- **WHEN** scoring runs with 120 forward trades before 2027-06-30
- **THEN** it refuses and the registry is unchanged

### Requirement: Stated power and scope
The pre-registration SHALL state the minimum detectable mean net R at 300 trades. It SHALL also state that:
- results are net of the recorded spread only, not slippage, swap or commission;
- no verdict is a trading instruction;
- the family never touches another family's registry, `models/` or `config.json`.

#### Scenario: Power stated
- **WHEN** the pre-registration is written
- **THEN** it states the minimum detectable effect and its calculation

### Requirement: Plain report
`results/rts_signals/report.md` SHALL show the forward trade count out of 300 and the days without a usable
report, by reason. Before scoring it SHALL show the running counts of targets, stops and time exits, but no
mean, total or test result, and say `not scored yet`. After scoring it SHALL show the verdict.

#### Scenario: Before scoring
- **WHEN** the report is built with 80 forward trades and no score
- **THEN** it shows `80 / 300` and the exit counts, and no mean R or total pips
