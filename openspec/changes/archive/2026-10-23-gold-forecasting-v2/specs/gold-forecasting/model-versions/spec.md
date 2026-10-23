# Spec Delta

## Purpose

Lets gold models be refreshed on a schedule without breaking the frozen forward test:
each refit runs beside the registered models as a shadow version, until the owner
deliberately promotes it.

## ADDED Requirements

### Requirement: Refits create new versions and never overwrite
A scheduled refit SHALL fit the configured cells on all data available at its run time.
It SHALL write them to a new folder `research_models/gold/versions/<YYYYMMDD>/`, with a
manifest of SHA-256 digests. It SHALL never modify the registered artifacts or their
manifest.

#### Scenario: Monthly refit
- **WHEN** the refit task runs on 2026-12-01
- **THEN** a new version folder and manifest exist, and `results/gold/artifact_manifest.csv` and `research_models/gold/<cell>/` are byte-identical to before

### Requirement: New versions forecast in shadow
The logger SHALL forecast with every active shadow version alongside the registered
models. Each shadow row SHALL carry its version's digest and a shadow flag. Shadow rows
SHALL never count toward a registered forward verdict.

#### Scenario: Forward report
- **WHEN** the forward report scores h1_lstm
- **THEN** it uses only rows from the registered digest, and shadow rows appear only on the leaderboard

### Requirement: Promotion is a manual, recorded decision
A version SHALL become the registered model only through an explicit promote command
run by the owner. The command SHALL refuse while a registered forward window is still
open, and SHALL record the old and new digests and the reason in `results/gold/RESULTS.md`.
A promotion SHALL start a new forward window.

#### Scenario: Promote during an open window
- **WHEN** the owner runs promote while the registered forward window has not ended
- **THEN** the command refuses and states when the window ends

#### Scenario: Promote after the window
- **WHEN** the window has ended and the owner promotes version 20261201 with a reason
- **THEN** the registered manifest points to the new digests, RESULTS.md records both digests and the reason, and a new window begins

### Requirement: A refit cannot run twice at once or block the logger
The refit SHALL take its own lock, run outside the logger's lock, and give up cleanly if
the previous refit is still running.

#### Scenario: Overlap
- **WHEN** a refit starts while the previous one still holds its lock
- **THEN** it exits without fitting, and records the skip in the health log
