# Spec Delta

## ADDED Requirements

### Requirement: The panel shows the second opinion
On a gold chart at H1 or lower, the panel SHALL add one line with each external model's
current gold size and whether it agrees with h1_lstm on size. The line SHALL be omitted
when no external forecast is current. It SHALL carry no trade words, and its hover text
SHALL carry "external model — research, no proven edge".

#### Scenario: Both models current
- **WHEN** Chronos-2 expects 0.18 %, TimesFM 0.15 %, h1_lstm 0.14 %, and the alert threshold is 0.151 %
- **THEN** the panel shows "Второ мнение: Chronos ±0,18 %, TimesFM ±0,15 % · размер: Chronos не е съгласен, TimesFM е съгласен с h1_lstm"

#### Scenario: Nothing current
- **WHEN** the external logger has not forecast the current hour
- **THEN** the panel has no second-opinion line
