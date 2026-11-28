# Spec Delta

## Purpose

Writes down Rockefeller Treasury Services' published trading rules as one executable definition and one
plain-language explanation, so that the simulation and the owner read the same rules.

## ADDED Requirements

### Requirement: One rule definition
The rules SHALL be held in one versioned definition with every number RTS states. The simulation SHALL read
only this definition. The numbers are:
- proportionality at 15 points;
- Rule 2 at 40 points, with stop 40 and target 60;
- Rule 3 fallback with stop 38 and target 44, at most three losses in spot;
- Rule 4 at 80 points, with stop 44 and target 48;
- Rule 5 at a 40-point gap plus 20;
- the Friday spot close at 12:00 New York.

#### Scenario: Changed number
- **WHEN** a rule number in the definition is edited
- **THEN** the definition's digest changes, and every result built with the old digest is refused as stale

### Requirement: Session clock
RTS times SHALL be read on the New York clock and converted to the Berlin bar-label clock for each date. This
covers the 16:00 close, the 18:00 open and the Friday 12:00 spot close. Daylight-saving mismatch weeks SHALL
convert correctly.

#### Scenario: Normal week
- **WHEN** the open of a session in a normal November week is converted
- **THEN** 18:00 New York becomes the bar label 00:00 Berlin of the next day

#### Scenario: Mismatch week
- **WHEN** the same open falls in the late-October mismatch week
- **THEN** it becomes the bar label 23:00 Berlin

### Requirement: Rule 0 and proportionality
A recommendation whose entry equals the 4 pm price SHALL enter at the session open. If the open is 15 or more
points from the 4 pm price, the stop and target SHALL keep their point distances from the actual open.

#### Scenario: Gap of 20 points
- **WHEN** EURUSD's 4 pm price is 1.0900, the stop 30 points away, and the open 1.0920
- **THEN** the entry is 1.0920 with the stop 30 points from 1.0920

### Requirement: Guerilla entry
A guerilla entry SHALL be filled only when the price touches it after the open, during that session. If it is
never touched, there is no trade.

#### Scenario: Never touched
- **WHEN** a sell guerilla at 1.1009 is never reached before the next 4 pm close
- **THEN** no base trade exists for that recommendation

### Requirement: Contingency rules
The full-rules simulation SHALL apply Rules 2, 3, 4 and 5 and the NPR restrictions exactly as the definition
states. It SHALL use each rule only in the cases RTS allows:
- Rule 2 never on a Friday in spot;
- Rule 3 not under NPR;
- no rule-based entry from a previous day's prices.

#### Scenario: Friday runaway
- **WHEN** a Friday spot move continues 40 points after a target
- **THEN** no Rule 2 trade is opened

### Requirement: Plain explanation
`results/rts_signals/README.md` SHALL explain every rule in plain words with one worked example each, in
Bulgarian and English. It SHALL state that RTS's results are hypothetical and gross of cost.

#### Scenario: README
- **WHEN** the README is built
- **THEN** it contains a section for each of Rules 0–5, proportionality, guerilla, NPR and the Friday close
