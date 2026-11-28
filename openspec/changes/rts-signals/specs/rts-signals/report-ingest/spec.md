# Spec Delta

## Purpose

Defines how Rockefeller Treasury Services report PDFs reach the project and become stored, checked rows:
fetching from Gmail or a drop folder, parsing the three report kinds, and converting their prices.

## ADDED Requirements

### Requirement: Gmail fetch, read-only
The family SHALL fetch new PDF attachments of RTS report emails from the owner's Gmail with read-only access.
It SHALL never send, label, move or delete mail. The credentials SHALL stay in a gitignored folder and SHALL
never be printed, logged or committed.

#### Scenario: New report email
- **WHEN** a new RTS email with a PDF attachment arrives and the fetch runs
- **THEN** the PDF is saved once to the family's gitignored inbox and the email is unchanged

#### Scenario: Credentials missing
- **WHEN** the fetch runs without an authorised token
- **THEN** it reports `gmail not authorised` and the drop-folder path still works

### Requirement: Drop folder
PDFs placed by hand in the family's drop folder SHALL be ingested exactly as fetched ones. A file already
ingested, identified by its SHA-256, SHALL never be ingested twice.

#### Scenario: Same PDF twice
- **WHEN** the same PDF is placed in the drop folder and also fetched from Gmail
- **THEN** its rows are stored once

### Requirement: Report kinds
Each PDF SHALL be classified as one of these kinds by its title text, or rejected as `unknown report`:
- Spot Afternoon;
- Morning;
- Futures, recognised only to be logged as `futures — not read`. By the owner's decision of 2026-11-28 (the
  owner trades spot only and has no futures on MT5), a futures report SHALL add no row.

Its report date SHALL be taken from the document, not from the file name.

#### Scenario: Futures report
- **WHEN** a Futures report is ingested
- **THEN** it is logged `futures — not read` and no recommendation, track-record or flag row is stored

#### Scenario: Morning report with a duplicate file name
- **WHEN** `112326 Morning Report ALL (1).pdf` and `112326 Morning Report ALL.pdf` hold the same document
- **THEN** both are classified Morning, dated 2026-11-23, and stored once

### Requirement: Spot recommendation rows
For each of POUND, EURO, AUD, YEN, SWISS and CAD, a Spot report SHALL yield one row with:
- the pair;
- the direction;
- the end-of-day position;
- the 4 pm price;
- the open position's stop, target and entry;
- the new entry, stop and target, or `NPR`.

#### Scenario: Recommendation with a new entry
- **WHEN** the 2026-11-25 spot report lists `YEN BUY SQUARE 15920 ... 15898 15858 15942`
- **THEN** the row is USDJPY, direction buy, new entry 158.98, stop 158.58 and target 159.42

#### Scenario: No position recommended
- **WHEN** a pair's new entry reads `NPR`
- **THEN** the row records NPR and no entry, stop or target

### Requirement: Price conversion
Spot prices SHALL be converted from RTS's integer form to MT5 prices with:
- 4 decimals for EURUSD, GBPUSD, AUDUSD, USDCHF and USDCAD;
- 2 decimals for USDJPY.

YEN, SWISS and CAD SHALL be the USD-base pairs.

#### Scenario: Swiss and CAD
- **WHEN** the spot report lists SWISS 8415 and CAD 14374
- **THEN** they are stored as USDCHF 0.8415 and USDCAD 1.4374

### Requirement: Guerilla entries
A new entry that differs from the 4 pm price SHALL be marked guerilla. An entry equal to the 4 pm price SHALL be
marked Rule 0, enter at the open.

#### Scenario: Guerilla
- **WHEN** EURO's new entry is 11009 and its 4 pm price is 10975
- **THEN** the row is marked guerilla at 1.1009

### Requirement: Track record rows
Each closed trade in the spot track record SHALL be stored with these fields:
- entry date, pair, side, entry price and rule number;
- exit date, exit price and points.

Numbers that the PDF text runs together, such as `0.81881` for entry 0.8188 with rule 1, SHALL be split by the
pair's known decimals.

#### Scenario: Merged entry and rule
- **WHEN** a spot track-record line reads `11/03/26 USD/CHF Sold 0.81881 11/03/26 0.8152 36 43.97 164.10`
- **THEN** the row is USDCHF, sell, entry 0.8188, rule 1, exit 0.8152, 36 points

### Requirement: Morning positions
A Morning report SHALL yield, for each listed currency, the spot, the current position, the signal date, the
signal rate and the gain or loss. These rows are information only and SHALL never be simulated as trades.

#### Scenario: Morning table
- **WHEN** the 2026-11-25 morning report lists `USD/MXN 19.6834 LONG USD 09/20/26 17.1960 14.46%`
- **THEN** one position row is stored for USDMXN with those values

### Requirement: Inconsistency flags
A report whose text contradicts itself SHALL be stored with a flag naming the conflict. The stated values SHALL
be kept unchanged. Conflicts include:
- a narrative price far from every listed level;
- a points figure that differs from entry minus exit.

#### Scenario: Typo in a narrative note
- **WHEN** the 2026-11-25 spot note says CAD short on Rule 2 at 13202 while the CAD levels are near 1.42
- **THEN** the row is flagged `narrative price off by more than 500 points` and no trade at 1.3202 is created

### Requirement: Stored rows
Parsed rows SHALL be written append-only under `results/rts_signals/`, each with:
- the source file's SHA-256;
- the report kind and date;
- the parser version.

Raw PDFs SHALL stay in the gitignored artifacts folder. They are paid third-party content.

#### Scenario: Re-parse
- **WHEN** the parser runs again on an already-stored PDF
- **THEN** no row is added or changed
