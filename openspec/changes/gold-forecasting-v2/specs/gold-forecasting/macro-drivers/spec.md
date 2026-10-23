# Spec Delta

## Purpose

Tests whether gold's documented macro drivers (real yields, the dollar, risk appetite,
silver and oil) add anything to the daily gold models, without letting a single future
value leak in.

## ADDED Requirements

### Requirement: Macro inputs are known at the as-of close
Each daily macro feature SHALL use the newest value published strictly before the as-of
bar's close. Each FRED series (DFII10, DTWEXBGS, VIXCLS) SHALL be lagged one US business
day. Each MT5 series (SILVER, Brent) SHALL use its bar that closed at or before the gold
bar. A forward-fill SHALL only ever carry a past value forward.

#### Scenario: Real yield on a Monday
- **WHEN** the gold D1 bar for Monday is the as-of bar
- **THEN** its real-yield feature is the DFII10 value dated the previous Friday or earlier

#### Scenario: Missing silver bar
- **WHEN** SILVER has no bar for a gold trading day
- **THEN** the feature carries the previous silver value and never a later one

### Requirement: Macro features are scale-free
Every macro feature SHALL be a change, a return, a z-score over a past window, or a
level that is stationary by construction (VIX, real yield in percent). No raw price
level SHALL enter a model.

#### Scenario: Silver level
- **WHEN** the feature builder sees a raw silver price column
- **THEN** it refuses with the gold scale-free error

### Requirement: Macro is tested as an add-on to the daily models
Each macro cell SHALL compare the same daily model fitted with and without the macro
block, on identical rows. Volatility and range SHALL be scored on paired error
differences, and return by Clark–West, at the macro family's alpha. Daily direction
SHALL be registered UNDERPOWERED unless the power rule admits it.

#### Scenario: Identical rows
- **WHEN** the macro and price-only variants are scored
- **THEN** both are scored on exactly the same validation rows, after dropping any row where a macro input is missing

### Requirement: Data sources fail visibly
If FRED or MT5 cannot supply a series, the study SHALL stop and name the series. The
live logger SHALL skip the macro cells for that pass and record a gap. Neither SHALL
substitute zeros.

#### Scenario: FRED unreachable live
- **WHEN** the FRED API and public CSV are unreachable and the cache lacks the needed date
- **THEN** the macro cells record a gap naming the series, and the price-only cells still forecast
