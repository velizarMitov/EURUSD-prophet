# Proposal

## Why

The owner wants two ideas that were tested on currencies run on gold (XAUUSD):

- **Bollinger Bands.** On currencies, both versions were DROP: `fx_bb` (mean reversion) and `fx_bb_sma` (the trade with an SMA20 exit).
- **The New York session range.** On currencies, `fx_ny_volatility` found that pre-session information predicts the range, and the linear HAR-X was the best model.

Gold behaves differently from currencies, so neither currency result carries over. Each idea needs its own pre-registered test on gold, decided once.

## What Changes

- **A new research package, `src/gold_bbnr/`**, with its own config `gold_bbnr_config.json`.
  - It sits outside `src/gold/`. It never edits `gold_config.json` or `gold_config_v2.json`, and never writes the gold registries, caches or `models/`.
  - Its frozen artifacts go to gitignored `research_models/gold_bbnr/`.
- **Data.** Gold H1 bars are read-only from `results/gold/gold_h1.csv`: bid OHLC on the Berlin label clock, 2015-01-02 .. 2026-11-20.
  - The clock is checked with `src.gold.clock.verify`.
  - D1 bars are built from H1 and close at 17:00 New York, as in `fx_bb`.
  - The gold cache is never refreshed from MT5.
- **The dollar index (owner request), from two sources, each fetched once into the family's own cache.** Each cache's digest is fixed in the pre-registration.
  - **Hourly DXY from MT5** (ActivTrades `USDIndDec26`). It has real H1 bars from 2017-01-03 and is on the broker's Berlin clock: its H1 returns mirror EURUSD's at lag 0 (correlation −0.91 to −0.95 every year). No roll jumps are visible. It is used in Part 2.
  - **Daily DXY from Yahoo** (`DX-Y.NYB`), covering the whole window. It is used in Part 1, always for the previous day.
- **Gross of cost, by owner decision.** No spread or slippage is modelled. The gold cache has no spread, and every price is the bid. No verdict is a trading result.
- **Part 1a: Bollinger mean reversion (as `fx_bb`).**
  - Bands: 20 days, 2 SD.
  - An event is the first D1 close outside the bands, with a 3-day cooldown.
  - Outcome: `r = dir × (close[t+3] − close[t]) / ATR20`.
  - Two cells: the rule's mean r > 0, and LightGBM against the base rate.
  - LightGBM uses 11 features: fx_bb's 10 without the cross-pair one, plus two previous-day dollar features (the DXY z-score in its own 20-day bands, and its 5-day return).
- **Part 1b: the Bollinger trade with an SMA20 exit (as `fx_bb_sma`).**
  - Entry at the D1 close of the event.
  - Take profit at `SMA20[t]`. The stop is at the same distance beyond the entry; a bar touching both counts as the stop.
  - Time stop at the close of t+10. Exits are scanned on H1 bars. One open trade at a time.
  - Result in R.
  - Two cells: the mean R > 0, and LightGBM against the base rate.
- **Part 2: the gold NY session range (as `fx_ny_volatility`).**
  - Target: the log of the range of the H1 bars labelled 15:00–21:00 Berlin, over the close of the 14:00 bar.
  - Fourteen fixed features, known at that close. The dollar feature is the DXY range from 00:00 to 14:00 over its 20-day mean, taken from MT5.
  - Because hourly DXY starts in 2017, Part 2 uses days from 2017 and scores from 2018-01-01.
  - Models: HAR, HAR-X (linear), LightGBM and a 5-seed CUDA MLP, in a walk-forward.
  - Three cells, decided by squared error: HAR-X vs HAR, LightGBM vs HAR-X, and MLP vs HAR-X.
- **Gates.**
  - A label report gives event counts, eligibility and power, and reads no outcome.
  - `PRE_REGISTRATION.md` and `.json` must be committed before any outcome is computed; `prereg.gate` checks git.
  - Each cell is scored once into its family's registry.
- **Power is stated before any outcome.**
  - Gold has about 179 first band breaches in 2015–2026. Six currency pairs were pooled before, so the Bollinger cells can see only a large effect.
  - The LightGBM cells skip folds with too few training events. This shrinks their samples further.

## Capabilities

### New Capabilities

- `gold-bb-nyrange/gold-bars`: read-only gold H1 input, the clock check, NY-close D1 bars, day validity and the gross/bid convention.
- `gold-bb-nyrange/dollar-index`: the one-time hourly DXY (MT5) and daily DXY (Yahoo) caches, their clock and quality checks, and their digests.
- `gold-bb-nyrange/bb-reversion`: band events, cooldown, the 3-day ATR outcome, the 9 features and the walk-forward LightGBM for Part 1a.
- `gold-bb-nyrange/bb-sma-trade`: the SMA20-exit trade, its H1 exit scan, the one-open-trade rule, the R outcome and the walk-forward LightGBM for Part 1b.
- `gold-bb-nyrange/ny-range`: the session-range target, the 14 features, the HAR, HAR-X, LightGBM and MLP walk-forward, and the frozen final models for Part 2.
- `gold-bb-nyrange/evaluation`: the label and power report, the pre-registration gate, the three registries and their cells, the bootstraps, and the report.

### Modified Capabilities

None. The `gold-forecasting`, `fx-ny-volatility`, `fx-bb-sma-exit` and `fx-ny-orb-hot` specs are unchanged.

## Impact

- **New:**
  - `src/gold_bbnr/` and `tests/test_gold_bbnr_*.py`.
  - `results/gold_bbnr/`: labels, pre-registration, three registries, the report and README.
  - `research_models/gold_bbnr/`.
- **Fetched once into `results/gold_bbnr/data/`:** hourly DXY from MT5 (read only, no orders) and daily DXY from Yahoo, using the generic client name.
- **Read only:**
  - `results/gold/gold_h1.csv`, plus `src.gold.clock`.
  - The `fx_session` EURUSD H1 cache, only for the DXY clock check.
  - The `fx_session` calendar helpers and `results/fomc_dates.csv`.
- **Docs:** a new `ARCHITECTURE_DOCS.md` section and a `CLAUDE.md` invariant for the family.
- **Untouched:**
  - `models/`, `config.json` and the `0.05/9` bar.
  - All gold registries, `gold_config*.json`, and every other family's files.
- **Facts recorded, not blockers:**
  - Gold's H1 history was already used by the gold family's validation and test blocks: the volatility LSTM CLEARED, and the range intervals were DROP.
  - The currency Bollinger families were DROP.
  - The currency HAR-X result is the reason HAR-X is registered here.
- **No** broker, order, sizing or execution code. No MT5 indicator in this change.
