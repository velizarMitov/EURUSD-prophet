# Proposal

## Why

Two H1 price-structure families ended without an edge on 2026-11-21. fx-fibonacci had
four DROPs, and fx-session had two DROPs and two cells NOT TRIGGERED. The owner's next
hypothesis changes the information source to interest rates and the horizon to a week.

Following Rockefeller (*The Foreign Exchange Matrix*), the 2-year yield differential
reacts most to ECB, Fed and BoE policy expectations. It is said to lead the exchange rate
over days to weeks. Earlier daily tests used the 10-year spread and policy rates at a
1-day horizon (#1, #3: KEEP-provisional, no proven edge). This family uses the 2-year
spread at a 5-day horizon, which avoids reading same-day co-movement as prediction.

## What Changes

- **A new research family, `fx_rates`, for EURUSD and GBPUSD.** It has its own package
  `src/fx_rates/`, config, registry and results directory.
- **Official daily yield caches.** Each source keeps its own append-only cache.

  | Series | Source |
  |---|---|
  | US 2Y | FRED `DGS2` |
  | US 10Y | FRED `DGS10`, for the US 2s10s slope only |
  | DE 2Y | Bundesbank Svensson term structure, residual maturity 2.0 years |
  | UK 2Y | Bank of England nominal spot curve at 2.0 years: the GLC archive plus the current-month file, read without new dependencies |

- **FX D1 bars from 2012-01-01.**
  - They are built read-only from the fx_session H1 caches, whose GBPUSD history
    already excludes the broker's shifted 2011 clock.
  - D1 bars with fewer than 12 H1 bars count as missing.
  - The target and the gross rule use the mid close.
- **Strictly causal alignment.**
  - The decision is taken at the close of D1 bar `t`.
  - Yields enter only from the previous FX trading day, `t−1`.
  - Holiday gaps are forward-filled for at most 3 business days and flagged by
    `yield_stale_flag`. A longer gap excludes the day. Back-fill is never used.
- **Target: the sign of the 5-day mid return, `close[t+5] − close[t]`.**
  - It is binary, ties are dropped, and there are no class weights.
  - The horizon is counted in FX trading days.
- **Twelve features.**
  - The 2Y spread: level, 5/20/60-day changes and 60-day z-score.
  - The US 2s10s slope.
  - FX moves: 5 and 20 days, scaled by realised volatility.
  - The FX-versus-rates divergence: the 20-day z-score of the FX move minus that of the
    spread change.
  - The MA20-versus-MA50 trend, the 20/60-day volatility ratio, and the stale flag.
- **An expanding walk-forward from 2015-01-01, in 6-month folds.**
  - The first fold has three years of history.
  - A 5-day purge plus a 5-day embargo separate every pair of slices.
  - The walk-forward is the only arbiter: there is no forward phase.
- **Four registered cells at alpha = 0.05/4, each scored once.**
  - Per pair, LightGBM against the class prior on log-loss.
  - Per pair, a CUDA MLP against LightGBM, only after that pair's LightGBM KEEP.
  - A cell that is not triggered keeps its alpha, which is never reassigned.
  - The test is a circular block bootstrap with 10-day blocks.
- **The power check comes before any model.**
  - `labels.md` gives the minimum detectable accuracy for overlapping 5-day targets,
    using the target's own sign autocorrelation, and for independent weeks.
  - The probe of 2026-11-21 measured about 55.3 %. That is a large effect, and the
    report says so.
- **Gross of spread and slippage, by owner decision.**
  - A non-overlapping weekly rule reports accuracy, Sharpe and profit factor.
  - It is descriptive only, and no verdict is a trading result.

## Capabilities

### New Capabilities
- `fx-rates/rates-data`: the FRED, Bundesbank and Bank of England daily yield caches,
  the FX D1 bars, and the causal alignment with the `t−1` lag, the limited forward-fill
  and the stale flag.
- `fx-rates/rates-features`: the fixed twelve-column feature list, scale-free and known
  at the close of `t`.
- `fx-rates/weekly-target`: the 5-day mid-sign target, ties and the overlap-aware power
  report.
- `fx-rates/evaluation`:
  - the purged and embargoed folds;
  - the LightGBM and conditional MLP models and the class-prior reference;
  - the four-cell registry, one-shot scoring and the config guard;
  - the descriptive gross weekly rule.

### Modified Capabilities
(none)

## Impact

- **New code and results.** The code lives in `src/fx_rates/` and the tests in
  `tests/test_fx_rates_*.py`. Results go to `results/fx_rates/`, and frozen artifacts to
  the gitignored `research_models/fx_rates/`.
- **Read-only inputs.** These are the fx_session H1 caches and the fx_fib/fx_session
  helpers: D1 aggregation, the block bootstrap, the fold and model code patterns, and
  the CUDA determinism helpers. Neither family's config or results change, and a test
  pins both digests.
- **Network, read only, with a generic client name.** The sources are FRED's public CSV,
  the Bundesbank REST API and the Bank of England yield-curve zips. No other vendor is
  used.
- **Nothing else changes.** `models/`, `config.json`, the served API, the production
  macro caches and every other registry stay untouched. The family never touches the
  `0.05/9` bar.
- **Docs.** `ARCHITECTURE_DOCS.md §3.16` and a CLAUDE.md invariant entry.
- **Dependencies.** None new. The Bank of England `.xlsx` files are read with the
  standard library (`zipfile` and `xml`), so `openpyxl` is not needed.
