# Design

## Context

The motivation is in proposal.md, and the requirements are in `specs/fx-rates/*`. The
source probe of 2026-11-21 established these facts. Its scripts are kept outside the repo;
the results are recorded here.

| Series | Endpoint | Coverage | Lag |
|---|---|---|---|
| US 2Y `DGS2`, US 10Y `DGS10` | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>`, a public CSV with no key | `DGS2` from 1976-06-01 | last value 2026-11-19, one business day |
| DE 2Y | `https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A?format=csv&lang=en` (the `/rest/download/` path returns 301 to it) | from 1997-08-07 | last value 2026-11-20 |
| UK 2Y | `https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalddata.zip`, about 39 MB, plus `.../latest-yield-curve-data.zip` | served as a zip of `.xlsx` files | archive to 2026-10-30; current-month file 2026-11-02 to 2026-11-19 |

**Client name.** All three sources accept the generic User-Agent `Mozilla/5.0 (research data client)`. FRED drops the connection for a fake Chrome User-Agent, presumably by bot detection; this was found on the first real refresh.

- **Bundesbank format.** The CSV has 9 metadata rows. Weekends hold `.` with the flag
  "No value available".
- **Bank of England format.** The archive holds yearly `.xlsx` files, back to 1979. The
  2.0-year point is the column whose header row reads `years: … 2 … 2.5` in sheet
  "4. spot curve". Sheet "3. spot, short end" has monthly maturities and must not be used.
- **IADB codes do not cover 2Y.** `IUDSNZC` is 5Y (2.97 % in January 2010) and
  `IUDMNZC` is 10Y.
- **`openpyxl` is not installed in `.venv`.** A stdlib reader (`zipfile` +
  `xml.etree`, shared strings, Excel serial dates) extracted 5 516 UK 2Y days from
  2005-01-04. January 2010 reads 1.45 %, which is plausible.
- **FX.** fx_session's H1 caches (EURUSD, GBPUSD, from 2012-01-02, with spread) give
  3 078 D1 out-of-sample days per pair from 2015-01-02. The overlapping 5-day sign
  targets autocorrelate at 0.59 / 0.40 / 0.23–0.27 / 0.09–0.12 at lags 1–4, so
  VIF ≈ 3.6–3.8.
  - Effective size is about 815–850.
  - The minimum detectable accuracy is about **55.3–55.4 %**, or **56.2 %** on about
    615 independent weeks, at alpha 0.0125 and 80 % power.
- **Holidays.** From 2015, the US series is missing on 111 FX trading days. DE is
  missing on 65 and UK on 93.

## Goals / Non-Goals

**Goals:**
- An honest test of the 2Y differential at a weekly horizon, with yields strictly from
  `t−1`.
- The fx_session discipline: a frozen config, a digest guard, one-shot scoring, a
  conditional MLP, a power report before models, and a reproducible report.

**Non-Goals:**
- Any hyper-parameter, feature, horizon or lag search.
- Foreign 10Y slopes, OIS curves, policy-rate data or a DXY or equity series.
- Intraday data, costs, sizing and execution.
- A forward logger, live inference or UI.

## Decisions

### D1. Its own package, `src/fx_rates/`, with read-only reuse

The family gets `fx_rates_config.json`, `fx_rates_hypothesis_log.csv` and
`results/fx_rates/`.

- **Imported read-only:**
  - `fx_session.data_loader.load_h1` and `load_points`, for the H1 caches;
  - `fx_fib.mtf_aligner.build_d1`;
  - `fx_fib.score.test_cell`;
  - the fx_session model and artifact helpers: `model_lgbm`, `model_mlp`, `artifacts`.
- **Written new:** a family registry module with its own `REGISTRY` name.
- **A test pins both earlier families.** It checks that fx_fib's and fx_session's
  configs still match the digests recorded with their verdicts.

*Alternative considered:* copying the helpers. Rejected, because duplicated scoring code
can drift.

### D2. FX D1 from fx_session's H1 caches, read-only

D1 bars come from `build_d1` on fx_session's H1 caches. A D1 bar needs at least 12 H1
bars. The mid close is the last H1 bar's bid close plus half its spread.

- **fx_session's per-week clock gate is not applied.** A one-hour label shift moves a
  daily close by one hour out of a 5-day horizon. History already starts in 2012, after
  the shifted GBPUSD year.
- **This family never refreshes those caches.** It reads them and records their SHA-256
  in `run_lgbm.json`. Refreshing stays fx_session's command, which is append-only.

*Alternative considered:* its own MT5 cache. Rejected, because it would be a third copy
of the same bars.

### D3. Yield caches and the Bank of England reader

`rates_loader.py` keeps `results/fx_rates/data/{us2y,us10y,de2y,uk2y}.csv` (columns
`date,value`), append-only, with revisions counted, and writes `refresh_log.csv`.

- **Bank of England.**
  - The first fill reads the archive files from 2005 on.
  - Every refresh also reads the current-month file, which wins where it overlaps.
  - The stdlib reader finds the sheet whose name contains "spot curve", the header row
    whose cells read 2.0 then 2.5, and Excel serial dates from 1899-12-30.
- **Errors per series.** Each series fails independently: that cache stays unchanged
  and the failure is logged.
- **Client and timeouts.** Requests use the generic User-Agent
  `Mozilla/5.0 (research data client)`, with no personal data, and a 180 s timeout.

### D4. Causal alignment: `t−1` plus a business-day fill limit

For FX trading date `t`, the lagged date is the FX trading date before `t`. For each
series, the value used is its latest observation dated on or before the lagged date.

- **Fill limit.** If that observation is more than 3 business days older than the lagged
  date, the row is excluded as "stale yields". Business days are counted with
  `numpy.busday_count`.
- **Stale flag.** `yield_stale_flag` is 1 if any of the four series used an observation
  older than the lagged date.
- **No back-fill.** This is a `merge_asof(direction='backward')` on dates. Nothing ever
  looks forward.

The FX features use closes up to and including `t`. The decision is at the close of
`t`, so `t`'s close is known.

### D5. The twelve features

| Column | Definition |
|---|---|
| `spread_2y_level` | foreign 2Y − US 2Y, lagged |
| `spread_2y_chg_5d`, `_20d`, `_60d` | level(t) − level(t−h), in FX trading days |
| `spread_2y_zscore_60d` | (level − mean60) / std60 |
| `us_2s10s_slope` | US10Y − US2Y, lagged |
| `fx_ret_5d`, `fx_ret_20d` | log(close_t / close_{t−h}) / (vol20 · √h) |
| `fx_vs_spread_div_20d` | z(fx log return over 20 days) − z(spread change over 20 days); each z uses the mean and std of the previous 252 days |
| `fx_ma20_vs_ma50` | log(MA20 / MA50) / vol20 |
| `fx_vol_20d_ratio` | vol20 / vol60 |
| `yield_stale_flag` | 0/1 |

`vol` is the standard deviation of daily log returns of the mid close. The warm-up is
252 + 20 FX days, so features start in early 2013. The first fold, in 2015, trains on
about 2 years.

*The owner asked whether to include `us_2s10s_slope`.* It needs only FRED `DGS10`, the
same source as `DGS2`, so it adds no new source. It is included. Foreign slopes would
need two more series and are left out.

### D6. Folds, purge and embargo

- **Folds.** 6-month folds from 2015-01-01, with shared boundaries, built over usable
  days.
- **Fit and validation.** `before` holds the usable days before the fold, minus the last
  10, a 5-day purge plus a 5-day embargo. Validation is the last 15 % of `before`. Fit
  is everything before validation, minus another 10 days.
- **Why the purge is needed.** Day `d`'s target spans the closes of `d+1 … d+5`, so the
  purge removes every fit row whose window reaches validation. The embargo adds
  margin.

### D7. Models and settings

- **LightGBM** uses the fx_session settings: 7 leaves, at least 50 rows per leaf,
  learning rate 0.02, feature and bagging fraction 0.8, L2 1.0, up to 2 000 trees with
  early stopping after 100, seed 42, deterministic. The binary objective has no
  weights.
- **The MLP** uses the fx_session architecture: `32-16`, LayerNorm and GELU, dropout
  0.2, BCE, AdamW 1e-3 with weight decay 1e-3, batch 256, patience 20, 5 seeds, CUDA
  only. It runs only behind a KEEP.
- **Both are read from this family's config.** fx_session's code is reused, but its
  config is not.

### D8. Scoring

- **Test.** It reuses `fx_fib.score.test_cell` with `block = 10` FX trading days. That
  is twice the target overlap, so the bootstrap keeps the lag 1–4 dependence inside its
  blocks.
- **Accuracy is descriptive only.** The registry records it, with a Wilson interval
  computed on the effective size `n / VIF`.
- **Two-stage pipeline, as in fx_session.** Stage 1 is
  `walk_forward --stage lgbm` → `score --stage lgbm`. Stage 2 is
  `walk_forward --stage mlp` → `score --stage mlp`, which records NOT TRIGGERED rows
  without training.

### D9. Descriptive weekly rule

For offsets 0–4, take every fifth out-of-sample day from the offset. The position is the
sign of P(up) − 0.5, held 5 days, with mid-to-mid profit in pips (EURUSD and GBPUSD
pip 0.0001).

The report gives, per offset and averaged:
- the accuracy, with Wilson 95 % on the weeks;
- mean and total pips;
- Sharpe from weekly pips at √52;
- the profit factor.

An always-long baseline is shown alongside.

## Risks / Trade-offs

- [The minimum detectable accuracy is about 55 %. Only a strong weekly edge can pass]
  → `labels.md` states this before any model. A DROP means "no edge of that size", not
  "no edge".
- [The Bank of England changes the file layout or URL] → the reader fails loudly on a
  missing "spot curve" sheet or 2.0/2.5 header. The cache stays unchanged, and nothing
  is guessed.
- [The archive download is about 39 MB] → it is needed only on the first fill. Refreshes
  read the current-month zip of about 0.4 MB, and the archive only if the cache has a
  gap.
- [Yield estimates from the Bundesbank Svensson model and the BoE spline may be revised]
  → revisions are counted, never rewritten. A large count is reported in `labels.md`.
- [A US holiday also closes FX liquidity on some days] → those days stay; the stale flag
  marks them.
- [The fx_session caches could be refreshed between the stage-1 run and scoring] → the
  run records their digests, and the label report is regenerated in the same run.
  Scoring uses only the saved predictions.

## Migration Plan

Nothing to migrate. Rolling back means deleting `src/fx_rates/`,
`tests/test_fx_rates_*.py`, `results/fx_rates/`, `research_models/fx_rates/` and the doc
entries.
