# Data provenance and offline reproducibility

**Everything needed to run every model in this repository is committed.** No API key, no
broker terminal, no network connection, and no training run is required to reproduce any
number reported in the notebooks.

Verify this claim in one command:

```bash
python verify_installation.py
```

It re-fits the headline model from the committed CSVs and compares the result against the
recorded figures. If the data were incomplete or altered, that check would fail.

---

## 1. What ships with the repository

Row counts and spans below are **asserted by**
`tests/test_input_data_provenance.py`, which fails if this table drifts away
from the committed files. Update the table in the same commit as any data
change — see §1.1.

| File | Rows | Size | Content |
|---|---:|---:|---|
| `results/eurusd_features.csv` | 15,760 | 4.4 MB | EUR/USD **daily** OHLCV + engineered features, 1971-01-11 → 2026-08-10 |
| `results/eurusd_h1.csv` | 60,056 | 4.1 MB | EUR/USD **hourly** OHLCV with tick volume, 2016-12-21 18:00 → 2026-08-18 14:00 UTC — **rolling cache, see §1.1** |
| `results/eurusd_m15.csv` | 350,000 | 22.9 MB | EUR/USD **15-minute** OHLCV with tick volume, 2012-06-25 21:30 → 2026-07-24 22:45 UTC |
| `results/pooled_h1/EURUSD_h1.csv` | 70,000 | 4.2 MB | EUR/USD hourly — **frozen** pooled snapshot, 2015-04-27 → 2026-07-28 UTC |
| `results/pooled_h1/GBPUSD_h1.csv` | 70,000 | 4.5 MB | GBP/USD hourly — replication instrument, same window |
| `results/pooled_h1/AUDUSD_h1.csv` | 70,000 | 4.1 MB | AUD/USD hourly — replication instrument, same window |
| `results/pooled_h1/EURUSD_h1_newyork.csv` | 70,000 | 6.7 MB | EUR/USD hourly re-stamped to New-York session time |
| `results/pooled_h1/EURUSD_m15_newyork.csv` | 350,000 | 33.5 MB | EUR/USD 15-minute re-stamped to New-York session time |
| `results/pooled_h1/retired/CHFUSD_h1.csv` | 70,000 | 7.1 MB | CHF/USD hourly — retired instrument (see `retired/README.md`) |
| `results/pooled_h1/retired/USDCHF_h1.csv` | 70,000 | 4.1 MB | USD/CHF hourly — retired instrument |
| `results/yield_differential.csv` | — | 804 KB | US 10Y − DE 10Y government bond yield spread |
| `results/policy_rate_differential.csv` | — | 432 KB | Fed funds − ECB deposit facility rate |
| `results/inflation_differential.csv` | — | 32 KB | US − DE CPI, year-on-year |
| `results/usd_index.csv` | — | 188 KB | Trade-weighted broad US dollar index |
| `results/vix.csv` | — | 308 KB | CBOE volatility index |
| `results/cot_positioning.csv` | — | 76 KB | CFTC Commitments of Traders, EUR and USD index |
| `results/fomc_dates.csv` | — | 8 KB | FOMC meeting calendar |
| `models/` | 46 files | 24 MB | All trained artifacts (see §4) |

Total working tree: **~277 MB**.

---

## 1.1 Frozen inputs vs. the rolling cache

Not every committed CSV is the same kind of object, and conflating the two is
what let a data rewrite go unnoticed. The inputs split into two classes, and
`tests/fixtures/input_data_protected_sha256.json` records which is which.

**FROZEN — byte-pinned.** `results/pooled_h1/*`, `results/eurusd_m15.csv` and
`results/eurusd_features.csv` are research inputs that no running code rewrites.
Each has one, or a small handful of, revisions in git history, and changes only
when a human deliberately regenerates it. Their SHA-256 digests are pinned;
`tests/test_input_data_provenance.py` fails loudly if one moves. **Three of the
four H1 families — `h1_direction`, `h1_multiday`, `pooled_h1` — read
`results/pooled_h1/EURUSD_h1.csv`, not the rolling cache**, so pinning this tree
is what actually protects their committed numbers.

**ROLLING — stamped, not pinned.** `results/eurusd_h1.csv` is an *operational
cache*. `src/live_data.py::fetch_h1_market_data` writes it on every successful
MT5/yfinance pull, so it legitimately changes whenever anyone runs a prediction
or a retrain; it has 25 revisions, the fixed-size MT5 window sliding forward each
time. Byte-pinning it would assert only that nobody had run the app — a
permanently red guard cannot detect the next change, so the repository
deliberately leaves it unpinned (the reasoning is recorded in the PINNING POLICY
docstring of `tests/test_h1_production.py`). Instead it carries a **recorded
provenance stamp** — digest, row count and span at the moment of stamping — plus
structural invariants (UTC, strictly increasing unique timestamps, schema, sane
OHLC), and the row count in §1 above must match the file on disk.

**Why the stamp matters.** Two commits rewrote input data under a title that
named something else:

| Commit | Title | What it also did |
|---|---|---|
| `f2645a0` (2026-08-15) | *Refactor code structure for improved readability and maintainability* | full production retrain: 30 model artifacts + `results/eurusd_h1.csv` |
| `c638f8d` (2026-08-18) | *Add unit tests for Yildirim/Toroslu/Fiore (2021) replication study* | appended 56 bars to `results/eurusd_h1.csv` and completed one partial bar |

Neither rewrite invalidated a committed H1 result — the affected file is the
rolling cache, and the families that matter read the frozen pooled snapshot —
but neither was visible in review either. A refresh of the cache is allowed; what
is not allowed is refreshing it silently. Re-stamp the fixture and §1 in the same
commit that refreshes the data, and say so in the commit message.

---

## 2. Where each series came from

### Price data — MetaTrader 5, ActivTrades EU

Daily, hourly and 15-minute OHLCV were pulled from a retail MT5 terminal
(`ActivTradesEU-Server`) via `src/live_data.py` and `src/curl_mt5_fetch.py`. Bars carry the
broker's `tick_volume` — the count of price updates its server published during the bar.

This is **not** consolidated traded volume; FX has no central tape and `real_volume` is zero
for retail FX brokers. For the purposes it is used for here — estimating how stale a
recorded closing price is — the broker's own update count is the correct quantity rather
than a limitation, because the staleness being measured is that of this feed. Its validity
was tested rather than assumed (§5).

Server timestamps are **CET/CEST**, not the more common EET default. Establishing this
required a month-by-month analysis of the weekly session boundary: the trading week opens at
server hour 23 in every month except March, when the United States has entered daylight
saving time and the European Union has not. A fixed-offset server shows no such excursion,
and a whole-year aggregate conceals it. All committed files are converted to **UTC**.

### Macroeconomic data — FRED (Federal Reserve Bank of St. Louis)

Retrieved through `src/macro_data.py` and cached to CSV. Series identifiers are declared in
`config.json`:

| Feature | FRED series |
|---|---|
| US 10-year Treasury yield | `DGS10` |
| German 10-year government yield | `IRLTLT01DEM156N` |
| Fed funds effective rate | `DFF` |
| ECB deposit facility rate | `ECBDFR` |
| US CPI, all urban consumers | `CPIAUCSL` |
| German harmonised CPI | `CP0000DEM086NEST` |
| Trade-weighted broad USD index | `DTWEXBGS` |
| CBOE volatility index | `VIXCLS` |

FRED data is US Government work and free of copyright restriction.

### Positioning and calendar data

CFTC Commitments of Traders (weekly, public) and the Federal Reserve's published FOMC
meeting calendar.

---

## 3. Offline operation, by design

Two independent ingestion chains, neither of which fails hard:

```
price:  MetaTrader 5  →  yfinance  →  bundled CSV        (src/live_data.py)
macro:  FRED API      →  FRED public CSV  →  on-disk cache  →  None   (src/macro_data.py)
```

Each stage degrades to the next without raising. The final stage in both chains is a
committed file, so the system always has data to work with.

Two consequences worth stating explicitly:

- **The `baseline` model variant consumes no macro columns at all.** It is immune to FRED
  outages by construction, not by exception handling.
- **`MetaTrader5` is a Windows-only package and is declared conditionally** in
  `requirements.txt` (`; sys_platform == "win32"`). On macOS and Linux pip skips it, and
  every MT5 import in the codebase is lazy — inside function bodies, never at module level —
  so no entry point imports it on a machine where it is absent.

---

## 4. Trained artifacts

All 46 are committed. Nothing needs retraining.

| Directory | Contents |
|---|---|
| `models/baseline/` | 7 artifacts: GBM classifier + regressor, multi-task LSTM, PCA, scalers |
| `models/with_macro/` | 7 artifacts, same structure, macro-augmented feature set |
| `models/volatility/` | 5-seed LSTM ensemble + its own PCA/scalers + `vol_metrics.json` |
| `models/calendar/` | The headline model — **ten numbers in a JSON file** |
| `models/h1_direction/`, `models/ti_lstm_h1/` | Hourly families |

`models/calendar/calendar_volatility.json` is worth opening directly. The entire model is
three GARCH parameters, one scale constant and six weekday multipliers — readable by eye,
and it outperforms the 46-file neural stack on the volatility target.

---

## 5. Was the data checked, or merely used?

Three checks are documented rather than assumed.

**Alignment.** Six currency pairs at one-minute resolution were joined on a strict common
timestamp index: 2,980,060 raw EUR/USD bars reduce to **2,959,641 rows** common to all six.
Alignment loss is 0.59–0.77% per pair. Unmatched bars are **dropped, never forward-filled** —
an interpolated bar in a cross-rate identity manufactures signal out of nothing.

**Gap census.** Of EUR/USD's 4,992 intraday gaps across eight years, 4,226 are a single
missing minute and only 8 exceed fifteen minutes. They cluster at the 21:00–22:00 UTC daily
rollover and in the 03:00–05:00 Asian session — the thin-liquidity regime, exactly where
one would expect them.

**Activity proxy validity.** Whether `tick_volume` carries real information was tested with
a permutation control rather than assumed. Theory predicts a specific exponent relating curl
variance to tick count; testing that exponent against its textbook value proved invalid
(errors-in-variables bias it away from the theoretical figure even on honest data). The
valid test compares the statistic against the same statistic computed on **shuffled** tick
counts:

| Simulated feed | z vs shuffled |
|---|---:|
| Honest | **−42.7** |
| 50% shuffled | −17.9 |
| Fully shuffled | **−0.6** |

The control correctly identifies an uninformative feed, which is what makes it a test.

---

## 6. Statistical-validity guarantees

Four invariants, each enforced by unit tests rather than convention:

1. Targets are built with `shift(-1)`; a window predicting row *t* never contains data from
   *t+1*.
2. Forward-fill carries **past** values forward only, never future values backward.
3. Scalers and PCA are fitted on the training block exclusively, then applied elsewhere by
   `.transform()`.
4. `TimeSeriesSplit` throughout; random K-fold appears nowhere in the repository.

Splits are chronological: train `[0:70%]`, validation `[70%:80%]`, test `[80%:100%]`. On the
modelled euro-era daily set (**8,605 rows**) that is **6,023 / 861 / 1,721**.
`src/calendar_volatility.py::build_daily_dataset` applies its own `dropna` and yields
**8,604** rows (6,022 / 861 / 1,721) — one row fewer, same validation and test blocks.

These counts grow as the daily history extends; earlier documents record 8,559 rows
(5,991 / 856 / 1,712). Any figure quoted against a row set is only comparable to another
figure from the same vintage — see §8.1.

---

## 7. What is *not* included, and why

**Raw one-minute parquet files** (`results/curl/raw/`, ~284 MB). Six files of eight-year M1
history for the microstructure study. Excluded because:

- no model, script or test reads them — only `src/curl_mt5_fetch.py`, which *produces* them;
- `pyarrow` is not a project dependency, so they could not be opened in the environment this
  project declares;
- they are regenerable with a single command on a machine with an MT5 terminal;
- redistributing a broker's bar history raises questions this project has no need to answer.

The derived coverage report, `results/curl/m1_coverage.csv`, **is** committed — so the
provenance evidence cited in the notebook survives without the bulk.

To regenerate them:

```bash
pip install pyarrow
python -m src.curl_mt5_fetch        # requires a running MT5 terminal (Windows)
```

---

## 8. Reproducing the reported results

```bash
pip install -r requirements.txt
python verify_installation.py          # environment + data + headline model
python -m pytest -q                    # 530 tests (see §8.1 for 5 known failures)
python -m uvicorn api:app --reload     # dashboard at http://127.0.0.1:8000
```

The headline result needs neither TensorFlow nor JAX — numpy and pandas suffice, and it fits
in about ten seconds on a CPU:

```python
from src.calendar_volatility import build_daily_dataset, chronological_masks, CalendarVolatilityModel

data = build_daily_dataset()                       # reads the committed CSV
train, val, test = chronological_masks(len(data))
model = CalendarVolatilityModel(use_dow=True).fit(
    data["log_return_pct"].to_numpy(),
    data["target_volatility_pct"].to_numpy(),
    data["dow"].to_numpy(),
    train,
)
print(model.params)                                # all ten parameters
```

Expected, to three decimals with a stated tolerance of **± 0.002**:

```
validation MAE  0.162 ± 0.002        test MAE  0.192 ± 0.002
GARCH  α = 0.0284   β = 0.9685   scale = 0.5495
Mon 1.289   Tue 1.232   Wed 1.273   Thu 1.354   Fri 0.274   Sun 1.058
```

**Why a tolerance on a deterministic model.** The model itself is pure numpy/pandas with no
RNG: on a *fixed* row set it reproduces bit-identically, run after run. But the daily history
extends, and the MAE is scored on whatever rows exist when you run it. The last growth of
8,559 → 8,605 rows moved validation MAE by 0.00004 and test MAE by 0.00082 (§8.1); ± 0.002
is roughly twice the larger of those, so it absorbs a comparable data refresh without
either becoming a promise the next `git pull` breaks or growing so wide it would accept a
genuine regression. Quoting five decimals here was a promise this repository cannot keep —
it was correct only for the row set it was measured on, and it silently expired.

`python verify_installation.py` checks the live figures against these values at this
tolerance and says which row set it used. If it reports a miss, compare row counts first.

The neural comparison carries a *different* kind of uncertainty and cannot be pinned this
way at all — see §8.1.

## 8.1 What reproduces, and to what precision

Re-measured 2026-08-19 against the committed files. Not every published figure is the same
KIND of number, and the difference matters more than the digits.

**Two distinct mechanisms move numbers in this repository. They are not the same defect and
must not be read as one:**

| Mechanism | Which model | What it means |
|---|---|---|
| **Run-to-run nondeterminism at a fixed seed** | 5-seed LSTM ensemble | The *same* code on the *same* data with the *same* seeds (42–46) returns a *different* number on every run. TensorFlow/oneDNN CPU kernels and thread-scheduled float reductions are not bit-reproducible. |
| **Changed inputs** | calendar model, GARCH(1,1) | The code is **fully deterministic**: identical inputs return bit-identical outputs. These figures moved because the *dataset grew* — 8,559 → 8,605 rows (§6), so **46 new rows entered the fit**. |

Only the first is a reproducibility defect. The second is arithmetic behaving correctly on a
longer sample, and re-running it on the old row set would return the old numbers exactly.

| Quantity | Published | Re-measured | Mechanism |
|---|---:|---:|---|
| Calendar validation MAE | 0.16209 | **0.16213** | deterministic — 46 new rows |
| Calendar test MAE | 0.19279 | **0.19197** | deterministic — 46 new rows |
| Calendar GARCH α, β, scale | 0.0284 / 0.9685 / 0.5495 | **bit-identical** | deterministic — unmoved by the new rows |
| Calendar DoW factors | Mon 1.291, Wed 1.271, Fri 0.275 | **Mon 1.289, Wed 1.273, Fri 0.274** | deterministic — 46 new rows. Tue, Thu, Sun unchanged |
| GARCH(1,1) validation MAE | 0.203794 | **0.203227** | deterministic — 46 new rows |
| 5-seed LSTM ensemble validation MAE | 0.18594 | **0.18452 – 0.18988** | **nondeterministic across runs at a fixed seed** |
| Ensemble verdict vs GARCH(1,1) | CLEARED | **CLEARED in 3/3 runs** | the verdict holds |

### The calendar model is deterministic; its inputs changed

`src/calendar_volatility.py` is pure numpy/pandas. It contains no RNG, no GPU kernel and no
threaded reduction, and it reproduces bit-identically given the same rows.

The re-measurement confirms this directly. **The GARCH block came back bit-identical** —
α = 0.0284, β = 0.9685, scale = 0.5495, unchanged in every digit that is printed. Only the
day-of-week factors moved, and they moved in the **third decimal**:

```
             frozen (8,559 rows)      re-measured (8,605 rows)
Mon               1.291          ->        1.289
Tue               1.232          ->        1.232      (unchanged)
Wed               1.271          ->        1.273
Thu               1.354          ->        1.354      (unchanged)
Fri               0.275          ->        0.274
Sun               1.058          ->        1.058      (unchanged)
```

**The cause is the 46 new rows, not nondeterminism.** Each weekday multiplier is an average
over the rows carrying that weekday; extending the sample by 46 daily bars adds roughly nine
observations per weekday and nudges three of the six averages by one unit in the third
decimal. Three factors did not move at all. The GARCH parameters, which are fitted over the
whole sample rather than per weekday, absorbed the new rows without a printed change.

Nothing about the calendar model is unstable. Re-run it on the 8,559-row vintage and the
frozen numbers return exactly.

### The LSTM ensemble is nondeterministic, at a fixed seed

This is the row to read carefully, and it is a different kind of problem. Three runs of the
*same code* on the *same data* with the *same seeds* (42–46) gave ensemble MAE
0.189877 / 0.184520 / 0.185618 — a spread of **0.00536**, which is comparable to the effect
being measured (GARCH-relative ΔMAE ≈ 0.018). TensorFlow/oneDNN on CPU is not
seed-deterministic, and `src/volatility.py::run_seed_ensemble_confirmation` says so in its
own docstring. The registered 0.18594 falls inside that band, so it was a fair draw — but it
is a draw from a distribution, not a fixed constant, and quoting it to six decimals
overstates what is known.

Re-running this on the old row set would **not** return the old number. That is what
separates it from the calendar row above.

**What this does and does not change.** The ship gate was that both bootstrap CIs exclude
zero at the Bonferroni bar α = 0.0167. That cleared in all three re-runs, so the volatility
family's conclusion stands. What does not stand is the precision: report the ensemble as
≈ 0.186 ± 0.003, not 0.185940. Per-run detail:
`results/volatility_verification/seed_ensemble_reproduction_2026-08-19.csv`.

The pre-registered arbiter record `results/volatility_seed_ensemble.csv` is **left
untouched**. It is the historical record of a pre-registered test at its own data vintage;
re-running a spent family's arbiter and overwriting the row would erase the audit trail
rather than correct it.

**Five known test failures.** All five are one defect: commit `f2645a0` (2026-08-15),
titled *"Refactor code structure for improved readability and maintainability"*, retrained
30 model artifacts without re-baselining the four `tests/fixtures/*_protected_sha256.json`
fixtures. The guards have been red since. The same commit also re-read and rewrote the
**one-shot test block** figures in `models/volatility/vol_metrics.json`
(n_test 1,712 → 1,721; test MAE 0.218973 → 0.216033) — a block the methodology reserves for
a single final report. Re-baselining pinned model artifacts is a deliberate decision and is
not bundled into this correction.
