# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/setup-gbrain`, `/retro`, `/investigate`, `/document-release`, `/document-generate`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.

EUR/USD next-day directional + return predictor. A research notebook trains the models; a shared inference service serves them behind two frontends. The deep reference for data flow, artifacts, and failure modes is **`ARCHITECTURE_DOCS.md`** — read it before any non-trivial change.

## Commands

Environment is **Windows / PowerShell**; a `.venv` is present. `MetaTrader5` is Windows-only.

```bash
# Tests (run from repo root — pytest discovers tests/ here)
python -m pytest -q                                  # full suite (~19 tests)
python -m pytest -q tests/test_unit.py               # one file
python -m pytest -q -k fetch_yield_differential      # one test / pattern

# Owner's way: "Start EUR-USD Prophet.vbs" in the repo root, or the desktop / Start-menu icon -> scripts/launch.vbs (hidden; reuses the server, restarts it when .py files are newer;
# log research_models/server/server.log). Create both icons once with scripts/install_shortcut.ps1. There is no start.bat.
# Run the app (FastAPI is the single entry point; serves dashboard + /api/predict + /history + /api/retrain)
python -m uvicorn api:app --reload                   # -> http://127.0.0.1:8000

# Retrain & regenerate the production artifacts in models/
python _train_pipeline.py

# Research notebook
jupyter notebook notebooks/01_data_preparation.ipynb
```

Pre-trained artifacts in `models/` **and** `results/eurusd_features.csv` are git-tracked, so the app runs immediately after `pip install -r requirements.txt` — **no training required**.

When a script prints non-ASCII (Cyrillic, ✓, →), set `PYTHONIOENCODING=utf-8` or the Windows `cp1252` console codec raises `UnicodeEncodeError`.

## Architecture: the single-source-of-truth contract

The whole design exists to stop **research-to-production drift**. Training (notebook + `_train_pipeline.py`) and serving (`api.py` → `src/inference.py`) import the **same** `src/features.py` so the feature matrix is byte-identical on both sides.

- **`src/features.py`** owns `FEATURE_COLUMNS` (the canonical **27** columns) and `LAG_COLUMNS` (6 autoregressive lags reduced by PCA). `compute_features()` is inference-safe (no target, no dropna → keeps the latest live bar); `add_advanced_features()` is the training variant (adds `target_return`/`target_direction`, drops NaNs).
- **Dual model variants** (`config.json → variants`): **`baseline`** = price-only (`PRICE_FEATURE_COLUMNS`, 23 cols, no FRED at all) and **`with_macro`** = the full 27-col set whose 4 macro features are statistically unproven (KEEP-provisional). Both train in one `_train_pipeline.py` run through ONE `train_variant()` body (never copy-paste it), on the IDENTICAL euro-era row set (only the columns differ), into `models/<variant>/`. Both serve on every prediction; the response carries `baseline` + `with_macro` blocks and a `variant_agreement` flag. The forward paper-trading ledgers (one per variant) decide which variant earns its keep.
- **`src/inference.py` `PredictionService`** loads every artifact once, with graceful-degradation gates per variant (`baseline_ready`/`macro_ready`/`models_ready`; per-family flags live inside `service.variants[name]`). A broken variant never crashes the other. It is the single serving path behind `api.py`.
- **Two fallback chains, neither hard-fails:** `src/live_data.py` (MT5 → yfinance → bundled history CSV) and `src/macro_data.py` (FRED API → FRED public CSV → on-disk cache → `None`). The baseline variant consumes no macro columns, so it is immune to FRED outages by construction.
- **Front door (`/`, openspec change `friendly-home-page`):** `/` serves the plain-language, bilingual home page `static/home.html`. The research console `static/index.html` is unchanged at `/advanced`. The home page renders from `GET /api/home` (`src/home_summary.py`), which is read-only: it runs no model and writes nothing, and it shows no spread, pips or P&L. Only its refresh button predicts, via the existing `POST /api/predict`. Its reliability strip is descriptive (Wilson 95 % vs a coin, N ≥ 30) and is never a hypothesis test.
- **App shell (openspec change `app-shell-and-gold-history`):** every page route carries one top bar, look and BG/EN and theme switch.
  - The source is `src/app_shell.py`, with the assets in `static/shell/` (served at `/ui/shell/`).
  - `api.py`'s `ShellLayer` injects the shell as each response leaves, so pinned page sources never change: `src/tracking.py`, `src/paper_trading.py` and `static/index.html` stay byte-identical.
  - `/advanced` equals `index.html` plus the shell; `app_shell.strip` proves it.
  - Add a new page to `PAGE_PATHS` and `NAV`. Never paste navigation markup into a page.
  - Details: `ARCHITECTURE_DOCS.md §5.4.1`.
  - `/gold-analytics` (openspec change `gold-market-analytics`, `src/gold_analytics/`, outside `src/gold` because it shows the spread) is descriptive only: no model, no volume of any kind, no trade word, MT5 read only. Build its spread cache with `python -m src.gold_analytics.refresh --days 250`. Details: `ARCHITECTURE_DOCS.md §5.4.2`.
- **Volatility family (`src/volatility.py` → `models/volatility/`):** predicts `target_volatility_pct` (|next-day log return|·100) — the ONLY neural family with a CI-confirmed edge over its honest baseline (5-seed multi-task LSTM ensemble vs train-only-fit GARCH(1,1); pre-registered gate in `results/volatility_seed_ensemble.csv`). ONE price-only model, NOT per-variant; its own hypothesis family (`results/volatility_hypothesis_log.csv`, now SPENT on the validation arbiter — new volatility claims need forward data). The validated object is the FULL 5-seed mean: `vol_ready` is all-or-nothing, never serve a partial ensemble. Details: `ARCHITECTURE_DOCS.md §3.5`.

## Invariants you must preserve (these caused real bugs when broken)

Changing any of these requires updating `config.json`, `_train_pipeline.py`, notebook **Section 19**, and `src/inference.py` **together**:

1. **Single `global_scaler.pkl` per variant, shared by that variant's GBM and LSTM.** There are no per-model scalers. Each variant's PCA (`lag_pca` + `lag_scaler`) and `global_scaler` are fit **only on the 0–80% train block** and live together under `models/<variant>/` — never mix one variant's scaler/PCA with another's models.
2. **`target_return` is in PERCENT, natively.** The `* 100` lives **only** in `src/features.py::add_advanced_features`. Both the GBM regressor and the LSTM return head train on and output percent — there must be **no `* 100` anywhere else** (not in `_predict_gbm`, not in the LSTM cell).
3. **Unified chronological split** from `config.json` (`train_fraction=0.80`, `val_fraction=0.10`): GBM trains `[0:80%]`; LSTM trains `[0:70%]` with `[70%:80%]` for early-stopping; **both** test on the identical held-out `[80%:100%]`.
4. **No look-ahead bias.** Targets via `shift(-1)`; `ffill` only carries a *past* value forward (never future backward); scaler/PCA fit train-only; `TimeSeriesSplit` everywhere (never random K-fold). The FRED/no-look-ahead unit tests guard this.

The 7 daily artifacts (`lag_scaler`, `lag_pca`, `global_scaler`, `best_gbm_eurusd`, `best_gbm_regressor_eurusd`, `lstm_multitask_eurusd.keras`, `lstm_time_steps`) exist **once per variant** under `models/baseline/` and `models/with_macro/` (14 total), produced by `_train_pipeline.py` and loaded by `PredictionService`. `test_smoke.py` asserts all of them exist. The H1 ensemble artifacts stay at `models/` root (price-only, shared by both variants); the volatility ensemble lives under `models/volatility/` (5 seed models + own PCA/scalers + `vol_metrics.json`, also smoke-asserted).

## Production Methodology (post-defense — the project now trades REAL money)

The exam is passed; a false-positive feature is now live capital risk. These rules govern **all new feature/model claims** and override the older test-block narrative where they conflict (full detail: `ARCHITECTURE_DOCS.md` → *Production Methodology*, `IMPROVEMENT_LOG.md` → *Production methodology hardening*):

1. **The historical test block `[80%:100%]` is SPENT for feature search.** It was reused as a repeated KEEP/DROP criterion (data-snooping). All feature ablation now runs on the **validation slice `[70%:80%]`** via `src/ablation.py` (fit on `[0:70%]` only; test block never indexed). The test block is a **one-shot final report** from `_train_pipeline.py`, never a search knob. Do **not** add or judge a feature by scoring it on the test block.
2. **Every KEEP clears a Bonferroni-corrected bar, not a flat 0.05.** `results/feature_hypothesis_log.csv` counts every feature hypothesis ever spent; the bar is `alpha = 0.05 / family_size` (now `0.05/9 ≈ 0.00556` — after the rejected `fomc_calendar_block` (#5, 2026-07-17), `cot_positioning_block` (#6, 2026-07-20), `fibonacci_retracement_block` (#7, 2026-07-26), `vix_regime_block` (#8, 2026-07-26), and `volatility_forecast_block` (#9, 2026-07-26) ADD-tests), printed in every `src/ablation.py` report. Register a genuinely new feature (it tightens the bar for all). All 4 macro features are **KEEP-provisional** — no proven edge; the five ADD-test bundles are DROPs. VIX (`src/vix_features.py`) rides the shared FRED framework (`macro.features.vix`) to maintain `results/vix.csv`, but its level is kept out of `MACRO_MERGE_COLUMNS` and its features out of `FEATURE_COLUMNS` (ablation-only). A built-but-unspent Fibonacci extension hypothesis (`src/fibonacci_fractals.py`) runs only if #7 clears — it did not, so it stays dormant. `volatility_forecast_block` (`predicted_vol_pct`, `src/volatility.py::load_frozen_volatility_ensemble`) is CROSS-FAMILY REUSE of the already-validated volatility ensemble (§ volatility family below) via pure batch inference on the frozen `models/volatility/` artifacts — no retraining — and also DROPped.
3. **The forward paper-trading ledgers are the primary production-worthiness signal.** `src/paper_trading.py` (`results/paper_trading_log_baseline.csv` + `results/paper_trading_log_macro.csv`, one per model variant, at `/paper-trading` + `/api/paper-trading`) accumulates **simulated**, cost-net P&L ledgers from live calls as sessions settle — whichever variant nets better over a meaningful forward window (**months**) is the honest winner; do not decide by re-analyzing the spent test block. It is **simulated only** — do **not** add broker/order-execution/position-sizing/stop-loss code; real execution is a separate risk-management conversation, only after a ledger shows a supported edge.

4. **A commit that touches `models/` must declare the retrain.** `.githooks/commit-msg`
   (enable once: `git config core.hooksPath .githooks`) refuses any commit staging a path
   under `models/` unless the message carries a `RETRAIN:` line. Six commits titled
   *"Refactor code structure for improved readability and maintainability"* moved production
   artifacts: `d222ecc` (2026-06-21, 10; its title also says *"and remove redundant
   sections"*), `0ece63c` (2026-07-07, 10), `b30599f` (2026-07-25, 18), `a73344e`
   (2026-08-08, 12), `3def541` (2026-08-08, 17), `f2645a0` (2026-08-15, 30) — 97 artifact
   modifications; `f2645a0` also re-read the one-shot test block while doing it. An earlier
   version of this list named `82b45ee` and `d61d033`; both touched zero model artifacts.
   They are same-titled serving-churn commits dated to the two clean-ups (`8a02c9b`,
   `5c0fe0d`), so the list had conflated *commits around the incidents* with *commits that
   hid a retrain*. Only `f2645a0` of the original three was correct. What the log can
   evidence is narrower than what git can: `POST /api/retrain` truncates
   `results/retrain.log` on every launch, and until 2026-09-21 nothing preserved the previous
   run, so **2026-09-11 is the earliest retrain the log can EVIDENCE, not the earliest that
   occurred** — git bounds `models/` movement back to 2026-06-20. Since 2026-09-21
   `src/retrain_log.py` archives each log to `results/retrain_logs/` before the next launch
   truncates it, and `src/artifact_provenance.py` (`GET /api/provenance`, dashboard banner)
   reports served `models/` artifacts that differ from the fixture-pinned declared set as
   *"undeclared for N days"*. Never bundle a retrain into a commit about something else, and
   never `--no-verify` past this.

5. **Horizon study and forward arbiter (`src/forecast_eval/`, openspec change
   `horizon-study-forward-evaluation`).** A standalone research system that trains its own
   *challengers* for every model type, imports the production feature code read-only, and
   never writes under `models/` (artifacts in gitignored `research_models/`, digests in
   `results/horizon_study/artifact_manifest.csv`). The historical study
   (`results/horizon_study/`, README there) is DESCRIPTIVE only. Horizon claims are decided
   solely on forward data logged every 15 minutes by `EURUSDProphet-ForwardLogger`
   (`results/forward_eval/`, RUNBOOK there). Scoring stays off until
   `results/forward_eval/PRE_REGISTRATION.md` is committed. The direction family is 12 of
   46 candidate cells at `alpha = 0.05/12`; the rest, including every daily-cadence cell,
   are `UNDERPOWERED — NO DECISION` by design. Its two families never touch the `0.05/9` bar.

6. **Bar labels are the broker server's wall clock, and it is `Europe/Berlin`.** Every
   cache in this repo stores MT5's epoch localised to UTC as-is, so a label is NOT UTC.
   `src/forecast_eval/m15_data.py` establishes this from the weekly market boundary
   (open label 23:00, and 22:00 in the US/EU DST-mismatch weeks of March and late
   October; 99.3 % of 425 weekends) and REFUSES a run whose weekly opens stop matching.
   The M15 session cells forecast only inside the owner's trading window — 15:30–23:00
   `Europe/Sofia`, which is the fixed label window 14:30–22:00 because Sofia is Berlin
   + 1 h all year — and a trade is eligible only if its as-of bar *and* its target bar
   fall inside it. Sub-hourly targets and costs use the **mid** price built from each
   bar's own spread: on bid closes alone the New York 17:00 rollover reads a spurious
   72 % one-hour "accuracy" because the bid falls when the spread quintuples. Every
   sub-hourly cell is scored on both price definitions and a gap beyond the declared
   1.0 pp tolerance labels it a spread artifact. Do not add an intraday rule without
   checking which clock its timestamps are on.

7. **Gold is its own research family (`src/gold/`, openspec change `gold-forecasting`,
   `ARCHITECTURE_DOCS.md §3.12`).**
   - It has three registries: `results/gold/gold_{direction,return,volatility}_hypothesis_log.csv`,
     with alpha 0.025 / 0.0125 / 0.025, fixed in `results/gold/PRE_REGISTRATION.md`.
     These never touch the `0.05/9` bar.
   - It is **gross of cost**: spread and tick volume are used nowhere, so no gold
     verdict is a trading result.
   - It never writes under `models/` and never reads `config.json`. Its frozen
     artifacts live in gitignored `research_models/gold/` with digests in
     `results/gold/artifact_manifest.csv`.
   - Its forward logger is the scheduled task `EURUSDProphet-GoldLogger`. Corrections
     go to `results/gold/RESULTS.md`.
   - **MT5 chart signals** (change `gold-mt5-chart-signals`, `src/gold/mt5_export.py`):
     - Each logger pass rebuilds `Common\Files\EURUSDProphet\gold_signals.tsv`.
     - The read-only indicator `mql5/Indicators/ProphetGoldSignals.mq5` reads it and
       draws h1_lstm arrows and alert markers at H1 and below for the last 5 trading days.
       On D1 it draws d1_lstm and d1_gbm arrows for the last 20.
     - Its probability-filter buttons are a display choice, never a registered rule.
     - That file is the only gold write outside the repo, and nothing reads it back.
     - Python builds all chart wording, and it must never name a trade.
     - Install with `scripts/gold/install_mt5_indicator.ps1`.
   - **GVZ morning check** (change `gold-gvz-morning-check`, `src/gold/gvz.py`): FRED
     `GVZCLS` + yfinance top-up, the first card on `/gold`. Its logs,
     `results/gold/gvz_log.csv` and `gvz_settlements.csv`, are descriptive and never
     scored. A GVZ claim needs its own pre-registration.
   - Gold features must stay scale-free: the price tripled, so a raw level puts the test
     block outside the training range.
   - **Round two** (change `gold-forecasting-v2`) adds the range, macro and direction2
     families, registered in `PRE_REGISTRATION_V2` and scored once: all DROP or
     UNDERPOWERED. Its settings live in `src/gold/gold_config_v2.json`.
     - **Never edit `gold_config.json`.** Round one pins its digest, and every study stage
       refuses if it changes.
     - **Refits are shadow versions** (`src/gold/versions.py`). Only
       `python -m src.gold.versions promote` changes the registered models, and it
       refuses while a forward window is open.
   - **Window strategy backtest** (change `gold-window-strategy`, package
     `src/gold_window/`, outside `src/gold` because it is net of cost): four fixed
     in-window rules with walk-forward models. Its `results/gold/window/backtest.md` is
     **descriptive only**, because the rules were chosen after the data was seen. It has
     no registry or verdict, and must never be cited as an edge.
   - **Acceleration direction study** (change `gold-acceleration-direction`, package
     `src/gold_accel/`): time-of-day acceleration features added to shadow copies of
     h1_gbm/h1_lstm plus a new `h1_accel`, walk-forward from 2018. Its
     `results/gold/accel/backtest.md` is **descriptive only** (the acceleration was seen
     first); it never registers, versions or promotes a model.

8. **External foundation models are a second opinion, `external_tsfm` (`src/external/tsfm/`,
   openspec change `tsfm-second-opinion`, `ARCHITECTURE_DOCS.md §3.13`).**
   - Chronos-2 and TimesFM 2.5 forecast the next-hour **size** for gold and EUR/USD.
     They are never an input, ensemble member or feature of the in-house models.
   - They run only in `worker.py` inside `.venv-tsfm` (`requirements-tsfm.txt`). **Never
     install their packages into `.venv`.** `api.py` imports only the read-only `view`.
   - Weights and LoRA adapters are pinned by digest in
     `results/external_tsfm/artifact_manifest.csv` and live in gitignored
     `research_models/external_tsfm/`.
   - `CLEAN_START = 2025-11-01`: nothing earlier is scored. The fine-tuning uses only the
     frozen history before it.
   - Its own family: 4 cells at `alpha = 0.05/4`, scored once at N = 1,500 or 2027-04-30
     against the frozen GARCH baselines. P(up) is shown but never registered. It never
     touches the `0.05/9` bar or the gold registries.
   - Rows are `dry_run` until `results/external_tsfm/PRE_REGISTRATION.md` and `.json` are
     committed. The forward logger is the task `EURUSDProphet-ExternalLogger` at :01,
     before the gold pass. Runbook: `results/external_tsfm/RUNBOOK.md`.

9. **Fischer Fibonacci + CUDA MLP is its own family, `fx_fib` (`src/fx_fib/`, openspec change
   `fx-fibonacci-mlp`, `ARCHITECTURE_DOCS.md §3.14`).**
   - It covers EURUSD and GBPUSD H1, with D1 as causal context, and is **gross of spread and
     slippage** by owner decision. No verdict from it is a trading result.
   - It is decided by its own purged walk-forward (owner decision 2026-11-21), in four
     cells at `alpha = 0.05/4` in `results/fx_fib/fx_fib_hypothesis_log.csv`. Each cell is
     scored once.
   - Earlier results (#7, the H1 direction block) are recorded as facts, not blockers. It
     never touches the `0.05/9` bar.
   - A changed `src/fx_fib/fx_fib_config.json` is a new hypothesis. Label, training and
     scoring stages refuse a config whose digest differs from the verdicts'.
   - It never writes `models/`. Its frozen artifacts are in `research_models/fx_fib/`.
   - All four cells were DROP on 2026-11-21.

10. **One forecast per day after the New York open is its own family, `fx_session`
    (`src/fx_session/`, openspec change `fx-session-ny`, `ARCHITECTURE_DOCS.md §3.15`).**
    - It covers EURUSD and GBPUSD; AUDUSD feeds only the dollar factor.
    - Features are taken at the close of the bar labelled 14:00 Berlin (16:00 Sofia). The
      target is the sign of the mid close at 21:00 Berlin (23:00 Sofia).
    - It is **gross of spread and slippage**, and decided by its own walk-forward only.
    - It has four cells at `alpha = 0.05/4` in `results/fx_session/fx_session_hypothesis_log.csv`:
      LightGBM vs prior, and an MLP vs LightGBM that runs only after that pair's LightGBM
      KEEP. Untriggered alpha is never reassigned.
    - History starts on 2012-01-01. The broker's GBPUSD H1 bars from December 2010 to
      November 2011 are one hour off, yet pass the 90 % clock check, so do not trust that
      threshold alone for this broker's old history.
    - A day is used only if its own week's weekly open is on the Berlin clock.
    - A changed `fx_session_config.json` is a new hypothesis; every stage refuses it.
    - It never writes `models/`. Its frozen artifacts are in `research_models/fx_session/`.
    - Result (2026-11-21): both LightGBM cells DROP, both MLP cells NOT TRIGGERED.

11. **The 2-year yield differential at a 5-day horizon is its own family, `fx_rates`
    (`src/fx_rates/`, openspec change `fx-rates-2y`, `ARCHITECTURE_DOCS.md §3.16`).**
    - It covers EURUSD and GBPUSD, with DE2Y−US2Y and UK2Y−US2Y.
    - Yields come from FRED (`DGS2`, `DGS10`), the Bundesbank Svensson curve and the BoE
      GLC spot curve at 2.0 years. Use the generic client name: FRED drops a fake
      browser User-Agent.
    - Yields always lag one FX trading day. The fill limit is 3 business days.
    - FX D1 bars are read-only from the fx_session H1 caches from 2012.
    - It is **gross of cost** and decided by its own walk-forward only.
    - It has four cells at `alpha = 0.05/4` in `results/fx_rates/fx_rates_hypothesis_log.csv`,
      with 10-day blocks for the overlapping 5-day targets.
    - The MLP cells run only behind a LightGBM KEEP.
    - It never writes `models/`. Its frozen artifacts are in `research_models/fx_rates/`.
    - Result (2026-11-21): both LightGBM cells DROP, both MLP cells NOT TRIGGERED.

12. **Bollinger Band (20, 2 SD) mean reversion on D1 is its own family, `fx_bb`
    (`src/fx_bb/`, openspec change `fx-bb-reversion`, `ARCHITECTURE_DOCS.md §3.17`).**
    - It covers EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD and USDCHF.
      - The first three are read-only from the fx_session caches.
      - USDJPY, USDCAD and USDCHF are in `results/fx_bb/data/`.
    - D1 bars close at 17:00 New York and are built from the Berlin labels. The close is
      the mid price.
    - **USDCHF on this broker is one hour off from 2012-01 to 2014-11**, yet it passes 96 %
      of the weekly-open check. It therefore starts on 2015-02-01. Each yearly lag check
      against EURUSD must peak at lag 0, on every refresh and every load.
    - **Corrupt spreads.** AUDUSD (300 000 points) and USDJPY (3 000 points) carry corrupt
      spreads on 2014-01-07.
      - A spread above 50 pips voids any mid price.
      - fx_session read AUDUSD only through its bid prices, so it is unaffected.
      - Check the spread field before using a mid price from these caches.
    - It is **gross of cost**. It has two cells at `alpha = 0.05/2` in
      `results/fx_bb/fx_bb_hypothesis_log.csv`: the rule itself on every event, and a pooled
      LightGBM against the base rate. Both use day-block bootstraps. No MLP.
    - It never writes `models/`. Its frozen artifacts are in `research_models/fx_bb/`.
    - Result (2026-11-22): both cells DROP. The rule averages +0.014 ATR with a 50.7 % win
      rate. LightGBM's log-loss is worse than the base rate's.

13. **The same Bollinger events as a trade with an SMA20 exit are their own family, `fx_bb_sma`
    (`src/fx_bb_sma/`, openspec change `fx-bb-sma-exit`, `ARCHITECTURE_DOCS.md §3.19`).**
    - **The trade.**
      - Entry at the D1 mid close of a first breach.
      - Take profit at `SMA20[t]`, and a stop at the same distance beyond the entry.
      - Exits scanned hour by hour on H1 bars. A bar touching both levels is a stop.
      - Time stop at the close of `t+10`, with one open trade per pair.
    - **Data.** It reads fx_bb's caches and checks read-only and never refreshes them. A test
      pins its event and feature config blocks to fx_bb's.
    - **It is the only fx family scored NET of cost.** The cost is the recorded bar spread
      only, half at entry and half at exit. Slippage, swap and commission are not
      modelled, so no verdict is a trading result.
    - **Cells.** Two cells at `alpha = 0.05/2` in
      `results/fx_bb_sma/fx_bb_sma_hypothesis_log.csv`, using day-block bootstraps keyed by
      entry date with **10-business-day blocks**, because a trade lives up to 10 days.
    - **The events were already seen in fx_bb.** Its notes say so.
    - It never writes `models/`. Its frozen artifacts are in `research_models/fx_bb_sma/`.
    - Result (2026-11-22): both cells DROP.
      - The rule averages −0.008 net R. Its exits split 32 / 32 / 35 % between take profit,
        stop and time stop. Its profit factor is 0.92 net.
      - LightGBM is within noise of the base rate.

14. **The New York session range is its own family, `fx_nyvol` (`src/fx_nyvol/`, openspec change
    `fx-ny-volatility`, `ARCHITECTURE_DOCS.md §3.18`).**
    - **Target.** `y = log(mid range of the bars labelled 15:00–21:00 Berlin / mid close14)`
      for EURUSD and GBPUSD. Features come at the close of the 14:00 bar.
    - **Data.** It reads fx_session's caches and helpers read-only. Eligibility is joint for
      both pairs, and a spread above 50 pips voids the day.
    - It is **gross of cost** and has its own registry,
      `results/fx_nyvol/fx_nyvol_hypothesis_log.csv`: four cells at `alpha = 0.05/4` with
      5-day blocks. It never touches the production volatility family or the `0.05/9` bar.
    - It never writes `models/`. Its frozen artifacts are in `research_models/fx_nyvol/`.
    - Result (2026-11-22):
      - LightGBM vs HAR is KEEP for both pairs (−9.6 % and −6.2 % MSE).
      - The MLP vs LightGBM is KEEP for EURUSD and DROP for GBPUSD.
    - **The descriptive linear HAR-X beats both non-linear models** (MSE 0.80 and 0.85 ×
      HAR's). The KEEPs show that pre-session information predicts the range, not that
      non-linearity helps. Do not cite LightGBM or the MLP as the best range model. Using
      HAR-X needs its own pre-registration.

15. **The NY opening-range breakout on hot days is its own family, `fx_ny_orb` (`src/fx_ny_orb/`,
    openspec change `fx-ny-orb-hot`, `ARCHITECTURE_DOCS.md §3.20`).**
    - **[B] Historical.** The owner's M15 breakout rule, 2018-08 .. 2026-08, net of the recorded
      spread plus 0.4 bps.
      - Two cells at `alpha = 0.05/2` in `results/fx_ny_orb/fx_ny_orb_hypothesis_log.csv`.
      - **No outcome may be computed before `PRE_REGISTRATION.md` and `.json` are committed.**
        `prereg.gate` checks git.
    - **[A] Forward.** A frozen OLS HAR-X against HAR on live days, two cells at `alpha = 0.05/2`,
      scored at 270 days or 2028-01-31.
    - **The clock.** Live passes use real time on the Berlin label clock, **never MT5's last
      tick**: a stale weekend tick once backfilled Friday. The logs are append-only and never
      backfilled.
    - **[C] The chart.** `ProphetNYOrb.mq5` draws from `Common\Files\EURUSDProphet\ny_orb.tsv`.
      - **A live entry arrow exists only behind a historical KEEP of `orb_hot_net_r`** (the owner's
        gate).
      - The indicator never places an order, and its text never names a lot or a promise.
    - It never writes `models/`, another family's files, the M1 parquets or the horizon-study cache.
    - Result (2026-11-22): both historical cells DROP. Hot days average −0.164 net R over 170 trades
      (non-hot −0.031 R over 1 951), with 66 % of hot trades stopped out. The live entry arrow stays
      off; the chart shows the range, band and realised range only.

16. **Bollinger Bands and the NY session range on gold are their own family, `gold_bbnr`
    (`src/gold_bbnr/`, openspec change `gold-bb-nyrange`, `ARCHITECTURE_DOCS.md §3.21`).**
    - **Inputs.** It reads `results/gold/gold_h1.csv` read-only, with the gold clock check. It
      lives outside `src/gold/` and never touches `gold_config*.json` or the gold registries.
    - **The dollar index.**
      - It is fetched once into `results/gold_bbnr/data/`: hourly from MT5 `USDIndDec26` (from
        2017; the contract expires on 2026-12-11) and daily from Yahoo `DX-Y.NYB`.
      - Both caches are pinned by digest in the pre-registration. **Never refetch them.**
      - The hourly series is lag-0-checked against EURUSD on every load.
    - **Gross of cost on bid prices**; no verdict is a trading result.
    - **Registries.** There are three: `gold_bb` (2 cells), `gold_bb_sma` (2 cells) and `gold_nyvol`
      (3 cells, HAR-X vs HAR, LightGBM vs HAR-X, MLP vs HAR-X). Each cell is scored once.
      **No outcome may be computed before `results/gold_bbnr/PRE_REGISTRATION.md` and `.json` are
      committed**; `prereg.gate` checks git, the config digest and the cache digests.
    - **Power.** The Bollinger cells are weak: gold has about 180 first breaches in 2015–2026.
    - It never writes `models/`. Its frozen Part 2 models are in `research_models/gold_bbnr/`.
    - **The gold cache keeps growing.** The study reads only rows up to `bars.PINNED_LAST`
      (2026-11-20 21:00), and the gate hashes only that prefix, with line endings normalised.
      Never widen the window or move that pin.
    - Result (2026-11-23):
      - All four Bollinger cells DROP. The reversion rule averages −0.251 ATR, and the SMA20 trade
        averages −0.087 R.
      - Gold tends to keep going after a breach. That is descriptive only and must never be cited
        as an edge.
      - In the range family, HAR-X vs HAR is KEEP, at 0.847 × HAR's MSE. LightGBM and the MLP are
        both DROP against HAR-X.
      - The linear HAR-X is the best range model, as in fx_nyvol. Using it live needs its own
        pre-registration.

## Notebook specifics

- It runs **from `notebooks/`**, so file paths are `../` (e.g. `../config.json`, `../models/`). Any `subprocess` call to pytest must pass `cwd=os.path.abspath('..')` or pytest collects 0 tests.
- It has **two distinct tracks** — do not conflate them: the **exploratory baseline** (Sections 13–17: no FRED, no PCA, binary target, saves `exploratory_*.pkl`) and the **production pipeline** (Section 19: FRED + PCA + percent targets, saves the real artifacts). `_train_pipeline.py` only mirrors the production track.
- **Known drift (dual-variant era):** the notebook's Section 19 still trains the pre-dual single 27-col pipeline into root-level `models/*.pkl` paths that production no longer loads. `_train_pipeline.py` is the sole producer of the real per-variant artifacts; treat the notebook training cells as historical/research until they are ported to `train_variant()` (tracked in `IMPROVEMENT_LOG.md`).

## Expected (not buggy) behaviour

Daily EUR/USD is near-efficient: **ROC-AUC ≈ 0.50** and the return regressor **shrinks predictions toward the ~0% mean** (Huber loss on a noisy, zero-mean target → predict the conditional mean; the trained MAE ≈ the "predict the mean" baseline). This is documented as a feature of mathematical honesty in `ARCHITECTURE_DOCS.md §4.2.1` and notebook Section 21 — do **not** try to "fix" the low return magnitudes by tweaking the model. The consensus carries a `CONFIDENCE_THRESHOLD = 0.52` guard that flags near-chance agreement as `"MIXED / LOW CONFIDENCE"`. The forecast date is weekend-aware (Fri/Sat roll forward to Monday — the next *trading* session).
