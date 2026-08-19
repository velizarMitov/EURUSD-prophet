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
   under `models/` unless the message carries a `RETRAIN:` line. Three commits titled
   *"Refactor code structure for improved readability and maintainability"* each hid a
   production retrain; `f2645a0` also re-read the one-shot test block while doing it. Never
   bundle a retrain into a commit about something else, and never `--no-verify` past this.

## Notebook specifics

- It runs **from `notebooks/`**, so file paths are `../` (e.g. `../config.json`, `../models/`). Any `subprocess` call to pytest must pass `cwd=os.path.abspath('..')` or pytest collects 0 tests.
- It has **two distinct tracks** — do not conflate them: the **exploratory baseline** (Sections 13–17: no FRED, no PCA, binary target, saves `exploratory_*.pkl`) and the **production pipeline** (Section 19: FRED + PCA + percent targets, saves the real artifacts). `_train_pipeline.py` only mirrors the production track.
- **Known drift (dual-variant era):** the notebook's Section 19 still trains the pre-dual single 27-col pipeline into root-level `models/*.pkl` paths that production no longer loads. `_train_pipeline.py` is the sole producer of the real per-variant artifacts; treat the notebook training cells as historical/research until they are ported to `train_variant()` (tracked in `IMPROVEMENT_LOG.md`).

## Expected (not buggy) behaviour

Daily EUR/USD is near-efficient: **ROC-AUC ≈ 0.50** and the return regressor **shrinks predictions toward the ~0% mean** (Huber loss on a noisy, zero-mean target → predict the conditional mean; the trained MAE ≈ the "predict the mean" baseline). This is documented as a feature of mathematical honesty in `ARCHITECTURE_DOCS.md §4.2.1` and notebook Section 21 — do **not** try to "fix" the low return magnitudes by tweaking the model. The consensus carries a `CONFIDENCE_THRESHOLD = 0.52` guard that flags near-chance agreement as `"MIXED / LOW CONFIDENCE"`. The forecast date is weekend-aware (Fri/Sat roll forward to Monday — the next *trading* session).
