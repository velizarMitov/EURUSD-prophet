# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

EUR/USD next-day directional + return predictor. A research notebook trains the models; a shared inference service serves them behind two frontends. The deep reference for data flow, artifacts, and failure modes is **`ARCHITECTURE_DOCS.md`** — read it before any non-trivial change.

## Commands

Environment is **Windows / PowerShell**; a `.venv` is present. `MetaTrader5` is Windows-only.

```bash
# Tests (run from repo root — pytest discovers tests/ here)
python -m pytest -q                                  # full suite (~19 tests)
python -m pytest -q tests/test_unit.py               # one file
python -m pytest -q -k fetch_yield_differential      # one test / pattern

# Run the app (serves the SAME PredictionService both ways)
python app.py                                        # Gradio UI  -> http://127.0.0.1:7860
python -m uvicorn api:app --reload                   # FastAPI    -> http://127.0.0.1:8000 (POST /api/predict, no body)

# Retrain & regenerate the production artifacts in models/
python _train_pipeline.py

# Research notebook
jupyter notebook notebooks/01_data_preparation.ipynb
```

Pre-trained artifacts in `models/` **and** `results/eurusd_features.csv` are git-tracked, so the app runs immediately after `pip install -r requirements.txt` — **no training required**.

When a script prints non-ASCII (Cyrillic, ✓, →), set `PYTHONIOENCODING=utf-8` or the Windows `cp1252` console codec raises `UnicodeEncodeError`.

## Architecture: the single-source-of-truth contract

The whole design exists to stop **research-to-production drift**. Training (notebook + `_train_pipeline.py`) and serving (`app.py`/`api.py` → `src/inference.py`) import the **same** `src/features.py` so the feature matrix is byte-identical on both sides.

- **`src/features.py`** owns `FEATURE_COLUMNS` (the canonical **24** columns) and `LAG_COLUMNS` (6 autoregressive lags reduced by PCA). `compute_features()` is inference-safe (no target, no dropna → keeps the latest live bar); `add_advanced_features()` is the training variant (adds `target_return`/`target_direction`, drops NaNs).
- **`src/inference.py` `PredictionService`** loads every artifact once, with graceful-degradation gates (`gbm_ready`/`lstm_ready`/`models_ready`). It is shared by both frontends so they can never diverge.
- **Two fallback chains, neither hard-fails:** `src/live_data.py` (MT5 → yfinance → bundled history CSV) and `src/macro_data.py` (FRED API → FRED public CSV → on-disk cache → `None`).

## Invariants you must preserve (these caused real bugs when broken)

Changing any of these requires updating `config.json`, `_train_pipeline.py`, notebook **Section 19**, and `src/inference.py` **together**:

1. **Single `global_scaler.pkl` for both models.** There are no per-model scalers. PCA (`lag_pca` + `lag_scaler`) and `global_scaler` are both fit **only on the 0–80% train block**.
2. **`target_return` is in PERCENT, natively.** The `* 100` lives **only** in `src/features.py::add_advanced_features`. Both the GBM regressor and the LSTM return head train on and output percent — there must be **no `* 100` anywhere else** (not in `_predict_gbm`, not in the LSTM cell).
3. **Unified chronological split** from `config.json` (`train_fraction=0.80`, `val_fraction=0.10`): GBM trains `[0:80%]`; LSTM trains `[0:70%]` with `[70%:80%]` for early-stopping; **both** test on the identical held-out `[80%:100%]`.
4. **No look-ahead bias.** Targets via `shift(-1)`; `ffill` only carries a *past* value forward (never future backward); scaler/PCA fit train-only; `TimeSeriesSplit` everywhere (never random K-fold). The FRED/no-look-ahead unit tests guard this.

The 7 production artifacts (`lag_scaler`, `lag_pca`, `global_scaler`, `best_gbm_eurusd`, `best_gbm_regressor_eurusd`, `lstm_multitask_eurusd.keras`, `lstm_time_steps`) are produced by `_train_pipeline.py` and notebook Section 19, and loaded by `PredictionService`. `test_smoke.py` asserts they exist.

## Notebook specifics

- It runs **from `notebooks/`**, so file paths are `../` (e.g. `../config.json`, `../models/`). Any `subprocess` call to pytest must pass `cwd=os.path.abspath('..')` or pytest collects 0 tests.
- It has **two distinct tracks** — do not conflate them: the **exploratory baseline** (Sections 13–17: no FRED, no PCA, binary target, saves `exploratory_*.pkl`) and the **production pipeline** (Section 19: FRED + PCA + percent targets, mirrors `_train_pipeline.py`, saves the real artifacts). `_train_pipeline.py` only mirrors the production track.

## Expected (not buggy) behaviour

Daily EUR/USD is near-efficient: **ROC-AUC ≈ 0.50** and the return regressor **shrinks predictions toward the ~0% mean** (Huber loss on a noisy, zero-mean target → predict the conditional mean; the trained MAE ≈ the "predict the mean" baseline). This is documented as a feature of mathematical honesty in `ARCHITECTURE_DOCS.md §4.2.1` and notebook Section 21 — do **not** try to "fix" the low return magnitudes by tweaking the model. The consensus carries a `CONFIDENCE_THRESHOLD = 0.52` guard that flags near-chance agreement as `"MIXED / LOW CONFIDENCE"`. The forecast date is weekend-aware (Fri/Sat roll forward to Monday — the next *trading* session).
