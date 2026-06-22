# AI-Assisted Development Project: EUR/USD Directional Predictor

## 1. Overview & System Requirements

**Project idea.** EUR/USD Prophet is a full machine-learning pipeline that predicts the next trading day's direction (UP/DOWN) and expected percentage return for the EUR/USD currency pair. The system is fully autonomous at inference time: it requires no manual date selection or data entry — it determines "today" from whichever live market source answers first, fetches exactly the historical context it needs, recomputes all technical indicators fresh, and produces a forecast for the next trading session.

**System requirements.**
- Fetch live OHLCV price data with a resilient fallback chain (MetaTrader5 terminal → Yahoo Finance → bundled historical snapshot), since no single data source can be assumed always available.
- Enrich price data with an exogenous macroeconomic signal — the US 10-Year/German 10-Year government bond yield differential — fetched from the FRED economic database, again with a fallback chain (official FRED API → FRED public endpoint → local cache).
- Engineer a reproducible, leakage-free feature set (24 technical/cyclical/macro features) shared identically between training and live inference, so the two paths can never silently diverge.
- Train two complementary model families — a Gradient Boosting dual pipeline (direction + return regression) and a Multi-Task LSTM (shared recurrent trunk, two output heads) — using strict chronological (walk-forward) validation to prevent look-ahead bias.
- Serve predictions through two interchangeable frontends: a Gradio web UI and a FastAPI REST endpoint, both backed by one shared inference service.
- Track every forecast against the realised market outcome over time, so prediction quality can be audited rather than assumed.
- Allow on-demand model retraining from the UI without blocking the running service.
- Validate every layer with an automated test suite (smoke, unit, integration) and document the full data/architecture flow for maintainability.

**Repository:** https://github.com/velizarMitov/EURUSD-prophet

---

## 2. System Architecture — Modules

The system was decomposed (with AI assistance) into the following technological modules:

1. **Live & Macro Data Ingestion** — resilient external data acquisition (`src/live_data.py`, `src/macro_data.py`)
2. **Feature Engineering & Dimensionality Reduction** — the shared feature contract (`src/features.py`)
3. **Model Training Pipeline** — GBM + Multi-Task LSTM training (`_train_pipeline.py`, plus a mirrored research notebook)
4. **Inference Service & Committee Consensus** — prediction logic shared by both frontends (`src/inference.py`)
5. **Web UI & REST API** — Gradio app and FastAPI service (`app.py`, `api.py`, `static/index.html`)
6. **Prediction Tracking & Automated Retraining** — outcome logging, history dashboard, background retraining (`src/tracking.py`, retrain endpoints)
7. **Testing & Validation Layer** — smoke, unit, and integration tests (`tests/`)

---

## 3. Development Process per Module

### 3.1 Live & Macro Data Ingestion

**Approach.** Live financial sources are individually unreliable, so ingestion is built as two ordered fallback chains: MT5 → Yahoo Finance → bundled history for prices, and FRED official API → FRED public CSV → on-disk cache for the macro yield-differential signal. **Workflow & testing.** A structured codebase audit surfaced a cache-overwrite defect that silently destroyed decades of cached macro history on every run; the fix merges incoming data into the existing cache instead of overwriting it, with the lost range recovered from git history. Each fallback tier is unit-tested in isolation via mocking, verifying call order and graceful degradation without a live network dependency. **AI tool choice.** Claude Code, for its ability to correlate the data module, its call sites, and git history within one session to find the defect's root cause.

**Representative directives:** *"Audit the macro-data ingestion path end-to-end for data-integrity risks."* · *"Fix the cache by merging new fetches into the existing file instead of overwriting it, and restore the lost historical range."*

---

### 3.2 Feature Engineering & Dimensionality Reduction

**Approach.** The principal risk is research-to-production drift, so all feature logic is centralized in one module (`src/features.py`) imported identically by training and inference; the scaler and PCA are fit strictly on a chronological training slice to prevent look-ahead leakage. **Workflow & testing.** Two previously separate scalers (fit on inconsistent 70%/80% splits) were consolidated into a single `global_scaler` fit on one unified 80% boundary, and the regression target was standardized to native percentage units for both model families, removing an inference-time `×100` rescaling inconsistency. Dedicated tests confirm the PCA is fit once and reused unmodified on held-out data, and that the macro forward-fill never carries a future value backward. **AI tool choice.** Claude Code, for coordinating one consistent change across the config, feature module, training script, and notebook simultaneously.

**Representative directives:** *"Unify the train/validation/test split boundaries and fit the PCA and scaler exclusively on the unified training block."* · *"Standardize the regression target so the GBM and the LSTM train on identical units."*

---

### 3.3 Model Training Pipeline (Dual GBM + Multi-Task LSTM)

**Approach.** Two model families were trained for direct comparison — a GBM dual pipeline (classifier and Huber-loss regressor) and a Multi-Task LSTM (shared trunk, two heads) — both validated with `TimeSeriesSplit`/chronological boundaries, never random shuffling. **Workflow & testing.** Following the preprocessing unification, both models were retrained and the resulting artifacts independently verified (feature dimensionality, target scale) by reloading and re-evaluating rather than trusting the training script's own output. The regressor was benchmarked against a trivial "predict the mean" estimator; a near-identical test MAE (0.2959% vs. 0.2958%) is documented as quantitative evidence that daily EUR/USD returns carry negligible exploitable signal. **AI tool choice.** Claude Code, to execute real training runs and cross-validate numeric claims directly against the persisted artifacts.

**Representative directives:** *"Retrain the dual Gradient Boosting ensemble on the unified training slice and report held-out metrics in percentage units."* · *"Benchmark the trained regressor against a constant-mean baseline to quantify learned signal."*

---

### 3.4 Inference Service & Committee Consensus

**Approach.** A single `PredictionService` class — artifact loading, feature-window resolution, per-model prediction, ensemble aggregation — is shared by both frontends, ensuring prediction logic can never diverge between them. **Workflow & testing.** Three production-safety gaps were closed: a confidence-threshold guard (0.52) now downgrades a unanimous-but-near-chance call to an explicit "MIXED / LOW CONFIDENCE" flag; a thin live data fetch now gracefully back-fills from historical data instead of raising a hard error; and forecast-date arithmetic was corrected to be business-day aware, rolling a Friday/Saturday data point forward to the next trading Monday. Unit tests exercise the consensus logic under synthetic agreement, disagreement, and low-confidence scenarios; an integration test drives the live endpoint end-to-end. **AI tool choice.** Claude Code, for tracing each live-edge failure mode through the call chain and verifying the date fix across all seven weekdays before deployment.

**Representative directives:** *"Add a confidence threshold so the ensemble does not report false confidence when both models are near chance level."* · *"Make the forecast date business-day aware so it always resolves to the next trading session."*

---

### 3.5 Web UI & REST API

**Approach.** Two interchangeable frontends sit on top of one shared service: a Gradio app for local interaction, and a FastAPI REST API serving a static dashboard for programmatic or browser-based use. **Workflow & testing.** Every backend change was verified against the actual running server via direct HTTP requests rather than code review alone. An integration test instantiates a FastAPI `TestClient` against the real application object and asserts the full prediction contract and dashboard availability. **AI tool choice.** Claude Code, operated with direct shell access to launch the server, poll for readiness, and issue real requests — closing the loop between "code changed" and "verified working."

**Representative directive:** *"Launch the service and confirm the live prediction endpoint returns a complete, correctly-typed response."*

---

### 3.6 Prediction Tracking & Automated Retraining

**Approach.** Two automation features close the loop between forecast and outcome: every prediction is logged once per trading day and scored against the realised market close on a dedicated history page; a Retrain control triggers model training as a detached background process — training takes 15–30 minutes and must never block a request — with a polling endpoint reporting live status and hot-reloading the refreshed artifacts on completion. **Workflow & testing.** Unit tests confirm idempotent logging and correct outcome scoring against a mocked closing price. The retraining endpoints were verified against a real running process: a concurrent second request correctly returned HTTP 409, and a deliberately terminated run was confirmed, via file timestamps, to leave the existing model artifacts untouched. **AI tool choice.** Claude Code, to start and terminate a genuine background training process and verify filesystem state directly rather than by inspection alone.

**Representative directives:** *"Add a Retrain control that runs training as a non-blocking background process and reports live status."* · *"Guarantee the retraining endpoint cannot run concurrently and that a failed run never corrupts existing artifacts."*

---

### 3.7 Testing & Validation Layer

**Approach.** A three-tier suite — smoke, unit, integration — treats time-series-specific correctness (no look-ahead, train-only preprocessing fits) as a first-class concern rather than an afterthought. **Workflow & testing.** The full 21-test suite was executed after every structural change elsewhere in the system, with failures investigated before proceeding, rather than deferring verification to the end of the project. **AI tool choice.** Claude Code, to author and *execute* tests directly within the same working session, closing the verification loop without a separate manual step.

**Representative directive:** *"Run the full test suite after this refactor and confirm there is no regression before proceeding."*

---

## 4. Challenges & Tool Comparison

**Biggest challenges encountered:**
- **Research-to-production drift.** Separate scalers fit on different splits, and a regression target trained in one unit but rescaled in another at inference, required coordinated changes across `config.json`, the training script, the inference service, and the notebook simultaneously.
- **Live-edge failure modes.** Bugs only visible at the historical/live data boundary: a cache-overwrite bug destroying cached macro data, a forecast date mechanically rolling onto a non-trading Saturday, and a thin live fetch that could hard-crash inference instead of degrading gracefully.
- **Distinguishing a real bug from an honest result.** Near-50% directional accuracy and near-zero predicted returns initially looked like defects; confirming this was the expected behaviour of a near-efficient market required comparing the model against a trivial mean-prediction baseline.
- **Environment friction.** Windows console encoding (`cp1252`) breaking on Unicode/emoji output, MLflow's file-store backend rejecting reads without an explicit opt-out flag, and port conflicts when restarting the live server.

**Which tool helped most, and why.** Claude Code was used throughout. Its main advantage was operating as a single agent with full access to the terminal, filesystem, and git history in one continuous session — reading the whole codebase to trace bugs to their root cause, making coordinated multi-file fixes, retraining real models, launching the actual server, and verifying behaviour with live HTTP requests, all without leaving the session. This closed-loop verification (change → run → observe real output → confirm) mattered more than code generation alone, especially for live-edge and data-integrity bugs only discoverable by running the system.
Copilot took me $40 credit and 1 subscription to Pro just dissaapeared for 3 pormpts... it was a disaster... of cource i rised a compliant and asked for refund....

**What I would improve if continuing the project:**
- Replace the file-based MLflow tracking store with a database backend (e.g., SQLite) for reliable experiment-history queries.
- Move background retraining from an in-memory subprocess handle to a persistent job queue, so status survives an API server restart.
- Explore longer-horizon or volatility-scaled targets, since daily-direction prediction is close to its theoretical ceiling on price history alone.

---

## 5. Working System Evidence

*[SCREENSHOT 1 — placeholder: the running Gradio/FastAPI dashboard after clicking "Fetch Live Market Data & Predict Tomorrow," showing the consensus prediction, individual GBM/LSTM cards, and market data source]*

*[SCREENSHOT 2 — placeholder: terminal output of `python -m pytest -q` showing the full test suite passing, and/or the `/history` prediction-vs-actual page]*

---

## 6. Repository

GitHub: **https://github.com/velizarMitov/EURUSD-prophet**
