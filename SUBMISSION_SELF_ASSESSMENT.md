# Self-assessment against the exam criteria

Submitted for the SoftUni Deep Learning final exam.
**Second submission** — the first was on 23 July 2026. Changes since then:
`CHANGELOG_SINCE_2026-07-23.md` and **Section 22** of
`notebooks/01_data_preparation.ipynb`.

Point ranges are the guideline's own. This table maps each criterion to concrete,
checkable evidence in the repository.

---

### Problem statement (0–10)

Next-day EUR/USD directional and return prediction — a real problem with real capital at
risk, and one with a well-known negative prior (Meese & Rogoff, 1983). Stated in
`README.md`, notebook Section 0, and formalised mathematically in Section 1.

The project answers the *correct* problem, which turned out not to be the one it started
with: after 56 registered hypotheses the honest answer on direction is ROC-AUC ≈ 0.50, and
the project reports that rather than manufacturing an edge (Section 21, Section 22.11).

### Layout (0–20)

22 numbered sections (Section 22 in eleven subsections) with a navigation table in cell 0, LaTeX for every mathematical
formulation, and a consistent structure per section (theory → code → empirical
interpretation). Supporting documents are separated by purpose:

| File | Role |
|---|---|
| `README.md` | orientation and how to run |
| `HOW_TO_RUN.md` | step-by-step execution |
| `ARCHITECTURE_DOCS.md` | 1,892-line technical reference |
| `IMPROVEMENT_LOG.md` | 2,025-line dated research journal |
| `CHANGELOG_SINCE_2026-07-23.md` | what changed since the first submission |

### Code quality (0–20)

54 modules in `src/`, each with a single responsibility and a module docstring stating
scope and constraints. The single-source-of-truth contract is the central design decision:
training and serving both import `src/features.py`, so the feature matrix is byte-identical
on both sides and research-to-production drift is structurally impossible.

Typed throughout (`pyrightconfig.json`, `mypy`), linted (`ruff`), configuration centralised
in `config.json`. Four invariants that previously caused real bugs are documented in
`CLAUDE.md` and enforced by tests.

**Nine trained neural networks ship with the repository** — `python -m src.dl_model_report`
prints the full model card. Two shared-trunk multi-task LSTMs (daily direction + return,
one per variant), a five-seed volatility ensemble, an hourly sequence-to-vector LSTM, and
an H1 technical-indicator LSTM. Every one was trained with a chronological split, early
stopping on a held-out slice, and dropout regularisation; the seed ensembling was forced by
measured framework nondeterminism rather than chosen for effect.

Beyond Keras, two further frameworks were used directly rather than through a wrapper: a
PyTorch `DirectionLSTM` (`src/h1_direction_model.py`), an LSTM with a dropout-regularised
MLP head written against `torch.nn`; and a JAX/Equinox continuous-time architecture with
custom surrogate gradients for a spiking readout (`src/ltc_spiking_arch.py`, 758 lines).

A tenth network is **not** ours: **Kronos**, a third-party pre-trained time-series
foundation model, integrated behind a commit-pinned HuggingFace loader
(`src/external/kronos/`) as an optional dependency, so the core app installs and serves
without it. It is the project's only comparison against a modern foundation model, and the
contamination it forced — its own training cutoff restricted the evaluation window to the
spent test block — is disclosed in the registry rather than worked around.

That inventory is the context for the headline finding. The calendar model wins **against
properly built networks**, which is what makes the comparison worth reporting.

### Previous research (0–10)

**Eleven cited sources**, notebook Section 22.10 — Fama (1970), Meese & Rogoff (1983), Harvey,
Liu & Zhu (2016), Bollerslev (1986), Hochreiter & Schmidhuber (1997), Diebold & Mariano
(1995), Parkinson (1980), Zhang, Mykland & Aït-Sahalia (2005), Clark (1973), López de Prado
(2018), Murphy (1973).

**Four distinct kinds of comparison**, which is where this criterion is genuinely met:

1. *Against our own previous submission* — Section 22.1–22.3, quantified.
2. *Against our own previous headline claim* — Section 22.4. The volatility ensemble was
   re-verified, replicated, and then shown to be a **calendar effect**: a six-number
   day-of-week lookup table beats the neural ensemble outright (MAE 0.20521 vs 0.21925).
3. *Against an external implementation* — Section 22.5. Kronos, a third-party pre-trained
   foundation model, carries incremental information (+0.1667, CI excludes zero) that does
   not convert into a distinguishable forecast gain.
4. *Against the incumbent production model, and winning* — Section 22.6. A ten-parameter
   calendar model beats the 5-seed neural ensemble on both blocks.

### Gathering / cleaning / formatting data (0–10)

Two independent ingestion chains, neither of which hard-fails: price data
(MT5 → yfinance → bundled CSV, `src/live_data.py`) and macro data
(FRED API → FRED public CSV → on-disk cache → `None`, `src/macro_data.py`).

Statistical validity is treated as a first-class concern: targets via `shift(-1)`, `ffill`
only ever carries a past value forward, scaler and PCA fitted on the training block only,
`TimeSeriesSplit` everywhere and never random K-fold. Dedicated no-look-ahead unit tests
guard each of these. Documented in `ARCHITECTURE_DOCS.md` §2.

**`DATA.md` documents provenance end to end**: every FRED series identifier, the MT5 server
clock forensics (CET/CEST, established from a March DST excursion), the strict-inner-join
alignment (2,980,060 raw bars → 2,959,641 common rows, unmatched bars dropped rather than
forward-filled), the gap census, and a **permutation control proving the activity proxy
carries real information** (z = −42.7 honest vs −0.6 shuffled). All data and all 46 trained
artifacts are committed; `python verify_installation.py` re-derives the headline result
offline and fails loudly if the checkout is incomplete.

### Testing (0–10)

**452 test functions** (up from 73 at the first submission),
covering smoke, unit, integration, no-look-ahead, and artifact-checksum tests that prevent
a retrain from silently altering production models.

Hypothesis testing is the project's organising principle: 15 families, 56 registered
hypotheses, Bonferroni-corrected bars (`α = 0.05 / family_size`), moving-block bootstrap
confidence intervals, McNemar tests for paired classification, mandatory replication on
other currency pairs when a claim clears, and power analysis before spending α (the G10
macro panel was stopped before fitting). **37 of 56 hypotheses are recorded as DROP.**

### The headline result

`src/calendar_volatility.py` — a GARCH(1,1) × six-weekday-multiplier model, ten parameters,
numpy only — beats the production 5-seed multi-task LSTM ensemble on **both** blocks of the
identical row set: validation MAE 0.16209 vs 0.18594, test 0.19279 vs 0.21897, higher R² on
both. Section 22.6. Its outstanding gaps are stated in the registry row rather than hidden.

`src/calibration_audit.py` shows the live `CONFIDENCE_THRESHOLD = 0.52` guard lifts accuracy
by +10.5 points on validation and by −0.005 on the test block — a textbook contamination
demonstration performed on our own production system. Section 22.7.

### Visualization (0–10)

Diagnostic exports in `results/`: learning curves, ACF/PACF, confusion matrices, residual
analysis, TimeSeriesSplit fold diagrams, feature-importance charts, FRED overlays. Plus two
live interactive surfaces served by `api.py` — the prediction dashboard and the
`/paper-trading` forward ledger.

Every plot is captioned with what it is meant to show, and the learning curves are used
specifically as evidence of *no data leakage* (Section 21.1) rather than as decoration.

### Communication (0–10)

The narrative arc is deliberate and, for a forecasting project, unusual: it argues its way
to a negative result on the first moment and defends that as the correct answer rather than
a failure. Section 22.11 states the position plainly.

Audience separation is explicit — `README.md` for a newcomer, the notebook for the grader,
`ARCHITECTURE_DOCS.md` for an engineer, `IMPROVEMENT_LOG.md` for anyone auditing how a
conclusion was reached.

---

## Known limitations, stated up front

- **Notebook Section 19 has drifted.** It still trains the pre-dual-variant single-27-column
  pipeline into root-level `models/*.pkl` paths that production no longer loads.
  `_train_pipeline.py` is the sole producer of the real per-variant artifacts. Tracked in
  `IMPROVEMENT_LOG.md`; the notebook training cells should be read as historical research.
- **One production model is live against its own DROP verdict** (H1 TI-LSTM, 2026-07-18).
  This was an owner decision and is recorded as an override with the contrary evidence
  beside it, not quietly.
- **The four macro features are KEEP-provisional.** None cleared the corrected bar; they
  are retained pending forward evidence and labelled as such everywhere.
- **The curl / discrete-Hodge work (Section 22.9) is unfinished.** It is validated on
  synthetic data only, no hypothesis is registered, and it touches no production code.
- **The calendar model (Section 22.6) is not fully registered.** Its row reads
  `PENDING-PAIRED-CI`: the paired bootstrap against the ensemble needs that model's per-row
  predictions, its GARCH is a numpy variance-targeting fit rather than the `arch` MLE used
  elsewhere, and it was built and scored in one pass with no pre-registration preceding
  measurement. All three are stated in the registry row.
- **`mit-deep-learning-book-pdf-master/`** is a third-party reference text included in the
  repository. It is not project code and is not used by any module.
