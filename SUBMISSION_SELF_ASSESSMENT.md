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
with: after 50 registered hypotheses the honest answer on direction is ROC-AUC ≈ 0.50, and
the project reports that rather than manufacturing an edge (Section 21, Section 22.8).

### Layout (0–20)

22 numbered sections with a navigation table in cell 0, LaTeX for every mathematical
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

53 modules in `src/`, each with a single responsibility and a module docstring stating
scope and constraints. The single-source-of-truth contract is the central design decision:
training and serving both import `src/features.py`, so the feature matrix is byte-identical
on both sides and research-to-production drift is structurally impossible.

Typed throughout (`pyrightconfig.json`, `mypy`), linted (`ruff`), configuration centralised
in `config.json`. Four invariants that previously caused real bugs are documented in
`CLAUDE.md` and enforced by tests.

### Previous research (0–10)

**Ten cited sources**, notebook Section 22.7 — Fama (1970), Meese & Rogoff (1983), Harvey,
Liu & Zhu (2016), Bollerslev (1986), Hochreiter & Schmidhuber (1997), Diebold & Mariano
(1995), Parkinson (1980), Zhang, Mykland & Aït-Sahalia (2005), Clark (1973), López de Prado
(2018).

**Three distinct kinds of comparison**, which is where this criterion is genuinely met:

1. *Against our own previous submission* — Section 22.1–22.3, quantified.
2. *Against our own previous headline claim* — Section 22.4. The volatility ensemble was
   re-verified, replicated, and then shown to be a **calendar effect**: a six-number
   day-of-week lookup table beats the neural ensemble outright (MAE 0.20521 vs 0.21925).
3. *Against an external implementation* — Section 22.5. Kronos, a third-party pre-trained
   foundation model, carries incremental information (+0.1667, CI excludes zero) that does
   not convert into a distinguishable forecast gain.

### Gathering / cleaning / formatting data (0–10)

Two independent ingestion chains, neither of which hard-fails: price data
(MT5 → yfinance → bundled CSV, `src/live_data.py`) and macro data
(FRED API → FRED public CSV → on-disk cache → `None`, `src/macro_data.py`).

Statistical validity is treated as a first-class concern: targets via `shift(-1)`, `ffill`
only ever carries a past value forward, scaler and PCA fitted on the training block only,
`TimeSeriesSplit` everywhere and never random K-fold. Dedicated no-look-ahead unit tests
guard each of these. Documented in `ARCHITECTURE_DOCS.md` §2.

### Testing (0–10)

**424 test functions across 13 files** (up from 73 across 4 at the first submission),
covering smoke, unit, integration, no-look-ahead, and artifact-checksum tests that prevent
a retrain from silently altering production models.

Hypothesis testing is the project's organising principle: 12 families, 50 registered
hypotheses, Bonferroni-corrected bars (`α = 0.05 / family_size`), moving-block bootstrap
confidence intervals, McNemar tests for paired classification, and mandatory replication
on other currency pairs when a claim clears. **34 of 50 hypotheses are recorded as DROP.**

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
a failure. Section 22.8 states the position plainly.

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
- **The curl / discrete-Hodge work (Section 22.6) is unfinished.** It is validated on
  synthetic data only, no hypothesis is registered, and it touches no production code.
- **`mit-deep-learning-book-pdf-master/`** is a third-party reference text included in the
  repository. It is not project code and is not used by any module.
