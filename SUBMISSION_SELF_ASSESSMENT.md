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
| `IMPROVEMENT_LOG.md` | 2,193-line dated research journal |
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

**533 test functions** across 19 files (556 collected tests; up from 73 at the first
submission), **all passing**, covering smoke, unit, integration, no-look-ahead, and
artifact-checksum tests that prevent a retrain from silently altering production models.

That guard has now fired twice, on both undeclared retrains, and caught every moved artifact
both times: nineteen on the 2026-08-08 retrain (re-baselined 2026-08-11) and thirty on the
2026-08-15 retrain in `f2645a0` (re-baselined 2026-08-19, after the reproducibility study had
measured the new artifacts). Every old→new digest of both events is preserved in
`tests/fixtures/PROTECTED_SET_REBASELINE_2026-08-11.json` and
`PROTECTED_SET_REBASELINE_2026-08-19.json`. Nothing was silently replaced.

Read the guard's record honestly: it did its job perfectly and it is still the *second* line
of defence. It reports a retrain after the fact; it cannot stop one arriving under a commit
message that describes something else. See *Known limitations*.

Hypothesis testing is the project's organising principle: 15 families, 56 registered
hypotheses, Bonferroni-corrected bars (`α = 0.05 / family_size`), moving-block bootstrap
confidence intervals, McNemar tests for paired classification, mandatory replication on
other currency pairs when a claim clears, and power analysis before spending α (the G10
macro panel was stopped before fitting). **37 of 56 hypotheses are recorded as DROP.**

### The headline result

`src/calendar_volatility.py` — a GARCH(1,1) × six-weekday-multiplier model, ten parameters,
numpy only — beats the production 5-seed multi-task LSTM ensemble on **both** blocks of the
identical row set: validation MAE 0.16213 vs 0.18919, test 0.19152 vs 0.21777, higher R² on
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
- **The calendar model (Section 22.6) now CLEARS its paired test, with two caveats standing.**
  The paired bootstrap has been executed (`python -m src.calendar_paired_bootstrap`): on the
  out-of-sample test block it beats the frozen 5-seed ensemble by ΔMAE **+0.02624**, CI95
  **[+0.02232, +0.02998]** — run twice, on row sets 45 bars apart, clearing both times.
  What that does *not* fix, and what stays in the registry row:
  its GARCH is a numpy variance-targeting fit rather than the `arch` MLE used elsewhere, and
  it was built and scored in one pass with no pre-registration preceding measurement. A
  cleared interval does not retire either.
- **The checksum guard fired on the 2026-08-08 retrain and was then re-baselined.** Nineteen
  protected artifacts moved; all nineteen were caught. They were re-baselined on 2026-08-11
  rather than left red, because a permanently failing guard has no power to detect the *next*
  change. Every old→new digest is preserved in
  `tests/fixtures/PROTECTED_SET_REBASELINE_2026-08-11.json`. The residual limitation is real
  and stated in §7.1 of the report: the registered volatility figures were earned against the
  pre-retrain artifacts, so the frozen ensemble is re-scored separately rather than reusing
  its logged number.
- **The one-shot test block was re-scored, repeatedly, and this is the whole record.**
  The production methodology reserves the test block `[80%:100%]` for a **single** final
  report; it is never a search knob and never a criterion. That rule was not honoured for
  `models/volatility/vol_metrics.json`. Disclosed here rather than corrected away: **no
  number below has been deleted or reverted.**

  The specific event that prompted this entry — commit `f2645a0`, 2026-08-15, titled
  *"Refactor code structure for improved readability and maintainability"* — retrained 30
  model artifacts and, as a side effect, re-read and overwrote the test-block figures:

  | Reading | Date | n_test | test ensemble MAE |
  |---|---|---:|---:|
  | before `f2645a0` (written by `3def541`) | 2026-08-08 | 1,712 | 0.218973 |
  | after `f2645a0` | 2026-08-15 | 1,721 | **0.216033** |

  Four things are true about it, and all four are stated rather than only the convenient one:

  1. **The block is reserved for a single final report, and it was scored more than once.**
     That is a methodology violation on its face, independent of what the numbers showed.
  2. **The second scoring was a side effect, not a decision.** `f2645a0` declared itself a
     readability refactor. Nobody chose to spend the block; a retrain rode in under a commit
     message that concealed it. This is the failure mode now blocked at commit time —
     see *Preventing the recurrence* below.
  3. **No verdict anywhere in this project used the second reading.** Every volatility
     verdict in `results/volatility_hypothesis_log.csv` is dated 2026-07-07 → 2026-08-06 and
     every one predates `f2645a0`. The SHIP decision was made on the validation arbiter
     (`results/volatility_seed_ensemble.csv`), whose `validation_decision` block —
     `mt_ensemble_mae` 0.18594, CI, `cleared_bar` — is **byte-unchanged through every one of
     these rewrites**. The contamination touched a reported figure, never a decision.
  4. **The difference carries no information.** 0.218973 → 0.216033 is a move of **0.00294**.
     The measured run-to-run spread of this same ensemble, at identical code, data and seeds,
     is **0.00536** (§8.1 of `DATA.md`). The change is smaller than the noise floor of the
     instrument that produced it. It is not a better measurement or a worse one; it is the
     same measurement drawn again.

  **The fuller record, which is worse than the event that prompted this entry.** Auditing
  the file's whole history for this disclosure showed the re-scoring was not a one-off. The
  test block has been re-read and overwritten on **nine** separate commits:

  | Date | Commit | n_test | test ensemble MAE |
  |---|---|---:|---:|
  | 2026-07-07 | `0ece63c` *(file created)* | 1,712 | 0.218756 |
  | 2026-07-10 | `21449dd` | 1,712 | 0.217550 |
  | 2026-07-17 | `994208d` | 1,712 | 0.218682 |
  | 2026-07-20 | `506a742` | 1,712 | 0.217309 |
  | 2026-07-25 | `b30599f` | 1,712 | 0.216235 |
  | 2026-08-05 | `7327e94` | 1,712 | 0.219248 |
  | 2026-08-08 | `a73344e` | 1,712 | 0.215174 |
  | 2026-08-08 | `3def541` | 1,712 | 0.218973 |
  | 2026-08-15 | `f2645a0` | 1,721 | 0.216033 |

  Point 4 holds across the entire table and holds harder: the full spread of all nine
  readings is **0.004074** (0.215174 – 0.219248), still **inside** the 0.00536 run-to-run
  noise of a single re-run. Nine re-scorings of the one-shot block produced no signal
  whatsoever — which is the strongest available evidence that nothing was learned from
  spending it, and no defence at all of having spent it. Point 3 also holds across the whole
  table: `mt_ensemble_mae` = 0.18594 in every one of the nine versions, so no verdict moved.

  The honest summary: a guard-rail was broken repeatedly and silently for five weeks, the
  breakage was invisible because each retrain arrived under a commit message describing
  something else, and it happened to cost nothing because the quantity being re-read was
  noise-dominated. The last clause is luck, not process, and it is not offered as mitigation.

- **Preventing the recurrence.** The three commits titled *"Refactor code structure for
  improved readability and maintainability"* that each carried a production retrain are the
  root cause behind both re-baselines and the spent test block above. The checksum fixtures
  catch a moved artifact, but only afterwards, and only if someone runs the suite and reads
  the failure. `.githooks/commit-msg` now refuses any commit that stages a path under
  `models/` unless the message carries an explicit `RETRAIN:` declaration, moving detection
  to the moment the commit is written. Enable with `git config core.hooksPath .githooks`;
  26 tests in `tests/test_retrain_commit_guard.py` cover it, including end-to-end proof that
  `git commit` is actually refused. `--no-verify` still bypasses it — this stops the
  accident, not a determined author, which is why the checksum fixtures stay.

- **`mit-deep-learning-book-pdf-master/`** is a third-party reference text included in the
  repository. It is not project code and is not used by any module.
