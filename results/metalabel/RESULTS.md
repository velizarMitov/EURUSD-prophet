# Metalabel family — results (C0, C1, C2)

**Nothing clears. Nothing was evaluated. Every cell is UNDERPOWERED — NO DECISION at the registered bar, C1 is INFEASIBLE-AS-SPECIFIED, and zero comparisons were spent.**

**Pre-registration:** [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md), committed at
**`6082fe5`** *before `src/metalabel.py` existed*. That commit contains the
protocol and the three registry rows and nothing else. This document is added
with the code, in the commit that follows.

**This is the outcome the pre-registration predicted, in writing, in advance**
(§7). It is a result, not a failure. A correctly-registered, correctly-powered
negative is worth exactly as much as a positive; here the power analysis
delivered the answer before any outcome needed to be looked at.

## What was tested

The question: does disagreement between sub-models carry information about
**when** a direction call is reliable? Three conditions, verbatim from the brief,
registered against the validation slices only — daily `[6023:6884]` (n = 861,
2018-05-02 → 2021-02-01) and H1 `[48544:58946]` after purge and embargo
(n = 10,378, 2023-03-16 04:00 → 2024-11-19 04:00 UTC). Test blocks were not
indexed on any path; `tests/test_metalabel.py` asserts it and `TestBlockGuard`
raises on the attempt.

Bar: `family_size = 9 + 3 = 12`, `alpha = 0.05/12 = 0.004167`, two-sided. Held
at 12 although C1 is infeasible — declaring a condition unrunnable must not
loosen the bar for the rest. The 80%-power MDE governs; just-significant is
reported alongside and never adjudicated on. Economic floor: 3 pp.

## What came back — the power table

Computed before any outcome was joined. Design effect **measured** on this
repository's own seven registered H1 runs (block-24 vs i.i.d. interval widths):
median 1.035, max 1.253. It is small because direction has no autocorrelation
(Ljung–Box on sign, lag 24, p = 0.115); the clustering lives in `|return|`.

| cell | n₁ | n₀ | n source | just-sig | **80% power** | × deff | required per cell | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **C0** daily disagree / agree | 353 | 508 | assumed (forward-log rate 40.8%) | 9.93 pp | **12.84 pp** | 13.06 pp | 7,634 — has **5%** | **UNDERPOWERED — NO DECISION** |
| **C1** as registered | — | — | **no n exists** | — | — | — | — | **INFEASIBLE-AS-SPECIFIED** |
| C1 *proposal* (not run) | 3,585 | 6,793 | measured (H_dir.2 McNemar b+c) | 2.96 pp | 3.83 pp | 3.89 pp | 7,634 — has 47% | would be UNDERPOWERED |
| **C2** top / bottom tercile | 3,459 | 3,459 | assumed (equal thirds) | 3.44 pp | **4.46 pp** | 4.53 pp | 7,634 — has **45%** | **UNDERPOWERED — NO DECISION** |

`results/metalabel/power_analysis.csv` and `run_meta.json` are the machine
copies; `python -m src.metalabel` regenerates them and evaluates nothing.

Two notes from the pre-commit adversarial check, neither changing a verdict:

- **Design-effect provenance.** One of the seven `h1_direction` registry rows
  whose interval widths feed the design effect is the `TESTBLOCK_CONFIRMATION`
  row. That is a published summary statistic read from a CSV, not an index into
  test-block data — but it is said here rather than left implicit. Excluding it
  gives a median deff of **0.990** (six validation rows); including it gives
  1.035. The larger, more conservative value was used. No MDE moves below the
  floor either way.
- **Rounding.** `PRE_REGISTRATION.md` §6 quotes C0's deff-corrected MDE as
  13.07 pp (deff rounded to 1.035 before multiplying); the machine output above is
  13.06 pp (deff 1.03467 unrounded). The pre-registration is immutable; the
  0.01 pp difference is recorded here, not edited there.

## C0 — structurally unanswerable, not merely underpowered

The brief asked how many years of daily data would resolve a 2–3 pp effect at
the registered alpha. MDE scales as `1/√n`:

| target | factor | n_val needed | years in the arbiter block | total history at 10% validation | euro-era data that exists |
|---|---:|---:|---:|---:|---|
| **3 pp** | 18.3× | 15,779 | **60 yrs** | **605 yrs** | 27 yrs — **22× short** |
| 2 pp | 41.2× | 35,503 | 136 yrs | 1,360 yrs | 27 yrs — 50× short |

605 years of daily EUR/USD does not exist and will not. **C0 cannot be answered
at daily frequency and should not be re-asked.** That is the finding.

Two further facts about C0's signal, recorded because they would have bitten
anyone who ran it: `variant_agreement` does not exist historically (it is
computed at serving time; 58 forward rows) and would have had to be
reconstructed; and it compares direction *strings*, so `MIXED / LOW CONFIDENCE`
counts as a direction — in the forward log 15 of the 20 disagreements are one
variant going MIXED, not opposite calls. MIXED rows are not trades and would
have been excluded from both cells, shrinking C0 further.

## C1 — infeasible as specified

The "H1→daily ensemble" of §3.4 predicts the **next day**, not the next hour
(`build_h1_datasets` aggregates H1 bars into daily statistics; 2,504-row daily
index). The `h1_direction` GBM predicts the **next hour**. They never call the
same event, so "they disagree" has no referent. Independently, the ensemble
trains through 2024-10-09, so **93.4%** (408 of 437) of its days inside the H1
validation window are in-sample for it. Independently again, all five ensemble
artifacts were among the 18 moved by the undeclared 2026-09-11 retrain and
carried no recorded provenance at registration.

`ti_lstm_h1` is also next-day. **Exactly one persisted model in this repository
emits an H1 next-bar direction, and it is the one under test.** There is no
second party.

**Not re-scoped.** The one same-target pair that exists (H_dir.1 GBM vs H_dir.2
LSTM) is recorded in `PRE_REGISTRATION.md` §4.2 as a **proposal for future
registration**, with its agreement rule stated in advance (forced at n = 2:
agree ⇔ same direction) and its cell sizes read from the registered McNemar
counts (3,585 / 6,793). Its MDE is 3.83 pp — also above the floor. Running it
now would be a condition chosen after seeing why the original failed.

## C2 — the best cell, still 45% of the sample it needs

Terciles of `|p − 0.5|` fitted on train, top-minus-bottom accuracy as the single
estimand. At 3,459 per bucket the 80%-power MDE is 4.46 pp against a 3 pp floor;
3 pp needs 7,634 per cell. The prior from §4.2.2 — the daily classifier's raw
`predict_proba` is Brier-**worse** than a constant (0.25063 vs 0.25013) — says
the margin carries approximately nothing, and a true meta-label effect above
4.5 pp in a market with ROC-AUC ≈ 0.50 is not plausible. Even executed perfectly,
C2 was near-certain to be uninformative.

The NY-session restriction was flagged and **not** taken: the NY breakdown in
`results/h1_direction_final/` is descriptive on the spent test block.

## What was NOT done, and why

- **No model was loaded and nothing was fitted.** The served `models/h1_direction/`
  is full-history and in-sample on validation (`"validated_out_of_sample": false`;
  trained 2026-07-30 18:11 UTC, seven hours *after* the test block was first
  read). The daily GBM trains on `[0:80%]`, which contains the whole validation
  slice. The only honest scorer is a `[0:70%]` reconstruction under the
  reproduction gate (`0.527462 ± 0.003`) — the repository's own idiom in
  `src/ablation.py` and `H_dir.7`. That is a `.fit()`, and it is **HOLD / NO GO**
  by the owner's ruling. The path is recorded in `PRE_REGISTRATION.md` §8 (one
  attempt, artifacts outside `models/`, no re-seeding) should a go ever be given.
  The power table says it should not be.
- **No test block was indexed.** Including for the join sanity check: the forward
  log's `as_of` dates (2026) fall inside the daily test block, so
  `reconstruct_as_of_close` against `prediction_log.csv` is refused by the guard.
  The check is implemented and verified to sub-pip tolerance on the validation
  slice instead (`test_reconstruct_as_of_close_agrees_sub_pip_on_the_validation_slice`).
- **No cell n was measured for C0 or C2.** Both depend on model output on
  validation; without the refit they are assumed and labelled so. That the cell
  sizes cannot be known without the refit is part of the NO DECISION.

## What would be required to resolve the underpowered cells

- **C0:** nothing achievable. 605 years of daily data. Do not re-ask at daily
  frequency.
- **C2:** 7,634 rows per tercile ≈ 22,900 validation bars, i.e. an H1 validation
  slice ~2.2× the current one. At the current 15% validation fraction that means
  a ~153,000-bar cache ≈ **25 years** of hourly EUR/USD; the frozen
  `pooled_h1/EURUSD_h1.csv` holds 11.25. Alternatively, forward data at the
  slice's own ~6,170 scored bars per year: **3.7 years** of live H1 forecasts to
  build a fresh 22,900-bar slice, or 2.0 years if pooled onto the existing
  validation slice (which mixes a historical arbiter with a forward one and
  would need its own registration). Either way the forward ledger does not yet
  log `h1_direction` probabilities per bar in a form that would support it.
- **C1 proposal:** same order as C2 (needs 7,634 in the smaller cell; has 3,585).
  Registration first; it does not inherit this family's alpha.

## What this family spends

**Zero comparisons.** Multiplicity correction counts comparisons made, not
hypotheses considered. The registry records it on every row. **The standing bar
for future feature claims remains `0.05/9`.** This family does not tighten it.

---

*Code: `src/metalabel.py` (31 tests in `tests/test_metalabel.py`). Registry:
`results/metalabel_hypothesis_log.csv`. Nothing under `models/`, `src/inference.py`,
`src/paper_trading.py` or any other family's log was touched.*
