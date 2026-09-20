# Pre-registration — META-LABELING: does sub-model disagreement flag reliable direction calls?

**Family:** `metalabel` (new). **Registry:** `results/metalabel_hypothesis_log.csv`.
**Status at the time of writing: NO CELL HAS BEEN EVALUATED. NO OUTCOME HAS BEEN JOINED.
ZERO COMPARISONS HAVE BEEN SPENT, AND — AS SECTION 8 SHOWS — NONE CAN BE.**

This document is committed BEFORE `src/metalabel.py` is written. The commit that
adds this file contains this protocol and the three registry rows and nothing
else; the commit that later adds code and results names this document's commit
hash. Same discipline as `results/density/PRE_REGISTRATION.md`.

Nothing in this file may be revised once committed. A deviation is recorded as a
deviation in the results commit and in the registry `notes`, never by editing the
text above it.

---

## 1. The question — and what it is NOT

**Does disagreement between sub-models carry information about WHEN a direction
call is reliable?** This is meta-labeling / stacking. The question is NOT "predict
direction better". It is: can the subset of calls that are more reliable than the
base rate be identified **ex ante**, from a signal already present in the served
response?

Three signals of that kind exist in this project, and each is one registered
condition (Section 3). The estimand for every cell is the same:

    delta = accuracy(direction call | signal says "reliable")
          - accuracy(direction call | signal says "unreliable")

on the **validation slice only**, with the bar and the power analysis fixed here
before any outcome is looked at.

## 2. Slices — fixed now, test blocks NEVER indexed

Two frequencies, two row sets, two split conventions. They are not interchangeable
and neither is the same as any other family's row set (density, for instance,
drops two edge rows and has n = 8603, not 8605).

**Daily** — the euro-era engineered matrix, `src/ablation.py::_canonical_split`
on `config.json` (`train_fraction 0.80`, `val_fraction 0.10`):

| block | rows | n | dates |
|---|---|---:|---|
| train | `[0 : 6023]` | 6,023 | 1999-01-04 → 2018-05-01 |
| **validation — the arbiter** | **`[6023 : 6884]`** | **861** | **2018-05-02 → 2021-02-01** |
| test | `[6884 : 8605]` | 1,721 | **NOT INDEXED. SPENT.** |

Arithmetic: `n = 8605`; `int(8605 × 0.70) = 6023`; `int(8605 × 0.80) = 6884`.

**H1** — `results/pooled_h1/EURUSD_h1.csv` (70,000 raw bars → 69,349 labelled
after the 200-bar warm-up and the 451 exact-zero-return drops),
`src/h1_direction_model.py::split_purge_embargo` with `TRAIN_FRAC 0.70`,
`VAL_FRAC 0.85`, purge 1, embargo 24:

| block | rows | n | dates (UTC) |
|---|---|---:|---|
| train | `[0 : 48544]`, last row purged | 48,543 | 2015-05-08 03:00 → 2023-03-15 02:00 |
| **validation — the arbiter** | **`[48544 : 58946]`, first 24 embargoed** | **10,378** | **2023-03-16 04:00 → 2024-11-19 04:00** |
| test | `[58946 : 69349]` | 10,403 | **NOT INDEXED. SPENT** (read once, 2026-07-30, `H_dir.1 TESTBLOCK_CONFIRMATION`). |

Arithmetic: `int(69349 × 0.70) = 48544`; `int(69349 × 0.85) = 58946`. Both
reproduce the registered `h1_direction_hypothesis_log.csv` rows exactly
(`n_train 48543`, `n_val 10378`).

A unit test in `tests/test_metalabel.py` asserts that no code path in
`src/metalabel.py` constructs an index ≥ `val_end` on either frequency.

## 3. The three conditions — VERBATIM from the registration brief

> **C0 — daily variant disagreement (expected to be underpowered; run it for the
> MDE, not for a verdict).** `baseline` and `with_macro` differ only in the 4 FRED
> macro columns, so "they disagree" literally means "the macro features push
> against the price signal". Condition: accuracy of the consensus direction call
> when `variant_agreement == False` vs `== True`, on the daily validation slice.
> Report the MDE prominently. Note in the report how many years of daily data
> would be needed to resolve a 2-3pp effect at the registered alpha.

> **C1 — H1 cross-model direction disagreement.** On the H1 validation slice,
> compare the `h1_direction` GBM's call against the sign of the H1→daily ensemble
> members (§3.4: XGBoost / RandomForest / SVM / LSTM — confirm from the artifacts
> what is actually available and what each one emits; the regressors give a
> direction via the sign of predicted return). Condition: accuracy of the primary
> `h1_direction` call, split by whether the other models agree with it.
> Pre-declare the agreement rule (e.g. unanimous vs split) in the pre-registration
> document and do not change it afterwards.

> **C2 — H1 probability margin.** Bucket `|p - 0.5|` from the `h1_direction` GBM
> into terciles fixed on the TRAIN slice only (never on validation). Condition:
> does accuracy increase monotonically across margin buckets? This is the most
> natural meta-label and it is directly relevant to §4.2.2, where calibration was
> evaluated but not adopted — reference that section's findings rather than
> re-deriving them.

No condition is added. No condition is re-scoped (Section 4.2 records one
proposal, explicitly NOT run).

## 4. Rules, fixed now

### 4.1 C0 — the signal and the sentinel

`variant_agreement` does not exist historically. It is computed at serving time
(`src/inference.py`, `len(set(consensus_directions.values())) == 1`) and lives only
in `results/prediction_log.csv` (58 rows at registration). On the validation
slice it must be **reconstructed**: both variants' consensus directions recomputed
per row, `variant_agreement := (baseline_direction == with_macro_direction)`.

**The sentinel.** The consensus direction takes one of three strings: `UP`, `DOWN`,
`MIXED / LOW CONFIDENCE` (`CONFIDENCE_THRESHOLD = 0.52`). A MIXED row is **not a
trade** — `src/paper_trading.py` books it flat. It is therefore **excluded from
both cells**, never counted as wrong. `variant_agreement` compares *strings*, so
`True` can mean "both said MIXED" and `False` can mean "one went MIXED". In the
forward log, 15 of the 20 disagreements are of that second kind and only 5 are
genuinely opposite calls. The scored row set is `{rows where the scored variant's
consensus ∈ {UP, DOWN}}`; the cell is `variant_agreement` on that row set.

**Which variant is scored:** `with_macro` (`pred_direction`, the production
lineage). Fixed here; not chosen after seeing anything.

### 4.2 C1 — INFEASIBLE-AS-SPECIFIED. Alpha unspent. NOT re-scoped.

The "H1→daily ensemble" of §3.4 (`models/h1_{xgb,rf,svm}_regressor.pkl`,
`models/h1_lstm.keras`) does not predict the next H1 bar. `build_h1_datasets`
aggregates H1 bars into **daily** statistics and predicts the **next-DAY** percent
return on a 2,504-row daily index. The `h1_direction` GBM predicts the **next
hour**. They never make a call about the same event; "they disagree" has no
referent.

Independently, the ensemble trains on `[0:80%]` of its own row set = through
**2024-10-09**. Of the 437 ensemble days inside the H1 validation window,
**408 (93.4%) are in-sample for the ensemble**; 29 are out-of-sample. A meta-label
built on that would look informative because the ensemble memorised the window.

Independently again, all five ensemble artifacts were among the 18 moved by the
undeclared 2026-09-11 retrain (declared post hoc in `4cc7920`), so at registration
they carried no recorded provenance. Even without the first two problems there
was no object to score.

`ti_lstm_h1` is also next-day ("NOT next-hour prediction", its own docstring).
There is **exactly one** persisted model in this repository that emits an H1
next-bar direction, and it is the one under test. C1 has no second party.

**Proposal for FUTURE registration — NOT RUN HERE.** The one same-target pair
that exists is the `H_dir.1` GBM against the `H_dir.2` LSTM (same rows, same
next-bar label, both reconstructable on `[0:70%]`). With exactly two binary
models the agreement rule is **forced, not chosen**: "unanimous vs split" does not
exist for n = 2 — *agree ⇔ both predict the same direction*. That rule is stated
here before any number is quoted for it. Its cell sizes are already knowable from
the registered `H_dir.2` row without any refit: with two binary models, opposite
calls are exactly the McNemar discordant pairs, so
`n_disagree = b + c = 1734 + 1851 = 3,585` (34.54%), `n_agree = 6,793`. Its
80%-power MDE at this family's bar is **3.83 pp i.i.d. / 3.89 pp design-effect
corrected** — above the 3 pp floor. It is recorded so that a future registrant
knows in advance that it, too, would be underpowered on this slice. **Running it
now would be HARKing with a step sideways** — a condition selected after seeing
why the original failed. It is not run.

### 4.3 C2 — terciles on TRAIN, monotonicity as the estimand

`|p − 0.5|` from the `h1_direction` GBM. Tercile boundaries `(q33, q67)` are
computed on the **train slice `[0:48543]` only** and applied unchanged to
validation. Validation bucket occupancy is therefore *not* guaranteed equal; the
assumed 3,459 / 3,459 / 3,460 in Section 6 holds only under identical train/val
margin distributions and is labelled ASSUMED.

**Estimand:** the top-vs-bottom tercile accuracy delta. Monotonicity across all
three is reported descriptively; the bar is applied to top-minus-bottom only.
One comparison, not three.

**Prior, from §4.2.2, cited not re-derived.** On the daily classifier, raw
`predict_proba` scored Brier **0.25063** against **0.25013** for a constant at the
training base rate — *worse than a constant* — and calibration was evaluated and
rejected because sigmoid scaling collapsed the range to `[0.461, 0.513]` and would
have pinned the consensus at permanent MIXED. Those numbers are documented in
`ARCHITECTURE_DOCS.md`, not traceable to a results CSV, and are cited as the
documented finding. C2 is the H1 twin of that question and the prior is that the
margin carries approximately nothing.

### 4.4 The NY-session tension — flagged, NOT acted on

`results/h1_direction_final/report.txt` and `testblock_by_ny_session.csv` show
the H1 direction edge concentrating in NY hours (NY-only +7.98 pp, London +4.92 pp,
Asia −0.26 pp). That is a **descriptive breakdown on the SPENT test block**.
Restricting C1/C2 to the NY session would define a subpopulation from a
test-block-derived observation. **Default: no restriction.** The decision is the
owner's; it was not taken.

## 5. Family size and the bar — fixed now, with the arithmetic

The registration brief instructs: count the current family from
`results/feature_hypothesis_log.csv`, add this family's conditions, use
`alpha = 0.05 / new_family_size`.

    feature_hypothesis_log.csv rows   :  9
    conditions registered here        :  3   (C0, C1, C2)
    family_size                       : 12
    alpha = 0.05 / 12                 :  0.004167   (two-sided → 99.58% CI)
    z_{alpha/2}                       :  2.8653

**Departure from repo convention, stated.** Every other family here is
self-contained (`density`: "does not touch `feature_hypothesis_log.csv`";
`h1_direction`: "NEW and independent"). That convention would give
`family_size = 3`, `alpha = 0.016667`. The brief's instruction is followed because
it is the owner's and because it is the **stricter** bar; the convention's number is
recorded so the departure is visible. Note also that `family_size` includes the new
hypotheses (`src/ablation.py:322`, `len(already | set(features))`), so the
`0.05/9 ≈ 0.00556` printed in CLAUDE.md is the bar *as of the last spent* feature
hypothesis, not the bar a new one faces (that is 0.05/10).

**C1 is infeasible and the family is HELD at 12, not dropped to 11.** Dropping it
would loosen alpha to 0.004545. Declaring a condition unrunnable must never make
the remaining conditions easier to pass — that is a perverse incentive. The MDE
difference is under 1%; holding costs nothing and removes the doubt. Stated here,
in advance.

**Just-significant vs 80% power.** H_dir.7's precedent reports both. Here the
**80%-power MDE governs**. Just-significant (`z_α · SE`) is ~50% power by
construction: if the true effect sits exactly at the economic floor, detection is a
coin flip, and whatever *is* declared significant is systematically inflated
(Type M). For a system that would size capital on the estimate, that is a route
to over-sizing. Just-significant is reported alongside and never adjudicated on.

## 6. POWER — computed before any outcome was joined

Unpaired two-proportion MDE (the cells are disjoint row subsets, so this is not
McNemar's paired SE):

    MDE = (z_{alpha/2} + z_beta) · sqrt( p(1−p) · (1/n_1 + 1/n_0) ),   p = 0.5
    z_{alpha/2} = 2.8653  (alpha = 0.004167),   z_beta = 0.8416  (80% power)

**Economic floor: 3 pp over base rate.** Any cell whose 80%-power MDE exceeds it
is reported `UNDERPOWERED — NO DECISION` and its accuracy is never reported as a
finding.

**Design effect.** The i.i.d. formula is optimistic under block dependence. Rather
than assume a factor, it was measured on this repository's own seven registered
H1 runs (block-24 CI half-width ÷ i.i.d. CI half-width): SE ratio **0.942–1.119,
median 1.017** → design effect **0.89–1.25, median 1.035**. It is small, and the
validation-slice ACF says why: **direction has no autocorrelation** (Ljung–Box on
sign at lag 24, p = 0.115; ACF(sign, 1) = −0.035), while `|return|` clusters
strongly (ACF 0.216 at lag 1, **0.188 at lag 24** — the diurnal cycle that
justifies `block_len = 24`). Blocking barely widens a direction-accuracy interval.
Both the i.i.d. and the median-deff numbers are reported.

| cell | n₁ | n₀ | source of n | just-sig | **80% power** | 80% × deff | required per cell for 3 pp |
|---|---:|---:|---|---:|---:|---:|---:|
| **C0** daily disagree / agree | 353 | 508 | ASSUMED — forward-log rate 20/49 = 40.8% | 9.93 pp | **12.84 pp** | 13.07 pp | 7,634 (has 353 → **5%**) |
| **C1** as registered | — | — | **no n exists** — see 4.2 | — | — | — | — |
| C1 *proposal* (4.2) | 3,585 | 6,793 | MEASURED — `H_dir.2` McNemar b+c | 2.96 pp | 3.83 pp | 3.89 pp | 7,634 (has 3,585 → 47%) |
| **C2** top / bottom tercile | 3,459 | 3,459 | ASSUMED — equal thirds | 3.44 pp | **4.46 pp** | 4.53 pp | 7,634 (has 3,459 → **45%**) |

`n required per cell = 0.5 · ((z_α + z_β) / 0.03)² = 7,634` at equal cells.

**Every cell exceeds the 3 pp floor at 80% power. Every cell is UNDERPOWERED —
NO DECISION.** This is known before a single outcome is joined, and it is the
reason no outcome is joined.

**Why the C0 and C2 n are ASSUMED, not measured.** Both depend on model output on
the validation slice (the reconstructed consensus for C0, the GBM margin
distribution for C2), and the only honest model for either is a `[0:70%]`
reconstruction (Section 8), which is on HOLD. That the cell sizes cannot be
known without the refit is itself part of the NO DECISION.

### 6.1 C0 — how much daily data would ever resolve it

    current : n_val = 861, MDE = 12.84 pp
    MDE ∝ 1/sqrt(n)  →  n_needed = 861 · (12.84 / target)²

| target | factor | n_val needed | years in the arbiter block | total history at 10% val | euro-era data that exists |
|---|---:|---:|---:|---:|---|
| **3 pp** | 18.3× | 15,779 | **60 yrs** | **605 yrs** | 27 yrs → **22× short** |
| 2 pp | 41.2× | 35,503 | 136 yrs | 1,360 yrs | 27 yrs → 50× short |

C0 is not underpowered by an amount more data could fix at this frequency. 605
years of daily EUR/USD does not and will not exist. **C0 is structurally
unanswerable on daily bars and should not be re-asked.** That is the finding the
brief asked for.

## 7. EXPECTED OUTCOME, STATED IN ADVANCE

Two layers, both predicted:

1. **No verdict can be reached** — Section 6 already shows every cell
   underpowered at the registered bar. This is the primary expected outcome and
   it is a *result*, not a failure. Rule 6 of the brief: a correctly-powered,
   correctly-registered negative is worth exactly as much as a positive.
2. **If a cell were somehow powered, the expected effect is ≈ 0.** Daily EUR/USD
   direction is near-efficient (ROC-AUC ≈ 0.50, §4.2.1); the daily classifier's
   probability is Brier-worse than a constant (§4.2.2); the H1 GBM's documented
   edge is +2.3 pp on validation. A meta-label with a true effect above 4.5 pp on
   top of that is not plausible. The best-powered cell here needs exactly that.

This paragraph exists so that neither outcome can later be dressed as a surprise
or reframed as "inconclusive, needs one more variant".

## 8. The frozen-artifact constraint — unsatisfiable as written; the refit path; HOLD

The brief requires "frozen artifacts and batch inference only". For every cell
that is unsatisfiable, for reasons established before registration:

- `models/h1_direction/` holds the **served full-history model**
  (`"validated_out_of_sample": false`, trained through 2026-07-28, and — per
  `report.txt` — trained at 18:11 UTC on 2026-07-30, **seven hours after** the
  test block was first read at 10:52). Scoring it on validation is in-sample.
- The daily GBM trains on `[0:80%]` (`_train_pipeline.py:229`), which **contains
  the whole validation slice**; the daily LSTM early-stops on it. Neither head is
  out-of-sample on `[6023:6884]`.
- The model that produced the `H_dir.1` registry rows was fit in memory and
  discarded — `h1_direction_final.py` is forbidden from writing to `models/`. Its
  anchor is a **behavioural reproduction gate**, not a digest.

The repository's own remedy is **refit on `[0:70%]` with frozen hyperparameters
under a reproduction gate** — `src/ablation.py` exists for exactly this, and
`H_dir.7` did it (refit validation accuracy must equal **0.527462 ± 0.003**).
That is a `.fit()`, so it breaks the constraint's letter while honouring its
intent (zero researcher degrees of freedom).

**Status: HOLD — NO GO.** The owner's ruling, recorded verbatim in substance: do
not refit while the tree carries undeclared `models/` modifications (cleared in
`4cc7920`, `0419645`); once cleared and on explicit go, fit on `[0:70%]`, fixed
hyperparameters, artifacts written **outside `models/`** at
`results/metalabel/frozen/`, **ONE attempt**, and if the gate misses on the first
run → INFEASIBLE, no re-seeding. If ever run, this document's results commit
records: achieved gate value, seed, library versions, SHA-256 of every artifact.
The go was not given, and Section 6 says it should not be: the best cell reaches
3.89 pp against a 3 pp floor.

## 9. Bootstrap — conventions and the refusal

- **H1:** paired moving-block (circular) bootstrap, **`block_len = 24` bars = one
  trading day**, `n_boot = 2000`, `random_state = 42`, percentile CI at
  `(alpha/2, 1 − alpha/2)` — `src/h1_direction_model.py::_block_bootstrap_delta`
  idiom, with `alpha` **passed explicitly** (that helper's default is the stale
  `FAMILY_ALPHA = 0.025`, 3.5× looser than the H_dir registry's 0.007143).
- **Daily:** `block_len = 5`, the convention `src/density_model.py` and
  `src/calendar_paired_bootstrap.py` use. (Both attribute it to "the volatility
  family"; `src/volatility.py::bootstrap_delta` is in fact plain i.i.d. The
  precedent cited is density/calendar, not volatility.)
- **Refusal:** if `n <= block_len`, return `(nan, nan)` rather than clamping the
  block — exactly `src/vol_scaled_backtest.py::bootstrap_delta_sharpe`. A block
  as long as the sample is a cyclic rotation; every resample gives the identical
  statistic and the "CI" collapses to a falsely confident point.

## 10. Join hazards — guarded by construction

1. **Sentinel.** `MIXED / LOW CONFIDENCE` (daily) and `MIXED / TIE` (H1 ensemble)
   are not directions. Rows carrying them are excluded, never scored wrong.
   `direction == "UP"` silently buckets both as DOWN and is forbidden.
2. **Timezone.** Daily rows are tz-naive server dates; H1 bars are tz-aware UTC.
   Join by `tz_localize(None)`, never `tz_convert`. The H1 feed offset
   (`results/h1_feed_offset.json`, currently +2 h) is inferred at runtime and is
   not a constant.
3. **Off-by-one.** Targets are `shift(−1)`; the forecast date is weekend-aware
   (Fri/Sat → Monday). The reconstruction of `as_of_close` from the price file
   must agree with the logged value to **< 0.5 pip (5e-5)** or the join is
   rejected.
4. **Two H1 files.** `results/eurusd_h1.csv` (rolling cache, 60,136 rows, window
   slides) vs `results/pooled_h1/EURUSD_h1.csv` (frozen, 70,000). The direction
   model reads the frozen one. Never mix.

## 11. Stopping rules

- Any cell whose 80%-power MDE > 3 pp: **stop — UNDERPOWERED — NO DECISION.**
  No accuracy number is reported for it as a finding. (Every cell trips this.)
- Any cell with no scorable second party: **stop — INFEASIBLE-AS-SPECIFIED.**
  (C1 trips this.)
- Any code path that constructs an index ≥ `val_end`: **abort**, test fails.
- Reproduction gate miss on the single permitted attempt: **INFEASIBLE**, no
  re-seed.
- No threshold, bucket boundary, agreement rule or cell definition is changed
  after this commit. Additional conditions are proposals for future registration
  only (4.2).

## 12. What this family spends — ZERO comparisons

A registered-but-unevaluated family spends nothing. Multiplicity correction counts
comparisons **made**, not hypotheses **considered**. Charging alpha for a
correctly declared underpowered refusal would create the incentive to *run* weak
tests rather than decline them — the opposite of what the correction is for.

- MDEs are computed at **12** (the registered bar). The "12" is the bar this
  family *would have faced*.
- The registry records **"zero comparisons spent"** on every row.
- **The standing bar for future feature claims remains `0.05/9`** — this family
  does not tighten it, because it made no comparison to tighten it with.

## 13. Test blocks

Daily `[6884:8605]` and H1 `[58946:69349]` are **not indexed by this family**, on
any path, for any purpose. Not for a verdict, not for a diagnostic, not for a
power calculation. `tests/test_metalabel.py` asserts it. There is no "one-shot
final report" for this family because there is no verdict to confirm.

---

*Written before `src/metalabel.py` existed. The commit adding this file contains
this document and the three registry rows and nothing else.*
