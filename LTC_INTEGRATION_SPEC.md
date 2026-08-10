# Work specification — integrating the LTC / CfC + LIF model into the project

**Family:** Idea 3 — liquid time-constant dynamics in learned business time with a spiking
abstention readout.
**Registry:** `results/ltc_hypothesis_log.csv` (2 hypotheses, α = 0.025 each, both PENDING).
**Modules:** `src/ltc_spiking_arch.py`, `src/ltc_data.py`, `src/ltc_experiment.py`.

This document is the assignment. It is staged, and **Stage 4 is gated** — the model does
not enter the serving path unless Stages 1–3 produce a cleared verdict. That gate is the
project's standing rule (`CLAUDE.md` → Production Methodology), not a preference.

---

## Stage 0 — Ship the exam first (do this before anything else)

The LTC work is research beyond the exam project. Mixing an unproven model into the repo
while the submission is being graded risks breaking a working deliverable for zero marks.

```
git add -A && git commit -m "docs: second exam submission" && git push origin main
```

**Acceptance:** the GitHub link contains Section 22, the changelog and the
self-assessment, the repo is public, and the exam is submitted. Only then continue.

---

## Stage 1 — Make it run

The JAX path in `src/ltc_experiment.py` has **never been executed**. Everything in
`src/ltc_data.py` has been run on real H1 data and its numbers are measured.

**Tasks**

1. Create `requirements-ltc.txt` with `jax`, `jaxlib`, `equinox`, `optax`.
   **Do not touch `requirements.txt`** — the production app must keep installing on a
   machine with no JAX. This mirrors how `requirements-kronos.txt` is already handled.
2. `python -m src.ltc_experiment --dry-run` (2,000 rows, 2 epochs).
3. Fix what breaks. Report tracebacks; do not silently work around them.
4. Run `warp_sanity_check`. On real H1, `dt_max = 74` in units of median bar spacing, and
   the cell composes `exp(-dt/tau)` — an unwarped weekend annihilates the hidden state.
   If `compression_ratio` is not comfortably below 20, re-initialise `BusinessTimeWarp`
   **before** training, or you are testing the optimiser's escape from a dead
   initialisation rather than the subordination hypothesis.
5. Time one epoch at `--seq-len 64` on the full training set. JAX is CPU-only on Windows.

**Acceptance:** dry run completes, warp check healthy, one-epoch wall time known.
**Do not proceed if a full run would take more than a few hours.** Reduce `seq_len` or
`hidden` first — those are free to change *now*, and frozen after Stage 2 begins.

---

## Stage 2 — Train and score

**Tasks**

1. `python -m src.ltc_experiment --timeframe H1 --epochs 30`.
2. Fit on `[0:70%]` only. Score on `validation[70:80]` only. The test block
   `[80%:100%]` is spent and must never be indexed.
3. Produce both pre-registered quantities:
   - **H_ltc.1** — ΔAURC against the GARCH(1,1) × day-of-week benchmark, paired
     moving-block bootstrap (`block_len=24`, `n=2000`).
     Benchmark to beat, already measured: **AURC = 0.03044**, mean CRPS 0.03875, on
     6,000 validation bars (2023-09-14 → 2024-09-02).
   - **H_ltc.2** — partial Spearman(`tau_eff`, `log_tick_rate` | `abs_return_pct`,
     `parkinson_pct`). The partial is the primary. The raw correlation is reported but
     does not gate, because a τ that tracks only volatility scores raw **−0.4966** —
     it would pass the raw test while having learned nothing.

**Hyperparameters are frozen at the end of Stage 1.** Any change to `seq_len`, `hidden`,
`lr` or `coverage_floor` after seeing a validation number converts the arbiter into a
training set and invalidates the run.

**Acceptance:** `results/ltc/hypothesis_scores.json` written, both quantities present,
coverage did not collapse. If the run aborted on the coverage floor, that is a **result**
— the model bought its score with silence. Report it; do not raise the floor and re-run.

---

## Stage 3 — Register the verdict

Replace the two PENDING rows in `results/ltc_hypothesis_log.csv` with the scored values
and today's date. **Both verdicts, including DROPs.** A registry that only records
successes measures nothing.

Then append a dated section to `IMPROVEMENT_LOG.md` in the existing format, stating the
hypothesis, the arbiter, the numbers, the verdict, and what it means.

**Acceptance:** no PENDING rows remain; `IMPROVEMENT_LOG.md` has the entry.

---

## Stage 4 — Integration into serving ⚠ GATED

**Only start this if H_ltc.1 cleared its bar** (bootstrap CI on ΔAURC excludes zero in the
model's favour at α = 0.025). If it did not clear, **stop at Stage 3.** A DROP is a
complete and publishable outcome; the modules stay in `src/` as research and the story is
told in the notebook.

If it cleared:

1. **Artifacts** under `models/ltc/` — never at `models/` root, never mixed with the
   `baseline/` or `with_macro/` variant directories. The model's own scaler, covariate
   column list and `seq_len` are saved beside it, exactly as each variant carries its own.
2. **`src/ltc_serving.py`**, following the `src/h1_direction_serving.py` pattern: load
   once at startup, expose an `ltc_ready` flag, and **degrade gracefully**. If JAX is not
   installed the import must fail softly and set `ltc_ready = False` — a machine without
   JAX must still serve every existing family. This is the same graceful-degradation
   contract as `baseline_ready` / `macro_ready` / `vol_ready`.
3. **Abstention is a first-class API state.** This model is selective; on a silent bar it
   has no forecast. The response must carry `{"status": "abstain"}` and the dashboard must
   render it as *abstain*, not as a neutral or zero prediction. Faking a number on a bar
   the model declined to speak on destroys the entire point of the architecture.
4. **Smoke test** asserting every `models/ltc/` artifact exists, in the style of the
   existing `test_smoke.py` checks.
5. **Feature parity.** If the serving path computes covariates, it imports them from the
   same code the training used — no re-implementation. This is the single-source-of-truth
   contract that the whole project exists to protect.

**Acceptance:** full test suite green; `api.py` starts and serves normally on a machine
with **no JAX installed**; abstention renders correctly.

---

## Stage 5 — Documentation

1. `ARCHITECTURE_DOCS.md` — new subsection under §3, in the existing style: what it is,
   what it consumes, what it emits, how it degrades, where its artifacts live.
2. `CLAUDE.md` — one line in the component map, plus the JAX-optional note.
3. Notebook — a new section covering the theory (Clark 1973 subordination, LTC/CfC,
   surrogate-gradient spiking), the pre-registered design, and the honest verdict.
4. `README.md` — one row in the structure table.

---

## Standing rules for every stage

- Nothing touches `src/inference.py`, `models/` or the serving path before Stage 4.
- Every fitted quantity — scalers, benchmark parameters, warp initialisation — is fitted
  on `[0:70%]` only.
- No centred rolling windows anywhere.
- The forward paper-trading ledger, over **months**, decides production worthiness. A
  cleared validation gate earns a place in the ledger, not real capital.
- Show numbers before narrative. If a result looks good, try to break it first.

---

## Prompts

Paste these into Claude in VS Code one at a time, in order. Do not paste Stage 4's prompt
until Stage 3 has produced a cleared verdict.

### Prompt 1 — Stage 1

> Read `LTC_INTEGRATION_SPEC.md`, then `src/ltc_data.py` and `src/ltc_experiment.py`.
> Execute **Stage 1 only**.
>
> Context you must know: the JAX path in `ltc_experiment.py` has never been executed —
> treat it as a draft. `ltc_data.py` has been run on real H1 data and its numbers are
> measured. On real H1, `dt_max = 74` in units of median bar spacing, so the weekend gap
> is the main structural risk.
>
> Create `requirements-ltc.txt` (do not modify `requirements.txt`), run
> `python -m src.ltc_experiment --dry-run`, fix what breaks, run `warp_sanity_check`, and
> time one epoch at `--seq-len 64` on the full training set.
>
> Report the dry-run output, the warp check, and the wall time. Do not start a full
> training run. Do not proceed to Stage 2.

### Prompt 2 — Stage 2 and 3

> Read `LTC_INTEGRATION_SPEC.md`. Execute **Stages 2 and 3**.
>
> Hyperparameters are now frozen. If you change `seq_len`, `hidden`, `lr` or
> `coverage_floor` after seeing any validation number, the run is invalid — say so rather
> than doing it.
>
> Train on `[0:70]`, score on `validation[70:80]`, never index the test block. The
> benchmark to beat is AURC = 0.03044 (already measured). For H_ltc.2 the primary is the
> **partial** Spearman controlling for `abs_return_pct` and `parkinson_pct` — the raw
> correlation is reported but does not gate, because a τ tracking only volatility scores
> raw −0.4966 and would pass a raw test having learned nothing.
>
> Then replace the two PENDING rows in `results/ltc_hypothesis_log.csv` with the scored
> values, and add a dated entry to `IMPROVEMENT_LOG.md`. Register DROPs as readily as
> KEEPs.
>
> Show me the numbers before any interpretation.

### Prompt 3 — Stage 4 (only if H_ltc.1 cleared)

> Read `LTC_INTEGRATION_SPEC.md` Stage 4, then `src/inference.py` and
> `src/h1_direction_serving.py` to match the existing patterns.
>
> H_ltc.1 cleared its bar. Wire the model into serving under the project's invariants:
> artifacts under `models/ltc/` with their own scaler and covariate list; a new
> `src/ltc_serving.py` with an `ltc_ready` flag that degrades gracefully when JAX is
> absent; abstention as a first-class `{"status": "abstain"}` API state rendered as
> *abstain* in the dashboard, never as a zero or neutral prediction; a smoke test for
> every artifact; and covariates imported from the same code training used, never
> re-implemented.
>
> Acceptance: full suite green, and `api.py` starts and serves every existing family on a
> machine with no JAX installed. Verify that last point explicitly.

### Prompt 4 — Stage 5

> Read `LTC_INTEGRATION_SPEC.md` Stage 5. Document the LTC family: a subsection under
> `ARCHITECTURE_DOCS.md` §3 in the existing style, a line in the `CLAUDE.md` component
> map, a notebook section covering the theory and the honest verdict, and a row in the
> `README.md` structure table. Match the surrounding tone — factual, with limitations
> stated plainly.
