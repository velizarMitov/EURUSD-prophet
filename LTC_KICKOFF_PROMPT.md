# Prompt — Idea 3 real-data run (paste into Claude in VS Code)

---

`src/ltc_spiking_arch.py` is built and tested. Two new modules are now in place:

- **`src/ltc_data.py`** — JAX-free pipeline, benchmark and selective-prediction metrics.
  Already run end-to-end on the real `results/eurusd_h1.csv`; the numbers below are real.
- **`src/ltc_experiment.py`** — training loop and pre-registered scoring.
  **Written but never executed** — JAX was unavailable where it was authored. Treat it as
  a draft until `--dry-run` passes.

`results/ltc_hypothesis_log.csv` exists with two PENDING rows at α = 0.025 each.

## Three corrections to the original plan, already applied

1. **Use H1, not M15.** `eurusd_m15.csv` is 350k rows; `eurusd_h1.csv` is 60k. Of the two,
   H1 is ~6× cheaper, and both already carry `tick_volume`. No MT5 fetch needed.
2. **Two hypotheses, not one with a secondary.** Registering the clock claim as a
   non-gating "secondary" would let us claim the scientifically interesting result while
   only paying for one test. Both are registered at α = 0.025.
3. **H_ltc.2 needed a confound control, and this is not a formality.** On the real
   validation slice, `Spearman(log tick rate, |return|) = +0.4966`. A fake τ that tracks
   *only volatility* scores **raw Spearman −0.4966** — the pre-declared negative sign, i.e.
   it would have **passed** the hypothesis as originally worded. Under the partial
   correlation it scores **+0.0122**, correctly nothing. The primary is now
   `partial_spearman(tau_eff, log_tick_rate | abs_return_pct, parkinson_pct)`.

## The benchmark you must beat (measured, real)

Fitted on `[0:70%]`, scored on `validation[70:80]` = 2023-09-14 → 2024-09-02, 6,000 bars:

| | value |
|---|---|
| GARCH(1,1) | α = 0.2421, β = 0.5875, persistence 0.8296 |
| Day-of-week factors | Sun 0.858, Mon 0.934, Tue 1.003, Wed 1.011, Thu 1.023, Fri 1.037 |
| Mean CRPS | 0.03875 |
| **Benchmark AURC** | **0.03044** |
| Risk at 10% / 25% / 50% / 100% coverage | 0.0253 / 0.0265 / 0.0300 / 0.0388 |

Risk rises monotonically with coverage, so the benchmark's own confidence ranking is
informative — this is not a straw man.

## Tasks

1. **Install and smoke-test.** Add `jax`, `jaxlib`, `equinox`, `optax` to a separate
   `requirements-ltc.txt` (do not touch `requirements.txt`; the production app must keep
   installing without them). Then:
   ```
   python -m src.ltc_experiment --dry-run
   ```
   2,000 rows, 2 epochs. Fix whatever breaks — the JAX path is unverified. Report the
   traceback rather than working around it silently.

2. **Run `warp_sanity_check` before any real training.** On real H1, `dt_p999 = 49` and
   `dt_max = 74` in units of median spacing. The cell composes `exp(-dt/tau)`, so an
   unwarped weekend annihilates the hidden state for any τ of order a few bars. If
   `compression_ratio` is not well under 20, re-initialise `BusinessTimeWarp` before
   training — otherwise you are testing whether Adam can escape a dead initialisation, not
   whether the subordination hypothesis holds.

3. **Time one epoch before committing.** `--epochs 2 --seq-len 64` on the full training
   set, and report wall time. JAX is CPU-only on Windows. Do not start a 30-epoch run
   without knowing what it costs.

4. **Full run, then score.** `--epochs 30`. The loop aborts if coverage collapses below
   half the floor for three consecutive epochs — that abort is a feature. If it fires, the
   model bought a good loss by refusing to speak; report it, do not raise the floor and
   pretend.

5. **Register.** Copy the scored numbers into `results/ltc_hypothesis_log.csv`, replacing
   the PENDING rows, with today's date. Both verdicts, including DROPs.

## Rules

- **Do not touch the test block.** `chronological_masks` returns it; nothing may index it.
- **Do not tune against the validation slice.** Hyperparameters (`seq_len`, `hidden`, `lr`,
  `coverage_floor`) are fixed from the dry run and left alone. Any change after seeing a
  validation number turns the arbiter into a training set.
- **Report the numbers before the narrative**, and if a result looks good, try to break it
  first — the fake-clock check above is the template.
- Nothing touches `src/inference.py`, `models/`, or the serving path.
- Add tests to `tests/` for anything new in `ltc_data.py`; keep the suite green.

Start with task 1 and show me the dry-run output.
