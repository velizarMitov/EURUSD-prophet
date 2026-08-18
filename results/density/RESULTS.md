# Density family — validation results (H_den.1, H_den.2)

**Pre-registration:** [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md), committed at
**`7524cab0237d325a144b7eef1f85ca759643b942`** *before `src/density_model.py`
existed*. That commit contains the protocol and the two registry rows and
nothing else — no model code, no numbers.

**Verdict: `H_den.1` DROPS. `H_den.2` NOT-RUN, alpha unspent.**
**This is the outcome the pre-registration predicted in writing, in advance.**

## The registered comparison

Arbiter: validation `[70:80]`, n = 860. Primary metric CRPS (lower better),
closed form only, paired per row. Bar: `alpha = 0.05/2 = 0.025` -> 97.5% CI.
Decision rule: KEEP only if the paired gap's CI lower bound > 0.

| | CRPS |
|---|---|
| Student-t(nu, 0, c_t * sigma_calendar) — **registered rival** | **0.191543** |
| MDN 5-seed ensemble, K=5 (25 components) — **challenger** | 0.191995 |

**delta (rival - MDN) = -0.000452, 97.5% CI [-0.002819, +0.001857].**

The interval covers zero *and* the point estimate favours the rival. The
registered rule is not met, and would not have been met even if the sign had
gone the other way. **DROP.**

## Every model on the arbiter block

| Model | CRPS | NLL | PIT mean | PIT var |
|---|---|---|---|---|
| Student-t(nu, 0, c_t·sigma_cal) | **0.191543** | 0.216405 | 0.5059 | 0.0742 |
| MDN 5-seed ensemble (K=5) | 0.191995 | **0.208247** | 0.4939 | 0.0730 |
| MDN symmetric ensemble | 0.193055 | 0.216287 | 0.4828 | 0.0711 |
| Gaussian(0, c_g·sigma_cal) | 0.194080 | 0.290726 | 0.5046 | 0.0605 |
| Student-t(nu, 0, c_h·sigma_garch) | 0.198868 | 0.368948 | 0.5020 | 0.0766 |
| Empirical unconditional | 0.207643 | — | 0.5031 | 0.0560 |

## All four baselines, as gaps against the MDN

Positive = MDN better. 97.5% CI, paired moving-block bootstrap, block_len = 5,
2000 resamples.

| Rival | delta | 97.5% CI | excludes 0? |
|---|---|---|---|
| **Student-t + calendar scale** *(registered primary)* | **-0.000452** | **[-0.002819, +0.001857]** | **no** |
| Gaussian + calendar scale | +0.002085 | [-0.000428, +0.004592] | no |
| Student-t + GARCH(1,1) | +0.006873 | [+0.004242, +0.009613] | yes |
| Empirical unconditional | +0.015647 | [+0.012547, +0.018811] | yes |

This is the part worth reading twice. The MDN is **not** a broken model: it
beats an unconditional distribution and a GARCH-scaled t decisively, both
intervals excluding zero. It is a competent conditional density forecaster. What
it cannot do is improve on **a fitted t-shape wrapped around a frozen 10-number
calendar scale**. The scale term is where essentially all of the achievable
skill lives, and a 10-parameter lookup already captures it.

## H_den.2 — not run, and why that matters

The pre-registration made the ablation **conditional**: evaluated only if
`H_den.1` clears, because a decomposition of "where the gain lives" is
meaningless when there is no gain. `H_den.1` dropped, so `H_den.2` is recorded
**NOT-RUN with its alpha unspent**.

The number was still computed and stored (`gaps.csv`) so the run is complete and
auditable: symmetric minus unrestricted = **+0.001060, 97.5% CI [-0.000078,
+0.002211]**, which also covers zero. It is recorded as a *number*, not a
verdict. Reading it as a claim would be exactly the post-hoc adaptation this
family was built to rule out.

**The family bar is not retroactively loosened.** `alpha = 0.025` was fixed by
`family_size = 2` before any measurement. Re-deriving it as `0.05/1 = 0.05`
now that only one hypothesis was tested would be the textbook version of the
thing being guarded against — and it would not change the verdict anyway, since
the interval covers zero at any level.

## Why 5 seeds was not optional

| | seed 42 | 43 | 44 | 45 | 46 | range |
|---|---|---|---|---|---|---|
| MDN | 0.192978 | 0.193387 | 0.193298 | 0.193632 | 0.192922 | **0.00071** |
| MDN symmetric | 0.193613 | 0.194045 | 0.193401 | 0.197507 | 0.194952 | 0.00410 |

**The seed-to-seed range (0.00071) is larger than the effect under test
(0.00045).** A single-seed run could have produced either sign at will. No
single-seed number appears anywhere in this family, and the scored object is
the full 5-seed ensemble mixture — all-or-nothing, never partial.

## K selection — validation only

| K | validation ensemble CRPS | selected |
|---|---|---|
| 2 | 0.192466 | |
| 3 | 0.192796 | |
| 5 | **0.191995** | yes |

The spread across K (0.0008) is also comparable to the effect under test. The
test block was never indexed to make this choice; `select_k` does not take a
test mask, and `tests/test_density_model.py` pins that with a tripwire object
that raises on any access.

## Calibration diagnostic — NOT a second test

Per the pre-registration, PIT is reported and **no p-value is computed from
it**. A flat PIT is not used to argue a KEEP the CRPS gap did not earn.

Ideal is PIT mean 0.5 and variance 1/12 = 0.0833. Every model here sits *below*
0.0833 (0.056-0.077), i.e. all of them are somewhat **too wide** — observations
land nearer the middle of the predictive distribution than a calibrated forecast
would put them. The unconditional empirical baseline is worst (0.0560), which is
what conditioning is supposed to fix and does. The MDN (0.0730) is not better
calibrated than the registered rival (0.0742) on this diagnostic either.

## What this establishes

1. **A conditional density model buys nothing here.** The calendar scale plus a
   fitted t-shape is a *sufficient* density forecast for next-day EUR/USD at the
   resolution 860 validation rows can resolve. This is the pre-registered
   prediction, reported as a finding.
2. **The scale term dominates, as predicted.** The MDN's clear wins over
   GARCH+t and the unconditional baseline show the machinery works; the absence
   of a win over calendar+t localises all the skill in the scale.
3. **A 10-number lookup table remains the thing to beat.** It already beat a
   5-seed LSTM ensemble on scale (`results/calendar_hypothesis_log.csv`); it now
   also holds its own on the strictly larger density estimand once given a
   two-parameter shape.

## What this does NOT establish

- Not "MDNs do not work" — this MDN beats two of its four baselines decisively.
- Not "there is no shape structure in EUR/USD returns" — only that any such
  structure is not separable from zero at n = 860 against a fitted t. The
  intervals are roughly +/-0.0023 wide, so a true effect smaller than about 1.2%
  of the CRPS level is undetectable here by construction.
- The secondary NLL favours the MDN (0.2082 vs 0.2164). NLL carries no verdict
  by pre-registration, and it is recorded here rather than promoted.

## Files

- `scores.csv` — CRPS / NLL / PIT summary per model per block
- `gaps.csv` — every paired bootstrap, including the registered primary
- `per_seed_crps.csv` — per-seed spread (diagnostic only, never a result)
- `k_selection.csv` — validation-only K sweep
- `pit_histograms.csv` — 20-bin rank histograms for every model
- `run_meta.json` — pre-registration hash, K, alpha, block_len, split sizes

---

# One-shot test-block report

Run **after** the validation verdict above was recorded and committed
(`dd4a87d`), exactly as section 10 of the pre-registration requires. The test
block was never indexed to select K, seeds, architecture, link constants, or the
verdict. **This section does not arbitrate anything** — the registered verdict
is the validation one and it stands regardless of what follows.

**Reproduction check.** The test-stage run refits from scratch. Its validation
numbers reproduced the committed run at `dd4a87d` to **0.0 absolute difference
on every model**, so the two blocks below are scored by the identical ensemble
and nothing here is confounded by run-to-run drift.

## Test `[80:100]`, n = 1721

| Model | CRPS | NLL | PIT mean | PIT var |
|---|---|---|---|---|
| Student-t(nu, 0, c_t·sigma_cal) | **0.218840** | 0.365283 | 0.4936 | 0.0753 |
| MDN 5-seed ensemble (K=5) | 0.220332 | **0.360981** | 0.4872 | 0.0781 |
| MDN symmetric ensemble | 0.221363 | 0.365211 | 0.4781 | 0.0762 |
| Gaussian(0, c_g·sigma_cal) | 0.221729 | 0.477853 | 0.4947 | 0.0624 |
| Student-t(nu, 0, c_h·sigma_garch) | 0.225439 | 0.483860 | 0.4949 | 0.0765 |
| Empirical unconditional | 0.231608 | — | 0.4961 | 0.0623 |

| Rival | delta (rival − MDN) | 97.5% CI | excludes 0? |
|---|---|---|---|
| **Student-t + calendar scale** *(registered primary)* | **-0.001492** | **[-0.003163, +0.000112]** | **no** |
| Gaussian + calendar scale | +0.001397 | [-0.000607, +0.003219] | no |
| Student-t + GARCH(1,1) | +0.005107 | [+0.003052, +0.007060] | yes |
| Empirical unconditional | +0.011276 | [+0.009137, +0.013332] | yes |

**The test block agrees with the arbiter on every point.** The model ordering is
identical, the registered gap again covers zero with the point estimate favouring
the rival (and slightly more so), and the MDN again beats GARCH+t and the
unconditional baseline with intervals excluding zero. The DROP is not a
single-slice artifact.

## H_den.2 on the test block — a number, still not a verdict

The symmetric ablation on test is **+0.001031, 97.5% CI [+0.000137, +0.001835]**,
which *excludes* zero.

**This is not a KEEP and is not read as one.** Two independent reasons, both
fixed in advance:

1. `H_den.2`'s precondition failed. The pre-registration made it conditional on
   `H_den.1` clearing; it did not, so `H_den.2` is NOT-RUN with its alpha
   unspent. A decomposition of "where the advantage lives" cannot be run on an
   advantage that does not exist — the unrestricted MDN loses to the registered
   rival, so beating its own hobbled variant establishes nothing about EUR/USD.
2. Even setting that aside, this is the **test block**, which is a one-shot
   report and not the arbiter. On the arbiter the same quantity was +0.001060
   with CI [-0.000078, +0.002211] — covering zero.

Promoting this cell to a finding would be precisely the post-hoc adaptation this
family was constructed to make impossible. It is recorded, in full, and left
unread.

## Seeds: what the spread does and does not show

Seed-to-seed CRPS range (0.00071) exceeds the effect under test (0.00045), so
**different seeds** genuinely could have produced either sign — which is what
makes the 5-seed all-or-nothing ensemble load-bearing rather than ceremonial.

To be precise about a related but different claim: **run-to-run** determinism
held perfectly here (0.0 difference across two independent full runs). The
TF/oneDNN nondeterminism measured elsewhere in this project did not reproduce
for this dense MDN. The seed argument stands; the nondeterminism argument does
not apply to this particular model and is not claimed.
