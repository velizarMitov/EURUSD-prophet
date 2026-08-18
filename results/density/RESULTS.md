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
