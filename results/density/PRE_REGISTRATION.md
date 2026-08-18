# Pre-registration — conditional DENSITY forecasting of next-day EUR/USD return

**Family:** `density` (new). **Registry:** `results/density_hypothesis_log.csv`.
**Status at the time of writing: NO MODEL EXISTS AND NO NUMBER HAS BEEN COMPUTED.**

This document is committed BEFORE `src/density_model.py` is written. The commit
that adds this file contains no model code and no results; the commit that later
adds results names this document's commit hash. The git history is the evidence,
and it is a deliverable in its own right — this family exists partly as the
project's answer to exam feedback that hypotheses appeared to be adapted after
seeing results.

Nothing in this file may be revised once committed. If a stated choice turns out
to be wrong or unworkable, the deviation is recorded as a deviation in the
results commit and in the registry `notes`, never by editing the text above it.

---

## 1. Why this is a new estimand and not a rematch

The calendar volatility model (`models/calendar/calendar_volatility.json`, 10
numbers: `omega, alpha, beta, uncond_var, scale` plus 6 day-of-week factors, one
of which is pinned to 1.0) predicts a **scale**: one non-negative number per day,
the expected size of tomorrow's move. It has no shape, no skew, no tail index,
and no way to acquire one — a point forecast of `|r|` cannot answer "what is
P(r < -1.2%)".

Density forecasting asks for the whole conditional law of tomorrow's return.
That is a strictly larger object. The calendar model cannot compete on it
directly; it can only enter through a distributional wrapper, which is exactly
what baselines 1-3 below are. So this is not a re-run of the contest deep
learning already lost in `results/calendar_hypothesis_log.csv` — the losing
quantity there (MAE on `|r|`) is not the quantity scored here.

That framing cuts both ways, and the honest version of it is stated in section 7:
a wrapped calendar scale may already be a complete answer.

## 2. Target, rows, and split — fixed now

- **Target** `y_t = log(close_{t+1} / close_t) * 100` — next-day **log** return in
  **percent**, signed.
  - Deliberately the log return, not `src/features.py::target_return` (which is
    the simple return in percent). `|y_t|` is then *exactly*
    `TARGET_VOLATILITY_COLUMN`, the quantity `models/calendar/` was fitted
    against, so the calendar scale enters the baselines in its native units with
    no unit reconciliation. The two conventions differ by O(r^2) and the choice
    is made here, in advance, for that alignment reason.
- **Rows** the euro-era daily row set produced by
  `src/volatility.py::build_volatility_matrix`, i.e. the same rows the volatility
  and calendar families use (n ~ 8559). No new row set is invented.
- **Features** `PRICE_FEATURE_COLUMNS` plus the lag-PCA reduction, taken from
  `build_volatility_matrix` unchanged. The single-source-of-truth contract
  applies: **no new feature is engineered for this family.** Price-only, so the
  family carries no FRED dependency and no macro-provenance question.
- **Split** chronological, from the volatility family's own boundaries:
  - train `[0 : 70%]` — MDN weights, PCA, scaler, and every baseline nuisance
    parameter are fit here and nowhere else.
  - validation `[70% : 80%]` — **the arbiter.** Early stopping, K selection, and
    the registered comparison all live here.
  - test `[80% : 100%]` — **one-shot final report only.** Not indexed during
    development, not consulted to choose anything. A dedicated unit test asserts
    that no selection step touches it.

## 3. Primary metric

**CRPS** (continuous ranked probability score), lower is better, on the
**validation block `[70:80]`**, paired row-by-row.

- Gaussian-mixture CRPS by the **closed form** (Grimit et al. 2006):

      CRPS = sum_i w_i A(y - mu_i, sigma_i^2)
             - 0.5 sum_i sum_j w_i w_j A(mu_i - mu_j, sigma_i^2 + sigma_j^2)
      A(m, s^2) = m (2 Phi(m/s) - 1) + 2 s phi(m/s)

- Student-t CRPS by the **closed form** (Jordan, Krueger and Lerch 2019), scaled:
  `CRPS(y; 0, sigma, nu) = sigma * CRPS_std(y/sigma; nu)`. Verified in the unit
  tests against numerical integration of `int (F(x) - 1{x >= y})^2 dx` on a fine
  grid.
- Empirical/unconditional CRPS by the exact identity
  `CRPS = E|X - y| - 0.5 E|X - X'|`, evaluated with the sorted-cumsum form.
- **No sampling estimator anywhere.** A Monte-Carlo CRPS would inject noise into
  the exact quantity under test.
- Cross-check, unit-tested: the mixture closed form with `K = 1` must agree with
  `src/ltc_data.py::crps_gaussian_np` to numerical precision.

## 4. Primary comparison

**MDN 5-seed ensemble** vs **Student-t(nu, 0, sigma_calendar)** — baseline 2, the
strongest rival, chosen as primary precisely because it already has the fat tail
that a mixture would otherwise be credited for discovering.

**The scored object is the ensemble, all-or-nothing**, exactly as the volatility
family defines it (`vol_ready` semantics): the ensemble predictive density is the
equal-weight mixture over the 5 seeds' mixtures, i.e. a single `5K`-component
Gaussian mixture with weights `w_{s,i} / 5`. That is itself a Gaussian mixture,
so the same closed form scores it with no approximation. **A partial ensemble is
never scored and never reported.** Seeds: `42, 43, 44, 45, 46`.

Single-seed numbers appear nowhere in any deliverable except a per-seed
diagnostic table explicitly labelled as spread, never as a result.

## 5. Baselines — all four, fit on train rows only

| # | Model | Fitted on train | Frozen |
|---|---|---|---|
| 1 | `Gaussian(0, c_g * sigma_cal)` | scalar `c_g` (MLE) | calendar params |
| 2 | `Student-t(nu, 0, c_t * sigma_cal)` — **PRIMARY RIVAL** | `nu`, `c_t` (joint MLE) | calendar params |
| 3 | `Student-t(nu, 0, c_h * sigma_garch)` | GARCH(1,1) plus `nu`, `c_h` | — |
| 4 | Empirical unconditional | train-block ECDF of `y` | — |

- **`models/calendar/` is loaded FROZEN and never refitted.** Its 10 numbers are
  read from JSON; `CalendarVolatilityModel.fit` is not called.
- `sigma_cal` is the calendar model's `predict()` output, which is a conditional
  scale for `|y|`, not a standard deviation. Turning it into a distribution
  parameter needs one link constant. That constant (`c_g`, `c_t`, `c_h`) is a
  **nuisance parameter of the baseline, not of the calendar model**, and is fit
  by maximum likelihood **on train rows only**. Fitting it is what gives the
  rival its best shot; declining to fit it would be a rigged comparison. This is
  declared here precisely because it is the one place "no refit" could be read
  two ways.
- Baseline 3's GARCH(1,1) is fit on train rows only via
  `src/calendar_volatility.py::fit_garch11` with a train mask — a *separate* fit
  from the frozen calendar model, so baselines 2 and 3 are genuinely different
  arms rather than the same recursion twice.

## 6. Model — fixed now

`src/density_model.py`, Mixture Density Network:

- shared trunk -> `K` components, each `(weight, mu, sigma)`; softmax weights.
- **`K` in `{2, 3, 5}`, selected on VALIDATION CRPS only.** The test block is
  never indexed for this or any other choice.
- `sigma = SIGMA_FLOOR + softplus(raw)` with **`SIGMA_FLOOR = 0.05`** (percent
  units; daily EUR/USD sigma is about 0.55%, so the floor sits ~11x below typical
  and cannot bind in normal operation). `log sigma` is carried explicitly through
  the log-likelihood for numerical stability.
- **Gradient clipping `clipnorm = 1.0`.** MDNs diverge when one component
  collapses onto a training point and the likelihood runs away; the floor plus
  clipping is the guard.
- **Every epoch asserts `min sigma > SIGMA_FLOOR` and aborts loudly on
  violation.** A silent NaN here would look like a result.
- Train on `[0:70%]`, early stop on `[70:80%]` (NLL). Scaler and PCA fit on
  `[0:70%]` only — inherited from `build_volatility_matrix`, not re-derived.

## 7. EXPECTED OUTCOME, STATED IN ADVANCE

**We expect the scale term to dominate and the shape gain to be small or
absent.** Daily EUR/USD is near-efficient; this project has already established
that conditional *mean* prediction shrinks to zero, and that a 10-number calendar
lookup beats a 5-seed LSTM ensemble on scale. The most likely result is that a
fitted Student-t on a frozen calendar scale is a *sufficient* density forecast
and the MDN buys nothing distinguishable from zero.

**A near-zero result is the predicted result and will be reported as such** — as
a finding, not a failure, and with the same prominence a positive result would
get. Same discipline as the curl STOP GATE. This paragraph exists so that this
outcome cannot later be dressed up as a surprise, or quietly reframed as an
inconclusive run that needs one more variant.

## 8. Family size, bar, and decision rule — fixed now

- **`family_size = 2`**, declared now, both members registered up front:
  - `H_den.1` — MDN ensemble vs Student-t(nu, 0, sigma_calendar). *Primary.*
  - `H_den.2` — symmetric-mixture ablation (components share a common mean),
    scored against the unrestricted MDN, to locate the gain as **skew** versus
    **tail/kurtosis**.
- **`alpha = 0.05 / 2 = 0.025` for each**, Bonferroni. Two-sided, so each verdict
  is read off a **97.5% CI**.
- This family is **self-contained**. It does not touch
  `results/feature_hypothesis_log.csv` and does not tighten any other family's
  bar, exactly as the volatility family is its own registry.
- **CI method** paired **moving-block bootstrap** on the per-row CRPS
  differences, **`block_len = 5`** (the volatility/calendar family convention),
  **2000 resamples**, `random_state = 42`. Paired on rows: both models score the
  identical validation rows and the bootstrap resamples the *difference* series.
- **DECISION RULE — KEEP only if the paired CRPS gap (rival minus MDN) excludes
  zero at the corrected bar**, i.e. the 97.5% CI lower bound > 0. Anything else
  is a DROP, including a favourable point estimate whose interval covers zero.
- `H_den.2` is evaluated **only if `H_den.1` clears.** If `H_den.1` drops, the
  decomposition has nothing to decompose; `H_den.2` is then recorded as NOT-RUN
  with its alpha unspent, and the family bar is **not** retroactively loosened to
  0.05 for `H_den.1`. The bar declared before measurement is the bar applied
  after it.

## 9. Secondary and diagnostic — explicitly NOT significance tests

- **NLL** on validation, reported for every model. Secondary. No verdict rides on
  it.
- **PIT / rank histogram** (20 bins) for every model, plus PIT mean and variance.
  This is a **CALIBRATION DIAGNOSTIC ONLY**. It is *not* a second significance
  test, no p-value is computed from it, and a uniform PIT will not be used to
  argue a KEEP that the CRPS gap did not earn. A model can be perfectly
  calibrated and still lose on CRPS; that is the point of carrying both.
- Per-seed CRPS spread, reported to show the ensemble is not masking a
  nondeterminism artifact of the same magnitude as the effect under test.

## 10. Test block

`[80% : 100%]` is a **one-shot final report**, run once after the validation
verdict is recorded, and never used to select K, seeds, architecture, link
constants, or the verdict itself. If it disagrees with validation, both numbers
are reported and the validation verdict stands as the registered one.

---

*Written before `src/density_model.py` existed. The commit adding this file
contains the registry row and this document and nothing else.*
