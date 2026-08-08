# Idea 2 / Step 1 — Robust curl on OHLCV bars

**Status: pre-registration draft. No hypothesis is spent yet.**
Code: `src/curl_stress.py`, `src/curl_null_simulation.py`, `tests/test_curl_stress.py`.
Simulation report: `results/curl_null_simulation_report.txt`.

---

## 1. The theoretical workaround

### 1.1 Start by conceding the point

For log rates on the currency graph, no-arbitrage at simultaneity makes the 1-cochain
`w` exact (`w = d(phi)`), so every triangle curl is **identically zero**. Therefore, on
real asynchronous data,

```
observed_curl  =  d(observation noise)
```

There is **no price signal in the curl. It is a pure measurement-error channel.** Your
worry — "this might all be broker feed noise" — is not a risk to be mitigated. It is
provably the whole content of the quantity, and any framing that hopes otherwise is
wrong from the first line.

That is the useful reframing, not a dead end. Staleness error scales as
`sigma * sqrt(time since last tick)`, so

```
E[curl^2]  ~  volatility  x  illiquidity
```

The curl is a **cross-sectional estimator of microstructure noise amplitude** — the same
object the two-scale realized-volatility literature (Zhang / Mykland / Aït-Sahalia)
estimates from signature plots across time scales, except measured across instruments at
one instant. That is a legitimate and, as far as I know, unexploited observable. The
estimand is the conditional **scale** of the noise, never a single bar's value.

### 1.2 Your three proposed workarounds, judged

| Proposal | Verdict |
|---|---|
| Rolling variance of curl instead of the raw absolute value | **Necessary but not sufficient.** Correct that the estimand is a scale. But raw curl scale rises whenever volatility rises, so an un-normalised rolling curl variance is largely a repackaging of volatility — and shipping it would be a false discovery of exactly the kind your volatility log already caught once. |
| Rolling z-score of the curl | **Wrong object as a primary.** Curl is mean-zero by construction, so `z = c / sigma_roll` divides out `sigma_roll` — which *is* the signal. You would be z-scoring away the only informative quantity. A z-score is fine as the **final** transform applied to the log-excess series, after the null normalisation below, never before it. |
| Use High/Low and volume | **Yes — this is the actual answer**, in two distinct roles. |

### 1.3 The construction

**Step 1 — a closed-form null for how much curl asynchronicity alone must produce.**
If edge `e`'s recorded Close is the last tick `tau_e` before the boundary, with `n_e`
Poisson ticks in a bar of length `D`, then for triangle `T = (A,B,C)`:

```
E[c_T^2]  =  sum_{X in T}  v_X * E|tau_e1(X) - tau_e2(X)|
E|tau_a - tau_b|  =  m_a + m_b - 2 m_a m_b / (m_a + m_b)          (independent exponentials)
m_e  =  (D / n_e) * [1 - n_e / (e^{n_e} - 1)]                      (truncated at one bar)
```

`v_X` are the per-currency variance rates. **Verified by Monte Carlo**: on an exactly
arbitrage-free synthetic world observed asynchronously, realised/theoretical curl variance
came in at **0.87–0.98** across all four triangles.

The truncation factor matters only for thin bars (< ~10 ticks) — which is exactly the
22:00 UTC rollover, holidays and the Sunday open. Without it the null over-predicts by an
order of magnitude there and therefore **hides** real stress precisely when it is most
likely.

**Step 2 — a feed-only proxy you can actually compute.** Replace `v_X` with per-edge
variance and per-edge tick counts:

```
P_T  =  sum_{e in T}  Var_bar(e) / n_ticks(e) * truncation(n_ticks(e))
```

`Var_bar` **must be the Parkinson High/Low estimator** `ln(H/L)^2 / (4 ln 2)`, lightly
smoothed (4 bars), not a rolling variance of close-to-close returns. This is the
load-bearing use of your High/Low columns, and it is not cosmetic:

| `Var_bar` source | corr(P, true E[c²]) | binned slope (theory 0.5) |
|---|---|---|
| rolling return variance, 96 bars | 0.295 | 0.097, collapses in the tail |
| **Parkinson H/L, 4 bars** | **0.941** | **0.541** |

The null needs an *instantaneous* variance; a 96-bar rolling variance is too lagged and
its outliers drive `P` in a direction that has nothing to do with staleness.

**Step 3 — excess curl.** Fit `E[c^2] = a + b*P_T` on the train slice, then

```
X_T  =  c_T^2 / (a + b * P_T)          E[X] = 1 under pure asynchronicity
```

Fit on **equal-count bin means of `P`, not raw bars.** Conditional on `P`, `c^2` is
`P * chi^2_1`, whose CV is `sqrt(2)`. That multiplicative noise attenuates the raw
per-bar `corr(c^2, P)` to roughly `CV(P)/sqrt(2)` — about **0.1–0.3 even when the null is
exactly true**. Reading that low correlation as "the null model doesn't work" is the trap;
binning removes the chi-square and the relationship appears cleanly (validated: bin ratios
flat at 0.40–0.60 across all ten deciles).

**Step 4 — aggregate, then smooth.** Average `X` over the **three independent** triangles
and take the log of a rolling mean (96 bars ≈ one day at M15). A single bar's excess is
chi-square noise; the scale needs a window.

> **K4 has `E - V + 1 = 3` independent cycles, not 4.** The four triangle curls are
> linearly dependent (numerically confirmed: rank 3). Any covariance/Mahalanobis treatment
> of all four is singular — `np.linalg.inv` will hand you garbage. Use `CYCLE_BASIS`.

**Step 5 — the High/Low infeasibility certificate.** Sum the per-edge log `[L, H]`
intervals around the triangle; `g_T` = distance from zero to that sum.
`g_T = 0` means some assignment of prices *inside the observed bars* satisfies no-arbitrage
exactly, so the raw curl is fully explained by within-bar timing — inconclusive, and the
common case. `g_T > 0` means **no within-bar timing can reconcile the triangle**.

Honest caveat: observed H/L are extremes of the *sampled* path, so the observed range is a
subset of the true range, biasing `g` upward — false positives, not false negatives. The
`buffer_log` argument (half-spread + tick allowance) restores conservatism. In simulation
the certificate fired on 0% of clean bars, 28.5% of a 15 bp dislocation and 100% of a
50 bp one, and **never** on 2 bp or 5 bp dislocations at M15. **It is an M1-and-events
tool, not an M15 tool** — the M15 bar range is simply much wider than a few bp.

### 1.4 The single sharpest diagnostic you can run tomorrow

`E[tau] = D / n` is independent of bar size at a fixed tick *rate*. So under pure
asynchronicity, **curl variance is flat across M1/M5/M15/H1** while return variance grows
linearly. Simulation confirms: RMS curl 0.94 / 0.86 / 0.91 / 0.81 bp across M1→H1, while
RMS return goes 3.4 → 8.1 → 14.1 → 19.2 bp.

- **Flat** → the curl is timing noise. Expected, correct, and it kills the "curl as
  arbitrage signal" reading immediately. The normalised excess survives as a liquidity proxy.
- **Rising with bar size** → either a genuine persistent dislocation or a stale feed.
  Separate them with the autocorrelation check: near-zero ACF = asynchronous sampling;
  persistent positive ACF = a frozen feed, i.e. a data bug.

---

## 2. Implementation plan

### 2.1 What already exists

`src/curl_stress.py`

| Function | Role |
|---|---|
| `align_bars` | strict inner-join on timestamps; **drops** unmatched bars, never ffills |
| `triangle_curl` / `curl_frame` | the curl itself, orientation-aware |
| `convention_check` | mean curl vs its standard error — run this **first** |
| `parkinson_variance` | H/L instantaneous variance |
| `staleness_null` | the feed-only null `P_T` |
| `null_calibration_table` | decile table — the honest way to read the null |
| `calibrate_null` / `excess_curl` | static fit (for the pre-registered test) |
| `causal_excess_curl` | expanding-window refit (for anything forward-looking) |
| `triangle_infeasibility` | H/L violation certificate |
| `stress_index` | aggregate over the 3-cycle basis, smoothed |
| `residualise_calendar` | strip hour-of-day and day-of-week |
| `frequency_scaling`, `curl_autocorrelation`, `shift_placebo`, `cycle_rank` | diagnostics |

`tests/test_curl_stress.py` — 17 invariant tests, all passing: exact-simultaneity gives
`< 1e-12` curl, cycle rank is 3, alignment shift inflates curl, calibration ignores
post-train rows, `causal_excess_curl` has no look-ahead.

### 2.2 Day-1 runbook

1. **Pull the six pairs from MT5** at M1, M5, M15, H1 over the longest common history.
   Broker symbols carry suffixes (`EURUSD.a`, `EURUSDm`, `EURUSD_raw`) — resolve via
   `mt5.symbols_get()`, do not hard-code. Convert `time` to UTC. Keep `tick_volume`
   (MT5's tick count; a well-established activity proxy in FX, and here it is the
   staleness estimator).
2. **`align_bars`.** Log how many bars each pair loses. If any pair loses more than a few
   percent, find out why before continuing.
3. **`convention_check`.** A large `t_stat` with small variance = a static level or
   convention mismatch, not stress. Simulation: an inconsistent quote convention produced
   a **constant 30.5 bp** curl — 30x larger than the genuine 0.9 bp timing noise. This
   will be the first thing you see and it is plumbing. *(Practical note: with n in the
   hundreds of thousands even a trivially small bias is "significant"; judge the mean
   curl against a pip, not against a p-value.)*
4. **`cycle_rank`** must return 3.
5. **`frequency_scaling`** across M1/M5/M15/H1 — the § 1.4 discriminator.
6. **`curl_autocorrelation`** and **`shift_placebo`** — stale-feed and misalignment checks.
   In simulation, one bar of misalignment inflated RMS curl 9–12x, so if your observed
   curl is anywhere near that scale relative to the null, you have an alignment bug.
7. **Hour-of-day and day-of-week profile of the stress index, before anything else.**
   Non-optional. Liquidity collapses at the 22:00 UTC rollover and through the Asian
   session, so raw curl has a large deterministic intraday shape. Skipping this step
   reproduces `results/volatility_hypothesis_log.csv` #3 exactly — an apparent edge that
   was a calendar table.

### 2.3 The STOP GATE — commit to it before you look

Stop, write it up as a negative result, and do not proceed to the predictive test if
**all** of the following hold:

- RMS curl is flat across timeframes (pure asynchronicity), **and**
- binned `c^2` vs `P` is consistent with the null (bin ratios roughly flat, no systematic
  excess), **and**
- what variation remains in the stress index is explained by hour-of-day + day-of-week +
  tick count.

That combination means the curl is exactly the measurement noise the theory predicts, with
nothing left over. It is a real finding, it costs no hypothesis slot, and it saves a month.

---

## 3. Testing predictive power without leaking

### 3.1 Registration

New family — **do not** put this in the direction family, whose bar is already
`0.05/9 ≈ 0.00556`. Create `results/curl_hypothesis_log.csv` with the same columns as
`results/volatility_hypothesis_log.csv`. Hypothesis #1 is judged at `alpha = 0.05`; each
subsequent registration tightens the ladder for all of them.

**Pre-register exactly one primary quantity, one horizon, one target, in writing, before
scoring.** Every additional horizon or target you try is another hypothesis and tightens
the bar retroactively.

### 3.2 Recommended primary

Mirror the design already precedented in your volatility log entry #9 (the Kronos
cross-family test):

> **Primary:** incremental information of the daily-aggregated stress index for next-day
> realized volatility, measured as the correlation between the stress index and the
> **residual of the frozen 5-seed volatility ensemble** on `validation[70:80]`, with a
> moving-block bootstrap CI (`block_len=5`, `n=2000`) at the family alpha.

Rationale: if the construct means what § 1.1 says, it is a volatility-times-illiquidity
proxy, so second-moment prediction is where its content should be. A direction test is
near-certain to DROP and would spend a slot to learn nothing. If you want the direction
ADD-test anyway, register it knowingly as hypothesis #2.

Aggregation M15 → daily: pre-declare the statistic (I would use the **mean** of
`stress_smooth` over the session, with the **max** as a pre-declared alternative only if
you register it as a separate hypothesis). Use only bars that close **strictly before**
the daily bar's close.

### 3.2b The confound residualisation is mandatory

On synthetic data **where the asynchronicity null is true by construction**, the raw stress
index still correlated **+0.40 with total tick count, −0.35 with volatility, −0.35 with
spread**. The null normalisation removes most of the liquidity/volatility channel, not all
of it. Testing the raw index would be testing a repackaged liquidity proxy that "works" for
reasons unrelated to the Hodge decomposition.

**Pre-register `stress_orthogonal`**, not `stress`:
`curl_mt5.residualise_against_confounds` regresses the index on log tick count, log summed
H/L range, log summed relative spread, and hour/day dummies, with coefficients fitted on
train rows only. After residualisation the confound correlations fall to 0.016 / −0.023 /
−0.014, and an injected 4 bp dislocation is still detected (+0.009 → +1.775).

Spread deserves its own line: MT5 bars are bid, so curl rises mechanically when spreads
widen, and spreads widen in stress. If excess curl adds nothing over spread, it is not a
new observable.

### 3.3 Leakage checklist

- **Arbiter is `validation[70:80]`.** The test block `[80:100]` is spent for feature
  search — do not index it.
- **Every fitted quantity is fitted on `[0:70]` only**: the null coefficients `(a, b)`,
  the calendar means in `residualise_calendar`, any z-score mean/sd. `calibrate_null`
  takes an explicit `train_mask` for exactly this reason, and
  `test_calibration_uses_train_rows_only` pins it.
- **Rolling windows are backward-only.** `stress_smooth` uses a trailing mean; the
  Parkinson estimator uses the current and prior bars. No centred windows anywhere.
- **The episode-in-the-training-window trap** (found during development): a stress episode
  inside the calibration slice inflates the fitted denominator and silently deflates the
  whole index. For forward use, prefer `causal_excess_curl`.
- **Static calibration drifts.** Under a *true* null, out-of-sample mean `X` wandered
  between 0.57 and 1.26 across simulation runs purely from regime persistence. A raw
  excess-curl *level* read years after its fit is partly stale normalisation, not stress.
- **Bootstrap in blocks.** Curl is at least weakly dependent at M15; i.i.d. resampling
  will overstate significance.

### 3.4 Power you can expect

On synthetic data the smoothed stress index separated an injected dislocation episode from
baseline with **AUC 0.996 at 2 bp**. So the machinery is not the limiting factor — if a
real dislocation of that size exists in your feed, this will see it. What it cannot do is
manufacture predictive content if the curl is what theory says it is.

### 3.5 The most likely outcome, stated in advance

Flat frequency scaling, near-zero curl autocorrelation, and a stress index whose variation
is mostly the intraday liquidity cycle. That is the null being true, and it is the result
the theory predicts. Write it into the log as a negative result and move to Idea 3.

The scenario worth staying alert for is the opposite one: curl variance that **rises** with
bar size. Check the feed before you get excited — a frozen or stale pair produces exactly
that signature, and it will be the most exciting-looking result in the study right up until
you find the bug.
