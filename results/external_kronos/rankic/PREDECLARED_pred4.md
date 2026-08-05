# KRONOS RankIC ON A FOREX CROSS-SECTION at pred_len=4 — pre-declared

**Written 2026-08-04, BEFORE the first window of this run was generated.**
Supersedes the pred_len=1 plan in `PREDECLARED.md`; that configuration is
retired (see `../SUPERSEDED_BY_CONFIGURATION.md`).

## STEP 0 probe — passed

Kronos-mini, pred_len=4, EURUSD, 50 windows: distinct endpoints out of 30 =
**mean 26.10, median 28, min 16, max 30**; 90.0% of windows above 20, none at or
below 10. Reference: **7.64** at pred_len=1, **29.87** at pred_len=24. The
single-step collapse is cleared at four steps, so the mean predicted return is
not built on a coarse grid. Gate was "above 20" → **PROCEED**.

## Configuration (no tuning)

| | value |
|---|---|
| model | **Kronos-mini primary** (beat base on every volatility measure at pred_len=24, 12.6× cheaper, and the paper's Forex column puts Kronos-S above Kronos-B). Base is a declared secondary, run only if mini shows something. |
| pred_len | 4 |
| sampling | every 4th hour, **non-overlapping**, stride = pred_len |
| sample_count / T / top_p / context | 30 / 1.0 / 0.9 / 512 |
| data | banked `rankic/raw/`, 19 instruments, 13,002-timestamp intersection grid, 2024-07-01 onward, fetched **with** the Market Watch sync fix |

**CHF is kept.** The RankIC raw data was re-fetched with the sync fix and audits
clean; the retirement decision concerns the corrupted `results/pooled_h1/` files
only. Dropping CHF would take the cross-section from 8 currencies to 7, the rank
bound from 7 to 6, and the minimum detectable RankIC from 0.0130 to 0.0140 —
from below the target to level with it.

## Power, before any result

* timestamps **n = 3,122** (every 4th hour from index 512 to 12,997)
* windows 19 × 3,122 = **59,318**
* measured 0.267 s/window steady state → **≈ 4.4 h**, inside the 8 h ceiling
* **minimum detectable RankIC ≈ 0.0133** (the brief's 0.0130 at n=3,250, scaled by √(3250/3122))
* **rank bound: 7** — 19 pairs from 8 currencies span at most 7 independent
  dimensions; PCA gives 7 components = 98.7%, 8th eigenvalue 0.008,
  participation dimension 5.85

**The paper's own absolute Forex targets:** Kronos-S **0.0177**, Kronos-B
**0.0141**, Kronos-L **0.0137**, TimesNet **0.0175**. All three Kronos columns
are **full-shot (fine-tuned per dataset)**; we run zero-shot. On Forex
specifically Kronos-base *loses* to a non-pretrained baseline. Expect little.
The MDE of 0.0133 sits below all three, so a null here will be a **powered**
null — it rules out an effect of the size the paper reports.

## STEP 4 — how this will be read

* **RankIC CI excludes zero AND exceeds momentum/reversal** → Kronos transfers to
  FX on its own primary task. First positive external directional result in this
  project; the serving configuration then deserves reconsideration.
* **RankIC CI includes zero while a simple ranking's CI excludes it** → the
  harness works, Kronos adds nothing. A clean, powered negative.
* **Nothing distinguishable, including the simple rankings** → the cross-section
  or the harness is inadequate; investigate before concluding anything about
  Kronos.

Benchmarks on identical timestamps and instruments: random ranking (empirical
noise floor), momentum (previous 4-bar return), reversal (its negative),
trailing 24-bar return. Statistics: circular moving-block bootstrap, block 24
timestamps, 2000 resamples, for the mean RankIC and every Kronos-minus-benchmark
difference; Newey-West t as a cross-check with the lag stated.
