# KRONOS — RankIC ON A FOREX CROSS-SECTION: pre-declared readings

**Written 2026-08-03, BEFORE the first window of this program was generated.**
Nothing below may be softened after a result is seen.

## What the paper actually reports (extracted before designing the test)

From arXiv 2508.02739, Table 14 (price series forecasting, Part 1) — **Forex row**:

| | Kronos_S | Kronos_B | Kronos_L | best baseline on Forex |
|---|---|---|---|---|
| IC | 0.0279 | 0.0292 | 0.0244 | NSTransformer 0.0228 |
| **RankIC** | **0.0177** | **0.0141** | **0.0137** | TimesNet 0.0175, DLinear 0.0168 |

Average over all 8 datasets: Kronos_L RankIC **0.0267**, Kronos_B 0.0258, Kronos_S 0.0254.
Best non-pre-trained baseline TimesNet **0.0143** → +87%. Best zero-shot TSFM
(Table 15) **0.0138** → +93%. Both headline numbers reproduce exactly.

**Three caveats recorded now, not after the fact:**

1. The Kronos columns of Table 14 sit under *Full-shot Time Series Models* —
   the headline numbers are **fine-tuned per dataset**. Fine-tuning is out of
   scope for this program, so we are testing **zero-shot Kronos-base** against a
   fine-tuned target.
2. The paper's *price series forecasting* RankIC is **not cross-sectional**.
   Appendix, Metric Calculation Details: "For each sample, the IC and RankIC are
   calculated between the predicted and true series for each of the four price
   channels (Open, High, Low, Close). The final reported metrics are the average
   across these four channels." That is a rank correlation **along the forecast
   path**, per sample. The cross-instrument RankIC this program computes is the
   Qlib/CSI300 convention used by their fine-tuning pipeline, not the convention
   behind the 0.0177.
3. The paper's horizon for 1-hour bars is **H = 12**, look-back 80 (Table 8).
   This program's fixed parameters use pred_len = 1.

Target magnitude to look for: **0.0137–0.0177** for the paper's own Forex
configuration; the industry reference band 0.02–0.05 is the "useful" bar.

## Fixed parameters (unchanged from every previous round, not tuned)

T = 1.0, top_p = 0.9, sample_count = 30, context = 512, pred_len = 1.
Clean window 2024-07-01 onward, whole context inside it. Seed **20260803**.

## STEP 5 — how this will be read

* **Kronos RankIC significantly above zero AND above momentum/reversal** → the
  model transfers to FX on its own task, the harness is validated, and the
  earlier directional null is confirmed as a property of the task rather than a
  bug. This would be the first positive external result in this project.
* **RankIC indistinguishable from zero, but momentum or reversal IS
  distinguishable** → the harness works and Kronos adds nothing on FX.
* **Nothing is distinguishable from zero, including the simple rankings** → the
  harness or the cross-section is inadequate; investigate before concluding
  anything about Kronos.

## Cross-section (fixed before generation)

19 instruments, all MT5, all resolved exactly, no substitution from any other
provider. Quote convention FOREIGN/USD throughout; USDJPY/USDCHF/USDCAD inverted
with the high/low swap. Timestamp grid = strict intersection, 13,002 bars,
0.0% loss, never forward-filled.

**Effective dimensionality, recorded before any RankIC is computed:** 6 principal
components explain 90% of contemporaneous H1 return variance, 7 explain 98.7%,
and the 8th eigenvalue is 0.008. This is not incidental — 19 pairs built from 8
currencies (EUR USD GBP AUD NZD JPY CHF CAD) span at most **7 independent
dimensions** by triangular construction. The cross-section is mathematically
rank-7 regardless of how many pairs are added. Every RankIC number below must be
read against that bound: ranking 19 instruments that carry 7 degrees of freedom
is a far weaker test than ranking 300 stocks.
