# Kickoff prompt — paste into Claude in VS Code

---

We are executing **Step 1 of Idea 2** in this repo: measuring discrete-Hodge "curl" on the
FX currency graph to decide whether a gauge-equivariant architecture is worth building.
This is a **measurement study, not a model**. Nothing may touch `src/inference.py`,
`models/`, or the serving path.

## Read first, in this order

1. `CURL_EXPERIMENT_PLAN.md` — the theory, the protocol, and the STOP GATE
2. `src/curl_stress.py` — core estimator (already written and unit-tested)
3. `src/curl_mt5.py` — real-data runner (already written; `run_day1` is the entry point)
4. `CLAUDE.md` → *Production Methodology*, and `ARCHITECTURE_DOCS.md` → *Production
   Methodology* — the rules that govern any new claim in this project

Do not restate these back to me. Read them and follow them.

## Non-negotiable context

- Daily EUR/USD is near-efficient. Nine direction-family hypotheses have all failed to
  clear their bar. The only family that ever cleared (volatility) turned out on
  verification to be a **Friday weekend-gap calendar effect** that a 6-number day-of-week
  lookup table beats outright. Assume by default that anything exciting is an artifact.
- The theory here says the observed curl is **pure measurement noise** (`w = d(phi)` is
  exact, so the true curl is identically zero). We are testing whether the *amplitude* of
  that noise, after normalising by what asynchronous sampling alone predicts, carries
  information. A "signal" in the raw curl is a bug, not a discovery.
- The historical test block `[80%:100%]` is **spent**. The arbiter is `validation[70:80]`.
- This is a **new hypothesis family**. `results/curl_hypothesis_log.csv` does not exist
  yet. Hypothesis #1 is judged at `alpha = 0.05`. Do not touch
  `results/feature_hypothesis_log.csv`.

## Task 1 — data pull (`src/curl_mt5_fetch.py`, new file)

Pull **M1** bars for the six pairs of the EUR/USD/JPY/GBP subgraph: EURUSD, EURJPY,
EURGBP, GBPUSD, GBPJPY, USDJPY.

- Use `mt5.copy_rates_range` and **keep `tick_volume` and `spread`**, not just OHLC. Both
  are load-bearing: `tick_volume` is the staleness estimator and `spread` is the control
  the index must beat.
- Resolve symbol names via `mt5.symbols_get()` — brokers add suffixes (`.a`, `m`, `_raw`).
  Do not hard-code.
- Convert `time` to UTC. Cache to `results/curl/raw/<SYMBOL>_M1.parquet` so we never
  re-pull.
- Report, per symbol: row count, first/last timestamp, and the count of gaps larger than
  one bar. MT5 M1 history is usually shallower than the daily history — tell me what we
  actually got before doing anything with it.

Follow the existing project conventions for MT5 access (see `src/live_data.py`,
`src/mt5_coverage.py`). Set `PYTHONIOENCODING=utf-8` when printing.

## Task 2 — run the day-one battery

```python
from src import curl_mt5 as m5
res = m5.run_day1(raw_frames, outdir="results/curl")
```

Then **interpret the output for me, in this order**, and stop at the first failure:

1. **`tick_volume` audit + `staleness_exponent_test`.** I do not know whether my broker's
   tick counts are real. If `pct_at_max` shows a ceiling, or `n_distinct` is tiny, or the
   permutation `z_vs_shuffled` is not clearly negative, then the tick counts carry no
   staleness information and the whole null degenerates. Say so plainly and stop.
2. **`convention_report`.** Judge `resid_bp` (the offset the bid convention does *not*
   explain), not `mean_curl_bp`. A large residual means an inverted pair.
3. **`cycle_rank` must be 3.**
4. **Frequency scaling.** Report the log-log slope. Flat ⇒ asynchronous sampling. Slope ≈ 1
   ⇒ a pair effectively printing once per bar, which is a data-quality finding.
5. **Curl autocorrelation.** Persistent positive ⇒ stale/frozen feed ⇒ audit before
   believing anything.

## Task 3 — the STOP GATE

`CURL_EXPERIMENT_PLAN.md` §2.3 defines it. Apply it honestly. If the curl is flat across
timeframes, consistent with the binned null, and its remaining variation is explained by
hour-of-day + day-of-week + tick count, then **write it up as a negative result and stop.**
That outcome costs no hypothesis slot and is the expected one. Do not go looking for a way
to keep the idea alive.

## Task 4 — only if the gate is passed

Write `results/curl_hypothesis_log.csv` with the same columns as
`results/volatility_hypothesis_log.csv`, and pre-register **in the log, before scoring**:

- Primary: incremental correlation of the daily-aggregated `stress_orthogonal` with the
  residual of the frozen 5-seed volatility ensemble, on `validation[70:80]`, moving-block
  bootstrap (`block_len=5`, `n=2000`), `alpha = 0.05`.
- Register `stress_orthogonal` (confound-residualised), **not** `stress`. On synthetic data
  where the null is true by construction, the raw index still correlated +0.40 with tick
  count and −0.35 with volatility and spread.
- One horizon, one target. Any additional one is another hypothesis and tightens the bar.

## Working rules

- Add tests to `tests/test_curl_stress.py` for anything new. All 17 currently pass.
- Every fitted quantity — null coefficients, calendar means, confound coefficients — is fit
  on `[0:70%]` only.
- No centred rolling windows anywhere.
- Show me the numbers before the narrative. If a result looks good, your first move is to
  try to break it.

Start with Task 1 and show me the coverage report before going further.
