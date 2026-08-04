# Every Kronos result in this directory dated before 2026-08-03 ran at `pred_len=1`

**The authors never do.** Their live demo (shiyu-coder.github.io/Kronos-demo)
runs 360-bar context, `pred_len=24`, `sample_count=30`, and publishes exactly two
derived numbers — *Upside Probability (Next 24h)* and *Volatility Amplification
(Next 24h)*. Their README example uses `lookback=400, pred_len=120`. Nothing in
the repository, the demo or the paper uses `pred_len=1`.

**Mechanism, which is why these are void rather than merely different.** At
`pred_len=1` the model takes ONE autoregressive step, so 30 samples are 30 draws
from a single-step categorical distribution over a coarse token vocabulary. That
structurally produces the ~7.6 distinct closes out of 30 and the 0.61× under-
dispersion these files record. At `pred_len=24` the paths compound over 24 steps
and the endpoint distribution is a convolution of 24 draws — what the demo's fan
chart shows. The collapse diagnosed in these files is a property of the
configuration, not of the model.

## Affected artifacts — superseded by configuration, retained for the record

| artifact | what it measured at `pred_len=1` |
|---|---|
| `kronos_clean_window_pup.csv`, `kronos_clean_window_result.json` | EURUSD H1 next-bar direction, 12,454 windows (already withdrawn on 2026-08-03 after the positive control failed) |
| `control/kronos_alibaba_pup.csv`, `control/kronos_btc_pup.csv` | positive controls, Alibaba 5-min and BTC 1h |
| `daily/kronos_m15_pup.csv`, `daily/kronos_m30_pup.csv`, `daily/kronos_d1_pup.csv` | four-frequency dose-response |
| `kronos_control_and_frequency.json` | scored output of the two above |

**No conclusion about tokenizer resolution, and no conclusion about Kronos on
FX, can be drawn from any of them.** Nothing here is deleted; nothing here is
edited. The replacement program is `pred24/`.

## Live-serving mismatch, recorded so it is not forgotten

`src/external/kronos/loader.py` sets `PRED_LEN = 1`, so the served
`/api/kronos-direction` endpoint and the forward ledger
(`kronos_ledger.csv`, `kronos_prediction_log.csv`) still generate at
`pred_len=1`. This program deliberately does **not** change that — a serving
change follows the result, not the other way round.
