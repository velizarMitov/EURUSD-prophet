# Gold results and corrections

The corrections document named by `PRE_REGISTRATION.md`. The pre-registration itself is
never edited.

## 2026-10-21 — the registration was not committed on its own

The registration was committed in `ab420d3` (2026-10-21 17:51 +0300, titled "Refactor code
structure for improved readability and maintainability"). VS Code Copilot made that commit
and pushed it. Task 5.2 required the registration to be committed alone. Instead, this
commit also carries:

- all of `src/gold/` and `tests/test_gold_*.py`;
- the OHLC caches;
- the openspec change;
- three EUR/USD `results/forward_eval/` files.

What still holds, checked the same day before any score:

- No validation score existed when it was committed. All three registries were empty, and
  no validation report existed.
- `scoring_allowed()` is True: both files are committed, unmodified, and ancestors of
  HEAD.
- The pinned D1 (3,466 rows) and H1 (70,015 rows) digests and the `gold_config.json`
  digest match the registration.

What is lost: a commit you can review as the registration alone. The pushed history is
not rewritten. Read `results/gold/PRE_REGISTRATION.*` at `ab420d3` as the registered
protocol.

## 2026-10-21 — validation verdicts (arbiter `validation[70:80]`, gross of cost)

One run (194 s). Each cell has exactly one row in its family registry.

| family | cell | result | verdict | registered expectation |
|---|---|---|---|---|
| volatility | d1_lstm | MAE 0.3887 % vs GARCH(1,1) 0.4222 %, CI [0.0150, 0.0503] | CLEARED | CLEARED ✓ |
| volatility | h1_lstm | MAE 0.0737 % vs GARCH-HOD 0.0814 %, CI [0.0058, 0.0096] | CLEARED | CLEARED ✓ |
| direction | h1_gbm | 51.04 % vs majority 50.74 %, CI [−0.0185, +0.0213] | DROP | possible CLEARED ✗ |
| direction | h1_lstm | 50.89 % vs majority 50.74 % | DROP | DROP ✓ |
| direction | d1_gbm / d1_lstm | 346 rows (43.9 % / 54.3 %) | UNDERPOWERED | UNDERPOWERED ✓ |
| return | all four | Clark–West p ≥ 0.0125 | DROP | DROP ✓ |

Nine of ten cells came out as registered. The H1 GBM direction edge seen on EUR/USD,
AUD/USD and CHF/USD does not carry over to gold on this slice: the point gain is +0.3 pp
against a CI of about ±2 pp. The D1 direction figures are descriptive only; 346 rows
cannot decide anything.

### Correction: Deflated Sharpe trial count

The first rendering of `validation_report.md` charged the descriptive Deflated Sharpe for
N = 1 trial. It was computed before the run appended its 16 GBM grid trials to
`trial_log.csv`. It has been recomputed from the saved per-row predictions
(`validation_predictions_{d1,h1}.csv`) with N = 16, and the per-trade Sharpe values match
the run exactly. DSR is now 0.14 (d1_gbm), 0.66 (d1_lstm), 0.17 (h1_gbm) and 0.14
(h1_lstm), down from 0.24, 0.78, 0.74 and 0.70. The code now computes it after the trials
are appended. No verdict depends on it.
