# gold-nyrange-forward — pre-registration

> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**

Written before any forward day. Commit this file and PRE_REGISTRATION.json unchanged; every pass and the scoring refuse otherwise.

Config digest `d52f0577d62c2752605a42bd1f626cf03e3ee474c942c3dc0a6f74cb5494131a`.

## The cell

- `gold_harx_vs_har_forward` in `results/gold_nyfwd/gold_nyfwd_hypothesis_log.csv`, alpha 0.05 (family size 1), one-sided.
- Statistic: mean over forward days of (HAR-X squared error - HAR squared error) < 0.
- Test: circular 5-day block bootstrap, 10000 resamples, seed 20261201. KEEP if the bootstrap upper (1 - alpha) bound of the mean difference is below 0, else DROP.
- Horizon: scored once on the first 60 forward days, as soon as they exist; with fewer on 2027-04-30, recorded UNDERPOWERED — NO DECISION. The horizon is never extended.
- A forward day is a settled ok day whose label date is after the commit date of PRE_REGISTRATION.json on the Berlin label clock.
- Target: y = log((max high - min low of the gold bars labelled 15:00-21:00 Berlin) / close of the 14:00 bar), bid prices; eligibility and target from gold_bbnr.ny_target unchanged.
- Dollar index: the USDInd contract with the nearest expiry at least 2 trading days after the day, lag-0 checked against EURUSD (lags [-3, -2, -1, 0, 1, 2, 3]); no qualifying contract = no dollar data, the day never counts and the horizon is not extended.

## Expected power — stated before any forward day

- Historical (gold_bbnr, 2211 days): mean difference -0.02755, block SE 0.003704, MSE 0.1522 vs 0.1798.
- At 60 days: SE_n = 0.003704 x sqrt(2211 / 60) = 0.02248; power = Phi(0.02755 / 0.02248 - 1.6449) = 0.337. **About 34%.**
- The test is weak by the owner's choice of horizon. A DROP is not evidence that the model fails; only a KEEP is a finding.

## The frozen model (copied from gold_bbnr, never refitted)

- Source: research_models/gold_bbnr/ols.json, SHA-256 `5c21860f75d0a63ef862162a15e70fc5ea3af3cd0f75abfed7f16c21ceb7848e`, fit on 2442 days 2017-01-31 .. 2026-11-20.
- Forecasts use these copied values. A pass refuses if ols.json exists and its digest differs.

| term | HAR-X | HAR |
|---|---:|---:|
| intercept | -0.6465131285 | -0.5984973538 |
| log_ny_range_lag1 | -0.009040317254 | 0.04956859747 |
| log_ny_range_ma5 | 0.1806640307 | 0.3622932547 |
| log_ny_range_ma20 | 0.4821614918 | 0.4633063983 |
| log_asia_range | 0.158780598 |  |
| log_london_range | 0.04302121989 |  |
| log_data_hour_range | 0.04360265874 |  |
| abs_data_hour_return | -2.169938388 |  |
| london_range_ratio | 0.07792549894 |  |
| data_hour_range_ratio | 0.1007803278 |  |
| dollar_pre_range_ratio | 0.02178025192 |  |
| day_of_week | 0.02409989832 |  |
| is_nfp_day | 0.07177342588 |  |
| is_fomc_day | 0.632744515 |  |
| dst_mismatch | -0.1606150471 |  |

## Facts recorded

- HAR-X and HAR are gold_bbnr's frozen OLS fits on every usable day up to 2026-11-20 (the range is in the model section; the hourly dollar index starts in 2017). gold-bb-nyrange was scored 2026-11-23: the historical cell gold_harx_vs_har was KEEP at 0.847 x HAR's MSE on 2 211 walk-forward days 2018-01-02 .. 2026-11-20.
- The frozen fit used all history, so the forward effect may be smaller than the walk-forward one. That is what this test measures.
- LightGBM and the MLP were DROP against HAR-X historically and are not forecast live.
- gold_bbnr's Bollinger cells were all DROP; nothing here concerns direction or a trade.
