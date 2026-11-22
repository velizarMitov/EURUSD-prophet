# fx-bb-sma-exit — Bollinger first breach, exit at SMA20 / symmetric stop / 10-day time stop

> **Net of the recorded bar spread only — slippage, swap and commission are not included; not a trading result.**

Openspec change `fx-bb-sma-exit`. Code `src/fx_bb_sma/`, tests `tests/test_fx_bb_sma_*.py`, docs
`ARCHITECTURE_DOCS.md §3.19`.

## Result (scored once, 2026-11-22): both cells DROP

| Cell | Question | Trades (dates) | Statistic | One-sided bound | Verdict |
|---|---|---|---|---|---|
| `sma_exit_rule_net_r` | mean net R per trade > 0 | 1 089 (821), 2012-02..2026-11 | −0.0082 R | 2.5 % quantile −0.0677 | **DROP** |
| `sma_exit_lightgbm_vs_base_rate` | pooled LightGBM log-loss below the base rate | 905 (661), 2015-01..2026-11 | −0.0034 (0.69113 vs 0.69452) | 97.5 % quantile +0.0028 | **DROP** |

alpha = 0.05/2 per cell, tested with a circular block bootstrap over entry business days:
10-day blocks, 10 000 resamples, seed 20261125. The minimum detectable effects were stated
in `labels.md` before any model:
- Cell 1: about 0.085 R on trades and 0.098 R on dates.
- Cell 2: about 54.7 % on trades and 55.4 % on dates.

What the trades did (descriptive, from `walk_forward_report.md`):
- **Exits.** 32.2 % at the take profit, 32.2 % at the stop and 35.5 % at the 10-day time
  stop.
- **Wins.** 49.2 % of trades won after spread.
- **Size.** The average win was +120.8 pips net, against an average loss of −126.7.
- **Profit factor.** 0.937 gross and 0.924 net.
- **Total.** −2.0 R gross and −9.0 R net over 1 089 trades.
- **Take profits and stops cancel.** Take profits average +1.000 R and stops −1.002 R (gap
  fills), and the time stops average about 0. So the rule is as likely to reach the stop
  as the mean.
- **No conflicts.** No H1 bar touched both levels, because they are about 2 R apart.
- **Bid/ask control.** −0.0074 R, a gap of 0.0008 R. Within tolerance.
- **Pairs and years scatter both ways.** USDCAD −0.10 R and AUDUSD +0.05 R; the years
  range from −0.26 R (2020) to +0.26 R (2019). None was tested.
- **The model.** Its slightly lower log-loss is within noise. The P(win) > 0.5 subset was
  chosen after the model. **Neither is an edge.**

**These are the same first-breach events as fx-bb-reversion** (DROP on 2026-11-22, mean
+0.014 ATR at 3 days). This family was designed after that result, and both registry notes
say so.

## Files

| File | What it is |
|---|---|
| `labels.md`, `labels.json` | trades, skipped breaches, exclusions and the power table, before any model; exit times only, no exit reason or result |
| `fx_bb_sma_hypothesis_log.csv` | the two verdict rows |
| `cell1_trades.csv.gz` | every scored trade (entry, take profit, stop, exit, reason, pips, R, bid/ask control), written when Cell 1 was scored |
| `oos_trades.csv.gz`, `run.json` | Cell 2's out-of-sample trades with `p_prior` and `p_lightgbm`; the folds and the six cache digests |
| `artifact_manifest.csv` | the digests of the frozen fold models in `research_models/fx_bb_sma/POOLED/` |
| `walk_forward_report.md` | the report, with the plain-language Bulgarian section and five example trades |

Data is read-only from `results/fx_bb/data` and `results/fx_session/data`, through fx_bb's
loader and checks. This family never refreshes it.

Stages, in order:
1. `python -m src.fx_bb_sma.labels`
2. `python -m src.fx_bb_sma.walk_forward`
3. `python -m src.fx_bb_sma.score --cell rule`, then `--cell model`
4. `python -m src.fx_bb_sma.report`

Every stage refuses a config whose digest differs from the one recorded by an earlier stage
or verdict.
