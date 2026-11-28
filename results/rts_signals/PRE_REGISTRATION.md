# rts-signals — pre-registration

> **Net of the recorded MT5 bar spread only; slippage, swap and commission are not modelled — not a trading result.**

Written before any forward trade. Commit this file and PRE_REGISTRATION.json unchanged; the ledger counts no forward trade and the scoring refuses otherwise.

Config digest `fe98518108a2bdd4a7dd38a3c525d0bd49fe2229766524a9571212e9c1ea57e7`. Rules digest `8e3226661f386a9f578b92ff19c428d4ad20432dbb4c9b4d7bff62cabf4c5daa` (src/rts_signals/rules.json).

## The cell

- `rts_spot_net_r` in `results/rts_signals/rts_signals_hypothesis_log.csv`, alpha 0.05 (family size 1), one-sided.
- Hypothesis: the mean net R of forward base trades of RTS spot recommendations is above 0.
- Statistic: sum of net R / number of trades over the first scored trades.
- Test: circular block bootstrap over business days keyed by report date, 5-day blocks, 10000 resamples, seed 20261202 (fx_bb.score.test_ratio, side 'greater'). KEEP only if the bootstrap lower alpha-quantile of the mean net R is above 0, else DROP.
- Horizon: scored once on the first 300 forward trades as soon as they exist; with fewer on 2027-06-30, recorded UNDERPOWERED — NO DECISION. Never extended.
- A forward trade is a filled base trade of a spot report dated after the commit date of PRE_REGISTRATION.json (Berlin label clock), received (Gmail arrival or drop-folder ingest) before its session ended; reports received later are ledgered `received late` and excluded.
- The base trade: src/rts_signals/simulate.py: Rule 0 with proportionality or Rule 1 guerilla, stop / target / time stop 16:00 New York (Friday 12:00), a bar touching both is the stop, M1 bid bars; net = gross less the recorded bar spread once (buy at entry, sell at exit); R = |entry - stop|.

## Power — stated before any forward trade

- Declared per-trade SD 1.0 R; about 4 trades a day with an assumed same-day correlation of 0.2 (design effect 1.60).
- SE = 1.0 R / sqrt(300) x sqrt(1 + (4 - 1) x 0.2) = 0.0730 R; MDE(80 %) = (1.645 + 0.842) x SE = 0.182 R per trade.
- **Minimum detectable mean net R at 300 trades: about 0.18 R.** A DROP means no detectable edge net of spread at this sample size, not proof of none.

## Scope

- Net of the recorded spread only, not slippage, swap or commission. No verdict is a trading instruction. The family never touches another family's registry, the production models directory or config.json.

## Facts recorded

- RTS's own November 2026 spot track record (105 closed trades, 2026-11-02 .. 2026-11-26) states +151 points gross. On the broker's prices 101 trades are confirmed and 4 show a different exit; less the recorded spread at each entry it is about -214 points. This was seen before the cell was designed and decides nothing.
- The base trades of the reports received before the commit (2026-11-23, -25, -27) were also seen: 8 filled. They are not forward trades.
- RTS enters Rule 0 trades at 18:00 New York, the broker's daily rollover, when the spread is widest.
- The Morning report's positions table is read by OCR and checked arithmetically; it is information only and never enters the cell.
- By the owner's decision of 2026-11-28 today's levels are shown on the chart before any verdict, labelled unverified.
