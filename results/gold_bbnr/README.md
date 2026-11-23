# gold-bb-nyrange — Bollinger Bands and the New York session range on gold

> **Gross of cost on bid prices (no spread, slippage or swap) — not a trading result.**

Openspec change `gold-bb-nyrange`. Code `src/gold_bbnr/`, tests `tests/test_gold_bbnr_*.py`.

## Three parts, three registries, seven cells

| Part | What | Registry | Alpha per cell |
|---|---|---|---|
| 1a | Bollinger (20, 2 SD) first breaches on NY-close D1 bars; 3-day reversion in ATR; LightGBM vs base rate | `gold_bb_hypothesis_log.csv` | 0.05/2 |
| 1b | The same breaches as a trade: TP at SMA20, symmetric stop, time stop t+10, H1 exit scan, R; LightGBM vs base rate | `gold_bb_sma_hypothesis_log.csv` | 0.05/2 |
| 2 | log(range of the bars labelled 15:00–21:00 Berlin / close14): HAR-X vs HAR, LightGBM vs HAR-X, MLP vs HAR-X | `gold_nyvol_hypothesis_log.csv` | 0.05/3 |

Every cell is scored once, and **only after `PRE_REGISTRATION.md` and `.json` are committed** unchanged.
`prereg.gate` checks git, the config digest and the digests of the three input caches.

## Data

- **Gold:** `results/gold/gold_h1.csv`, read-only. It holds 70 527 H1 bars from 2015-01-02 to 2026-11-20, which give 3 068 valid NY-close days.
- **Hourly DXY:** MT5 ActivTrades `USDIndDec26`, fetched once into `data/dxy_h1.csv`. It holds 53 046 bars from 2017-01-03.
  - In every year, its returns correlate with EURUSD's most strongly at lag 0 (−0.90 to −0.95).
  - It has one 10-day hole, in June 2024.
  - Only one day opens more than 1 % away from the previous close: 2025-02-03, a weekend tariff gap.
- **Daily DXY:** Yahoo `DX-Y.NYB`, fetched once into `data/dxy_d1.csv`. It holds 3 140 days, and only closes dated before the gold day are used.

## Power, from `labels.md`, before any outcome

| Cell | Sample | Minimum detectable effect (80 % power) |
|---|---|---|
| 1a rule | 173 events | mean r 0.257 ATR |
| 1a model | 106 events (folds from 2020; the 2018–2019 folds have < 60 training events) | accuracy 0.636 |
| 1b rule | ≤ 179 trades | mean R ≥ 0.209 R |
| 1b model | ≤ 110 trades | accuracy ≥ 0.634 |
| 2 (each) | 2 211 days, 2018–2026 | 0.063 SD of the squared-error difference |

**The Bollinger cells can see only a large effect.** Gold is one instrument, with 180 first breaches in 12 years. A DROP there means "not visible", not "no effect".

## Stages

```
python -m src.gold_bbnr.dxy --mt5-path <terminal64.exe>    # once, done
python -m src.gold_bbnr.dxy --yahoo                         # once, done
python -m src.gold_bbnr.labels
python -m src.gold_bbnr.prereg --write                      # then the owner commits both files
python -m src.gold_bbnr.walk_forward --part bb|bb_sma|nyvol # gated
python -m src.gold_bbnr.score --family bb|bb_sma|nyvol      # gated, once each
python -m src.gold_bbnr.frozen                              # gated; Part 2 refit
python -m src.gold_bbnr.report
```
