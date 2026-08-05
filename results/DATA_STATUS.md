# DATA STATUS — MT5-sourced files

Sibling annotation. **No hypothesis-log schema is changed and no log row is
deleted or edited.** Audit run 2026-08-04; guard is `src/mt5_coverage.py`.

## Audit verdict

| file | timeframe | bars | span | verdict |
|---|---|---:|---|---|
| **`results/eurusd_h1.csv`** (live cache, feeds the production H1 ensemble) | H1 | 60,152 | 2016-12-01 → 2026-08-04 | **CLEAN** |
| `results/eurusd_m15.csv` | M15 | 350,000 | 2012-06-25 → 2026-07-24 | **CLEAN** |
| `results/pooled_h1/EURUSD_h1.csv` | H1 | 70,000 | 2015-04-27 → 2026-07-28 | **CLEAN** |
| `results/pooled_h1/GBPUSD_h1.csv` | H1 | 70,000 | 2015-04-27 → 2026-07-28 | **CLEAN** |
| `results/pooled_h1/AUDUSD_h1.csv` | H1 | 70,000 | 2015-04-27 → 2026-07-28 | **CLEAN** |
| `results/pooled_h1/EURUSD_h1_newyork.csv` | H1 | 70,000 | 2015-04-27 → 2026-07-28 | **CLEAN** |
| `results/pooled_h1/EURUSD_m15_newyork.csv` | M15 | 350,000 | 2012-06-25 → 2026-07-24 | **CLEAN** |
| **`retired/USDCHF_h1.csv`** | H1 | 70,000 | 2015-03-16 → 2026-07-28 | **HOLED** |
| **`retired/CHFUSD_h1.csv`** | H1 | 70,000 | 2015-03-16 → 2026-07-28 | **HOLED** |

Every other >72h gap in every file is a **market-wide year-end closure**,
confirmed by all peer instruments missing the identical span. Measured closures
run 73.0–77.0 h, which is why the guard applies a wider allowance across the
turn of the year rather than a flat 72 h bound.

## The hole

**Affected instrument: CHF only.** `USDCHF_h1.csv` is missing
**2026-06-15 06:00 → 2026-07-28 00:00 (1026.0 h = 42.8 days)**;
`CHFUSD_h1.csv` is derived from it by inversion and inherits the identical gap.
Cause: an MT5 symbol not selected in Market Watch returns a partially synced
history block, and `copy_rates_from_pos` reaches further back to satisfy the
requested bar count — so 70,000 bars and a correct last timestamp coexist with
six missing weeks.

## Blast radius — computed, not assumed

The gap sits at **99.977%–100%** of the file's row index (69,984 of 70,000 bars
precede it). Every completed result that consumed these files used a `[70:85]`
validation slice ending in October 2024, twenty months earlier.

| result | verdict in log | slice used | gap intersects | status |
|---|---|---|---|---|
| **H_dir.5** `_replication_CHFUSD` | **KEEP** +3.1063pp, McNemar p=6e-06 | rows 49,000–59,500 = **2023-01-27 19:00 → 2024-10-04 10:00** | **No** | **UNAFFECTED — stands** |
| **H_pool.1** GBM pooled vs EURUSD | DROP | common index [70:85] = 2023-02-10 → 2024-10-11 | **No** | UNAFFECTED |
| **H_pool.2** LSTM pooled vs EURUSD | DROP | common index [70:85] = 2023-02-10 → 2024-10-11 | **No** | UNAFFECTED |
| H_macro.1 / H_macro.2 (G10 panel) | REGISTERED-UNSPENT | — | — | **different data path** — FRED `DEXSZUS`, never MT5 |
| H_tA.1 / H_tA.2 (Tier-A carry) | DROP | monthly G10 panel from 1976 | — | **different data path** — FRED, never MT5 |

The CHF hole removes 737 bars from the pooled common index, all of them after
2026-06-15 and therefore outside every slice any completed result indexed.

**No re-run is required for any result.**

## Retirement of CHF — an owner decision

USDCHF and CHFUSD are retired from **active** use at the owner's request. This
is **not** a data-driven withdrawal: H_dir.5's KEEP was measured on intact data
and is unaffected, as computed above.

* The two CSVs are **moved, not deleted**, to `results/pooled_h1/retired/`;
  sha256 verified identical across the move, and the protected-set fixture entry
  was repointed to the new path with the same digest.
* `src/pooled_h1_data.py` gains `ACTIVE_INSTRUMENTS` / `ACTIVE_PAIRS`, which
  exclude CHF and are now the default for `fetch_pooled_h1()`. The historical
  `POOLED_INSTRUMENTS` / `POOLED_PAIRS` tuples are **left intact** — they
  describe H_pool.1/H_pool.2 as they were actually run, and the protected
  `src/pooled_h1_model.py` imports them. Rewriting them would silently redefine
  a finished result.
* `load_pooled_h1()` falls back to `retired/`, so the completed programs still
  reproduce.

**Exception, deliberate:** CHF is **kept** in the Kronos RankIC cross-section
(`results/external_kronos/rankic/raw/`). That data was fetched separately *with*
the Market Watch sync fix and audits clean at 0.0% intersection loss. Dropping it
would take the cross-section from 8 currencies to 7 and the rank bound from 7 to
6, raising the minimum detectable RankIC from 0.0130 to 0.0140 — from below the
paper's target to level with it.

## Protected-set re-baseline, 2026-08-04

The four `tests/fixtures/*_protected_sha256.json` fixtures were re-baselined for
**17 `models/` entries each**. This was a deliberate, owner-approved action, not
silent drift.

**Cause.** A retrain triggered through the running server at 19:09 was partway
through rewriting `models/` when it was mistaken for a test side effect and
partially reverted with `git checkout`, leaving the artifact set **torn** — some
files from the old commit, some from the new run. That violates the invariant
that a variant's scaler, PCA and both models come from one run, and it would have
left the 5-seed volatility ensemble partially replaced (`vol_ready` is
all-or-nothing over that mean). The remedy chosen was a clean end-to-end
`_train_pipeline.py` run.

**Gate before re-baselining.** Re-baselining a torn set would certify the tear as
correct and make it permanently invisible, so the set was verified first on two
independent axes:

* **temporal** — all 14 per-variant artifacts, all 5 volatility seeds and their
  PCA/scalers written inside one 20.4-minute window (20:53:30 → 21:13:55), zero
  stale files;
* **structural** — `baseline` 23 features and `with_macro` 27 across
  `global_scaler`, both GBMs and the LSTM input; `lag_scaler` width matching
  `lag_pca` input; LSTM timesteps matching `lstm_time_steps` in both variants.

Both passed. Drift was confined to `models/`; the re-baseline script refuses to
touch any non-`models/` entry.

**Not affected.** `models/h1_direction/` (H_dir.1) is **not** produced by
`_train_pipeline.py` and did not drift. Its only writer is
`src/train_h1_direction.py`, which is imported solely by tests — nothing in
`api.py`, `src/inference.py` or the pipeline retrains it, and the served H_dir.1
does not change on any schedule.

**Measured, not assumed:** hashing every file under `models/` before and after a
full `pytest` run showed **the test suite changes nothing** under `models/`.
