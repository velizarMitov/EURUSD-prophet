# Retired: USDCHF_h1.csv and CHFUSD_h1.csv

Moved here 2026-08-04. **Not deleted, not edited** — sha256 verified identical
before and after the move, and `src/pooled_h1_data.py::load_pooled_h1` searches
this directory as a fallback so every completed program that consumed these
files still reproduces byte-for-byte.

## The corruption

```
missing 2026-06-15 06:00:00  ->  2026-07-28 00:00:00
        1026.0 hours = 42.8 days
```

`CHFUSD_h1.csv` is derived from `USDCHF_h1.csv` by inversion, so it inherits the
identical hole. Monthly bar counts across the gap:

| month | USDCHF / CHFUSD | EURUSD reference | |
|---|---:|---:|---|
| 2026-05 | 504 | 504 | ok |
| 2026-06 | **247** | 528 | 47% |
| 2026-07 | **16** | 551 | 3% |

## Why every count-based check passed

Both files report **70,000 bars** and a last timestamp of **2026-07-28**, which
look correct. An MT5 symbol that is not selected in Market Watch has only a
partially synced history block on disk; `copy_rates_from_pos` does not error, it
simply reaches **further back** to satisfy the requested bar count. So the total
stays right, the newest timestamp stays right, and the hole sits in the middle.

A bar count and a last timestamp are not evidence of completeness. The guard
that now catches this is `src/mt5_coverage.py` (interior-gap + monthly-density
checks, plus `sync_symbol()` prevention before every fetch).

## Retirement is an OWNER DECISION, not a withdrawal of any result

**No hypothesis-log row was deleted or edited.** In particular:

**H_dir.5** (`H_dir.5_replication_CHFUSD`, 2026-07-28) — **KEEP**, delta_acc
**+3.1063pp**, McNemar **p = 0.000006**, block CI [+0.014085, +0.048236],
alpha 0.008333, cleared_bar True — **is not affected by this corruption and
stands.** Its validation slice is rows 49,000–59,500 =
**2023-01-27 19:00 → 2024-10-04 10:00**. The gap begins 2026-06-15, **twenty
months later**, and sits at 99.977%–100% of the file's row index. The KEEP was
measured on intact data.

No future reader should assume this result was withdrawn for being wrong. It was
not withdrawn at all.

See `results/DATA_STATUS.md` for the full blast-radius assessment.
