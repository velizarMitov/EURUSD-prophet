# Tasks

Before starting:
- Ask the owner to turn off VS Code Copilot auto-commit and auto-push (memory:
  project-copilot-auto-commits).
- Take a full-suite baseline (`python -m pytest -q`) and note any pre-existing failures.

Each group ends with no new failures against that baseline, and with `git status` showing
no change under `models/`, `config.json`, `results/forward_eval/`, any gold registry,
registration or manifest, or `src/gold/gold_config.json`.

## 1. Configuration

- [ ] 1.1 Add the top-level `gvz` block from design D9 to `src/gold/gold_config_v2.json`.
  Leave `study`, `alerts`, `retrain` and `paths` untouched.

  Verify with a new test in `tests/test_gold_gvz.py`:
  - `prereg2.study_digest(load_config_v2())` equals `config_v2_study_sha256` in
    `results/gold/PRE_REGISTRATION_V2.json`;
  - the SHA-256 of `src/gold/gold_config.json` equals the round-one pinned digest.

## 2. Acquisition and the read half (`src/gold/gvz.py`)

- [ ] 2.1 Implement the read half.
  - `read_series(cfg2)` merges `gvzcls.csv` (FRED) and `gvz_yfinance.csv` (top-up),
    keeping FRED for every FRED date and tagging each value with its source.
  - `reading(cfg2, now, gold_close, close_bar)` returns the D7 block: percentile from
    `history_start` up to and including the value's date, the implied move GVZ / √252 in
    % and $, and stale after `stale_after_days`.
  - Both touch files only, and the module has no top-level network import.

  Verify with unit tests on temporary caches:
  - FRED wins on a shared date, and yfinance fills only later dates;
  - GVZ 25.2 at a close of 4,000.00 gives ±1.59 % and ±63.50 $;
  - a value above 91 % of stored values reads percentile 91;
  - Monday with Friday's value is not stale, and Thursday with the previous Friday's is;
  - no stored value gives `state: none`;
  - with no gold close, the percent is present and the dollar value is `None`;
  - importing `src.gold.gvz` does not import `yfinance` (check `sys.modules`).
- [ ] 2.2 Implement `refresh(cfg2, now, fred_fetch=None, yahoo_fetch=None)` with
  injectable fetchers.
  - FRED goes through `src.macro_data.fetch_fred_feature` into `gvzcls.csv`, over the
    full window on the first run and the last 30 days afterwards.
  - yfinance `^GVZ` (`period='10d'`, imported inside the function) is merge-written by
    date into `gvz_yfinance.csv`.
  - Values dated on or after today's `America/New_York` date are dropped.
  - Return a short status string.

  Verify with tests using stub fetchers:
  - at 17:00 Sofia on 2026-10-27, a 2026-10-27 yfinance bar is not stored;
  - a FRED failure with a cache keeps the cached values and reports the failure;
  - when both fail, the function raises a clear error naming GVZ;
  - no test touches the network.

## 3. Logger integration (`src/gold/forward_logger.py`)

- [ ] 3.1 Add `log_gvz(cfg, now_iso, frames, ...)`, called in `run_once` right after
  `log_alerts` and wrapped like the other optional steps.
  - It refreshes only when the newest stored date is before the last completed US
    session (design D4).
  - On failure it appends a `failures.csv` row with cell `gvz`.
  - It never alters round-one, range, alert, shadow or v2 output.
  - Put its status in `status['gvz']`, and add a short `gvz:` segment right after
    `alerts:` in the `run_with_health` note (design D10).

  Verify with tests in `tests/test_gold_forward_logger.py` using stub frames and stub
  fetchers:
  - a fresh cache contacts no source at 14:03 Sofia on Tuesday;
  - when both sources raise, a `gvz` failure row and a `gvz: failed` note appear, and the
    pass's predictions and settlements are identical to a run without GVZ;
  - the health note still contains `alerts:` and `gvz:` within 300 characters.
- [ ] 3.2 Add `observe(...)` and `settle(...)` to `gvz.py`, and call them from
  `log_gvz`.
  - **Observe:** on the first Monday-to-Friday pass at or after 13:00 Europe/Sofia, append
    one row to `results/gold/gvz_log.csv` with the D6 columns. It takes the gold close from
    the last `frames['H1']` bar and the newest `h1_lstm` row from `predictions.csv`, and
    leaves blank any field unknown at that moment.
  - **Settle:** for each logged date without a settlement, once `frames['H1']` holds a bar
    labelled after that date's 21:00, append one row to `results/gold/gvz_settlements.csv`.
    The row holds the range of the bars labelled 12:00–21:00 as a percent of the first
    open, and the bar count, with the range blank below 9 bars.

  Verify with tests:
  - passes at 12:03, 13:03 and 14:03 Sofia write exactly one row, at 13:03;
  - Saturday writes none;
  - a missing h1_lstm forecast leaves the fields blank, and later passes do not fill them;
  - 10 bars settle with a range, 7 bars settle blank with count 7, and nothing settles
    at 20:03 Sofia;
  - `gvz_log.csv` is byte-identical before and after settlement.
- [ ] 3.3 Verify the logs are never scored: add a test that runs `log_gvz` over a
  simulated month and asserts that every file under `results/gold/` named in
  `cfg['paths']['registries']`, `PRE_REGISTRATION*.json`, `artifact_manifest*.csv` and
  `leaderboard` output is byte-identical. Add another asserting that
  `src/gold/forward_report.py` and `src/gold/leaderboard.py` never reference `gvz`.

## 4. API (`src/gold/summary.py`)

- [ ] 4.1 In `build()`, add the `gvz` block through `gvz.reading(...)` (design D5, D7).
  - The gold close and bar come from the newest H1 row in `predictions.csv`.
  - `lstm_vol_pct` and `lstm_as_of_bar` come from the current `h1_lstm` entry.
  - It sits inside `try/except`, so an error yields `{"state": "none"}`.

  Verify with tests in `tests/test_gold_summary.py`:
  - the block is present and correct from temporary caches;
  - the existing `test_build_writes_nothing` still passes, with GVZ caches present;
  - `test_summary_imports_no_model_library` still passes, and `yfinance` is not in
    `sys.modules` after `build()`;
  - a corrupt GVZ cache yields `state: none` with the rest of the payload unchanged.
  - The `tests/test_gold_isolation.py` suite stays green.

## 5. The card (`static/gold.html`)

- [ ] 5.1 Add `<div id="gvz">` at the top of `#panel-now` and `renderGvz()`, called from
  `render()` and on `ps:lang`. Add the BG/EN strings to `I18N` with the layout and states
  of design D8: ok, stale (the warning line uses the existing `stall` colour) and none.
  The card has no direction arrows and no trade or cost words.

  Verify:
  - `tests/test_gold_view.py`: `/gold` returns 200 and contains `id="gvz"`, and the
    existing `COST_WORDS` test still passes;
  - a new test asserts that the GVZ strings in both languages contain none of `buy`,
    `sell`, `stop`, `entry`, `target`, `купи`, `продай` or `стоп`.
- [ ] 5.2 Check the card in the running app: restart via the launcher, open `/gold` in
  light and dark themes at phone and desktop widths, switch BG/EN, and stub a stale cache
  once to see the warning. Record the check in this task.

## 6. Documentation

- [ ] 6.1 Add a short GVZ morning-check paragraph to `ARCHITECTURE_DOCS.md §3.12` (gold).
  It covers:
  - the sources and the cache files;
  - the logger step;
  - the API block;
  - the observation and settlement logs;
  - "descriptive, never scored".

  Add one line to the gold item in `CLAUDE.md`. Verify that both name
  `results/gold/gvz_log.csv` and say the logs are never scored.

## 7. Integration

- [ ] 7.1 Run the full suite `python -m pytest -q`. Verify there are no new failures
  against the baseline and that `git status` shows only the files this change names.
- [ ] 7.2 Run one live logger pass (`python -m src.gold.forward_logger`). Verify:
  - `results/gold/macro/gvzcls.csv` holds history from 2013;
  - `gvz_yfinance.csv` holds only dates after FRED's newest;
  - the health row's note has a `gvz:` segment;
  - `/api/gold` carries a `gvz` block with today's served value and its date.
