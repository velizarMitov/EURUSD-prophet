# Tasks

## 1. Isolated runtime and pinned weights

- [x] 1.1 Create `requirements-tsfm.txt` (exact pins: `chronos-forecasting`, `timesfm[torch]`,
  `peft`, CUDA 12.8 `torch`) and `scripts/external_tsfm/install.ps1`, which creates
  `.venv-tsfm`, installs it and reports whether the GPU is visible (design D1). Add `.venv-tsfm/` and `research_models/external_tsfm/` to
  `.gitignore`. Verify: the script runs, `git diff requirements.txt` is empty and the main
  `.venv` `pip freeze` is unchanged.
- [x] 1.2 Create `src/external/tsfm/vendor.py` (design D2):
  - per-model repo id, revision and package version;
  - `--fetch` into `research_models/external_tsfm/<model>/`;
  - the SHA-256 manifest `results/external_tsfm/artifact_manifest.csv`;
  - `verify(model)`, which raises on a missing or altered file.

  Verify with `tests/test_external_tsfm_vendor.py`: an altered file and a missing file are
  refused, and an unchanged file passes.
- [x] 1.3 Create `src/external/tsfm/worker.py`. It reads JSON on stdin and writes JSON
  deciles on stdout, for both models, both input modes, zero-shot or a verified
  fine-tuned adapter, contexts up to the model's maximum, batched windows,
  `HF_HUB_OFFLINE=1`, GPU when free and CPU otherwise, and imports nothing from the repo. Add the caller
  `src/external/tsfm/client.py`, which applies the 120 s timeout and turns errors into
  failures.

  Verify:
  - the install script's smoke test returns 9 increasing deciles for a synthetic series
    from each model;
  - a test with a fake runner covers a timeout and a crash, each yielding a failure, not
    an exception.
- [x] 1.4 Add the isolation tests (`tests/test_external_tsfm_isolation.py`):
  - the production modules never import `src.external.tsfm`;
  - `worker.py` imports no repo module;
  - the package names no write target under `models/`, `results/gold/`,
    `results/forward_eval/` or `results/horizon_study/`.

  Verify that they pass and that each fails on an injected violation.

## 2. Quantiles to size and P(up)

- [x] 2.1 Create `src/external/tsfm/forecast.py` (design D3):
  - the piecewise-linear CDF through the deciles, with linear extension and a [0, 1]
    clip;
  - the size as median |R| by bisection;
  - P(up) = 1 − F(0), clipped to [0.1, 0.9];
  - the closes → returns mapping.

  Verify with `tests/test_external_tsfm_forecast.py`:
  - the spec scenarios: symmetric with q25/q75 at ±0.10 gives size 0.10 and P(up) 0.50;
    zero at the 40th percentile gives P(up) 0.60;
  - a heavily skewed case;
  - size grows as the deciles widen;
  - the closes mode matches the returns mode on the same distribution.

## 3. Fine-tuning on the full history (owner's request, 2026-11-02)

- [x] 3.1 Create `src/external/tsfm/finetune.py --freeze` (design D10). It writes each
  instrument's entire H1 history before 2025-11-01 (gold from `gold_h1.csv`, EUR/USD from
  the deepest MT5 read) to `results/external_tsfm/finetune/<instrument>_h1_history.csv`,
  with digests, and refuses any bar on or after 2025-11-01.

  Verify:
  - a test that an injected 2025-11-01 bar makes the freeze refuse;
  - the files exist and report their first and last bars and row counts (gold about
    11 years, EUR/USD about 15; corrected at implementation from the planning estimates
    of 16 and 10).
- [x] 3.2 Add `finetune.py --train`, run in `.venv-tsfm` on the GPU: LoRA per model and
  instrument.
  - **Stage 1:** train on bars before 2024-11-01 and score the holdout year every 100
    steps, up to 3,000 steps.
  - **Stage 2:** refit from the base weights on every bar before 2025-11-01 for the chosen
    step count, with seed 42.
  - **Outputs:** the adapters go to `research_models/external_tsfm/finetuned/` and their
    digests to the manifest. `finetune_meta.json` records the steps, learning rate, loss
    curves, versions and GPU.
  - **Failure:** a fine-tune that fails, or never beats zero-shot on the holdout, is
    recorded and dropped.

  Run it for the 4 model × instrument pairs. Verify:
  - a test that the training loader only ever reads the frozen history files;
  - a reload of each adapter reproduces its holdout quantiles exactly;
  - `finetune_meta.json` lists all 4 pairs as trained or failed, with the reason.

## 4. Clean-window replay and baselines

- [x] 4.1 Freeze the replay data (design D4):
  - gold from `results/gold/gold_h1.csv`, and EUR/USD from a fresh MT5 read through
    `src/forecast_eval/mt5_reader`;
  - the window `results/external_tsfm/replay/<instrument>_h1_window.csv` with digests;
  - any request before 2025-11-01 is refused.

  Verify: a test of the refusal message, and that both files exist with recorded digests.
- [x] 4.2 Fit the frozen EUR/USD GARCH(1,1) × hour-of-day baseline on bars from
  2020-11-01 to 2025-10-31, with gold's `h1_garch_hod` estimator imported read-only.
  Save it under `research_models/external_tsfm/baseline_eurusd/` with its digest (design
  D5). Verify with a test that the fit sees no bar on or after 2025-11-01, and that a
  reload gives identical forecasts.
- [x] 4.3 Create `src/external/tsfm/replay.py`:
  - the {zero-shot, fine-tuned} × {returns, closes} × {2,048, long} grid per model and
    instrument, without the fine-tuned entries of a failed fine-tune;
  - the choice by lowest first-half MAE;
  - the descriptive second half, plus the baseline and gold h1_lstm errors;
  - `results/external_tsfm/replay/REPLAY.md` and `replay.json`, labelled descriptive.

  Run it on the GPU. Verify:
  - the files list all eight first-half errors per model and instrument (fewer only where
    a fine-tune failed), and the chosen configuration;
  - a test shows that a run after `PRE_REGISTRATION.json` exists keeps the frozen choice.

## 5. Forward logger

- [x] 5.1 Create `src/external/tsfm/forward_logger.py` (design D6):
  - its own lock;
  - gold via `src/gold/data.fetch_live` plus its clock check, and EUR/USD via
    `src/forecast_eval/mt5_reader` plus its clock check;
  - the newest closed bar, with a 60 s poll;
  - gaps, never back-filled;
  - one worker call per model;
  - `predictions.csv`, `failures.csv` and `health.csv` under
    `results/external_tsfm/forward/`;
  - the EUR/USD baseline forecast in the same pass.

  Verify with `tests/test_external_tsfm_forward.py`, using injected frames and a fake
  worker:
  - a normal hour logs 2 rows per instrument;
  - three missed hours give gaps;
  - MT5 closed gives a health row with zero forecasts;
  - one model failing leaves the other model's rows;
  - no file outside `results/external_tsfm/` and `research_models/external_tsfm/`
    changes.
- [x] 5.2 Add settlement:
  - gold targets and eligibility from `src/gold/targets`, and EUR/USD from
    `src/forecast_eval/targets`;
  - copy `baseline_size_pct`, plus `inhouse_size_pct` from gold's h1_lstm row, into each
    settlement.

  Verify:
  - a parity test shows gold realised |r| equals the gold logger's settlement for the
    same as-of bar, and EUR/USD equals `volatility_pct_target`;
  - a market-break row settles ineligible.
- [ ] 5.3 Create `scripts/external_tsfm/install_task.ps1` and `uninstall_task.ps1` for
  `EURUSDProphet-ExternalLogger`: hourly at :01, WakeToRun, StartWhenAvailable,
  IgnoreNew, 20-minute limit, hidden runner. Write
  `results/external_tsfm/RUNBOOK.md`, covering install, the pass order, files, health,
  phases and rollback.

  Verify:
  - after install, `Get-ScheduledTask` shows those settings;
  - one by-hand pass (`python -m src.external.tsfm.forward_logger`) writes one health row;
  - every RUNBOOK command runs as written.

## 6. Registration and scoring

- [x] 6.1 Create `src/external/tsfm/prereg.py --write`. It writes
  `results/external_tsfm/PRE_REGISTRATION.md` and `.json`, fixing the items of design D7:
  - the cells and the frozen configurations;
  - the baseline digests;
  - α = 0.0125;
  - the bootstrap settings;
  - N = 1,500, the minimum of 500, the cap 2027-04-30, and the power statement.

  Also add the phase rule (`FP.committed_and_clean`). Verify with tests that rows are
  `dry_run` while the file is uncommitted or modified, and `registered` after a commit, in
  a temp git repo.
- [x] 6.2 Create `src/external/tsfm/forward_report.py --score`:
  - it refuses before registration;
  - it scores each cell once, at N or the cap, with the block bootstrap from
    `src/forecast_eval/stats` (block 24, 2,000 resamples, seed 42) on paired rows;
  - KEEP when the lower bound > 0, DROP otherwise, UNDERPOWERED below 500;
  - it writes append-only `verdicts.json`.

  Verify with tests on synthetic settlements: a clear edge gives KEEP, no edge gives DROP,
  430 rows at the cap give UNDERPOWERED, and a second run never re-scores a cell.
- [x] 6.3 Document the family:
  - an `external_tsfm` section in `ARCHITECTURE_DOCS.md`: models, the full-history
    fine-tuning and its boundary, clean window, replay, cells, α, baselines, power, and
    why direction is not registered;
  - the bullet in `CLAUDE.md`, under Production Methodology, next to the horizon-study
    and gold families.

  Verify that every path and command named there exists.
- [x] 6.4 Run the fine-tuning (3.2), the replay (4.3) and `prereg --write` on this PC and show the owner
  `REPLAY.md` and `PRE_REGISTRATION.md`. Commit them only after the owner approves.
  Verify: the commit exists and the next pass logs `registered` rows.

## 7. Second-opinion view

- [x] 7.1 Create `src/external/tsfm/view.py` (design D8):
  - a read-only payload per instrument: current forecasts, Sofia hour, % and $ size,
    P(up) marked "not registered", and the last-forecast time;
  - the agreement rules: gold size against the alert threshold and h1_lstm, gold
    direction against h1_lstm, EUR/USD size against the baseline, and EUR/USD direction
    against h1_gbm at h = 1;
  - the descriptive forward record per cell, with "too few to say" below 30 and the
    recorded verdict only after scoring;
  - the BG/EN texts with the honesty label.

  Verify with `tests/test_external_tsfm_view.py`:
  - every spec scenario: disagreement at 0.18 % against 0.14 % with threshold 0.151;
    "nothing to compare"; 12 rows giving "too few to say"; the recorded verdict shown
    unchanged; ±0.18 % ≈ 7.02 $ at 3897.67;
  - the trade-word scan;
  - no model library imported.
- [x] 7.2 Add `GET /api/second-opinion` and `/second-opinion` (`static/second_opinion.html`)
  to `api.py`. Add the page to `PAGE_PATHS` and to `NAV` under "More" in
  `src/app_shell.py`. Extend `tests/test_gold_isolation.py`'s api import rule to allow only
  `src.external.tsfm.view`.

  Verify with tests:
  - the endpoint returns both instruments;
  - the page is served with the shell and BG by default;
  - the More menu marks it `aria-current="page"`;
  - no file changes during a request.

  Note the page in `ARCHITECTURE_DOCS.md` §5.4.1's page list.
- [x] 7.3 Add the second-opinion card to `static/gold.html`. It fetches
  `/api/second-opinion`, shows both models' sizes and their agreement with h1_lstm, and
  links to the page, in BG and EN. Verify with tests that the card's dictionary keys exist
  in both languages and that the `/api/gold` payload keys are unchanged.

## 8. MT5 panel line

- [x] 8.1 Add the `second_bg` / `second_en` meta line to `src/gold/mt5_export.py`, built from
  external rows whose hour is current, and empty when none is. Make the indicator show it
  in the hourly panel, with the honesty label as its tooltip (design D9). Recompile and
  reinstall with `scripts/gold/install_mt5_indicator.ps1`.

  Verify with tests:
  - the spec scenario text (Chronos 0,18 % disagrees, TimesFM 0,15 % agrees, threshold
    0,151 %);
  - an empty line when nothing is current;
  - the trade-word scan;
  - `results/external_tsfm/` read-only.

  Then verify that it compiles with 0 errors in all three terminals and update the
  RUNBOOK chart section.

## 9. Integration check

- [ ] 9.1 With the task installed and the family registered, let one :01 external pass and
  the following :03 gold pass run. Verify:
  - `results/external_tsfm/forward/health.csv` shows `ok`, with 4 forecasts;
  - `/second-opinion` and the `/gold` card show the current hour for both models;
  - the MT5 gold panel shows the second-opinion line;
  - nothing under `results/gold/`, `results/forward_eval/` or `models/` changed except
    what their own loggers wrote.

  Then run `python -m pytest -q tests/test_external_tsfm_*.py tests/test_gold_*.py` and
  confirm they pass.
