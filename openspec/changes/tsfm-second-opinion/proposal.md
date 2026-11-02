# Proposal

## Why

On 2026-11-02 the owner asked for an external model that helps the project's models or makes
their forecasts more trustworthy. The project's evidence points to where such a model can
help:
- **Direction.** Every DROP so far came from price, macro or positioning information. The
  one external model tried, Kronos, failed on direction: its Brier skill was −0.62 at
  `pred_len=24`.
- **Volatility.** Kronos passed an incremental test against the in-house volatility
  ensemble (KEEP, correlation 0.17, CI excluding 0).
- **Literature.** Time-series foundation models (TSFMs) help on realised volatility, not on
  direction ([arXiv 2505.11163](https://arxiv.org/pdf/2505.11163),
  [arXiv 2507.07296](https://arxiv.org/html/2507.07296v1)).

An external model cannot make a forecast valid on its own. Only forward data can. What it
can be is an honest **second opinion** on the one thing this project forecasts with skill,
the size of the next move. It gets its own pre-registered verdict, and its agreement with
the in-house models is shown next to them.

The owner chose:
- Chronos-2 (Amazon) and TimesFM 2.5 (Google), both open weights under Apache-2.0, run
  locally;
- both instruments, gold and EUR/USD;
- the role of an independent challenger with an agreement display, not an input to the
  in-house models.

## What Changes

- **Two pinned, train-free challengers.**
  - Chronos-2 (120M) and TimesFM 2.5 (200M) forecast the next H1 bar for gold and EUR/USD,
    zero-shot.
  - Weights are pinned by Hugging Face revision and SHA-256, and are never updated
    silently.
  - They run in a separate Python environment, so the real-money app's dependencies never
    change.
- **Fine-tuning on the full history** (owner's request, 2026-11-02).
  - Each model is also fine-tuned (LoRA) on all of each instrument's history before
    2025-11-01: about 11 years of gold H1 (64,295 bars from 2015-01) and about 15 years
    of EUR/USD H1 (93,763 bars from 2010-10, the deepest MT5 read).
  - The number of steps is chosen on the last pre-window year. The model is then refit on
    the whole pre-window history with that step count, and frozen.
  - Long context is offered too: up to 8,192 bars for Chronos-2 and 16,256 for TimesFM
    (its 16,384 limit includes the 128-step output patch), about 1.5 to 2.7 years of
    hours.
- **A clean-window replay.**
  - Both models were released in autumn 2025, so bars from 2025-11-01 on cannot be in
    their pretraining or in the fine-tuning.
  - A replay on that window picks one configuration per model and instrument, by a rule
    declared before the run: zero-shot or fine-tuned × returns or closes × 2,048 or the
    long context.
  - It is descriptive, never a verdict.
- **A forward logger.**
  - It is its own scheduled task, `EURUSDProphet-ExternalLogger`.
  - Every hour it forecasts the newest closed H1 bar of each instrument: the expected size
    of the move and P(up).
  - It settles each forecast against the instrument's own target and eligibility rules, and
    writes a health row.
  - It never writes to the gold or EUR/USD loggers' files. It only reads them, for the
    comparison.
- **A pre-registered family, `external_tsfm`.**
  - It has 4 cells: {Chronos-2, TimesFM} × {gold, EUR/USD}.
  - Each tests whether the model's next-hour size forecast beats that instrument's frozen
    GARCH baseline on forward data, at α = 0.05/4.
  - Each cell is scored once, at a declared sample size or date cap.
  - Direction is logged and shown, but it is not registered and never judged.
  - The family never touches the EUR/USD `0.05/9` bar or the gold registries.
- **A second-opinion view.**
  - A read-only `/second-opinion` page and API show, for both instruments, each model's
    current forecast, its agreement with the in-house model, and the forward record so
    far, labelled descriptive.
  - `/gold` gains a "second opinion" card.
  - The MT5 gold panel gains one line, e.g. "Второ мнение: Chronos ±0,18 %, TimesFM
    ±0,15 % · размер: Chronos не е съгласен, TimesFM е съгласен с h1_lstm".

## Capabilities

### New Capabilities

- `external-forecasts/tsfm-challengers`: pinning, the isolated runtime, the clean-window
  replay, forward logging and settlement, the registered family and its scoring, and
  isolation from production.
- `external-forecasts/second-opinion`: the read-only page, the API and the `/gold` card, the
  agreement rules and the honesty labels.

### Modified Capabilities

- `app-shell`: the "More" menu also lists the second-opinion view.
- `gold-forecasting/mt5-chart-signals`: the panel adds one second-opinion line. This
  capability is still in the unarchived change `gold-mt5-chart-signals`, so that change
  should be archived first.

## Impact

- **Code.**
  - New package `src/external/tsfm/`:
    - `vendor.py`: pins and digests;
    - `worker.py`: runs in the side environment, series in and quantiles out;
    - `forecast.py`: quantiles to size and P(up);
    - `finetune.py`: the history freeze, and LoRA fine-tuning in the side environment;
    - `replay.py`;
    - `forward_logger.py`;
    - `prereg.py`;
    - `view.py`: read-only.
  - `api.py`: `GET /api/second-opinion` and the `/second-opinion` page.
  - `static/second_opinion.html`, a card in `static/gold.html`, and `src/app_shell.py`
    (`PAGE_PATHS`, `NAV`).
  - `src/gold/mt5_export.py` and the indicator panel: one line.
- **Dependencies.**
  - A new `requirements-tsfm.txt` with exact pins installs into a separate `.venv-tsfm`:
    chronos-forecasting, timesfm, peft, and torch with CUDA for fine-tuning and the
    replay, falling back to CPU for the hourly forecasts.
  - The main `requirements.txt` and `.venv` are unchanged.
- **Data and artifacts.**
  - Gitignored weights go to `research_models/external_tsfm/`.
  - Digests go to `results/external_tsfm/artifact_manifest.csv`.
  - Forward logs go to `results/external_tsfm/forward/`, and the replay to
    `results/external_tsfm/replay/`.
  - The pre-registration is `results/external_tsfm/PRE_REGISTRATION.md`.
- **Scheduling.** The new task `EURUSDProphet-ExternalLogger` runs hourly at :01, with its
  own lock and log. That is before the gold pass at :03, so the gold MT5 export already
  carries the current second opinion.
- **Tests.** New `tests/test_external_tsfm_*.py`:
  - quantile maths;
  - pin refusal;
  - isolation;
  - no writes outside the external folders;
  - settlement parity with each instrument's own targets;
  - the agreement rules;
  - no trade words.
- **Docs.** `ARCHITECTURE_DOCS.md`, `results/external_tsfm/RUNBOOK.md` and `CLAUDE.md`.
- **Not touched.**
  - `models/`, `config.json`, `_train_pipeline.py`, `src/inference.py`, `src/features.py`
    and `src/paper_trading.py`;
  - the gold and EUR/USD registries, pre-registrations and loggers, which are only read;
  - `results/horizon_study/study_record.json`.
- **Cost.**
  - No API fees and no keys.
  - About 5 GB of disk for the side environment with CUDA torch, the weights and 4 LoRA
    adapters.
  - The fine-tuning and the replay run once on the GPU, a few hours in all.
  - A few seconds per hourly pass.
