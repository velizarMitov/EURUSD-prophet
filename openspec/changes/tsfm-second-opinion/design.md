# Design

## Context

- **What already exists.**
  - The project has already run one external foundation model: Kronos, through
    `src/external/kronos/` and the train-free challenger `src/forecast_eval/challengers/kronos.py`.
  - Kronos's weights are pinned (`KRONOS_COMMIT`). It is scored only on its clean window,
    after its training cutoff.
  - It failed on direction and passed an incremental volatility test.
  - The EUR/USD forward logger logs it as `kronos_volatility` beside `vol_ensemble`.
- **The in-house comparators already logged every hour:**
  - gold `h1_lstm` (size, P(up)) and `h1_garch_hod` (size), in
    `results/gold/forward/predictions.csv`;
  - EUR/USD `h1_gbm` (P(up)) in `results/forward_eval/predictions.csv`.

  EUR/USD has no hourly in-house size model and no hourly GARCH row.
- **What must not move.**
  - The study record `results/horizon_study/study_record.json` and the gold round-one
    files are pinned. Adding challengers to them would change registered objects.
- **Runtime.**
  - The main `.venv` runs Python 3.13 with torch 2.11 (CUDA 12.8) and TensorFlow for the
    LSTMs. It serves real-money decisions.
  - Chronos-2 is installed through `chronos-forecasting` (Apache-2.0, 120M parameters,
    quantile output, released 2025-10-20).
  - TimesFM 2.5 is installed through `timesfm[torch]` (Apache-2.0, 200M parameters,
    continuous quantile head with mean plus deciles, context up to 16k, released autumn
    2025).
  - Either package may pull its own `transformers`, `numpy` or other pins.
  - Both models support context and fine-tuning:
    - Chronos-2 takes up to 8,192 bars of context and supports LoRA fine-tuning. AutoGluon's
      wrapper defaults to learning rate 1e-5, 1,000 steps, batch 32 and a fine-tuning
      context of 2,048.
    - TimesFM 2.5 takes up to 16,384 and ships a LoRA/PEFT fine-tuning example. It
      normalises its input internally (RevIN), so raw values go in.
- **History available** (measured at implementation, 2026-11-02).
  - Gold: `results/gold/gold_h1.csv` holds 64,295 H1 bars before 2025-11-01, from
    2015-01 (about 11 years). The broker's earlier months are daily bars labelled H1 and
    are trimmed from the cache.
  - EUR/USD: the deepest MT5 read is about 100,000 bars: 93,763 before 2025-11-01, from
    2010-10 (about 15 years).

## Goals / Non-Goals

**Goals:**
- A size forecast from each external model that can be compared row by row with the
  in-house forecast for the same as-of bar, and judged once on forward data.
- Nothing about the existing loggers, registries, models or app dependencies changes.
- The owner sees the second opinion where they already look: `/gold`, MT5 and one new
  page.

**Non-Goals:**
- No refit during the forward window. Each fine-tune (D10) happens once, before
  registration, and is frozen. A later refit would be a new configuration and need a new
  registration.
- No direction verdict. P(up) is logged and shown as descriptive only, because the prior
  is weak (Kronos failed, and the literature agrees).
- No input into the in-house models, no ensemble with them, and no feature. Each of those
  would spend a hypothesis in the `0.05/9` family.
- No D1 cells. About 21 rows a month makes a forward verdict on daily data take years.
- No paid API, no keys, no network call at forecast time. Weights are downloaded once, at
  install.

## Decisions

### D1. A separate environment and a worker process
- `requirements-tsfm.txt` pins exact versions of `chronos-forecasting`, `timesfm[torch]`,
  `peft` and `torch` for a separate `.venv-tsfm`.
- The logger runs in the main `.venv`. Per model, it calls
  `.venv-tsfm\Scripts\python.exe -I src/external/tsfm/worker.py` with a JSON request on
  stdin: model, configuration, the series and the quantile levels. It receives JSON
  quantiles on stdout.
- The worker imports only the standard library, numpy, torch and the model package. It
  never imports repo modules.
- Each call has a 120 s timeout.

Alternatives considered:
- **Installing into `.venv`.** One `pip install` could change `transformers` or `numpy`
  under the TensorFlow LSTMs that serve real money.
- **A long-running local server.** Another process to keep alive, with no benefit at one
  call per hour.

`.venv-tsfm` gets the CUDA 12.8 build of torch, the same build as the main `.venv`, because
the fine-tuning (D10) and the long-context replay need the GPU. The hourly worker uses the
GPU when it is free and CPU otherwise. One series per call with a long context takes a few
seconds on CPU, since both models patch the input to about 512 tokens.

### D2. Pinning
- `src/external/tsfm/vendor.py` holds, per model:
  - the Hugging Face repo id and revision commit;
  - the package version.
- `python -m src.external.tsfm.vendor --fetch` downloads each revision into
  `research_models/external_tsfm/<model>/` and writes the SHA-256 of every file to
  `results/external_tsfm/artifact_manifest.csv`.
- The logger and the replay verify the files against the manifest before every load, in
  the same way as gold's `verify_cell`. The worker loads from that local folder with
  network access disabled (`HF_HUB_OFFLINE=1`).

### D3. From quantiles to size and P(up)
- **Quantile levels.** Both models are asked for the deciles 0.1…0.9. TimesFM's head gives
  exactly these, and Chronos-2 accepts them as quantile levels. A common grid keeps the
  two comparable.
- **CDF.** The predictive CDF `F` is piecewise-linear through the decile points. It is
  extended linearly beyond q10 and q90 with the end segments' slopes, then clipped to
  [0, 1].
- **Size.** The size is the `m ≥ 0` with `F(m) − F(−m) = 0.5`, solved by bisection. That is
  the median of |R|, the point forecast that minimises mean absolute error. MAE is the
  registered score.
- **P(up).** `1 − F(0)`, clipped to [0.1, 0.9], because outside the decile grid it is
  extrapolation. It is descriptive only, so the clip costs no verdict.
- **Two input modes.**
  - `returns`: the series is the hourly log returns in percent, and the quantiles are
    returns.
  - `closes`: the series is the closes, and each quantile `q` becomes
    `100·ln(q / last close)`.
- Weekend gaps are consecutive steps, as for Kronos. Every row records
  `spans_weekend_gap`.

### D4. Clean-window replay chooses the configuration
- **Window.** As-of bars from 2025-11-01 to 2026-10-31, after both release dates. The bars
  are frozen into `results/external_tsfm/replay/<instrument>_h1_window.csv`, with digests,
  so a re-run sees the same data.
  - Gold comes from `results/gold/gold_h1.csv`, which is append-only.
  - EUR/USD comes from a fresh MT5 read through `src/forecast_eval/mt5_reader`. It does
    not come from `results/eurusd_h1.csv`, which shrank below its DATA.md count on
    2026-11-02.
- **Grid.** {zero-shot, fine-tuned} × {returns, closes} × {2,048, long}, where long is
  8,192 for Chronos-2 and 16,256 for TimesFM (its 16,384 limit includes the 128-step
  output patch). That is 8 configurations per model and
  instrument. The rule is the lowest first-half MAE (2025-11-01 to 2026-04-30). The second
  half is a descriptive check. A fine-tune that failed removes its four configurations.
- **Cost.** Rolling forecasts are batched (64 windows per worker call) on the GPU. The full
  replay is about 2 models × 8 configurations × ~11,500 bars, run once, a few hours.
- **Context.** A long context reaches before 2025-11-01, which is allowed: only the scored
  as-of bars must be clean (see Risks).
- **Report.** `results/external_tsfm/replay/REPLAY.md` gives:
  - every configuration's error;
  - the frozen GARCH baseline's error on the same rows;
  - for gold, also h1_lstm's error on the rows where the gold study saved a prediction.
  
  It is labelled descriptive and choice-only. The comparisons inform no choice.

### D5. Baselines
- **Gold.** The `h1_garch_hod` forecast the gold logger made for the same as-of bar, read
  from its predictions file. It is a frozen, registered gold cell.
- **EUR/USD.** A GARCH(1,1) × hour-of-day, fitted once on EUR/USD H1 bars from 2020-11-01
  to 2025-10-31 with the same estimator class as gold's `h1_garch_hod`, imported
  read-only. It is saved under `research_models/external_tsfm/baseline_eurusd/` with its
  digest. The logger computes its forecast in the same pass.
- **Pairing.** A row is paired only when both the model and its baseline forecast exist
  for that as-of bar. The settlement copies the baseline value into the row, so scoring
  never re-reads another logger.

### D6. The forward logger
- **Schedule.** `EURUSDProphet-ExternalLogger` runs hourly at :01. Its settings are the
  same as gold's task: WakeToRun, StartWhenAvailable, IgnoreNew, 20-minute limit. The lock
  is `research_models/external_tsfm/forward.lock`.
- **Why :01.** The gold pass at :03 ends with the MT5 chart export. Running first means the
  export already sees this hour's external forecasts. If MT5 has not yet closed the bar,
  the pass polls for up to 60 s using gold's closed-bar rule, then records a gap.
- **Data.** Gold uses `src/gold/data.fetch_live` and its clock check. EUR/USD uses
  `src/forecast_eval/mt5_reader` and that arbiter's clock check. All are imported
  read-only.
- **Files.** Under `results/external_tsfm/forward/`:
  - `predictions.csv`: the key, model, instrument, as-of bar and close, `config_id`,
    `q10`…`q90`, `size_pct`, `p_up`, `model_digest`, `phase`, `spans_weekend_gap` and
    `logged_at`;
  - `settlements.csv`: the realised return and |return|, `eligible`, `baseline_size_pct`,
    `inhouse_size_pct` (gold `h1_lstm`) and `settled_at`;
  - `gaps.csv`, `failures.csv` and `health.csv`.
- **Settlement.** It uses `src/gold/targets` for gold and `src/forecast_eval/targets` for
  EUR/USD, on the same price series those arbiters use. A test checks that realised moves
  equal theirs for the same bars.

### D7. Registration and scoring
- `results/external_tsfm/PRE_REGISTRATION.md` and `.json` are written by
  `python -m src.external.tsfm.prereg --write` after the replay. They contain:
  - the cells and the frozen configurations;
  - the baselines with their digests;
  - α = 0.0125;
  - the bootstrap: block length 24, 2,000 resamples, seed 42;
  - N = 1,500, the minimum of 500 and the cap 2027-04-30.
- **Phase.** The phase follows gold's rule: `registered` only once the file is committed
  and unmodified (`FP.committed_and_clean`). Before that, every row is `dry_run`.
- **Scoring.** `python -m src.external.tsfm.forward_report --score` refuses before
  registration. It scores a cell once N or the cap is reached and appends to
  `results/external_tsfm/verdicts.json`. A scored cell is never re-scored.
- **Verdicts.**
  - KEEP when the lower bound of the 98.75 % interval of (baseline MAE − model MAE)
    is > 0;
  - DROP otherwise;
  - UNDERPOWERED — NO DECISION below 500 rows at the cap.
- **Power.** gold `h1_lstm`'s own edge over GARCH was 0.0113 pp of MAE. The SE implied
  by its test-block CI is ≈ 0.00111 at n = 12,780; the planning draft's 0.00128 was
  corrected at implementation. At n = 1,500 the SE is ≈ 0.0032, so the detectable edge at
  80 % power is ≈ 0.011 pp. An edge as small as h1_lstm's would be borderline.
  `prereg.power_statement` computes this from `results/gold/test_report.json`, and the
  pre-registration states it. About 110 eligible hours a week means N arrives in roughly
  14 weeks.

### D8. The second-opinion view
- **Data source.** `src/external/tsfm/view.py` reads only:
  - the external forward files;
  - gold's `predictions.csv` (`h1_lstm`) and the alert threshold in
    `gold_config_v2.json['alerts']`;
  - EUR/USD's `predictions.csv` (`h1_gbm`, horizon 1);
  - the frozen EUR/USD baseline's logged values.

  It imports no model library, like `summary.py`.
- **Endpoints.** `api.py` adds `GET /api/second-opinion` and `/second-opinion`
  (`static/second_opinion.html`). `src/app_shell.py` lists the page under "More".
- **Gold card.** `static/gold.html` gains a card that fetches the new endpoint, so the
  `/api/gold` payload is unchanged.
- **Wording.** All texts are in one BG/EN dictionary in the view module. A test scans them
  for the trade-word list used by `tests/test_gold_mt5_export.py`.

### D9. The MT5 line
- `src/gold/mt5_export.py` gains a `second_<lang>` meta line, built from external rows
  whose hour is current at export time. It reads `results/external_tsfm/forward/`
  read-only. When nothing is current the line is empty.
- The indicator adds that line to the hourly panel when it is non-empty, with the
  "external model" label as its tooltip.
- The file format stays version 2. An unknown meta key is ignored by older indicators,
  which simply show no line.

### D10. Fine-tuning on the full pre-window history
- **Data.** `src/external/tsfm/finetune.py --freeze` writes each instrument's entire H1
  history before 2025-11-01 to `results/external_tsfm/finetune/<instrument>_h1_history.csv`,
  with a digest. Gold comes from `gold_h1.csv`, and EUR/USD from the deepest MT5 read the
  reader allows. Any bar on or after 2025-11-01 makes the freeze refuse. The fine-tune
  reads only these files, so the boundary is checked in one place.
- **Method.** LoRA on each model, in `.venv-tsfm` on the GPU (as implemented):
  - **Chronos-2:** the pinned `chronos-forecasting` 2.3.2 has its own entry point,
    `Chronos2Pipeline.fit(finetune_mode='lora')`. It is used at the package's
    recommended LoRA rate of 1e-5, batch 32 and context 2,048, so AutoGluon is not needed.
  - **TimesFM 2.5:** the pinned package has no trainer. The worker injects LoRA adapters
    (`peft`, r = 8) into the attention and feed-forward layers and trains them through
    the same first-step path the forecasts use, at 1e-4. That path matches the package's
    `forecast()` to 1e-6. The batch is 32, in two micro-batches of 16, which keeps the
    peak under 4.3 GB of the 8 GB card.
  - Both stages use a constant learning rate, so step k of stage 1 and a stage-2 run of
    k steps see the same rate.
  - **Target and loss.** Next-bar log return in percent. Each model uses its own quantile
    loss, with the input mode fixed as in the replay grid. Fine-tuning uses the returns
    series, and the closes mode reuses the same adapter.
- **Two stages.**
  1. Train on bars before 2024-11-01. Score the holdout year 2024-11-01 to 2025-10-31 every
     100 steps, up to 3,000 steps. Choose the step count with the lowest holdout quantile
     loss.
  2. Refit from the base weights on every bar before 2025-11-01, for exactly that many
     steps, with seed 42.

  The adapters go to `research_models/external_tsfm/finetuned/<model>_<instrument>/`. Their
  digests go into the artifact manifest. A `finetune_meta.json` records the steps, the
  learning rate, the loss curves, the versions and the GPU.
- **Why two stages.** Stage 1 alone would leave the newest year, the one closest to
  today's market, out of the weights. The owner asked for the whole history. Stage 2 uses
  all of it, and stage 1 only picks how long to train.
- **Failure.** A fine-tune that fails or diverges (holdout loss never below the zero-shot
  loss on the holdout year) is recorded and dropped from the replay grid. The zero-shot
  model stays.

## Risks / Trade-offs

- **Fine-tuning learns the past too well.** Sixteen years of hourly gold include regimes
  unlike today. → The holdout year chooses the step count. The clean window then decides
  between zero-shot and fine-tuned on unseen bars, and forward data judges. If the
  zero-shot model wins the replay, the fine-tune costs nothing but compute.
- **The fine-tuning APIs are young.** The Chronos-2 entry point and TimesFM's example
  script may change. → Exact pins plus a vendored commit. A fine-tune that cannot run is a
  recorded failure, not a blocker.
- **GPU memory.** Long contexts at batch 32 may not fit. → The fine-tuning context is
  2,048, as in AutoGluon's default. Only inference uses the long context, and the batch is
  halved until it fits, recorded in the meta.

- **The owner treats agreement as confirmation.** Two models agreeing is not evidence of
  skill. → Every text carries "external model — research, no proven edge".
  Agreement is labelled descriptive. The view shows KEEP/DROP only from the registered
  score.
- **The upstream packages change their APIs.** → Exact pins in `requirements-tsfm.txt`, the
  pinned revisions, and a worker smoke test against both models in the install task.
- **TimesFM 2.5's packaging is in flux.** PyPI and the repository disagree about the 2.5
  install. → The worker uses whichever entry point the pinned version exposes, chosen and
  recorded at install. If neither works, TimesFM is reported unavailable, Chronos-2 runs
  alone, and the TimesFM cells read UNDERPOWERED at the cap.
- **The replay overlaps gold's spent one-shot test block.** → The replay chooses a
  configuration only from the external model's own errors. Comparisons with h1_lstm there
  are descriptive. Verdicts come only from forward rows.
- **MT5 contention at :01–:03.** The EUR/USD logger already reads MT5 every 15 minutes
  alongside the gold pass, so concurrent reads are existing practice. The external pass
  takes about 30 s.
- **The clean window is short for 2048-bar contexts.** The first forecasts in November 2025
  need context reaching back into training-period bars. → The context may include earlier
  bars, because only the scored as-of bars must be clean. The model saw at most the
  inputs, never these targets after its cutoff.

## Migration Plan

1. Run `scripts/external_tsfm/install.ps1`. It creates `.venv-tsfm`, fetches and pins the
   weights, and runs a worker smoke test.
2. Run `python -m src.external.tsfm.finetune --freeze`, then `--train` (GPU, about an hour
   per model and instrument). Then run `python -m src.external.tsfm.replay`, then
   `python -m src.external.tsfm.prereg --write`. Review the files, then commit them.
3. Run `scripts/external_tsfm/install_task.ps1`. Rows are `registered` from the first pass
   after the commit.
4. **Rollback:** `uninstall_task.ps1`, and delete `.venv-tsfm` and
   `research_models/external_tsfm/`. Nothing else depends on them. The view shows "no
   external forecasts".

## Open Questions

- Resolved at implementation (2026-11-02):
  - `.venv-tsfm` is Python 3.11 with `torch 2.11.0+cu128`, `chronos-forecasting 2.3.2`,
    `timesfm 2.0.2`, `peft 0.21.2` and `transformers 5.18.0`.
  - `timesfm 2.0.2` is the newest release that ships only TimesFM 2.5. From 3.0.0 the
    package also carries `timesfm3`, whose weights (2026-08) may have seen the clean
    window.
  - The revisions are the weight-upload commits: `amazon/chronos-2@95a9710` (2025-10-30)
    and `google/timesfm-2.5-200m-pytorch@d418f3e` (2025-10-01). Every later commit of
    both repos edits only the README.
