# Tasks

## 1. Package, config and protection

- [x] 1.1 Create `src/rts_signals/` with `__init__.py` and `rts_signals_config.json`. The config covers:
  - the paths `results/rts_signals` and `research_models/rts_signals` (with `inbox/`, `drop/`, `secrets/`
    and `cache/`);
  - the six pairs and their price factors;
  - the terminal path;
  - the Gmail senders and query;
  - forward 300 trades or 2027-06-30, and scoring alpha 0.05, 5-day blocks, 10 000 resamples, seed 20261202;
  - the export file and 20 display days.

  Verify with `tests/test_rts_signals_config.py`: the config loads and the digest is stable.
- [x] 1.2 Add `tests/test_rts_signals_protected.py`. Verify that:
  - no source names a write target outside the family trees, apart from the export;
  - no source contains an order or position call;
  - `secrets/` is gitignored;
  - an end-to-end run on fixtures leaves `models/`, `config.json` and every other registry byte-identical.

## 2. Report ingest (spec report-ingest)

- [x] 2.1 Copy the 12 uploaded PDFs into `research_models/rts_signals/drop/`. Write `ingest.py` with SHA-256
  de-duplication and `parse.py` with kind detection, the Spot and Morning parsers (Futures logged, never read), including the
  track-record splitter and the price conversion. Use the extracted text of the PDFs as test fixtures under
  `tests/fixtures/rts_signals/` (text only, no PDFs). Verify with `tests/test_rts_signals_parse.py`:
  - the YEN 2026-11-25 row converts to 158.98 / 158.58 / 159.42;
  - SWISS and CAD convert correctly;
  - EURO 11009 is guerilla;
  - `0.81881` splits to 0.8188 and rule 1;
  - NPR rows have no levels;
  - the duplicate morning report is stored once;
  - the CAD `13202` note is flagged;
  - a re-parse adds nothing.
- [x] 2.2 Write `gmail.py`, read-only (`gmail.readonly`), with an injectable client, and
  `scripts/rts_signals/authorize_gmail.ps1`. Verify with a fake client in `tests/test_rts_signals_gmail.py`:
  - a new attachment is saved once;
  - a missing token reports `gmail not authorised`;
  - no send, modify or delete method is called;
  - the token path is gitignored.

## 3. Rule book (spec rule-book)

- [x] 3.1 Write `rules.json` and `rules.py`. They cover:
  - the rule numbers and their digest;
  - New York to Berlin label conversion for the 16:00 close, the 18:00 open and the Friday 12:00 close;
  - a README generator in Bulgarian and English with one worked example per rule.

  Verify with `tests/test_rts_signals_rules.py`:
  - in November the 18:00 New York open becomes 00:00 Berlin, and 23:00 in the late-October mismatch week;
  - the README has every rule section;
  - editing a number changes the digest.

## 4. Simulation (spec trade-simulation)

- [x] 4.1 Write `bars.py`: a read-only MT5 M1 feed (injectable) and append-only per-pair caches with the spread.
  Verify that a repeated top-up stores each bar once and that the source has no order calls.
- [x] 4.2 Write `simulate.py` for the base trade: Rule 0 with proportionality, guerilla entries, stop, target
  and the time stop including the Friday spot close, the tie-as-stop rule, gross and net of spread, pips and R,
  and the `no price data` and `entry not reached` reasons. Verify with `tests/test_rts_signals_simulate.py` on
  synthetic minute bars:
  - target first and stop first;
  - a tie;
  - the 20-point gap proportionality case;
  - an untouched guerilla;
  - the Friday noon exit;
  - net below gross by exactly the recorded spreads.
- [x] 4.3 Write `replay.py` (full rules, Rules 2–5 and NPR) and `reconcile.py`. Verify with synthetic paths for
  Rule 2 (and none on Friday), Rule 3 with the 38/44 fallback and the three-loss cap, Rule 4 and Rule 5. Then
  reconcile the November spot track record against the real MT5 cache. Verify that every RTS trade gets one
  label, and report the counts.

## 5. Evaluation (spec evaluation)

- [x] 5.1 Write `history.py`, which builds `results/rts_signals/history.md`: per pair, RTS's stated points, our
  gross and net base results and the reconciliation counts, marked `descriptive — no verdict`. Verify that it
  runs on the real cache and is deterministic apart from the generated-at line.
- [x] 5.2 Write `ledger.py`, `prereg.py` (git gate, config and rules digests, power at 300 trades with its
  calculation), `registry.py` and `score.py` (cell `rts_spot_net_r`). Verify with
  `tests/test_rts_signals_score.py` in a temporary git repo:
  - an untracked pre-registration refuses;
  - a late report is excluded;
  - a planted edge is KEEP and zero is DROP;
  - 120 trades before the deadline refuse;
  - fewer than 300 at the deadline are UNDERPOWERED;
  - re-scoring leaves the registry byte-identical.
- [x] 5.3 Write `report.py`. Verify that the pre-score report shows the count out of 300 and the exit counts but
  no mean, total or test result.
- [x] 5.4 Run `prereg --write`. Verify that both files exist and that no forward row exists.
- [ ] 5.5 **Owner step:** review and commit `results/rts_signals/PRE_REGISTRATION.md` and `.json`. Verify with
  `python -m src.rts_signals.prereg --check`.

## 6. Chart (spec chart-display)

- [x] 6.1 Write `export.py`, which rebuilds `rts_signals.tsv` with today's levels, the last 20 days' trades and
  the Morning positions, all texts in Bulgarian and English. Verify with `tests/test_rts_signals_export.py`:
  - the 2026-11-25 USDJPY row has the right levels;
  - labels start with `RTS:` and today's ends `непроверено`;
  - a failed write reports `export failed`;
  - nothing contains a lot size.
- [x] 6.2 Write `mql5/Indicators/ProphetRTS.mq5` with `scripts/rts_signals/install_mt5_indicator.ps1` and
  `uninstall_mt5_indicator.ps1`. Verify that:
  - it compiles in MetaEditor with 0 errors;
  - a test finds no order functions;
  - the uninstall, run against a fake APPDATA, leaves the other Prophet indicators untouched.

## 7. Task, docs and integration

- [x] 7.1 Write `run.py` (fetch → parse → simulate settled sessions → ledger → export) and
  `scripts/rts_signals/` (`run_hidden.vbs`, `run_module.cmd`, `install_tasks.ps1` with `-DryRun` for
  `EURUSDProphet-RTSInbox` hourly at :20, and `uninstall_tasks.ps1`). Verify that the `-DryRun` output is one
  task at :20 with a command of at most 261 characters, and that a second `run.py` changes nothing.
- [x] 7.2 Write `results/rts_signals/README.md` (generated rules explanation) and `RUNBOOK.md` (Gmail
  authorisation, drop folder, chart, refusals). Add `ARCHITECTURE_DOCS.md` §3.23 and a `CLAUDE.md` invariant 18.
  Verify that every RUNBOOK command runs as written (`--help` or `-DryRun`).
- [x] 7.3 Integration check. Run the full `tests/test_rts_signals_*` suite plus the other families' protection
  tests. Then run one real `run.py` on the 12 PDFs and the MT5 cache. Confirm:
  - the TSV exists;
  - `history.md` is built;
  - `models/` and every other registry are unchanged.

## Workflow follow-up

- After 5.5 the owner authorises Gmail, installs the task and the indicator per the RUNBOOK.
- Score once at 300 forward trades or on 2027-06-30 with `python -m src.rts_signals.score`, then record the
  result in the docs.
- Archive the change after implementation; scoring may come later.
