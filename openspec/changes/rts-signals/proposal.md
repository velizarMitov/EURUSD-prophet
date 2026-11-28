# Proposal

## Why

The owner receives Rockefeller Treasury Services (RTS) reports by email every trading day. The 4 pm New York
**Spot Afternoon Report** gives, for six pairs, a direction, an entry, a stop and a target for the next
session. The reports carry RTS's own track record, but it is hypothetical, gross of spread and self-reported.
For November 2026 it shows about +138 points over about 110 spot trades. The spread on these pairs is roughly
1–2 pips a trade, so that is close to zero after cost.

The owner wants three things:
- to see every recommendation on the MT5 chart;
- to check the recommendations against the broker's own prices;
- to know, honestly and in time, whether they are reliable.

Only about one week of reports exists in Gmail (2026-11-20 .. 27), so reliability can only be decided on
**forward** data.

## What Changes

- **New family `rts_signals`** (package `src/rts_signals/`). Display and evaluation only: no order, broker,
  sizing or execution code. RTS levels are a third party's suggestions, never this project's signal.
- **Report ingest.**
  - PDFs come from Gmail automatically (the RTS sender, `has:attachment`), or from a drop folder for files
    saved by hand. The 12 PDFs already uploaded are the first input.
  - Text is extracted with `pypdf`. Two report kinds are parsed: Spot Afternoon and Morning. Futures reports
    are logged and never read (owner decision 2026-11-28: the owner trades spot only).
  - Each parsed row is stored with its source file digest. Raw PDFs stay in the gitignored
    `research_models/rts_signals/`, because they are paid third-party content.
  - Inconsistencies in a report are flagged, never silently fixed. For example, the 11/25 spot note says
    CAD "Rule 2 at 13202" for 14202.
- **Rule book.** RTS's rules are written down as data and code: Rule 0 (open), proportionality, Rule 1
  (guerilla), Rule 2 (runaway 40/60, never on Fridays in spot), Rule 3 (recovery, 38/44 adjustment, at most
  three losses), Rule 4 (reversal 44/48), Rule 5 (opposite gap), NPR, and the Friday-noon close in spot. A
  plain-language explanation goes in the README.
- **Simulation on the owner's MT5 prices.** It uses M1 bid bars and the recorded spread of
  `EURUSD, GBPUSD, AUDUSD, USDJPY, USDCHF, USDCAD`, read-only from the `mt5app` terminal. It comes in two
  levels:
  - **Base trade.** One trade per recommendation: entry, stop and target, with exits scanned minute by minute.
    It is reported both gross and **net of the recorded spread**.
  - **Full rules.** Descriptive only: the chain of Rules 2–5. It is reconciled trade by trade against the
    track record printed in RTS's own reports.
- **Evaluation.**
  - **Historical (November 2026 track record and the uploaded week).** Descriptive only, because RTS's gross
    result was seen first. It shows how many RTS fills our prices confirm, and the result net of spread.
  - **Forward.** One pre-registered cell, `rts_spot_net_r`: the mean net R of base trades, measured after the
    pre-registration commit, is above 0, at alpha 0.05. It is scored once at 300 trades or on 2027-06-30
    (`UNDERPOWERED — NO DECISION` if fewer). It has its own registry and never touches any other family's bar.
- **MT5 chart (owner decision 2026-11-28: "искам да виждам всичко").**
  - A new read-only indicator, `ProphetRTS.mq5`, reads `Common\Files\EURUSDProphet\rts_signals.tsv`. It shows:
    - **today's** recommendation as soon as the report arrives: direction, entry, stop and target, labelled
      as RTS's unverified suggestion;
    - past days' trades with their result net of spread;
    - RTS's Morning Report positions in the corner.
  - It differs from the fx_ny_orb gate, where a live entry sign waits for a KEEP. That difference is the
    owner's explicit choice and is stated on the chart: there is no verdict until the forward cell is scored.
- **Scheduled task** `EURUSDProphet-RTSInbox`, hourly at :20, so it never collides with :01/:03/:05/:08. It
  fetches new reports, simulates settled days and rebuilds the chart file.

## Capabilities

### New Capabilities
- `rts-signals/report-ingest`: fetching RTS PDFs from Gmail or a drop folder, parsing the three report
  kinds, unit conversion and the stored, digested rows.
- `rts-signals/rule-book`: RTS's trading rules as executable definitions and their plain explanation.
- `rts-signals/trade-simulation`: base-trade and full-rules simulation on MT5 M1 bars, gross and net of spread,
  on the New York / Berlin clocks.
- `rts-signals/evaluation`: the descriptive historical reconciliation, the forward ledger, the
  pre-registration and the one scored cell.
- `rts-signals/chart-display`: the TSV export and the read-only MT5 indicator.

### Modified Capabilities
- None.

## Impact

- **New code:** `src/rts_signals/`, `tests/test_rts_signals_*.py`, `mql5/Indicators/ProphetRTS.mq5` and
  `scripts/rts_signals/` (task, indicator install and uninstall, Gmail one-time authorisation).
- **New dependency:** `google-api-python-client` and `google-auth-oauthlib` in `.venv`, for the Gmail API.
  - The OAuth client file and token live in the gitignored `research_models/rts_signals/secrets/`. They are
    never committed or printed.
  - The scope is `gmail.readonly`.
- **New results:** `results/rts_signals/` holds the parsed rows, simulations, ledger, pre-registration,
  registry, report and RUNBOOK. Raw PDFs and caches go in the gitignored `research_models/rts_signals/`.
- **One file outside the repo:** `Common\Files\EURUSDProphet\rts_signals.tsv`.
- **Docs:** a new `ARCHITECTURE_DOCS.md` section and a `CLAUDE.md` invariant.
- **No change** to production, `models/`, `config.json` or any other family.
