# Design

## Context

- **Inputs seen.** The 12 uploaded PDFs cover 2026-11-20 .. 27 and three kinds: Morning, Spot Afternoon and
  Futures. `pypdf`, already in `.venv`, extracts their tables as clean text lines, for example
  `YEN BUY SQUARE 15920 -- -- -- 11/24/26 -- 15898 15858 15942`.
- **Track records.** The track-record pages list closed trades. The PDF text sometimes runs the entry and the
  rule number together, for example `0.81881` and `122390`.
- **Gmail.** It holds about 13 RTS emails with PDFs. Senders are `ber@rts-forex.com` and the RTS assistant's
  address.
- **MT5.** The `mt5app` terminal lists EURUSD, GBPUSD, AUDUSD, USDJPY, USDCHF and USDCAD with M1 history from
  2026-08 (99 999 bars). That covers RTS's whole November track record.
- **Patterns to copy from fx_ny_orb and gold_nyfwd:**
  - the Berlin label clock (`forecast_eval.m15_data.label_now`, zoneinfo for New York);
  - append-only logs and the git pre-registration gate;
  - `fx_bb.score`'s day-block bootstrap;
  - a TSV plus read-only MQL5 indicator, with schtasks via a hidden VBS runner.

## Goals / Non-Goals

**Goals:**
- A parser that never invents a trade: any doubt becomes a flag, never a guess.
- A base-trade simulation simple enough to be checked by hand from the minute bars.
- A forward verdict decided only on data after the pre-registration.

**Non-Goals:**
- Executing, copying or sizing RTS trades, or any order code.
- Futures. Their reports are logged `futures — not read` and add no row (owner decision 2026-11-28: spot only).
- Scoring the full-rules chain, or the Morning report's long-term positions.
- Re-deciding anything from RTS's gross track record.

## Decisions

**D1. A new package `src/rts_signals/`.** RTS is a third-party source with its own life cycle (inbox, parser,
ledger), unrelated to any model family. A separate package keeps its writes and tests isolated.

**D2. Gmail API with `gmail.readonly`, plus a drop folder.**
- `scripts/rts_signals/authorize_gmail.ps1` runs the one-time OAuth flow in the browser. The owner creates the
  Google Cloud OAuth client.
- The client file and token live in the gitignored `research_models/rts_signals/secrets/`.
- The query is `from:(ber@rts-forex.com OR <assistant address>) has:attachment filename:pdf`. The addresses
  are kept in the family config, which is part of the repo, and never in logs.
- Alternatives:
  - IMAP with an app password. It needs 2-step verification and grants full mail access, so it was rejected.
  - Claude's own Gmail connector. It works only inside a conversation, not in a scheduled task.
- The drop folder `research_models/rts_signals/drop/` always works, also without Gmail.

**D3. Parsing by line patterns, not layout.**
- Each kind has a small set of regular expressions keyed on its title line. The report date is read from the
  document's own date line.
- Track-record numbers are split by the pair's decimals:
  - spot entries have 4 decimals (USDJPY too, as 158.9500).
- Every stored row carries the PDF SHA-256 and `parser_version`.
- An unparseable line is stored as `unparsed` with its text, never dropped.

**D4. Price conversion table.**

| Pair | Factor |
|---|---|
| POUND, EURO, AUD | /10 000 |
| SWISS, CAD (USD base) | /10 000 |
| YEN (USD base) | /100 |

- Track-record P/L figures stay in RTS's units.
- A spot row whose converted price differs from the MT5 M1 close at its 16:00 New York label by more than
  50 points is flagged `price mismatch`. This catches unit errors early.

**D5. The base trade is the scored object.**
- It enters per Rule 0, with proportionality, or per the guerilla rule, inside the session
  [18:00 New York on the report date, the next 16:00 New York close).
- It exits at the stop, the target or the time stop. The time stop is 16:00 New York, or 12:00 on a Friday
  for spot.
- It is scanned on M1 bid bars with the recorded spread:
  - buy: entry ask = bid + spread; target when bid high ≥ target; stop when bid low ≤ stop;
  - sell: mirrored on the ask side;
  - a tie counts as the stop.
- R = |entry − stop| in price terms.
- The full-rules chain is a separate, descriptive replay (D7). This keeps the scored object free of RTS's
  discretionary edge cases, such as "it's your call" or Brexit-style exceptions.

**D6. The rule definition is data.** `src/rts_signals/rules.json` holds every number from the rule book, and
its digest is part of the pre-registration. The plain-language README is generated from it, so the
explanation and the code cannot drift apart.

**D7. Full-rules replay and reconciliation.**
- The replay walks each pair's minute bars across sessions. It keeps one open position at a time and applies
  Rules 2–5 and NPR.
- Reconciliation matches RTS track-record trades to our replay by pair, side, entry date and entry within
  3 points. It labels each `confirmed`, `different exit` or `not reproducible`.
- The broker's spot prices differ from RTS's eSignal composite by a pip or two, so a 3-point tolerance is
  used and stated.

**D8. Scoring.**
- One cell at alpha 0.05, one-sided. The statistic is the mean net R of the first 300 forward base trades.
- It uses `fx_bb.score`'s day-block bootstrap, keyed by report date with 5-day blocks, 10 000 resamples and
  seed 20261202. Blocks are used because six pairs on one day share the dollar move.
- Power is stated in the pre-registration with a declared per-trade SD of 1 R, scaled for the blocks.
- Forward trades are reports dated after the commit day on the label clock and fetched before their session
  ended.

**D9. Clock and task.**
- `EURUSDProphet-RTSInbox` runs every hour at :20 Sofia, every day. :20 is free; the other loggers use :01, :03,
  :05 and :08.
- Each run fetches, parses, simulates settled sessions and rebuilds the export.
- The 4 pm report arrives about 22:00–23:00 Sofia. The session opens at 01:00 Sofia, so today's levels are on
  the chart before the open.

**D10. Chart.**
- `rts_signals.tsv` follows the layout of `gold_ny_range.tsv`: `#meta` and `#text` lines, then rows.
- `ProphetRTS.mq5` copies ProphetNYOrb's reader and objects. It draws only on the six pairs, at M1–H1.
- Trade words such as SELL, stop and target are allowed here because they are RTS's published levels.
- Every label starts with `RTS:` and today's ends with `непроверено / unverified`. There is no lot size and no
  promise.

## Risks / Trade-offs

- **RTS changes the PDF layout.** → The parser stores `unparsed` lines and the report shows a non-zero
  unparsed count. A test fixture holds each known layout.
- **The Gmail token expires or is revoked.** → The pass reports `gmail not authorised`, the drop folder still
  works and the RUNBOOK says how to re-authorise.
- **The owner adds PDFs selectively after seeing outcomes.** → Reports fetched after their session ended are
  ledgered `received late` and excluded from the cell.
- **Broker prices differ from RTS's.** → There is a 3-point reconciliation tolerance, and results are always
  on the owner's own prices, which is what matters for the owner.
- **Paid content.** → Raw PDFs and their text are never committed. Only extracted numbers and digests are.
- **Low power.** About 75 report days give 300 trades. A DROP means no detectable edge net of spread, not
  proof of none. → The pre-registration states the minimum detectable effect.
- **The live levels lead the owner to trade before any verdict.** → The chart corner states that no verdict
  exists. The family contains no execution code. This is the owner's explicit choice.

## Migration Plan

1. Build, and test against the 12 uploaded PDFs as fixtures.
2. The owner authorises Gmail once.
3. Run the historical report (descriptive).
4. Write the pre-registration, which the owner commits.
5. Install the task and the indicator.
6. Rollback: the uninstall scripts remove the task, indicator and TSV. Stored rows remain.

## Open Questions

- What is the exact sender address used for the assistant's mails? It is read from the Gmail search at
  implementation, and both known senders are accepted.
