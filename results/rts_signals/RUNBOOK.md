# RTS signals — RUNBOOK / ръководство

> Levels are Rockefeller Treasury Services' suggestions, not this project's signal. Results are net of the
> recorded MT5 spread only (no slippage, swap or commission) and no verdict is a trading instruction.
> / Нивата са предложения на RTS. Резултатите са нето само от спреда; нищо тук не е търговски съвет.

All commands run from the repository root in PowerShell.

## 1. Gmail, once / Gmail, еднократно

1. In Google Cloud (Google Auth Platform) create a project, enable **Gmail API**, set the audience to
   **External**, add yourself as a test user, add the scope `https://www.googleapis.com/auth/gmail.readonly`,
   then create an OAuth client of type **Desktop app** and download its JSON.
   Then publish the app to **In production**: in Testing mode Google revokes the token every 7 days.
2. Save the JSON as `research_models\rts_signals\secrets\client_secret.json`. Never paste it anywhere.
3. Authorise (a browser opens; on "Google hasn't verified this app" choose Advanced → continue):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\rts_signals\authorize_gmail.ps1
   powershell -ExecutionPolicy Bypass -File scripts\rts_signals\authorize_gmail.ps1 -Status
   ```

   `-Status` prints `gmail authorised` or `gmail not authorised`. If it says not authorised later (token
   revoked or expired), run step 3 again. The drop folder works without Gmail.

## 2. Drop folder / ръчно пускане

Save any RTS PDF into `research_models\rts_signals\drop\`. The next pass ingests it once (by SHA-256).
A report saved after its session ended is recorded `received late` and never counts in the forward test.

```powershell
.venv\Scripts\python.exe -m src.rts_signals.ingest
```

## 3. One pass by hand / едно минаване на ръка

```powershell
.venv\Scripts\python.exe -m src.rts_signals.run              # fetch, ingest, bars, simulate, ledger, chart
.venv\Scripts\python.exe -m src.rts_signals.run --no-gmail   # drop folder only
```

Each part also runs alone: `src.rts_signals.bars`, `.simulate`, `.reconcile`, `.history`, `.report`,
`.export`, `.rules --readme` (all accept `--help`).

## 4. The pre-registration — the owner's step / предварителна регистрация

```powershell
.venv\Scripts\python.exe -m src.rts_signals.prereg --write
git add results/rts_signals/PRE_REGISTRATION.md results/rts_signals/PRE_REGISTRATION.json
git commit -m "rts-signals: pre-registration"
.venv\Scripts\python.exe -m src.rts_signals.prereg --check
```

Until `--check` prints `committed and unchanged`, no forward trade counts. Forward trades are reports dated
after the commit day.

## 5. The scheduled task / задачата

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rts_signals\install_tasks.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\rts_signals\install_tasks.ps1
powershell -ExecutionPolicy Bypass -File scripts\rts_signals\uninstall_tasks.ps1
```

`EURUSDProphet-RTSInbox` runs every hour at :20. Log: `research_models\rts_signals\EURUSDProphet-RTSInbox.log`.

## 6. The chart / графиката

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rts_signals\install_mt5_indicator.ps1
powershell -ExecutionPolicy Bypass -File scripts\rts_signals\uninstall_mt5_indicator.ps1
```

In MT5: Navigator → Indicators → EURUSDProphet → ProphetRTS, onto an EURUSD, GBPUSD, AUDUSD, USDJPY,
USDCHF or USDCAD chart at H1 or lower.
- Today: blue entry line with an arrow, red dashed stop, green dashed target, label `RTS: … непроверено`.
- Past: entry-to-exit line, green for a net gain and red for a net loss, or a dotted entry with the
  reason there was no trade.
- Corner: the forward count `N / 300 · няма присъда`, and RTS's Morning positions.

## 7. Scoring, once / оценката, веднъж

```powershell
.venv\Scripts\python.exe -m src.rts_signals.score
```

It refuses before 300 forward trades unless it is 2027-06-30 or later (then `UNDERPOWERED — NO DECISION`).
A second run refuses and changes nothing.

## Refusals and messages / откази

| message | meaning |
|---|---|
| `gmail not authorised` | no or revoked token: section 1, step 3 |
| `mt5 unreachable` | the mt5app terminal is closed; bars catch up on the next pass |
| `export failed` | the MT5 Common Files folder is not writable; stored rows are unaffected |
| `not counting: … is not committed` | section 4 |
| `ConfigChanged` | the config or `rules.json` changed after the pre-registration: a new hypothesis |
| `price mismatch` | RTS's 4 pm price is more than 50 points from MT5's: a unit or typo problem |
| `received late` | the report arrived after its session ended; excluded from the cell |
