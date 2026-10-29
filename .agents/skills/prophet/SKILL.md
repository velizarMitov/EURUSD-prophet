---
name: prophet
description: "Operate the EUR/USD Prophet app in this repo: a read-only health and status report, starting or opening the app, and diagnosing why it will not start or a page fails. This is the project's launch skill for /run. Use for: how is Prophet today, is the app OK, status, health check, start / run / open the app, it won't start, port 8000 busy, is the gold logger running, did the retrain finish, scheduled tasks. Bulgarian: как е Prophet днес, как е приложението, статус, пусни приложението, отвори го, не тръгва, защо не работи, работи ли gold логерът, мина ли retrain-ът. Never predicts, retrains, restarts without a yes, or gives trading advice. Do NOT use for changing code, adding features or hypotheses, retraining, or research and statistics questions."
---

# EUR/USD Prophet: operator

The app is one FastAPI server on `http://127.0.0.1:8000`. Four scheduled tasks
log forecasts in the background, and simulated ledgers track them. This skill
reports how the app is doing, starts it, and finds out why it fails. It changes
nothing along the way. `AGENTS.md` governs any work on the code; this skill
does not.

## Route the request

| The owner asks | Do |
|---|---|
| How is it / status / is everything OK | Full pass → read `references/reading-status.md` → [reply](#reply) |
| Is it up? / one area (gold logger, retrain, tasks) | `--fast` pass → report that signal using its section in `reading-status.md` |
| Start / open / run it (including `/run`) | `--fast` pass → `references/launch.md` |
| It will not start / a page errors | `references/launch.md` → *It will not start*; full pass if the server is up |
| Should I buy / sell / what to trade | [Hard limit 4](#hard-limits) |
| Change code, add a feature, retrain, research | Not this skill |

## Run the pass

From the repo root:

```
.venv\Scripts\python.exe .Codex\skills\prophet\scripts\status.py          # full, up to ~70 s
.venv\Scripts\python.exe .Codex\skills\prophet\scripts\status.py --fast   # ~10 s, skips the hit rate
```

It prints one JSON document:
- `overall`;
- `signals`: each with `id`, `level`, `summary`, `see` and `details`;
- `performance`: entries at level `info`.

The pass always exits 0 when it printed a report. A source that fails becomes
`unknown` with the reason, and the other signals still report. When the server
is down, the endpoint signals are `unknown`, and the file, task and process
signals still report.

Use the pass instead of calling endpoints by hand. If you need more detail than
it gives, read the file named in `see`. Do not call another endpoint (see the
hard limits).

## Hard limits

1. **No predicting and no retraining.**
   - Never call `POST /api/predict`. It appends a row to the owner's forward
     record.
   - Never call `POST /api/retrain`. It rewrites `models/`.
2. **No reads that write.**
   - `GET /api/paper-trading` rewrites the ledger CSVs.
   - `GET /api/retrain/status` can hot-reload the models.
   - `GET /history` rewrites `results/prediction_history.html`.

   The pass already reads their files instead.
3. **Starting is allowed. Restarting and stopping need a yes.**
   - When the server is down, start it with the launcher. On stale code,
     running the launcher restarts the server, so ask first.
   - Never stop a process that the launcher does not own.
4. **No trading advice.**
   - Never say buy, sell or hold, a size, a time or a ranking.
   - Report what the app shows and how reliable it is.
   - The ledgers are a simulation, not a supported edge.
5. **Report a fix, do not perform it.** These are the owner's decisions, so the
   skill reports them and acts only after a yes:
   - re-pinning fixtures or committing;
   - restoring `models/`;
   - reinstalling a scheduled task;
   - installing a package.

   Any commit follows `AGENTS.md`.
6. **Link project rules, do not copy them.** Name the rule and where it lives,
   for example "`AGENTS.md` → Production Methodology, item 4".

## Reply

- **Language:** write in the owner's language. A question in Bulgarian gets a
  Bulgarian answer.
- **Line 1:** the overall state in plain words. Never answer with only
  "red/yellow"; give the reason.
- **Then each `red` signal, then each `yellow` one,** one short line each:
  - what is wrong and since when;
  - where to see it (from `see`);
  - who acts or what fixes it.
- **One line for everything else that is fine,** naming it.
- **Performance in one line:** the direction hit rate with N and its 95 %
  interval, or "too early" when N is below 30. Give pips or P&L only when the
  owner asks about the ledgers.
- **At most one question at the end,** and only if a decision is needed, for
  example a restart.
- **No JSON and no tables of every field.**

An example (2026-10-29):

> Приложението работи, но има две неща за решаване и три за наглеждане.
> - **Модели без декларация от 15 дни:** 13 артефакта в `models/` се различават
>   от декларираните. Те съвпадат с retrain-а от 14.10. Виж банера на
>   `/advanced`. Решението е твое: декларираш с `RETRAIN:` commit или връщаш
>   старите.
> - **Gold logger-ът е спрял:** 7 пропуснати часа за 24 ч. Виж `/gold`. Компютърът
>   трябва да е буден, а MT5 отворен.
> - Днешната прогноза още не е направена: бутонът за опресняване на `/`.
> - Последното пускане на задачата Refit е отказано от Task Scheduler (`0x800710E0`).
> - Един пропуснат проход на forward logger-а: MT5 не е отговорил.
>
> Сървърът, retrain-ът и логът са наред. Посока: 21 от 53 (40 %, 95 % интервал
> 28–53 %), неразличимо от монета.
