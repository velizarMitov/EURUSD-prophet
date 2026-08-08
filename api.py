import os
import sys
import json
import time
import subprocess
import threading
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, '.env'))

from src.inference import PredictionService

with open(os.path.join(BASE_DIR, 'config.json')) as f:
    CONFIG = json.load(f)

app = FastAPI(title="EURUSD Multi-Task Production Predictor API")
service = PredictionService(BASE_DIR, CONFIG)


@app.post("/api/predict")
def predict_endpoint():
    """
    Fully automated: fetches the latest available EURUSD bar itself (live,
    falling back to the bundled historical tail if no fresher data is
    reachable), runs it through the serialized PCA/scaling/model pipeline,
    and returns both models' predictions plus a committee consensus. Takes
    no request body -- there is nothing left for a caller to supply.
    """
    if not service.models_ready:
        raise HTTPException(status_code=503, detail=f"Model artifacts missing. Errors: {service.load_errors}")

    try:
        return service.predict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Automated data pipeline failed: {e}")


# Ensure the UI path maps correctly
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_root():
    """Serve the zero-input dashboard (static/index.html) at the API root,
    or a minimal JSON health summary if the static/ directory is absent."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API Active. Baseline (price-only) ready: " + str(service.baseline_ready) +
            ", With-macro ready: " + str(service.macro_ready) + ", H1 ready: " + str(service.h1_ready)}


# ── Background retraining ──────────────────────────────────────────────────
# _train_pipeline.py takes ~15-30 min (GridSearch + 3 LSTM fits), so it must run
# as a detached subprocess rather than blocking an HTTP request. The frontend
# fires POST /api/retrain once, then polls GET /api/retrain/status.
#
# The log is the ONLY thing the owner can see while a run is in flight, so three
# properties are load-bearing (all three failed together on 2026-08-07, costing
# four hours of waiting on a run that had stopped making progress after 21 min):
#
#   1. The child runs UNBUFFERED. CPython block-buffers stdout at 8 KB when it is
#      a file rather than a TTY, while stderr stays line-buffered. That asymmetry
#      let library warnings (stderr) through while swallowing every print() from
#      the volatility stage onward; because the run then hung instead of exiting,
#      the buffer was never flushed and that output was lost outright, not merely
#      delayed. `-u` plus PYTHONUNBUFFERED (which also reaches the nested 12C
#      subprocess) removes the buffer entirely.
#   2. The log ALWAYS ends with _MARKER. "Is it done?" must be answerable by
#      reading the last line -- never by comparing artifact mtimes.
#   3. A run is judged by evidence that survives a server restart. The in-memory
#      Popen handle used to be the only record, so restarting the server reported
#      "idle" for a finished run and silently skipped the hot-reload, leaving the
#      API serving stale models with fresh ones already on disk.
_retrain = {"proc": None, "started_at": None, "reloaded": False,
            "returncode": None, "finished_at": None, "pid": None}
RETRAIN_LOG = os.path.join(BASE_DIR, "results", "retrain.log")
RETRAIN_STATE = os.path.join(BASE_DIR, "results", "retrain_state.json")
_MARKER = "=== RETRAIN EXIT"
# A log that has not grown in this long, with no marker, is not "working" --
# it is stalled or orphaned, and the status must say so rather than "running".
_STALL_SECONDS = 300


def _tail(path, n=15):
    """Last n lines of a file, tolerant of the subprocess writing concurrently."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _log_age():
    """Seconds since the log was last written, or None if there is no log."""
    try:
        return max(0.0, time.time() - os.path.getmtime(RETRAIN_LOG))
    except OSError:
        return None


def _marker_returncode():
    """Exit code parsed from the log's completion marker, or None if the run
    never reached one (killed, hung, or still going)."""
    try:
        with open(RETRAIN_LOG, encoding="utf-8", errors="replace") as f:
            for line in reversed(f.readlines()):
                if line.startswith(_MARKER):
                    for tok in line.split():
                        if tok.startswith("rc="):
                            return int(tok[3:])
    except (OSError, ValueError):
        pass
    return None


def _write_marker(returncode, elapsed):
    """Append the single line that terminates EVERY run, whatever its fate."""
    try:
        with open(RETRAIN_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{_MARKER} rc={returncode} elapsed={elapsed:.1f}s "
                    f"at={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except OSError:
        pass


def _save_state():
    try:
        with open(RETRAIN_STATE, "w", encoding="utf-8") as f:
            json.dump({k: _retrain[k] for k in
                       ("pid", "started_at", "returncode", "finished_at")}, f)
    except OSError:
        pass


def _watch_retrain(proc, started_at):
    """Reap the child and stamp the log. Runs on a daemon thread so the marker
    is written even when the run fails or is killed -- the cases where the owner
    most needs the log to say so."""
    returncode = proc.wait()
    # started_at is re-stamped (not merely read) so the persisted state is
    # complete on its own -- a restart needs it to report elapsed time.
    _retrain.update(returncode=returncode, finished_at=time.time(),
                    started_at=started_at)
    _write_marker(returncode, time.time() - started_at)
    _save_state()


def _recover_retrain_state():
    """Rebuild what is knowable about a run this process did not launch."""
    try:
        with open(RETRAIN_STATE, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return
    if not saved.get("started_at"):
        return
    _retrain.update(pid=saved.get("pid"), started_at=saved.get("started_at"),
                    returncode=saved.get("returncode"),
                    finished_at=saved.get("finished_at"))
    if _retrain["returncode"] is None:
        # The marker outlives the process that wrote it; trust it over our memory.
        _retrain["returncode"] = _marker_returncode()


_recover_retrain_state()


def _resolve_state():
    """(state, returncode) for the current or most recent run.

    Uses the live Popen handle when we own the child, and falls back to durable
    on-disk evidence (exit marker + log freshness) when we do not."""
    proc = _retrain["proc"]
    if proc is not None:
        returncode = proc.poll()
        if returncode is None:
            return "running", None
        return ("completed" if returncode == 0 else "failed"), returncode

    if _retrain["started_at"] is None:
        return "idle", None

    returncode = _retrain["returncode"]
    if returncode is not None:
        return ("completed" if returncode == 0 else "failed"), returncode

    # No marker: the child outlived (or was killed with) the server that spawned
    # it. A log still growing proves it is alive; a cold one proves nothing good.
    age = _log_age()
    if age is not None and age < _STALL_SECONDS:
        return "running", None
    return "interrupted", None


@app.post("/api/retrain")
def start_retrain():
    """Launch _train_pipeline.py in the background (non-blocking). 409 if a run
    is already in progress so concurrent clicks can't spawn parallel trainings
    -- two pipelines interleaving writes into models/ is how the artifact set
    got torn once before."""
    if _resolve_state()[0] == "running":
        raise HTTPException(status_code=409, detail="A retraining run is already in progress.")

    os.makedirs(os.path.dirname(RETRAIN_LOG), exist_ok=True)
    started_at = time.time()
    logf = open(RETRAIN_LOG, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "_train_pipeline.py"],
            cwd=BASE_DIR, stdout=logf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        # Popen duplicates the handle for the child, so the parent's copy is dead
        # weight -- and leaking one per run eventually exhausts the handle table.
        logf.close()

    _retrain.update(proc=proc, started_at=started_at, reloaded=False,
                    returncode=None, finished_at=None, pid=proc.pid)
    _save_state()
    threading.Thread(target=_watch_retrain, args=(proc, started_at), daemon=True).start()
    return {"state": "started", "pid": proc.pid}


@app.get("/api/retrain/status")
def retrain_status():
    """Poll the background retrain. On success, hot-reload the freshly written
    artifacts ONCE so subsequent predictions use the new models without a server
    restart."""
    global service
    state, returncode = _resolve_state()
    if state == "idle":
        return {"state": "idle"}

    elapsed = int((_retrain["finished_at"] or time.time()) - _retrain["started_at"]) \
        if _retrain["started_at"] else 0
    age = _log_age()
    payload = {"state": state, "elapsed_seconds": elapsed,
               "log_age_seconds": None if age is None else int(age),
               "log_tail": _tail(RETRAIN_LOG)}

    if state == "running":
        # Surfaced so a silent run reads as silent. The 2026-08-07 hang showed a
        # climbing elapsed time and nothing else; log_age_seconds is the number
        # that distinguishes "working" from "wedged".
        payload["supervised"] = _retrain["proc"] is not None
        payload["stalled"] = age is not None and age >= _STALL_SECONDS
        return payload

    if state == "completed":
        if not _retrain["reloaded"]:
            service = PredictionService(BASE_DIR, CONFIG)   # reload new artifacts in-place
            _retrain["reloaded"] = True
        payload["models_ready"] = service.models_ready
        return payload

    if state == "failed":
        payload["returncode"] = returncode
        return payload

    # interrupted: no exit marker and a cold log -- the run died without ever
    # reporting. Artifacts on disk may be a partial set; say so instead of
    # guessing "completed" and hot-reloading a torn model directory.
    payload["detail"] = ("The retrain stopped without writing a completion marker "
                         "(server restarted, or the process was killed/hung). "
                         "Artifacts may be from a partial run -- verify before trusting them.")
    return payload


@app.get("/history", response_class=HTMLResponse)
def prediction_history():
    """
    Render the prediction-vs-actual comparison table: every logged forecast,
    scored against the realised EUR/USD close once its forecast date has closed.
    The page is rebuilt live (fresh actuals) on each request and also written to
    results/prediction_history.html for offline viewing.
    """
    from src.tracking import build_history_html

    tracking_cfg = CONFIG.get('tracking', {})
    log_path = os.path.join(BASE_DIR, tracking_cfg.get('log_path', 'results/prediction_log.csv'))
    html = build_history_html(log_path, CONFIG['data'])

    try:
        out_path = os.path.join(BASE_DIR, tracking_cfg.get('html_path', 'results/prediction_history.html'))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
    except OSError:
        pass

    return HTMLResponse(content=_with_h1_nav(html))


@app.get("/api/paper-trading")
def paper_trading_summary():
    """JSON scorecards + ledgers for the simulated forward paper-trading
    harness — ONE ledger PER MODEL VARIANT (baseline price-only vs with_macro),
    both driven by the same prediction log. Rebuilt live from the log joined to
    realised closes; the per-variant CSVs are refreshed on each call. Simulated
    only — no broker orders are placed."""
    from src.paper_trading import build_all_ledgers

    pt_cfg = CONFIG.get('paper_trading', {})
    log_path = os.path.join(BASE_DIR, CONFIG.get('tracking', {}).get('log_path', 'results/prediction_log.csv'))
    spread_pips = pt_cfg.get('spread_pips', 1.5)

    all_ledgers = build_all_ledgers(log_path, CONFIG['data'], pt_cfg, base_dir=BASE_DIR)
    return {
        "spread_pips": spread_pips,
        "variants": {
            name: {"summary": blk['summary'], "ledger": blk['ledger'].to_dict(orient='records')}
            for name, blk in all_ledgers.items()
        },
    }


@app.get("/paper-trading", response_class=HTMLResponse)
def paper_trading_page():
    """Render BOTH variants' simulated paper-trading ledgers + running
    scorecards (cumulative net P&L, win rate, Sharpe-like ratio, max drawdown),
    net of a realistic retail spread. Whichever variant nets better cost-
    adjusted P&L over a meaningful forward window is the honest winner — these
    ledgers are the primary production-worthiness arbiter going forward
    (ARCHITECTURE_DOCS.md Production Methodology)."""
    from src.paper_trading import build_all_ledgers, render_html

    pt_cfg = CONFIG.get('paper_trading', {})
    log_path = os.path.join(BASE_DIR, CONFIG.get('tracking', {}).get('log_path', 'results/prediction_log.csv'))
    spread_pips = pt_cfg.get('spread_pips', 1.5)

    all_ledgers = build_all_ledgers(log_path, CONFIG['data'], pt_cfg, base_dir=BASE_DIR)
    return HTMLResponse(content=_with_h1_nav(render_html(all_ledgers, spread_pips)))


# Cross-links to the H1 direction ledger, injected HERE rather than edited into
# src/tracking.py and src/paper_trading.py. Those two modules are byte-pinned --
# src/paper_trading.py by a pre-existing test that diffs it against git HEAD, and
# both by the protected-set fixtures -- so adding navigation at the render layer
# gives the pages their links without modifying a single page module.
_H1_NAV = (
    '<p style="max-width:1100px;margin:1.25rem auto;padding:0 1rem;font-size:.85rem;'
    'font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#555;">'
    'Also: <a href="/h1-direction">⏱ H1 next-bar direction ledger</a> (observational) '
    '&nbsp;·&nbsp; <a href="/kronos-volatility">🜂 Kronos volatility record</a> (external) '
    '&nbsp;·&nbsp; <a href="/">dashboard</a> '
    '&nbsp;·&nbsp; <a href="/history">prediction-vs-actual history</a> '
    '&nbsp;·&nbsp; <a href="/paper-trading">paper-trading ledgers</a></p>')


def _with_h1_nav(html: str) -> str:
    """Append the cross-link block without touching the page modules."""
    if '</body>' in html:
        return html.replace('</body>', _H1_NAV + '</body>', 1)
    return html + _H1_NAV


@app.get("/api/h1-direction")
def h1_direction_endpoint():
    """
    H_dir.1 next-H1-bar direction, ON DEMAND. One call, one prediction.

    The base bar is always the last FULLY CLOSED hourly bar — the currently
    forming hour is never the base. `minutes_remaining` is a first-class field,
    not a nicety: the model predicts the CURRENT, still-forming hour, so a call
    at 14:05 leaves 55 minutes of the predicted move ahead while a call at 14:50
    leaves 10. When the forecast bar has ALREADY closed (weekend, holiday or a
    stale feed) the response says so via `forecast_bar_status` and never presents
    a bar that is already history as though it were actionable.

    Observational. Simulated ledger only. Not a trading instruction. Returns a
    clear `available: false` payload rather than a 500 when the artifacts or the
    H1 feed are missing, and can never affect /api/predict, /history or
    /paper-trading.
    """
    if not service.h1_dir_ready:
        return {
            "available": False,
            "reason": "H1 direction artifacts missing (models/h1_direction/).",
            "errors": [e for e in service.load_errors if e.startswith("H1 direction")],
            "disclaimer": "Observational. Simulated ledger only. Not a trading instruction.",
        }
    try:
        result = service.predict_h1_direction()
    except Exception as e:
        return {
            "available": False,
            "reason": f"H1 data pipeline unavailable: {e}",
            "disclaimer": "Observational. Simulated ledger only. Not a trading instruction.",
        }

    try:
        from src.h1_direction_serving import log_prediction, build_ledger
        log_prediction(result, log_path=os.path.join(BASE_DIR, 'results/h1_direction_log.csv'))
        ledger = build_ledger(log_path=os.path.join(BASE_DIR, 'results/h1_direction_log.csv'),
                              base_dir=BASE_DIR)
        settled = int((ledger['model_version'] == result['model_version']).sum()) if len(ledger) else 0
    except Exception:
        settled = 0                       # logging must never break the response

    result["available"] = True
    result["forward_observations"] = settled
    return result


@app.get("/h1-direction", response_class=HTMLResponse)
def h1_direction_page():
    """Forward ledger for the observational H1 direction model, grouped BY
    model_version. A blended hit rate across a retrain would describe a model
    that is no longer served, so the view never leads with one."""
    from src.h1_direction_serving import build_and_save, read_log, render_html

    log_path = os.path.join(BASE_DIR, 'results/h1_direction_log.csv')
    ledger = build_and_save(log_path=log_path, base_dir=BASE_DIR)
    return HTMLResponse(content=render_html(ledger, read_log(log_path), base_dir=BASE_DIR))


@app.get("/api/kronos-volatility")
def kronos_volatility_endpoint():
    """
    Kronos next-24-hour VOLATILITY forecast, ON DEMAND. EXTERNAL foundation model
    (Kronos-mini, 4.1M), zero-shot, at the authors' own demo configuration.

    This REPLACED the direction channel, which was measured to carry no
    information three separate ways. Volatility is the channel with measured
    signal: amplification AUC 0.689 [0.645, 0.731], n=519.

    WHAT IS THE MODEL'S AND WHAT IS OURS. `p_vol_amp_raw` and `pred_vol_pct_24h`
    are the model's output. `p_vol_amp_calibrated` and `pred_vol_pct_24h_scaled`
    apply OUR frozen corrections from models/external_kronos/vol_calibration.json
    — loaded, never refitted at request time — and are labelled as ours.

    GRACEFUL DEGRADATION. torch, the checkpoint and the calibration file are
    OPTIONAL. Anything missing returns a clear `available: false` and can NEVER
    affect /api/predict, /history, /paper-trading or /h1-direction.

    OBSERVATIONAL. Not tested against this project's own volatility ensemble.
    Simulated ledger only. Not a trading instruction.
    """
    unavailable = {
        "available": False,
        "disclaimer": ("External foundation model, zero-shot. Observational, "
                       "simulated only, not a trading instruction."),
    }
    if not getattr(service, "kronos_ready", False):
        unavailable["reason"] = (
            getattr(service, "kronos_error", None)
            or "Kronos unavailable (see requirements-kronos.txt).")
        return unavailable
    try:
        result = service.predict_kronos_volatility()
    except Exception as e:
        unavailable["reason"] = f"Kronos volatility unavailable: {e}"
        return unavailable

    try:
        from src.external.kronos.vol_serving import build_ledger, log_prediction
        log_prediction(result, base_dir=BASE_DIR)
        ledger = build_ledger(base_dir=BASE_DIR)
        settled = int((ledger['model_version'] == result['model_version']).sum()) if len(ledger) else 0
    except Exception:
        settled = 0                        # logging must never break the response

    result["available"] = True
    result["forward_observations"] = settled
    return result


@app.get("/kronos-volatility", response_class=HTMLResponse)
def kronos_volatility_page():
    """Forward record for the Kronos volatility channel, grouped by
    model_version and leading with CALIBRATION — the clean window showed the raw
    probability discriminates but is badly calibrated, so whether our frozen
    correction holds forward is the open question worth watching."""
    from src.external.kronos.vol_serving import build_and_save, read_log, render_html

    ledger = build_and_save(base_dir=BASE_DIR)
    return HTMLResponse(content=render_html(ledger, read_log(base_dir=BASE_DIR),
                                            base_dir=BASE_DIR))


@app.get("/api/kronos-direction")
def kronos_direction_endpoint():
    """
    RETIRED. Kept responding for one release so nothing that calls it breaks
    silently; it never returns a direction.

    Direction was measured dead three separate ways: next-bar AUC 0.509 (an
    out-of-sample isotonic recalibration fails to beat a constant), 24-bar AUC
    0.517 with Brier skill −0.62, and cross-sectionally a RankIC of +0.0199
    entirely attributable to a one-line reversal ranking (orthogonalised CI
    [−0.00127, +0.01768] at a powered MDE of 0.0133).

    A served number with no information is worse than no number, because it
    looks like information.
    """
    from src.external.kronos.loader import DIRECTION_RETIRED_REASON
    return {
        "available": False,
        "retired": True,
        "reason": DIRECTION_RETIRED_REASON,
        "replacement": "/api/kronos-volatility",
        "disclaimer": ("External foundation model, zero-shot. Observational, "
                       "simulated only, not a trading instruction."),
    }


@app.get("/kronos-direction", response_class=HTMLResponse)
def kronos_direction_page():
    """RETIRED view. Reachable, but its content is the retirement note and a
    link — the historical ledger stays on disk and is not deleted."""
    from src.external.kronos.loader import DIRECTION_RETIRED_REASON
    return HTMLResponse(content=(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Kronos direction — retired</title>'
        '<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;'
        'margin:2rem auto;max-width:46rem;padding:0 1rem;color:#1a1a1a;background:#fafafa}'
        '.warn{background:#fff8e1;border-left:4px solid #ffa000;padding:.75rem 1rem}'
        '.nav{font-size:.9rem;color:#555}'
        '@media (prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}'
        '.warn{background:#2b2410}.nav{color:#aaa}}</style></head><body>'
        '<h1>Kronos direction — retired</h1>'
        '<div class="warn"><strong>%s</strong></div>'
        '<p>Direction was measured to carry no information three separate ways:</p>'
        '<ul><li>next-bar, AUC 0.509 — an out-of-sample isotonic recalibration '
        'fails to beat a constant forecast</li>'
        '<li>24-bar, AUC 0.517, and Brier skill got <em>worse</em> (−0.62): the model '
        'became confident without becoming informative</li>'
        '<li>cross-sectionally, RankIC +0.0199 entirely attributable to a one-line '
        'reversal ranking; orthogonalised CI [−0.00127, +0.01768] at a powered '
        'minimum detectable effect of 0.0133</li></ul>'
        '<p>The historical direction ledger is retained on disk and was not deleted. '
        'The replacement is <a href="/kronos-volatility">the Kronos volatility '
        'forward record</a>.</p>'
        '<p class="nav">Also: <a href="/">dashboard</a> &middot; '
        '<a href="/history">history</a> &middot; '
        '<a href="/paper-trading">paper trading</a> &middot; '
        '<a href="/h1-direction">H1 direction</a> &middot; '
        '<a href="/kronos-volatility">Kronos volatility</a></p>'
        '</body></html>' % DIRECTION_RETIRED_REASON))


if __name__ == "__main__":
    # Makes this file directly runnable: `python api.py` (or a double-click via
    # start.bat) launches the server, instead of silently importing the module
    # and exiting. Equivalent to `python -m uvicorn api:app`.
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting EUR/USD Prophet -> http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("api:app", host=host, port=port, reload=False)
