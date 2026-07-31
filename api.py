import os
import sys
import json
import time
import subprocess
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
_retrain = {"proc": None, "started_at": None, "reloaded": False}
RETRAIN_LOG = os.path.join(BASE_DIR, "results", "retrain.log")


def _tail(path, n=15):
    """Last n lines of a file, tolerant of the subprocess writing concurrently."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


@app.post("/api/retrain")
def start_retrain():
    """Launch _train_pipeline.py in the background (non-blocking). 409 if a run
    is already in progress so concurrent clicks can't spawn parallel trainings."""
    proc = _retrain["proc"]
    if proc is not None and proc.poll() is None:
        raise HTTPException(status_code=409, detail="A retraining run is already in progress.")

    os.makedirs(os.path.dirname(RETRAIN_LOG), exist_ok=True)
    logf = open(RETRAIN_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "_train_pipeline.py"],
        cwd=BASE_DIR, stdout=logf, stderr=subprocess.STDOUT,
    )
    _retrain.update(proc=proc, started_at=time.time(), reloaded=False)
    return {"state": "started", "pid": proc.pid}


@app.get("/api/retrain/status")
def retrain_status():
    """Poll the background retrain. On success, hot-reload the freshly written
    artifacts ONCE so subsequent predictions use the new models without a server
    restart."""
    global service
    proc = _retrain["proc"]
    if proc is None:
        return {"state": "idle"}

    returncode = proc.poll()
    elapsed = int(time.time() - _retrain["started_at"]) if _retrain["started_at"] else 0

    if returncode is None:
        return {"state": "running", "elapsed_seconds": elapsed, "log_tail": _tail(RETRAIN_LOG)}

    if returncode == 0:
        if not _retrain["reloaded"]:
            service = PredictionService(BASE_DIR, CONFIG)   # reload new artifacts in-place
            _retrain["reloaded"] = True
        return {"state": "completed", "elapsed_seconds": elapsed,
                "models_ready": service.models_ready, "log_tail": _tail(RETRAIN_LOG)}

    return {"state": "failed", "returncode": returncode,
            "elapsed_seconds": elapsed, "log_tail": _tail(RETRAIN_LOG)}


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
    '&nbsp;·&nbsp; <a href="/kronos-direction">🜂 Kronos forecast ledger</a> (external) '
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


@app.get("/api/kronos-direction")
def kronos_direction_endpoint():
    """
    Kronos next-H1-bar forecast, ON DEMAND. EXTERNAL foundation model, zero-shot.

    `p_up` is the headline field, not `direction`: Kronos emits a distribution
    over sampled price paths and p_up is the share closing above the last actual
    close. `mc_noise_estimate` rides along so a reader can see how precise that
    number is (8.17pp run-to-run at 30 paths) rather than over-reading it.

    GRACEFUL DEGRADATION. torch and a 102M-parameter checkpoint are OPTIONAL. A
    missing package, missing checkpoint, absent GPU or load failure returns a
    clear `available: false` and can NEVER affect /api/predict, /history,
    /paper-trading or /h1-direction.

    Observational. Simulated ledger only. Not a trading instruction.
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
        result = service.predict_kronos_direction()
    except Exception as e:
        unavailable["reason"] = f"Kronos prediction unavailable: {e}"
        return unavailable

    try:
        from src.external.kronos.serving import build_ledger, log_prediction
        log_prediction(result, base_dir=BASE_DIR)
        ledger = build_ledger(base_dir=BASE_DIR)
        settled = int((ledger['model_version'] == result['model_version']).sum()) if len(ledger) else 0
    except Exception:
        settled = 0                        # logging must never break the response

    result["available"] = True
    result["forward_observations"] = settled
    return result


@app.get("/kronos-direction", response_class=HTMLResponse)
def kronos_direction_page():
    """Forward ledger for the external Kronos model, grouped by model_version and
    leading with CALIBRATION rather than hit rate — the clean-window evaluation
    showed the model is sharp, so whether p_up means anything is the open
    question worth watching."""
    from src.external.kronos.serving import build_and_save, read_log, render_html

    ledger = build_and_save(base_dir=BASE_DIR)
    return HTMLResponse(content=render_html(ledger, read_log(base_dir=BASE_DIR),
                                            base_dir=BASE_DIR))


if __name__ == "__main__":
    # Makes this file directly runnable: `python api.py` (or a double-click via
    # start.bat) launches the server, instead of silently importing the module
    # and exiting. Equivalent to `python -m uvicorn api:app`.
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting EUR/USD Prophet -> http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("api:app", host=host, port=port, reload=False)
