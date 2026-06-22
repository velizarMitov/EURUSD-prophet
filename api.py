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
    return {"message": "API Active. GBM ready: " + str(service.gbm_ready) + ", LSTM ready: " + str(service.lstm_ready)}


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

    return HTMLResponse(content=html)
