import os
import sys
import json
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
