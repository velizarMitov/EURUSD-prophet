import os
import sys
import warnings

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.features import load_history, build_live_features

app = FastAPI(title="EURUSD Multi-Task Production Predictor API")

MODELS_DIR = os.path.join(BASE_DIR, 'models')
GBM_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'best_gbm_eurusd.pkl')
GBM_REGRESSOR_PATH = os.path.join(MODELS_DIR, 'best_gbm_regressor_eurusd.pkl')
GBM_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_gb_eurusd.pkl')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_multitask_eurusd.keras')
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_lstm_multitask.pkl')
LSTM_TIME_STEPS_PATH = os.path.join(MODELS_DIR, 'lstm_time_steps.pkl')

gbm_classifier = None
gbm_regressor = None
gbm_scaler = None
lstm_model = None
lstm_scaler = None
lstm_time_steps = None
history_df = None
load_errors = []

try:
    gbm_classifier = joblib.load(GBM_CLASSIFIER_PATH)
    gbm_regressor = joblib.load(GBM_REGRESSOR_PATH)
    gbm_scaler = joblib.load(GBM_SCALER_PATH)
except Exception as e:
    load_errors.append(f"GBM dual pipeline: {e}")

try:
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from tensorflow.keras.models import load_model
    lstm_model = load_model(LSTM_MODEL_PATH)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    lstm_time_steps = joblib.load(LSTM_TIME_STEPS_PATH)
except Exception as e:
    load_errors.append(f"Multi-Task LSTM: {e}")

try:
    history_df = load_history()
except Exception as e:
    load_errors.append(f"Historical feature context: {e}")

gbm_ready = gbm_classifier is not None and gbm_regressor is not None and gbm_scaler is not None
lstm_ready = lstm_model is not None and lstm_scaler is not None and lstm_time_steps is not None
models_ready = (gbm_ready or lstm_ready) and history_df is not None


class PricingData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    tick_volume: float


def _predict_gbm(feature_row):
    scaled = gbm_scaler.transform(feature_row.to_frame().T)
    prob_up = float(gbm_classifier.predict_proba(scaled)[0, 1])
    pred_class = int(gbm_classifier.predict(scaled)[0])
    predicted_return = float(gbm_regressor.predict(scaled)[0])
    return {
        "direction": "UP" if pred_class == 1 else "DOWN",
        "confidence": prob_up if pred_class == 1 else (1 - prob_up),
        "predicted_return_pct": predicted_return * 100,
    }


def _predict_lstm(window_df):
    scaled = lstm_scaler.transform(window_df.values)
    window_3d = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
    predicted_return, prob_up = lstm_model.predict(window_3d, verbose=0)
    predicted_return = float(predicted_return.ravel()[0])
    prob_up = float(prob_up.ravel()[0])
    return {
        "direction": "UP" if prob_up >= 0.5 else "DOWN",
        "confidence": prob_up if prob_up >= 0.5 else (1 - prob_up),
        "predicted_return_pct": predicted_return * 100,
    }


@app.post("/api/predict")
def predict_endpoint(data: PricingData):
    if not models_ready:
        raise HTTPException(status_code=503, detail=f"Model artifacts missing. Errors: {load_errors}")

    new_bar = {
        "open": data.open, "high": data.high, "low": data.low,
        "close": data.close, "tick_volume": data.tick_volume,
    }
    max_window = max(lstm_time_steps or 1, 1)

    try:
        feature_window = build_live_features(history_df, new_bar, time_steps=max_window)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature computation failed: {e}")

    response = {}
    if gbm_ready:
        response["gbm"] = _predict_gbm(feature_window.iloc[-1])
    if lstm_ready:
        if len(feature_window) < lstm_time_steps:
            response["lstm_error"] = "Not enough historical context for the LSTM sliding window."
        else:
            response["lstm"] = _predict_lstm(feature_window.tail(lstm_time_steps))

    return response


# Ensure the UI path maps correctly
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API Active. GBM ready: " + str(gbm_ready) + ", LSTM ready: " + str(lstm_ready)}
