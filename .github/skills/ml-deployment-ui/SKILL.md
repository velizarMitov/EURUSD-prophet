---
name: ml-deployment-ui
description: 'MLOps and Full-Stack Developer for deploying the trained EURUSD model as an interactive app. Use when: building a Gradio UI for model predictions, extracting inference logic from notebooks into predict.py/app.py, saving and loading trained models with joblib or Keras, adding interactive Plotly visualizations to the app, structuring code for FastAPI Web API readiness. Triggers: Gradio, deploy model, prediction UI, model persistence, joblib, pickle, Keras save, app.py, predict.py, inference script, interactive chart, Plotly, FastAPI, MLOps, web interface, model serving.'
argument-hint: 'Describe the deployment task (e.g. "build Gradio UI", "extract predict.py from notebook", "save and load LSTM model", "add Plotly chart to app")'
---

# ML Deployment Engineer & UI Designer

## Role
You are an expert MLOps and Full-Stack Developer. The model is trained — now you build a clean, interactive application so users can interact with EURUSD predictions without touching the notebook.

## When to Use
- Extracting trained model inference logic from `.ipynb` into reusable `.py` scripts
- Saving and loading models (joblib, Keras, PyTorch)
- Building a Gradio web interface for predictions
- Adding interactive Plotly visualizations of forecasts
- Structuring the backend for future FastAPI/cloud deployment

---

## Target File Structure

```
eurusdprophet/
├── notebooks/               # Research only — NOT imported by the app
├── src/
│   ├── features.py          # Feature engineering (shared between training & inference)
│   ├── predict.py           # Pure inference logic — loads model, returns prediction
│   └── visualize.py         # Plotly chart builders
├── models/
│   ├── scaler.pkl           # Fitted StandardScaler
│   ├── best_model.pkl       # Classical model (joblib)
│   └── lstm_best.keras      # Deep learning model (Keras)
├── app.py                   # Gradio UI entry point
├── api.py                   # FastAPI Web API (Web API readiness)
└── requirements.txt
```

---

## Step 1 — Model Persistence

### Classical Models (scikit-learn)

```python
# In your training notebook — run once after final model selection
import joblib

# Save
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler,     'models/scaler.pkl')
print("Models saved.")

# Load (used in predict.py and app.py)
best_model = joblib.load('models/best_model.pkl')
scaler     = joblib.load('models/scaler.pkl')
```

> Use `joblib` over `pickle` for scikit-learn objects — it handles large numpy arrays more efficiently and is the officially recommended method.

### Deep Learning Models (Keras / TensorFlow)

```python
# Save — use the native Keras format (.keras), NOT legacy .h5
model.save('models/lstm_best.keras')

# Load
from tensorflow import keras
model = keras.models.load_model('models/lstm_best.keras')
```

### PyTorch Models

```python
# Save state dict (preferred — portable across architectures)
import torch
torch.save(model.state_dict(), 'models/lstm_best.pt')

# Load
model = build_lstm(SEQ_LEN, N_FEATURES)  # rebuild architecture first
model.load_state_dict(torch.load('models/lstm_best.pt', map_location='cpu'))
model.eval()
```

---

## Step 2 — Modular Inference Script (`src/predict.py`)

Extract all inference logic from the notebook into a single, importable module.

```python
# src/predict.py
"""
Inference module for EURUSD prediction.

Separating inference from the research notebook ensures the app can
load and run predictions without any notebook runtime dependency.
"""
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

MODEL_PATH  = Path("models/best_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# Load once at module import — avoids reloading on every request
_model  = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering pipeline used during training.
    MUST be kept in sync with the training notebook's feature set.
    """
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)

    df['return_1d']   = df['Close'].pct_change(1)
    df['return_5d']   = df['Close'].pct_change(5)
    df['log_return']  = np.log(df['Close'] / df['Close'].shift(1))
    df['rsi_14']      = ta.rsi(df['Close'], length=14)
    df['ema_20']      = ta.ema(df['Close'], length=20)
    df['ema_50']      = ta.ema(df['Close'], length=50)
    df.dropna(inplace=True)
    return df


def predict_next_close(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame of recent OHLCV data, return the predicted next close.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least 60 rows of OHLCV data with a DatetimeIndex.

    Returns
    -------
    dict with keys: prediction (float), confidence_range (tuple), last_close (float)
    """
    df_feat   = engineer_features(df.copy())
    feature_cols = [c for c in df_feat.columns if c != 'target']
    X_scaled  = _scaler.transform(df_feat[feature_cols].tail(1))
    pred      = float(_model.predict(X_scaled)[0])
    last_close = float(df['Close'].iloc[-1])

    return {
        "prediction":       round(pred, 5),
        "last_close":       round(last_close, 5),
        "predicted_change": round(pred - last_close, 5),
        "direction":        "UP" if pred > last_close else "DOWN"
    }
```

---

## Step 3 — Interactive Visualizations (`src/visualize.py`)

```python
# src/visualize.py
"""
Plotly chart builders for the Gradio UI.
Returns fig objects that Gradio renders natively.
"""
import plotly.graph_objects as go
import pandas as pd


def plot_forecast(df: pd.DataFrame, predicted_price: float) -> go.Figure:
    """
    Plot historical EURUSD close prices with the model's next-day forecast.
    Uses a candlestick for recent history and a marker for the prediction.
    """
    recent = df.tail(60)

    fig = go.Figure()

    # Historical candlestick
    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent['Open'], high=recent['High'],
        low=recent['Low'],   close=recent['Close'],
        name='EURUSD',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))

    # Forecast point — 1 trading day ahead
    next_date = recent.index[-1] + pd.tseries.offsets.BDay(1)
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[predicted_price],
        mode='markers',
        marker=dict(color='#ff9800', size=14, symbol='star'),
        name='Forecast'
    ))

    fig.update_layout(
        title='EURUSD — Historical Price & Next-Day Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        height=450
    )
    return fig


def plot_residuals(y_true, y_pred) -> go.Figure:
    """Residual scatter for model diagnostics tab."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers', marker=dict(size=4, opacity=0.5),
        name='Residuals'
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(
        title='Residuals vs Fitted',
        xaxis_title='Fitted Values',
        yaxis_title='Residuals',
        template='plotly_dark'
    )
    return fig
```

---

## Step 4 — Gradio UI (`app.py`)

```python
# app.py
"""
Gradio web interface for EURUSD prediction.

Run with:  python app.py
Access at: http://localhost:7860
"""
import gradio as gr
import yfinance as yf
import pandas as pd
from src.predict   import predict_next_close
from src.visualize import plot_forecast

TICKER   = "EURUSD=X"
INTERVAL = "1d"
PERIOD   = "6mo"


def run_prediction(period: str, show_sma: bool):
    """
    Fetch live EURUSD data, run inference, and return results for the UI.
    This function is called every time the user clicks 'Predict'.
    """
    df = yf.download(TICKER, period=period, interval=INTERVAL, progress=False)
    if df.empty:
        return "Failed to fetch data.", None

    result = predict_next_close(df)
    fig    = plot_forecast(df, result["prediction"])

    summary = (
        f"**Last Close:**  {result['last_close']}\n\n"
        f"**Prediction:**  {result['prediction']}\n\n"
        f"**Change:**      {result['predicted_change']:+.5f}\n\n"
        f"**Direction:**   {'🟢 UP' if result['direction'] == 'UP' else '🔴 DOWN'}"
    )
    return summary, fig


# --- UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="EURUSD Prophet") as demo:
    gr.Markdown("# 💱 EURUSD Prophet\nNext-day exchange rate forecast powered by ML.")

    with gr.Row():
        period_dd  = gr.Dropdown(
            choices=["1mo", "3mo", "6mo", "1y"],
            value="6mo",
            label="Historical Period"
        )
        show_sma   = gr.Checkbox(label="Show SMA overlays", value=False)
        predict_btn = gr.Button("🔮 Predict", variant="primary")

    with gr.Row():
        output_text = gr.Markdown(label="Prediction Summary")
        output_chart = gr.Plot(label="EURUSD Chart")

    predict_btn.click(
        fn=run_prediction,
        inputs=[period_dd, show_sma],
        outputs=[output_text, output_chart]
    )

    gr.Markdown(
        "_Model trained on daily OHLCV data. Not financial advice._",
        elem_id="disclaimer"
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
```

---

## Step 5 — Web API Readiness (`api.py`)

Structure the backend so prediction logic is a single function call — making FastAPI wrapping trivial.

```python
# api.py
"""
FastAPI Web API — exposes the same predict_next_close() function over HTTP.

Run with:  uvicorn api:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from src.predict import predict_next_close

app = FastAPI(
    title="EURUSD Prophet API",
    description="Next-day EURUSD forecast endpoint.",
    version="1.0.0"
)


class PredictionResponse(BaseModel):
    prediction:       float
    last_close:       float
    predicted_change: float
    direction:        str


@app.get("/predict", response_model=PredictionResponse)
def predict(period: str = "6mo"):
    """
    Fetch live EURUSD data and return next-day price prediction.
    
    Parameters
    ----------
    period : str
        yfinance period string (e.g. '3mo', '6mo', '1y')
    """
    df = yf.download("EURUSD=X", period=period, interval="1d", progress=False)
    if df.empty:
        raise HTTPException(status_code=503, detail="Failed to fetch market data.")
    return predict_next_close(df)


@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## Step 6 — `requirements.txt` (App-specific)

```
# Core ML
scikit-learn>=1.4
tensorflow>=2.15
joblib>=1.3

# Data
yfinance>=0.2
pandas>=2.0
pandas-ta>=0.3
numpy>=1.26

# UI
gradio>=4.0

# Visualization
plotly>=5.18

# API
fastapi>=0.110
uvicorn>=0.29
pydantic>=2.0
```

---

## Deployment Checklist

```
[ ] Training notebook fully executed with outputs saved
[ ] models/best_model.pkl (or .keras) present and loads without error
[ ] models/scaler.pkl present
[ ] src/predict.py runs standalone: python -c "from src.predict import predict_next_close"
[ ] app.py launches without error: python app.py
[ ] Gradio UI accessible at localhost:7860
[ ] api.py health check returns 200: curl http://localhost:8000/health
[ ] requirements.txt complete and pinned
[ ] No hardcoded absolute paths — all paths relative to project root
[ ] No model training code inside app.py or api.py
[ ] README updated with "How to run the app" section
```

---

## Anti-Patterns to Avoid

| Anti-Pattern | Risk | Fix |
|-------------|------|-----|
| Training code inside `app.py` | Retrains on every request | Move all training to notebook; app only calls `load + predict` |
| Importing from `.ipynb` directly | Fragile, notebook-order dependent | Extract logic into `src/*.py` |
| Fitting scaler inside `predict_next_close()` | Data leakage on inference | Load pre-fitted scaler from `models/scaler.pkl` |
| Hardcoded file paths | Breaks in CI/CD or cloud | Use `pathlib.Path` relative to project root |
| Saving model as `.pkl` with pickle for Keras | Unstable across TF versions | Use `.keras` native format |
