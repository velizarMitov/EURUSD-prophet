import gradio as gr
import json
import warnings

import os
import sys

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.inference import PredictionService

with open(os.path.join(BASE_DIR, 'config.json')) as f:
    CONFIG = json.load(f)

service = PredictionService(BASE_DIR, CONFIG)

DATA_SOURCE_LABELS = {
    "MT5": "Live MT5 terminal session",
    "yfinance": "Live Yahoo Finance fetch",
    "history_fallback": "Historical fallback (no live source reachable)",
}


def fetch_and_predict():
    """
    Fully autonomous: no manual OHLCV entry, and no manual date selection.
    The pipeline figures out "today" from whichever live source answers
    first (MT5, then Yahoo Finance, then the bundled historical tail),
    fetches exactly enough bars to satisfy the SMA_200/lag warm-up, and
    forecasts the very next trading day.
    """
    if not service.models_ready:
        msg = f"Critical MLOps Failure: Model artifacts not loaded. Errors: {service.load_errors}"
        return msg, msg, msg

    try:
        result = service.predict()
    except Exception as e:
        msg = f"Automated data pipeline failed: {e}"
        return msg, msg, msg

    bar = result['bar_used']
    source_label = DATA_SOURCE_LABELS.get(result['data_source'], result['data_source'])
    market_state = (
        f"Data 'As Of' {result['as_of_date']}  ({source_label})\n"
        f"O={bar['open']:.5f}  H={bar['high']:.5f}  L={bar['low']:.5f}  C={bar['close']:.5f}  Vol={bar['tick_volume']:.0f}"
    )

    direction_parts, return_parts = [], []
    for name in ('gbm', 'lstm'):
        if name in result:
            p = result[name]
            arrow = "📈" if p['direction'] == "UP" else "📉"
            direction_parts.append(f"{name.upper()}: {p['direction']} {arrow} ({p['confidence']:.2%})")
            return_parts.append(f"{name.upper()}: {p['predicted_return_pct']:+.4f}%")
    if 'lstm_error' in result:
        direction_parts.append(f"LSTM: {result['lstm_error']}")
        return_parts.append("LSTM: n/a")
    model_breakdown = "  |  ".join(direction_parts) + "\n" + "  |  ".join(return_parts)

    consensus_text = "N/A — only one model produced a prediction."
    if 'consensus' in result:
        c = result['consensus']
        arrow = "📈" if c['direction'] == "UP" else "📉"
        agree_label = "✅ Models agree" if c['agreement'] else "⚠️ Models disagree — showing higher-confidence call"
        consensus_text = (
            f"Forecasting Date: {result['forecasting_date']}  (t+1)\n"
            f"{c['direction']} {arrow}  |  {c['confidence']:.2%} confidence  |  {c['predicted_return_pct']:+.4f}% avg. return  —  {agree_label}"
        )

    return market_state, model_breakdown, consensus_text


# Gradio Component Construction
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as ui:
    gr.Markdown("# EURUSD Multi-Task Predictor (GBM + Multi-Task LSTM)")
    gr.Markdown(
        "Fully autonomous — knows today's date itself, fetches live market data (MT5, falling back to "
        "Yahoo Finance), recomputes real technical indicators, and queries both production models. "
        "No manual data entry, no date picker."
    )

    fetch_btn = gr.Button("🔄 Fetch Live Market Data & Predict Tomorrow", variant="primary", size="lg")

    out_market_state = gr.Textbox(label="Today's Market State (Data 'As Of')", placeholder="Not yet fetched...", interactive=False, lines=2)
    out_breakdown = gr.Textbox(label="Individual Model Confidences", interactive=False, lines=2)
    out_consensus = gr.Textbox(label="Prediction for Next Period — Committee Consensus", interactive=False, lines=2)

    gr.Markdown("---")
    if not service.models_ready:
        gr.Markdown(f"**Startup Warning:** Some model artifacts failed to load: {service.load_errors}")

    fetch_btn.click(
        fn=fetch_and_predict,
        inputs=[],
        outputs=[out_market_state, out_breakdown, out_consensus],
        show_progress="full",
    )

if __name__ == "__main__":
    ui.launch()
