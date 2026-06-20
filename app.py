import gradio as gr
import joblib
import json
import warnings

import os
import sys

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.features import load_history, build_live_features, apply_lag_pca, LAG_COLUMNS

with open(os.path.join(BASE_DIR, 'config.json')) as f:
    CONFIG = json.load(f)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LAG_SCALER_PATH = os.path.join(MODELS_DIR, 'lag_scaler.pkl')
LAG_PCA_PATH = os.path.join(MODELS_DIR, 'lag_pca.pkl')
GBM_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'best_gbm_eurusd.pkl')
GBM_REGRESSOR_PATH = os.path.join(MODELS_DIR, 'best_gbm_regressor_eurusd.pkl')
GBM_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_gb_eurusd.pkl')
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_multitask_eurusd.keras')
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_lstm_multitask.pkl')
LSTM_TIME_STEPS_PATH = os.path.join(MODELS_DIR, 'lstm_time_steps.pkl')

lag_scaler = None
lag_pca = None
gbm_classifier = None
gbm_regressor = None
gbm_scaler = None
lstm_model = None
lstm_scaler = None
lstm_time_steps = None
history_df = None
load_errors = []

try:
    lag_scaler = joblib.load(LAG_SCALER_PATH)
    lag_pca = joblib.load(LAG_PCA_PATH)
except Exception as e:
    load_errors.append(f"PCA lag reduction: {e}")

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
    history_df = load_history(os.path.join(BASE_DIR, CONFIG['data']['history_csv_path']))
except Exception as e:
    load_errors.append(f"Historical feature context: {e}")

pca_ready = lag_scaler is not None and lag_pca is not None

gbm_ready = pca_ready and gbm_classifier is not None and gbm_regressor is not None and gbm_scaler is not None
lstm_ready = pca_ready and lstm_model is not None and lstm_scaler is not None and lstm_time_steps is not None
models_ready = (gbm_ready or lstm_ready) and history_df is not None


def _predict_gbm(model_input_row):
    assert gbm_classifier is not None and gbm_regressor is not None and gbm_scaler is not None
    scaled = gbm_scaler.transform(model_input_row.to_frame().T)
    prob_up = float(gbm_classifier.predict_proba(scaled)[0, 1])
    pred_class = int(gbm_classifier.predict(scaled)[0])
    predicted_return = float(gbm_regressor.predict(scaled)[0])
    confidence = prob_up if pred_class == 1 else (1 - prob_up)
    return ("UP" if pred_class == 1 else "DOWN"), confidence, predicted_return * 100


def _predict_lstm(model_input_window):
    assert lstm_model is not None and lstm_scaler is not None
    scaled = lstm_scaler.transform(model_input_window.values)
    window_3d = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
    predicted_return, prob_up = lstm_model.predict(window_3d, verbose=0)
    predicted_return = float(predicted_return.ravel()[0])
    prob_up = float(prob_up.ravel()[0])
    confidence = prob_up if prob_up >= 0.5 else (1 - prob_up)
    return ("UP" if prob_up >= 0.5 else "DOWN"), confidence, predicted_return * 100


def predict_eurusd(open_p, high_p, low_p, close_p, volume_p):
    """
    Real inference: appends the submitted bar to genuine EURUSD price history,
    recomputes the actual rolling SMA/volatility/lag indicators (no mocked
    constants), then queries both production models.
    """
    if not models_ready:
        msg = f"Critical MLOps Failure: Model artifacts not loaded. Errors: {load_errors}"
        return msg, msg

    new_bar = {"open": open_p, "high": high_p, "low": low_p, "close": close_p, "tick_volume": volume_p}
    max_window = max(lstm_time_steps or 1, 1)

    try:
        feature_window = build_live_features(history_df, new_bar, time_steps=max_window)
        model_input_window = apply_lag_pca(feature_window, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    except Exception as e:
        msg = f"Feature computation failed: {e}"
        return msg, msg

    direction_parts = []
    return_parts = []

    if gbm_ready:
        direction, confidence, predicted_return = _predict_gbm(model_input_window.iloc[-1])
        arrow = "📈" if direction == "UP" else "📉"
        direction_parts.append(f"GBM: {direction} {arrow} ({confidence:.2%})")
        return_parts.append(f"GBM: {predicted_return:+.4f}%")

    if lstm_ready:
        if len(model_input_window) < lstm_time_steps:
            direction_parts.append("LSTM: insufficient history")
            return_parts.append("LSTM: insufficient history")
        else:
            direction, confidence, predicted_return = _predict_lstm(model_input_window.tail(lstm_time_steps))
            arrow = "📈" if direction == "UP" else "📉"
            direction_parts.append(f"LSTM: {direction} {arrow} ({confidence:.2%})")
            return_parts.append(f"LSTM: {predicted_return:+.4f}%")

    return "  |  ".join(direction_parts), "  |  ".join(return_parts)


# Gradio Component Construction
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as ui:
    gr.Markdown("# EURUSD Multi-Task Predictor (GBM + Multi-Task LSTM)")
    gr.Markdown(
        "Input the current daily trading parameters. The app appends this bar to real historical "
        "EURUSD price data, recomputes genuine technical indicators, and queries both the Gradient "
        "Boosting dual pipeline and the Multi-Task LSTM (shared trunk, dual heads) for next-bar "
        "direction and exact % return."
    )

    with gr.Row():
        with gr.Column():
            open_val = gr.Number(label="Open Price", value=1.1000, step=0.0001)
            high_val = gr.Number(label="High Price", value=1.1050, step=0.0001)
            low_val = gr.Number(label="Low Price", value=1.0950, step=0.0001)
            close_val = gr.Number(label="Close Price", value=1.1020, step=0.0001)
            vol_val = gr.Number(label="Daily Tick Volume", value=85000, step=100)

            predict_btn = gr.Button("Evaluate Market Direction", variant="primary")

        with gr.Column():
            out_direction = gr.Textbox(label="Predicted Market Direction (T+1)", placeholder="Waiting for parameters...")
            out_return = gr.Textbox(label="Predicted Exact Return (T+1, %)", placeholder="...")

    gr.Markdown("---")
    if not models_ready:
        gr.Markdown(f"**Startup Warning:** Some model artifacts failed to load: {load_errors}")
    gr.Markdown(
        "**Note:** Features (SMA/volatility/lags) are computed from genuine historical EURUSD bars "
        "(`results/eurusd_features.csv`) with your submitted bar appended as the most recent observation — "
        "no values are hardcoded or approximated."
    )

    predict_btn.click(predict_eurusd, inputs=[open_val, high_val, low_val, close_val, vol_val], outputs=[out_direction, out_return])

if __name__ == "__main__":
    ui.launch()
