import gradio as gr
import pandas as pd
import numpy as np
import joblib
import warnings

import os

# Suppress specific non-critical numerical extraction warnings for production
warnings.filterwarnings('ignore')

# Determine absolute paths for robust deployment execution regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_gbm_eurusd.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler_gb_eurusd.pkl')

# Mathematical Baseline Definitions matching the Notebook's structure precisely
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'tick_volume', 'log_return', 
    'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200', 'volatility_20', 
    'bar_dynamics', 'return_lag_1', 'dynamics_lag_1', 'return_lag_2', 
    'dynamics_lag_2', 'return_lag_3', 'dynamics_lag_3', 'day_sin', 'day_cos'
]

# Loading serialized ML pipeline artifacts computationally via absolute paths
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
except Exception as e:
    model = None
    scaler = None
    model_loaded = False
    error_msg = str(e)
    print(f"Artifacts missing: {e}. Please ensure 'models/' directory exists and contains trained .pkl files.")

def predict_eurusd(open_p, high_p, low_p, close_p, volume_p):
    """
    Inference mapping endpoint. Integrates the extracted inputs, fakes computational 
    long-term rolling dependencies matching standard mathematical averages natively, 
    and pipes data into the verified Gradient Boosting Decision Trees. 
    """
    if not model_loaded:
        return f"Critical MLOps Failure: Model not loaded. Error: {error_msg}", "N/A"
    
    # Building a simulation matrix structurally matching our 20-feature dimensional constraint
    inference_vector = np.zeros((1, len(FEATURE_COLUMNS)))
    
    # 1. Base Variables Mapping
    inference_vector[0, 0] = open_p
    inference_vector[0, 1] = high_p
    inference_vector[0, 2] = low_p
    inference_vector[0, 3] = close_p
    inference_vector[0, 4] = volume_p
    
    # 2. Approximate Dependent Technical Indicators to enable isolated demo functionality 
    # (In formal architectures, these require streaming real history. We mock stable approximations here).
    log_return = np.log(close_p / open_p) if open_p > 0 else 0
    bar_dynamics = (high_p - low_p) / open_p if open_p > 0 else 0
    
    inference_vector[0, 5] = log_return               # log_return
    inference_vector[0, 6] = close_p * 0.999          # SMA_21 
    inference_vector[0, 7] = close_p * 0.995          # SMA_50
    inference_vector[0, 8] = close_p * 0.990          # SMA_100
    inference_vector[0, 9] = close_p * 0.985          # SMA_200
    inference_vector[0, 10] = 0.005                   # volatility_20
    inference_vector[0, 11] = bar_dynamics            # bar_dynamics
    inference_vector[0, 12] = 0.001                   # return_lag_1
    inference_vector[0, 13] = 0.003                   # dynamics_lag_1
    inference_vector[0, 14] = -0.002                  # return_lag_2
    inference_vector[0, 15] = 0.002                   # dynamics_lag_2
    inference_vector[0, 16] = 0.0015                  # return_lag_3
    inference_vector[0, 17] = 0.004                   # dynamics_lag_3
    inference_vector[0, 18] = 0.974                   # day_sin (Wednesday approx)
    inference_vector[0, 19] = -0.222                  # day_cos (Wednesday approx)
    
    # 3. Formally transform vector avoiding data-leakage via isolated weights
    scaled_vector = scaler.transform(inference_vector)
    
    # 4. Infer probability and predicted discrete structural bounds
    prob_up = model.predict_proba(scaled_vector)[0, 1]
    pred_class = model.predict(scaled_vector)[0]
    
    direction = "UP 📈" if pred_class == 1 else "DOWN 📉"
    confidence = f"{prob_up:.2%}" if pred_class == 1 else f"{(1 - prob_up):.2%}"
    
    return direction, confidence

# Gradio Component Construction
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as ui:
    gr.Markdown("# EURUSD Directional Extrapolator via Gradient Boosting")
    gr.Markdown("Input the current daily trading parameters. Our computational matrix evaluates 20 temporal features structurally extracted and returns the probabilistic directional mapping.")
    
    with gr.Row():
        with gr.Column():
            open_val = gr.Number(label="Open Price", value=1.1000, step=0.0001)
            high_val = gr.Number(label="High Price", value=1.1050, step=0.0001)
            low_val = gr.Number(label="Low Price", value=1.0950, step=0.0001)
            close_val = gr.Number(label="Close Price", value=1.1020, step=0.0001)
            vol_val = gr.Number(label="Daily Tick Volume", value=85000, step=100)
            
            predict_btn = gr.Button("Evaluate Market Direction", variant="primary")
            
        with gr.Column():
            out_direction = gr.Textbox(label="Predicted Target Axis (T+1)", placeholder="Waiting for parameters...")
            out_prob = gr.Textbox(label="Evaluator Algorithmic Confidence", placeholder="...")
            
    gr.Markdown("---")
    gr.Markdown("**Technical Limitation Warning:** *Mock application assumes continuous SMA averages aligned linearly with current inputs for UI functional demonstration. Actual API executions mandate rolling cache computations.*")

    predict_btn.click(predict_eurusd, inputs=[open_val, high_val, low_val, close_val, vol_val], outputs=[out_direction, out_prob])

if __name__ == "__main__":
    ui.launch()