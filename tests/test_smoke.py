import os

def test_smoke_model_resolution():
    """Smoke test ensuring exactly that critical artifact payloads exist precisely locally."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_artifacts = [
        os.path.join('models', 'best_gbm_eurusd.pkl'),
        os.path.join('models', 'best_gbm_regressor_eurusd.pkl'),
        os.path.join('models', 'scaler_gb_eurusd.pkl'),
        os.path.join('models', 'lstm_multitask_eurusd.keras'),
        os.path.join('models', 'scaler_lstm_multitask.pkl'),
        os.path.join('models', 'lstm_time_steps.pkl'),
        os.path.join('results', 'eurusd_features.csv'),
    ]

    for relative_path in required_artifacts:
        full_path = os.path.join(BASE_DIR, relative_path)
        assert os.path.exists(full_path), f"Strict structural failure: payload omitted at {full_path}"