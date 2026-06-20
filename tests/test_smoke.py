import os

def test_smoke_model_resolution():
    """Smoke test ensuring exactly that critical artifact payloads exist precisely locally."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'models', 'best_gbm_eurusd.pkl')
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler_gb_eurusd.pkl')
    
    assert os.path.exists(model_path), f"Strict structural failure: Model payload omitted at {model_path}"
    assert os.path.exists(scaler_path), f"Strict structural failure: Scaler payload omitted at {scaler_path}"