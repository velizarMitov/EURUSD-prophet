import sys
import os

# Append root to pathway explicitly for TestClient targeting
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_integration_predict_endpoint():
    """Integration test verifying the fully-automated /api/predict endpoint
    (no request body) drives the live-fetch-or-fallback pipeline end-to-end
    without arbitrary HTTP 500 runtime crashes."""

    response = client.post("/api/predict")

    # Strictly evaluating operational capability
    if response.status_code == 200:
        data = response.json()
        assert data.get('data_source') in ("MT5", "yfinance", "history_fallback"), "API omitted/mis-labeled the automated data source."
        assert "as_of_date" in data, "API omitted the 'Data As Of' date."
        assert "forecasting_date" in data, "API omitted the t+1 forecasting date."
        assert "bar_used" in data, "API omitted the bar actually used for inference."
        for field in ("date", "open", "high", "low", "close", "tick_volume"):
            assert field in data['bar_used'], f"bar_used missing '{field}'."

        assert "gbm" in data or "lstm" in data, "API returned no predictions from either model."
        for key in ("gbm", "lstm"):
            if key in data:
                assert data[key]['direction'] in ["UP", "DOWN"], f"{key} mapping bound invalid explicit response."
                assert 0.0 <= data[key]['confidence'] <= 1.0, f"{key} mapping bounds corrupted statistical probability bounds."
                assert isinstance(data[key]['predicted_return_pct'], float), f"{key} return head must surface a numeric % return."

        if "gbm" in data and "lstm" in data:
            assert "consensus" in data, "Committee consensus must be present when both models produced a prediction."
            c = data['consensus']
            assert c['direction'] in ["UP", "DOWN"]
            assert isinstance(c['agreement'], bool)
            assert 0.0 <= c['confidence'] <= 1.0
    else:
        # Evaluates gracefully falling back to artifact erroring without severe web crashes
        assert response.status_code in (400, 503), "API returned undocumented explicit computational failure bounds."
        
def test_static_ui_resolves():
    """Evaluate Web UI mounts dynamically"""
    response = client.get("/")
    assert response.status_code == 200, "Web API Static Route execution failed to implicitly resolve index.html."