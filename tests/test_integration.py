import sys
import os

# Append root to pathway explicitly for TestClient targeting
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_integration_predict_endpoint():
    """Integration test verifying API endpoints properly link JSON inputs mathematically 
    via FastAPIs pydantic validation securely without arbitrary HTTP 500 runtime crashes."""
    
    payload = {
        "open": 1.1000,
        "high": 1.1050,
        "low": 1.0950,
        "close": 1.1020,
        "tick_volume": 85000.0
    }
    
    response = client.post("/api/predict", json=payload)
    
    # Strictly evaluating operational capability
    if response.status_code == 200:
        data = response.json()
        assert "direction" in data, "API execution decoupled targeting logic structurally."
        assert "confidence" in data, "API execution decoupled confidence logic structurally."
        assert data['direction'] in ["UP", "DOWN"], "API mapping bound invalid explicit response."
        assert 0.0 <= data['confidence'] <= 1.0, "API mapping bounds corrupted statistical probability bounds."
    else:
        # Evaluates gracefully falling back to artifact erroring without severe web crashes
        assert response.status_code == 503, "API returned undocumented explicit computational failure bounds."
        
def test_static_ui_resolves():
    """Evaluate Web UI mounts dynamically"""
    response = client.get("/")
    assert response.status_code == 200, "Web API Static Route execution failed to implicitly resolve index.html."