import sys
import os

# Append root to pathway explicitly for TestClient targeting
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

VARIANTS = ("baseline", "with_macro")


def _assert_committee_block(block, variant):
    """One variant's committee block must carry sane gbm/lstm/consensus values."""
    assert "gbm" in block or "lstm" in block, f"[{variant}] returned no predictions from either model."
    for key in ("gbm", "lstm"):
        if key in block:
            assert block[key]['direction'] in ["UP", "DOWN"], f"[{variant}] {key} mapping bound invalid explicit response."
            assert 0.0 <= block[key]['confidence'] <= 1.0, f"[{variant}] {key} mapping bounds corrupted statistical probability bounds."
            assert isinstance(block[key]['predicted_return_pct'], float), f"[{variant}] {key} return head must surface a numeric % return."

    if "gbm" in block and "lstm" in block:
        assert "consensus" in block, f"[{variant}] committee consensus must be present when both models predicted."
        c = block['consensus']
        assert c['direction'] in ["UP", "DOWN", "MIXED / LOW CONFIDENCE"]
        assert isinstance(c['agreement'], bool)
        assert 0.0 <= c['confidence'] <= 1.0


def test_integration_predict_endpoint(tmp_path, monkeypatch):
    """Integration test verifying the fully-automated /api/predict endpoint
    (no request body) drives the live-fetch-or-fallback pipeline end-to-end
    without arbitrary HTTP 500 runtime crashes, returning BOTH model variants
    (baseline price-only + with_macro) side by side plus the variant_agreement
    comparison flag.

    The prediction-log write inside predict() is redirected to a tmp file, so
    the test NEVER touches the real tracked results/prediction_log.csv
    (production log). Running the suite must leave that file byte-for-byte
    unchanged — asserted here by confirming the write landed in the tmp path."""
    import api
    isolated_log = tmp_path / "prediction_log.csv"
    monkeypatch.setitem(api.service.config['tracking'], 'log_path', str(isolated_log))

    response = client.post("/api/predict")

    # Strictly evaluating operational capability
    if response.status_code == 200:
        # A successful forecast logs itself; that write must have gone to the
        # redirected tmp path, proving the tracked production log was untouched.
        assert isolated_log.exists(), \
            "predict() logged to the redirected tmp path (never the tracked production log)."
        data = response.json()
        assert data.get('data_source') in ("MT5", "yfinance", "history_fallback"), "API omitted/mis-labeled the automated data source."
        assert "as_of_date" in data, "API omitted the 'Data As Of' date."
        assert "forecasting_date" in data, "API omitted the t+1 forecasting date."
        assert "bar_used" in data, "API omitted the bar actually used for inference."
        for field in ("date", "open", "high", "low", "close", "tick_volume", "yield_differential", "macro_source"):
            assert field in data['bar_used'], f"bar_used missing '{field}'."

        # Dual-variant response shape: both variants present, each a full committee.
        for variant in VARIANTS:
            assert variant in data, f"API omitted the '{variant}' variant block."
            _assert_committee_block(data[variant], variant)

        # variant_agreement: bool when both consensuses exist, None otherwise.
        assert "variant_agreement" in data, "API omitted the variant_agreement comparison flag."
        both_consensus = all("consensus" in data[v] for v in VARIANTS)
        if both_consensus:
            assert isinstance(data['variant_agreement'], bool), "variant_agreement must be a bool when both variants produced a consensus."
            expected = data['baseline']['consensus']['direction'] == data['with_macro']['consensus']['direction']
            assert data['variant_agreement'] == expected, "variant_agreement must reflect the two consensus directions."
        else:
            assert data['variant_agreement'] is None
    else:
        # Evaluates gracefully falling back to artifact erroring without severe web crashes
        assert response.status_code in (400, 503), "API returned undocumented explicit computational failure bounds."


def test_static_ui_resolves():
    """Evaluate Web UI mounts dynamically"""
    response = client.get("/")
    assert response.status_code == 200, "Web API Static Route execution failed to implicitly resolve index.html."
