import os

def test_smoke_model_resolution():
    """Smoke test ensuring exactly that critical artifact payloads exist precisely locally.
    Dual-variant layout: every daily artifact exists PER VARIANT under
    models/<variant>/ (baseline price-only + with_macro), plus the shared data/config."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    VARIANT_ARTIFACTS = [
        'lag_scaler.pkl',
        'lag_pca.pkl',
        'best_gbm_eurusd.pkl',
        'best_gbm_regressor_eurusd.pkl',
        'global_scaler.pkl',
        'lstm_multitask_eurusd.keras',
        'lstm_time_steps.pkl',
    ]
    required_artifacts = [
        os.path.join('models', variant, artifact)
        for variant in ('baseline', 'with_macro')
        for artifact in VARIANT_ARTIFACTS
    ] + [
        os.path.join('results', 'eurusd_features.csv'),
        'config.json',
        '.env.example',
    ]

    for relative_path in required_artifacts:
        full_path = os.path.join(BASE_DIR, relative_path)
        assert os.path.exists(full_path), f"Strict structural failure: payload omitted at {full_path}"
