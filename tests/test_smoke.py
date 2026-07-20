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
    # The volatility family ships as a 5-seed ensemble: the VALIDATED object is
    # the mean over ALL seeds (results/volatility_seed_ensemble.csv), so every
    # seed model must exist — a partial ensemble is a different, unvalidated
    # predictor and inference refuses to serve one (vol_ready gate).
    VOLATILITY_ARTIFACTS = [
        'volatility_lstm_seed42.keras',
        'volatility_lstm_seed43.keras',
        'volatility_lstm_seed44.keras',
        'volatility_lstm_seed45.keras',
        'volatility_lstm_seed46.keras',
        'global_scaler.pkl',
        'lag_scaler.pkl',
        'lag_pca.pkl',
        'lstm_time_steps.pkl',
        'vol_metrics.json',
    ]
    # Observational H1 TI-LSTM: in production by explicit owner override
    # DESPITE a DROP verdict (no demonstrated edge) — the artifacts must exist
    # for the ti_h1_ready gate, but their presence is NOT validation (see
    # IMPROVEMENT_LOG.md "owner override", ti_metrics.json carries
    # validated: false).
    TI_H1_ARTIFACTS = [
        'ti_lstm_h1.keras',
        'ti_scaler.pkl',
        'ti_config.pkl',
        'ti_metrics.json',
    ]
    required_artifacts = [
        os.path.join('models', variant, artifact)
        for variant in ('baseline', 'with_macro')
        for artifact in VARIANT_ARTIFACTS
    ] + [
        os.path.join('models', 'volatility', artifact)
        for artifact in VOLATILITY_ARTIFACTS
    ] + [
        os.path.join('models', 'ti_lstm_h1', artifact)
        for artifact in TI_H1_ARTIFACTS
    ] + [
        os.path.join('results', 'eurusd_features.csv'),
        'config.json',
        '.env.example',
    ]

    for relative_path in required_artifacts:
        full_path = os.path.join(BASE_DIR, relative_path)
        assert os.path.exists(full_path), f"Strict structural failure: payload omitted at {full_path}"
