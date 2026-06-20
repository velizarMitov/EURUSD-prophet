import os

import joblib
import numpy as np
import pandas as pd

from .features import (
    load_history, compute_features, apply_lag_pca,
    LAG_COLUMNS, FEATURE_COLUMNS,
)
from .live_data import fetch_live_market_data


class PredictionService:
    """
    Loads every trained artifact once and serves Multi-Task EURUSD
    predictions from automatically fetched live market data. Shared by
    api.py and app.py so the two frontends can never drift apart.
    """

    def __init__(self, base_dir: str, config: dict):
        self.config = config
        models_dir = os.path.join(base_dir, 'models')
        self.load_errors = []

        self.lag_scaler = self.lag_pca = None
        self.gbm_classifier = self.gbm_regressor = self.gbm_scaler = None
        self.lstm_model = self.lstm_scaler = self.lstm_time_steps = None
        self.history_df = None

        try:
            self.lag_scaler = joblib.load(os.path.join(models_dir, 'lag_scaler.pkl'))
            self.lag_pca = joblib.load(os.path.join(models_dir, 'lag_pca.pkl'))
        except Exception as e:
            self.load_errors.append(f"PCA lag reduction: {e}")

        try:
            self.gbm_classifier = joblib.load(os.path.join(models_dir, 'best_gbm_eurusd.pkl'))
            self.gbm_regressor = joblib.load(os.path.join(models_dir, 'best_gbm_regressor_eurusd.pkl'))
            self.gbm_scaler = joblib.load(os.path.join(models_dir, 'scaler_gb_eurusd.pkl'))
        except Exception as e:
            self.load_errors.append(f"GBM dual pipeline: {e}")

        try:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            from tensorflow.keras.models import load_model
            self.lstm_model = load_model(os.path.join(models_dir, 'lstm_multitask_eurusd.keras'))
            self.lstm_scaler = joblib.load(os.path.join(models_dir, 'scaler_lstm_multitask.pkl'))
            self.lstm_time_steps = joblib.load(os.path.join(models_dir, 'lstm_time_steps.pkl'))
        except Exception as e:
            self.load_errors.append(f"Multi-Task LSTM: {e}")

        try:
            self.history_df = load_history(os.path.join(base_dir, config['data']['history_csv_path']))
        except Exception as e:
            self.load_errors.append(f"Historical feature context: {e}")

        self.pca_ready = self.lag_scaler is not None and self.lag_pca is not None
        self.gbm_ready = self.pca_ready and None not in (self.gbm_classifier, self.gbm_regressor, self.gbm_scaler)
        self.lstm_ready = self.pca_ready and None not in (self.lstm_model, self.lstm_scaler, self.lstm_time_steps)
        self.models_ready = (self.gbm_ready or self.lstm_ready) and self.history_df is not None

    def _resolve_latest_window(self, time_steps: int):
        """
        Automated data pipeline (no manual input ever required): knows
        "today" implicitly as whatever the live source's most recent bar is.

        Tries a live MT5 terminal session first, then Yahoo Finance, fetching
        exactly enough daily bars to satisfy the SMA_200 warm-up plus the
        LSTM's sliding window. Only falls back to the bundled historical CSV
        tail if neither live source is reachable at all.
        """
        bars_needed = max(self.config['data'].get('live_fetch_bars', 250), 200 + time_steps)
        mt5_symbol = self.config['data']['symbol']
        yf_symbol = self.config['data'].get('live_symbol', 'EURUSD=X')

        ohlcv_df, data_source = fetch_live_market_data(mt5_symbol, yf_symbol, bars=bars_needed)
        if ohlcv_df is None or len(ohlcv_df) < 200 + time_steps:
            ohlcv_df = self.history_df.tail(bars_needed)
            data_source = "history_fallback"

        engineered = compute_features(ohlcv_df).dropna(subset=FEATURE_COLUMNS)
        if len(engineered) < time_steps:
            raise RuntimeError(
                f"Insufficient bars after SMA_200/lag warm-up: got {len(engineered)} usable rows, need {time_steps}."
            )

        feature_window = engineered[FEATURE_COLUMNS].tail(time_steps)
        model_input_window = apply_lag_pca(feature_window, self.lag_scaler, self.lag_pca, lag_columns=LAG_COLUMNS)

        as_of_date = engineered.index[-1]
        last_row = ohlcv_df.loc[as_of_date]
        bar_used = {
            "date": as_of_date.date().isoformat(),
            "open": float(last_row['open']),
            "high": float(last_row['high']),
            "low": float(last_row['low']),
            "close": float(last_row['close']),
            "tick_volume": float(last_row['tick_volume']),
        }
        forecasting_date = (as_of_date + pd.Timedelta(days=1)).date().isoformat()

        return model_input_window, data_source, bar_used, as_of_date.date().isoformat(), forecasting_date

    def _predict_gbm(self, model_input_row):
        scaled = self.gbm_scaler.transform(model_input_row.to_frame().T)
        prob_up = float(self.gbm_classifier.predict_proba(scaled)[0, 1])
        pred_class = int(self.gbm_classifier.predict(scaled)[0])
        predicted_return = float(self.gbm_regressor.predict(scaled)[0])
        return {
            "direction": "UP" if pred_class == 1 else "DOWN",
            "confidence": prob_up if pred_class == 1 else (1 - prob_up),
            "predicted_return_pct": predicted_return * 100,
        }

    def _predict_lstm(self, model_input_window):
        scaled = self.lstm_scaler.transform(model_input_window.values)
        # `scaler_lstm_multitask` was fit across the full 1971-2026 history, whose
        # earliest decades carry degenerate placeholder tick_volume (~1-10) that
        # drags the fitted mean/std far below genuine live broker volumes
        # (~1e5-2e5). That single feature can land 8-10 std devs out for real
        # data, and unlike the GBM trees (scale-invariant), the LSTM extrapolates
        # wildly on out-of-distribution inputs. Clip to a still-generous +/-5 std
        # so a single contaminated feature can't blow up the regression head.
        scaled = np.clip(scaled, -5.0, 5.0)
        window_3d = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
        predicted_return, prob_up = self.lstm_model.predict(window_3d, verbose=0)
        predicted_return = float(predicted_return.ravel()[0])
        prob_up = float(prob_up.ravel()[0])
        return {
            "direction": "UP" if prob_up >= 0.5 else "DOWN",
            "confidence": prob_up if prob_up >= 0.5 else (1 - prob_up),
            "predicted_return_pct": predicted_return * 100,
        }

    @staticmethod
    def compute_consensus(predictions: dict) -> dict:
        """Simple committee logic: if every model agrees on direction, average
        their confidence/return; otherwise defer to whichever model is more
        confident and flag the disagreement rather than silently averaging
        across opposite-signed predictions."""
        directions = {name: p['direction'] for name, p in predictions.items()}
        agreement = len(set(directions.values())) == 1

        if agreement:
            direction = next(iter(directions.values()))
            confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
            predicted_return_pct = sum(p['predicted_return_pct'] for p in predictions.values()) / len(predictions)
        else:
            _, best = max(predictions.items(), key=lambda kv: kv[1]['confidence'])
            direction = best['direction']
            confidence = best['confidence']
            predicted_return_pct = best['predicted_return_pct']

        return {
            "direction": direction,
            "agreement": agreement,
            "confidence": confidence,
            "predicted_return_pct": predicted_return_pct,
        }

    def predict(self) -> dict:
        if not self.models_ready:
            raise RuntimeError(f"Model artifacts not loaded. Errors: {self.load_errors}")

        max_window = max(self.lstm_time_steps or 1, 1)
        model_input_window, data_source, bar_used, as_of_date, forecasting_date = self._resolve_latest_window(max_window)

        predictions = {}
        response = {
            "as_of_date": as_of_date,
            "forecasting_date": forecasting_date,
            "data_source": data_source,
            "bar_used": bar_used,
        }

        if self.gbm_ready:
            predictions['gbm'] = self._predict_gbm(model_input_window.iloc[-1])

        if self.lstm_ready:
            if len(model_input_window) < self.lstm_time_steps:
                response['lstm_error'] = "Not enough historical context for the LSTM sliding window."
            else:
                predictions['lstm'] = self._predict_lstm(model_input_window.tail(self.lstm_time_steps))

        response.update(predictions)
        if predictions:
            response['consensus'] = self.compute_consensus(predictions)
        return response
