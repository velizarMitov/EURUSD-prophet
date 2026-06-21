import os

import joblib
import pandas as pd

from .features import (
    load_history, compute_features, apply_lag_pca, merge_macro_features,
    LAG_COLUMNS, FEATURE_COLUMNS,
)
from .live_data import fetch_live_market_data
from .macro_data import fetch_yield_differential


class PredictionService:
    """
    Loads every trained artifact once and serves Multi-Task EURUSD
    predictions from automatically fetched live market data. Shared by
    api.py and app.py so the two frontends can never drift apart.
    """

    # Minimum averaged confidence for a unanimous call to count as a genuine
    # agreement. The direction heads sit near chance (ROC-AUC ~0.50), so a
    # coin-flip "agreement" must not be surfaced as a confident ensemble call.
    CONFIDENCE_THRESHOLD = 0.52

    def __init__(self, base_dir: str, config: dict):
        """
        Load every serialized artifact (PCA, the single global feature scaler,
        both GBM heads, the Multi-Task LSTM + its time_steps, and the bundled
        historical OHLCV CSV) exactly once at process start-up. Each load is
        independently try/excepted into self.load_errors rather than failing
        fast, so e.g. a missing LSTM file still leaves the GBM pipeline (or
        vice versa) servable -- see self.gbm_ready/self.lstm_ready/
        self.models_ready below for how callers should gate on this. Both model
        families share ONE global_scaler (no per-model scalers anymore).
        """
        self.config = config
        self.base_dir = base_dir
        models_dir = os.path.join(base_dir, 'models')
        self.load_errors = []

        self.lag_scaler = self.lag_pca = None
        self.global_scaler = None
        self.gbm_classifier = self.gbm_regressor = None
        self.lstm_model = self.lstm_time_steps = None
        self.history_df = None

        try:
            self.lag_scaler = joblib.load(os.path.join(models_dir, 'lag_scaler.pkl'))
            self.lag_pca = joblib.load(os.path.join(models_dir, 'lag_pca.pkl'))
        except Exception as e:
            self.load_errors.append(f"PCA lag reduction: {e}")

        # Single global StandardScaler shared by BOTH model families (replaces
        # the former separate scaler_gb / scaler_lstm). Fitted once in
        # _train_pipeline.py on the unified 0-80% train block.
        try:
            self.global_scaler = joblib.load(os.path.join(models_dir, 'global_scaler.pkl'))
        except Exception as e:
            self.load_errors.append(f"Global feature scaler: {e}")

        try:
            self.gbm_classifier = joblib.load(os.path.join(models_dir, 'best_gbm_eurusd.pkl'))
            self.gbm_regressor = joblib.load(os.path.join(models_dir, 'best_gbm_regressor_eurusd.pkl'))
        except Exception as e:
            self.load_errors.append(f"GBM dual pipeline: {e}")

        try:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            from tensorflow.keras.models import load_model
            self.lstm_model = load_model(os.path.join(models_dir, 'lstm_multitask_eurusd.keras'))
            self.lstm_time_steps = joblib.load(os.path.join(models_dir, 'lstm_time_steps.pkl'))
        except Exception as e:
            self.load_errors.append(f"Multi-Task LSTM: {e}")

        try:
            self.history_df = load_history(os.path.join(base_dir, config['data']['history_csv_path']))
        except Exception as e:
            self.load_errors.append(f"Historical feature context: {e}")

        self.pca_ready = self.lag_scaler is not None and self.lag_pca is not None
        self.scaler_ready = self.global_scaler is not None
        self.gbm_ready = self.pca_ready and self.scaler_ready and None not in (self.gbm_classifier, self.gbm_regressor)
        self.lstm_ready = self.pca_ready and self.scaler_ready and None not in (self.lstm_model, self.lstm_time_steps)
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

        # Graceful live-edge handling. A live fetch can return fewer bars than
        # the SMA_200 + lag/LSTM warm-up needs (thin session, holiday week, a
        # freshly listed symbol). Rather than hard-failing, back-fill the
        # missing PRECEDING rows from the bundled history so the rolling-window
        # warm-up is always satisfied, while keeping whatever fresh live bars we
        # did get. Duplicate dates are resolved in favour of the live bar.
        if ohlcv_df is None or len(ohlcv_df) == 0:
            ohlcv_df = self.history_df.tail(bars_needed)
            data_source = "history_fallback"
        elif len(ohlcv_df) < bars_needed:
            missing = bars_needed - len(ohlcv_df)
            preceding = self.history_df[self.history_df.index < ohlcv_df.index.min()].tail(missing)
            combined = pd.concat([preceding, ohlcv_df])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            ohlcv_df = combined
            data_source = f"{data_source}+history_backfill"

        macro_cfg = self.config.get('macro', {})
        macro_df, macro_source = fetch_yield_differential(
            ohlcv_df.index.min(), ohlcv_df.index.max(),
            series_ids=macro_cfg.get('fred_series'),
            cache_path=os.path.join(self.base_dir, macro_cfg.get('cache_path', 'results/yield_differential.csv')),
        )
        if macro_df is not None:
            ohlcv_df = merge_macro_features(ohlcv_df, macro_df)
        else:
            ohlcv_df = ohlcv_df.assign(yield_differential=0.0)
            macro_source = "unavailable"

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
            "yield_differential": float(last_row['yield_differential']),
            "macro_source": macro_source,
        }
        # The model's "next bar" is the next *trading* session, not the next
        # calendar day: FX closes Friday night, and the shift(-1) targets were
        # built over history that already skips weekends, so the row after a
        # Friday is a Monday. Roll Friday/Saturday forward to Monday so the
        # displayed forecast date matches what the model actually predicts.
        weekday = as_of_date.weekday()           # Mon=0 .. Fri=4, Sat=5, Sun=6
        days_ahead = {4: 3, 5: 2}.get(weekday, 1)  # Fri->Mon, Sat->Mon, else +1
        forecasting_date = (as_of_date + pd.Timedelta(days=days_ahead)).date().isoformat()

        return model_input_window, data_source, bar_used, as_of_date.date().isoformat(), forecasting_date

    def _predict_gbm(self, model_input_row):
        """
        Run the GBM dual pipeline on a single flat feature row (no sliding
        window -- tree ensembles consume one observation at a time, unlike
        the LSTM). `model_input_row` must already be PCA-reduced and in
        FEATURE_COLUMNS/model_input_columns() order; it is scaled here with
        the single global_scaler (the same instance the LSTM uses) before
        either head sees it.
        """
        scaled = self.global_scaler.transform(model_input_row.to_frame().T)
        prob_up = float(self.gbm_classifier.predict_proba(scaled)[0, 1])
        pred_class = int(self.gbm_classifier.predict(scaled)[0])
        # The regressor is now trained on target_return in PERCENT units (see
        # src/features.py), so its output is already a percentage -- no *100.
        predicted_return = float(self.gbm_regressor.predict(scaled)[0])
        return {
            "direction": "UP" if pred_class == 1 else "DOWN",
            "confidence": prob_up if pred_class == 1 else (1 - prob_up),
            "predicted_return_pct": predicted_return,
        }

    def _predict_lstm(self, model_input_window):
        """
        Run the Multi-Task LSTM on a `(time_steps, n_features)` sliding
        window (the model's two Functional-API heads: `return_output` for
        the continuous % return, `direction_output` for the UP/DOWN
        probability). Scaled with the SAME global_scaler the GBM uses -- both
        families now share one scaler fit on the unified 0-80% train block
        (see config.json `split` and _train_pipeline.py).
        """
        # Pass the named DataFrame (columns already in model_input_columns()
        # order) so the scaler validates feature names instead of warning.
        scaled = self.global_scaler.transform(model_input_window)
        window_3d = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
        predicted_return, prob_up = self.lstm_model.predict(window_3d, verbose=0)
        # Both heads are trained on target_return in PERCENT units, so this
        # output is already a percentage and is symmetric with _predict_gbm
        # (neither multiplies by 100).
        predicted_return_pct = float(predicted_return.ravel()[0])
        prob_up = float(prob_up.ravel()[0])
        return {
            "direction": "UP" if prob_up >= 0.5 else "DOWN",
            "confidence": prob_up if prob_up >= 0.5 else (1 - prob_up),
            "predicted_return_pct": predicted_return_pct,
        }

    @staticmethod
    def compute_consensus(predictions: dict) -> dict:
        """Committee logic with a low-confidence guard. If every model agrees on
        direction, average their confidence/return -- UNLESS that averaged
        confidence is strictly below CONFIDENCE_THRESHOLD, in which case the
        unanimous-but-coin-flip call is downgraded to "MIXED / LOW CONFIDENCE"
        with agreement=False (a near-chance head must not dictate the ensemble).
        On genuine disagreement, defer to whichever model is more confident
        rather than averaging across opposite-signed predictions."""
        directions = {name: p['direction'] for name, p in predictions.items()}
        agreement = len(set(directions.values())) == 1

        if agreement:
            direction = next(iter(directions.values()))
            confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
            predicted_return_pct = sum(p['predicted_return_pct'] for p in predictions.values()) / len(predictions)
            if confidence < PredictionService.CONFIDENCE_THRESHOLD:
                # Unanimous direction, but neither head is meaningfully above a
                # coin flip -- do not advertise this as a confident agreement.
                agreement = False
                direction = "MIXED / LOW CONFIDENCE"
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
        """
        End-to-end, zero-input inference for t+1: resolve the latest live
        feature window (Section 2/2B's MT5/yfinance + FRED fallback chains),
        run whichever of the GBM/LSTM pipelines is loaded, and assemble a
        single response dict with both models' predictions plus a committee
        consensus (see compute_consensus). Raises RuntimeError if no model
        artifacts loaded successfully at all.
        """
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
