import os

import joblib
import pandas as pd

from .features import (
    load_history, compute_features, apply_lag_pca, merge_macro_features,
    LAG_COLUMNS, FEATURE_COLUMNS,
)
from .live_data import fetch_live_market_data, drop_incomplete_bars
from .macro_data import fetch_yield_differential


class PredictionService:
    """
    Loads every trained artifact once and serves Multi-Task EURUSD
    predictions from automatically fetched live market data. Used by api.py
    (the single web entry point) and by the training notebook for evaluation.
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
        # Auxiliary H1->Daily ensemble (loaded independently below).
        self.h1_xgb = self.h1_rf = self.h1_svm = self.h1_lstm_model = None
        self.h1_feature_scaler = self.h1_lstm_scaler = None
        self.h1_feature_columns = self.h1_lstm_config = None

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
            # Single-row inference doesn't benefit from GPU; force CPU so the
            # artifact stays portable to machines without a CUDA-capable device.
            # get_booster() is XGBoost-specific — safe to call only on XGBoost models.
            for _m in (self.gbm_classifier, self.gbm_regressor):
                if hasattr(_m, 'get_booster') and hasattr(_m, 'set_params'):
                    _m.set_params(device='cpu')
        except Exception as e:
            self.load_errors.append(f"GBM dual pipeline: {e}")

        try:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            import tensorflow as tf
            for _gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(_gpu, True)
            from tensorflow.keras.models import load_model
            self.lstm_model = load_model(os.path.join(models_dir, 'lstm_multitask_eurusd.keras'))
            self.lstm_time_steps = joblib.load(os.path.join(models_dir, 'lstm_time_steps.pkl'))
        except Exception as e:
            self.load_errors.append(f"Multi-Task LSTM: {e}")

        try:
            self.history_df = load_history(os.path.join(base_dir, config['data']['history_csv_path']))
        except Exception as e:
            self.load_errors.append(f"Historical feature context: {e}")

        # Auxiliary H1->Daily predictor: four return regressors + their two
        # scalers + column/config metadata. Loaded on its own try/except so its
        # absence never blocks the daily GBM/LSTM service (it is supplementary).
        try:
            self.h1_xgb = joblib.load(os.path.join(models_dir, 'h1_xgb_regressor.pkl'))
            self.h1_rf = joblib.load(os.path.join(models_dir, 'h1_rf_regressor.pkl'))
            self.h1_svm = joblib.load(os.path.join(models_dir, 'h1_svm_regressor.pkl'))
            # Force XGBoost to CPU for portable single-row inference (same
            # rationale as the daily GBM heads above).
            if hasattr(self.h1_xgb, 'get_booster') and hasattr(self.h1_xgb, 'set_params'):
                self.h1_xgb.set_params(device='cpu')
            self.h1_feature_scaler = joblib.load(os.path.join(models_dir, 'h1_feature_scaler.pkl'))
            self.h1_lstm_scaler = joblib.load(os.path.join(models_dir, 'h1_lstm_scaler.pkl'))
            self.h1_feature_columns = joblib.load(os.path.join(models_dir, 'h1_feature_columns.pkl'))
            self.h1_lstm_config = joblib.load(os.path.join(models_dir, 'h1_lstm_config.pkl'))
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            from tensorflow.keras.models import load_model
            self.h1_lstm_model = load_model(os.path.join(models_dir, 'h1_lstm.keras'))
        except Exception as e:
            self.load_errors.append(f"H1->Daily predictor: {e}")

        self.pca_ready = self.lag_scaler is not None and self.lag_pca is not None
        self.scaler_ready = self.global_scaler is not None
        self.gbm_ready = self.pca_ready and self.scaler_ready and None not in (self.gbm_classifier, self.gbm_regressor)
        self.lstm_ready = self.pca_ready and self.scaler_ready and None not in (self.lstm_model, self.lstm_time_steps)
        self.h1_ready = None not in (
            self.h1_xgb, self.h1_rf, self.h1_svm, self.h1_lstm_model,
            self.h1_feature_scaler, self.h1_lstm_scaler, self.h1_feature_columns,
        )
        self.models_ready = (self.gbm_ready or self.lstm_ready) and self.history_df is not None

    @staticmethod
    def _fetch_bar_count(bars_needed: int, now=None) -> int:
        """
        Exact replacement for an empirical over-fetch multiplier. MT5 emits
        one bar per Mon-Fri weekday plus one partial Sunday bar per week,
        all of which _drop_incomplete_bars later strips except the weekday
        ones; today's still-forming bar is stripped too. Rather than guess
        that inflation with a flat percentage, count precisely how many
        Sundays actually fall inside the calendar window that must contain
        `bars_needed` weekday sessions (5 weekdays per 7 calendar days), and
        request exactly that many extra bars plus 1 for today's forming bar.

        The one thing that genuinely cannot be counted exactly from the
        weekday alone is bank holidays (Christmas, Good Friday, etc.) -- those
        don't follow a fixed weekly cycle, so a small fixed pad covers them;
        it only pads holiday weeks, it no longer pads every single week the
        way the old 1.45x multiplier did.
        """
        HOLIDAY_PAD_DAYS = 10
        now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
        today = now.normalize()
        lookback_days = -(-bars_needed * 7 // 5) + HOLIDAY_PAD_DAYS  # ceil(bars_needed * 7/5) + pad
        window = pd.date_range(today - pd.Timedelta(days=lookback_days), today, freq='D')
        sundays_in_window = int((window.weekday == 6).sum())
        forming_today_bar = 1
        return bars_needed + sundays_in_window + forming_today_bar

    # Kept as a PredictionService-scoped alias (the canonical implementation
    # now lives in live_data.drop_incomplete_bars so tracking.py can reuse it
    # too without a circular import -- see that function's docstring).
    _drop_incomplete_bars = staticmethod(drop_incomplete_bars)

    def _resolve_latest_window(self, time_steps: int):
        """
        Automated data pipeline (no manual input ever required): knows
        "today" implicitly as whatever the live source's most recent
        *completed* bar is (today's still-forming bar and MT5's partial
        weekend bar are dropped up front -- see _drop_incomplete_bars).

        Tries a live MT5 terminal session first, then Yahoo Finance, fetching
        exactly enough daily bars to satisfy the SMA_200 warm-up plus the
        LSTM's sliding window. Only falls back to the bundled historical CSV
        tail if neither live source is reachable at all.
        """
        bars_needed = max(self.config['data'].get('live_fetch_bars', 250), 200 + time_steps)
        mt5_symbol = self.config['data']['symbol']
        yf_symbol = self.config['data'].get('live_symbol', 'EURUSD=X')

        # Over-fetch so that dropping today's forming bar AND every weekend bar
        # (below) still leaves at least `bars_needed` completed weekday bars --
        # otherwise the back-fill path would trigger on every live MT5 call and
        # mislabel the data_source. _fetch_bar_count counts the actual number
        # of Sundays in the lookback window instead of guessing with a flat
        # percentage; yfinance is already weekday-only, so over-fetching there
        # just yields harmless extra warm-up.
        fetch_bars = self._fetch_bar_count(bars_needed)
        ohlcv_df, data_source = fetch_live_market_data(mt5_symbol, yf_symbol, bars=fetch_bars)

        # Restrict the live feed to fully-closed weekday sessions BEFORE any
        # warm-up/back-fill accounting, so as_of_date can only ever be the last
        # completed bar. This drops today's still-forming bar and MT5's partial
        # weekend bar (see _drop_incomplete_bars). The bundled history and the
        # back-fill rows are already weekday-only and free of a forming bar, so
        # they need no trimming.
        if ohlcv_df is not None and len(ohlcv_df) > 0:
            ohlcv_df = self._drop_incomplete_bars(ohlcv_df)

        # Graceful live-edge handling. A live fetch can return fewer bars than
        # the SMA_200 + lag/LSTM warm-up needs (thin session, holiday week, a
        # freshly listed symbol). Rather than hard-failing, back-fill the
        # missing PRECEDING rows from the bundled history so the rolling-window
        # warm-up is always satisfied, while keeping whatever fresh live bars we
        # did get. Duplicate dates are resolved in favour of the live bar.
        if ohlcv_df is None or len(ohlcv_df) == 0:
            if self.history_df is None:
                raise RuntimeError("No live data and no history fallback available.")
            ohlcv_df = self.history_df.tail(bars_needed)
            data_source = "history_fallback"
        elif len(ohlcv_df) < bars_needed and self.history_df is not None:
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

    def _predict_h1(self, now=None):
        """
        Run the auxiliary H1->Daily ensemble on the latest COMPLETE trading day.

        Fetches recent H1 bars (MT5 -> yfinance -> cache), engineers that day's
        flattened features + 24-hour tensor, and runs all four regressors. Each
        outputs a next-day % return; its direction is that return's sign. The
        RBF SVM consumes the scaled flat features; the scale-invariant tree
        models consume the raw ones; the LSTM consumes the per-hour-scaled
        tensor. Returns (per_model_dict, as_of_date_iso).

        Kept deliberately separate from the daily GBM/LSTM committee
        (compute_consensus): these are return-only regressors with no calibrated
        probability, so they carry their own agreement-based consensus
        (compute_h1_consensus) rather than polluting the probability-tuned daily
        one.
        """
        from .h1_features import build_h1_inference_sample

        h1_cfg = self.config.get('h1', {})
        cache = os.path.join(self.base_dir, h1_cfg.get('cache_path', 'results/eurusd_h1.csv'))
        flat_row, seq, as_of = build_h1_inference_sample(cache_path=cache, now=now)

        X_raw = flat_row[self.h1_feature_columns].values   # enforce trained column order
        X_scaled = self.h1_feature_scaler.transform(X_raw)
        nf = seq.shape[2]
        seq_s = self.h1_lstm_scaler.transform(seq.reshape(-1, nf)).reshape(seq.shape).astype('float32')

        raw = {
            'h1_xgboost': float(self.h1_xgb.predict(X_raw)[0]),
            'h1_random_forest': float(self.h1_rf.predict(X_raw)[0]),
            'h1_svm': float(self.h1_svm.predict(X_scaled)[0]),
            'h1_lstm': float(self.h1_lstm_model.predict(seq_s, verbose=0).ravel()[0]),
        }
        per_model = {
            name: {"direction": "UP" if r > 0 else "DOWN", "predicted_return_pct": r}
            for name, r in raw.items()
        }
        return per_model, as_of.date().isoformat()

    @staticmethod
    def compute_h1_consensus(per_model: dict) -> dict:
        """Aggregate the four return-only H1 regressors into one call. Direction
        is the majority sign; confidence is the fraction of models on that side
        (a genuine [0.5, 1.0] agreement measure, NOT a probability); predicted
        return is the mean across all models. agreement=True only on a unanimous
        sign."""
        dirs = [p['direction'] for p in per_model.values()]
        n = len(dirs)
        up = dirs.count("UP")
        down = n - up
        return {
            "direction": "UP" if up >= down else "DOWN",
            "agreement": up == n or down == n,
            "confidence": max(up, down) / n,
            "predicted_return_pct": sum(p['predicted_return_pct'] for p in per_model.values()) / n,
            "n_models": n,
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
            if self.lstm_time_steps is None or len(model_input_window) < self.lstm_time_steps:
                response['lstm_error'] = "Not enough historical context for the LSTM sliding window."
            else:
                predictions['lstm'] = self._predict_lstm(model_input_window.tail(self.lstm_time_steps))

        response.update(predictions)
        if predictions:
            response['consensus'] = self.compute_consensus(predictions)

        # Supplementary H1->Daily ensemble (separate intraday predictor). It has
        # its own data feed and its own agreement-based consensus, surfaced as a
        # self-contained block so it never perturbs the daily committee above.
        # Any failure here (no H1 feed, etc.) degrades to an h1_error note.
        if self.h1_ready:
            try:
                h1_per_model, h1_as_of = self._predict_h1()
                response['h1'] = {
                    "as_of_date": h1_as_of,
                    "predictions": h1_per_model,
                    "consensus": self.compute_h1_consensus(h1_per_model),
                }
            except Exception as e:
                response['h1_error'] = str(e)

        # Best-effort: log this forecast for the /history prediction-vs-actual
        # table. Logging must never break a prediction, so swallow any error.
        try:
            from .tracking import log_prediction
            log_path = self.config.get('tracking', {}).get('log_path', 'results/prediction_log.csv')
            log_prediction(response, os.path.join(self.base_dir, log_path))
        except Exception:
            pass

        return response
