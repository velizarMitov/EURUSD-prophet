import json
import os

import joblib
import pandas as pd

from .features import (
    load_history, compute_features, apply_lag_pca, merge_macro_features,
    LAG_COLUMNS, FEATURE_COLUMNS, PRICE_FEATURE_COLUMNS, variant_feature_columns,
)
from .live_data import fetch_live_market_data, drop_incomplete_bars
from .macro_data import fetch_macro_features


class PredictionService:
    """
    Loads every trained artifact once and serves Multi-Task EURUSD predictions
    from automatically fetched live market data. Used by api.py (the single web
    entry point) and by the training notebook for evaluation.

    Dual-variant architecture (see ARCHITECTURE_DOCS.md): TWO complete daily
    model families are loaded and served side by side on every prediction —
    'baseline' (price-only, no macro features) and 'with_macro' (the full
    27-column set whose macro features remain statistically unproven, KEEP-
    provisional under the Bonferroni bar). Each variant has an independent
    readiness gate (baseline_ready / macro_ready) following the existing
    graceful-degradation pattern: a missing/broken variant never crashes the
    other. predict() returns both variants' committees plus a
    `variant_agreement` flag — a disagreement is direct evidence the macro
    block is actually changing the decision.
    """

    # Minimum averaged confidence for a unanimous call to count as a genuine
    # agreement. The direction heads sit near chance (ROC-AUC ~0.50), so a
    # coin-flip "agreement" must not be surfaced as a confident ensemble call.
    CONFIDENCE_THRESHOLD = 0.52

    def __init__(self, base_dir: str, config: dict):
        """
        Load every serialized artifact exactly once at process start-up: for
        EACH variant its own PCA, global feature scaler, both GBM heads and
        Multi-Task LSTM (+ time_steps) from models/<variant>/, plus the shared
        bundled historical OHLCV CSV and the shared H1 ensemble. Each load is
        independently try/excepted into self.load_errors (prefixed with the
        variant name) rather than failing fast, so e.g. a broken with_macro
        LSTM still leaves the baseline committee fully servable.
        """
        self.config = config
        self.base_dir = base_dir
        models_dir = os.path.join(base_dir, 'models')
        self.load_errors = []

        self.variant_names = config.get('variants', ['baseline', 'with_macro'])
        self.variants = {
            name: self._load_variant_artifacts(os.path.join(models_dir, name), name)
            for name in self.variant_names
        }

        self.history_df = None
        # Auxiliary H1->Daily ensemble (shared across variants — it is price-only
        # by construction and independent of the daily feature sets).
        self.h1_xgb = self.h1_rf = self.h1_svm = self.h1_lstm_model = None
        self.h1_feature_scaler = self.h1_lstm_scaler = None
        self.h1_feature_columns = self.h1_lstm_config = None

        try:
            self.history_df = load_history(os.path.join(base_dir, config['data']['history_csv_path']))
        except Exception as e:
            self.load_errors.append(f"Historical feature context: {e}")

        # Auxiliary H1->Daily predictor: four return regressors + their two
        # scalers + column/config metadata. Loaded on its own try/except so its
        # absence never blocks the daily variants (it is supplementary).
        try:
            self.h1_xgb = joblib.load(os.path.join(models_dir, 'h1_xgb_regressor.pkl'))
            self.h1_rf = joblib.load(os.path.join(models_dir, 'h1_rf_regressor.pkl'))
            self.h1_svm = joblib.load(os.path.join(models_dir, 'h1_svm_regressor.pkl'))
            # Force XGBoost to CPU for portable single-row inference (same
            # rationale as the daily GBM heads).
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

        self.h1_ready = None not in (
            self.h1_xgb, self.h1_rf, self.h1_svm, self.h1_lstm_model,
            self.h1_feature_scaler, self.h1_lstm_scaler, self.h1_feature_columns,
        )

        # Next-day realized-volatility ensemble (models/volatility/): the
        # 5-seed multi-task LSTM ensemble whose volatility head cleared the
        # pre-registered validation ship gate against GARCH(1,1) (see
        # src/volatility.py + results/volatility_seed_ensemble.csv). Loaded on
        # its own try/except — its absence never blocks the daily variants.
        # The validated object is the FULL seed ensemble, so vol_ready demands
        # every seed model: serving a partial ensemble would be a silently
        # different (unvalidated) predictor.
        self.vol_models = []
        self.vol_lag_scaler = self.vol_lag_pca = self.vol_global_scaler = None
        self.vol_time_steps = None
        self.vol_metrics = None
        try:
            from .volatility import VOL_MODEL_DIR, VOL_MODEL_FILES
            vol_dir = os.path.join(base_dir, VOL_MODEL_DIR)
            self.vol_lag_scaler = joblib.load(os.path.join(vol_dir, 'lag_scaler.pkl'))
            self.vol_lag_pca = joblib.load(os.path.join(vol_dir, 'lag_pca.pkl'))
            self.vol_global_scaler = joblib.load(os.path.join(vol_dir, 'global_scaler.pkl'))
            self.vol_time_steps = joblib.load(os.path.join(vol_dir, 'lstm_time_steps.pkl'))
            with open(os.path.join(vol_dir, 'vol_metrics.json')) as f:
                self.vol_metrics = json.load(f)
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            from tensorflow.keras.models import load_model
            self.vol_models = [
                load_model(os.path.join(vol_dir, fname)) for fname in VOL_MODEL_FILES
            ]
        except Exception as e:
            self.load_errors.append(f"Volatility ensemble: {e}")

        self.vol_ready = (
            len(self.vol_models) > 0
            and None not in (self.vol_lag_scaler, self.vol_lag_pca,
                             self.vol_global_scaler, self.vol_time_steps)
        )

        # H1 TI-LSTM (models/ti_lstm_h1/): shipped by EXPLICIT OWNER OVERRIDE
        # (2026-07-18) DESPITE a DROP verdict on its own hypothesis bar — it has
        # NO demonstrated edge (test AUC ~0.51, ΔAUC CI vs the H1 ensemble
        # includes 0 with a negative point estimate). Served for transparent
        # forward observation ONLY; every response block carries
        # `validated: false` + the real numbers, and the UI must keep the
        # warning framing (never the volatility model's validated badge).
        # The .keras artifact was trained on the Keras3/torch backend but is
        # backend-portable (standard layers), so it loads here under tf.keras —
        # serving has no torch dependency. All-or-nothing gate like vol_ready.
        self.ti_model = None
        self.ti_scaler = self.ti_config = self.ti_metrics = None
        try:
            ti_dir = os.path.join(models_dir, 'ti_lstm_h1')
            self.ti_scaler = joblib.load(os.path.join(ti_dir, 'ti_scaler.pkl'))
            self.ti_config = joblib.load(os.path.join(ti_dir, 'ti_config.pkl'))
            with open(os.path.join(ti_dir, 'ti_metrics.json')) as f:
                self.ti_metrics = json.load(f)
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            from tensorflow.keras.models import load_model
            self.ti_model = load_model(os.path.join(ti_dir, 'ti_lstm_h1.keras'))
        except Exception as e:
            self.load_errors.append(f"H1 TI-LSTM (observational): {e}")

        self.ti_h1_ready = None not in (
            self.ti_model, self.ti_scaler, self.ti_config, self.ti_metrics,
        )

        # H1 NEXT-BAR DIRECTION (models/h1_direction/): H_dir.1, the only claim in
        # this project to survive a clean one-shot out-of-sample confirmation.
        # Shipped OBSERVATIONALLY, exactly as ti_h1 was: served, logged, and given
        # its own simulated forward ledger. NOT a real-money signal.
        #
        # The SHIPPED model is a full-history refit, so the confirmation does NOT
        # transfer to it (it saw that test block in training). meta carries
        # validated_out_of_sample=False and the serving response must never echo
        # the test-block numbers -- they belong to a different model.
        #
        # All-or-nothing gate like vol_ready. A failure here can never affect
        # /api/predict, /history or /paper-trading: it only appends to
        # self.load_errors and leaves h1_dir_ready False.
        self.h1_dir_model = self.h1_dir_scaler = self.h1_dir_meta = None
        try:
            import xgboost as xgb
            h1d_dir = os.path.join(models_dir, 'h1_direction')
            with open(os.path.join(h1d_dir, 'h1_direction_meta.json')) as f:
                self.h1_dir_meta = json.load(f)
            self.h1_dir_scaler = joblib.load(
                os.path.join(h1d_dir, 'h1_direction_scaler.pkl'))
            model = xgb.XGBClassifier()
            model.load_model(os.path.join(h1d_dir, 'h1_direction_gbm.json'))
            # Single-row inference does not benefit from GPU; force CPU so the
            # artifact stays portable to machines without a CUDA device.
            model.set_params(device='cpu')
            self.h1_dir_model = model
        except Exception as e:
            self.load_errors.append(f"H1 direction (observational): {e}")

        self.h1_dir_ready = None not in (
            self.h1_dir_model, self.h1_dir_scaler, self.h1_dir_meta,
        )
        # Independent per-variant readiness gates (the names the API surfaces).
        self.baseline_ready = self._variant_ready('baseline')
        self.macro_ready = self._variant_ready('with_macro')
        # Servable at all: at least one variant can predict and history loaded.
        self.models_ready = (
            any(self._variant_ready(n) for n in self.variant_names)
            and self.history_df is not None
        )

    def _load_variant_artifacts(self, variant_dir: str, name: str) -> dict:
        """Load one variant's full artifact set from models/<variant>/ into a
        self-contained dict with per-family readiness flags. Every failure is
        recorded as '[<variant>] ...' in self.load_errors and leaves the other
        families of this variant (and the other variant entirely) untouched."""
        v = {
            'feature_columns': variant_feature_columns(name),
            'lag_scaler': None, 'lag_pca': None, 'global_scaler': None,
            'gbm_classifier': None, 'gbm_regressor': None,
            'lstm_model': None, 'lstm_time_steps': None,
        }

        try:
            v['lag_scaler'] = joblib.load(os.path.join(variant_dir, 'lag_scaler.pkl'))
            v['lag_pca'] = joblib.load(os.path.join(variant_dir, 'lag_pca.pkl'))
        except Exception as e:
            self.load_errors.append(f"[{name}] PCA lag reduction: {e}")

        try:
            v['global_scaler'] = joblib.load(os.path.join(variant_dir, 'global_scaler.pkl'))
        except Exception as e:
            self.load_errors.append(f"[{name}] Global feature scaler: {e}")

        try:
            v['gbm_classifier'] = joblib.load(os.path.join(variant_dir, 'best_gbm_eurusd.pkl'))
            v['gbm_regressor'] = joblib.load(os.path.join(variant_dir, 'best_gbm_regressor_eurusd.pkl'))
            # Single-row inference doesn't benefit from GPU; force CPU so the
            # artifact stays portable to machines without a CUDA-capable device.
            for _m in (v['gbm_classifier'], v['gbm_regressor']):
                if hasattr(_m, 'get_booster') and hasattr(_m, 'set_params'):
                    _m.set_params(device='cpu')
        except Exception as e:
            self.load_errors.append(f"[{name}] GBM dual pipeline: {e}")

        try:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            import tensorflow as tf
            for _gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(_gpu, True)
            from tensorflow.keras.models import load_model
            v['lstm_model'] = load_model(os.path.join(variant_dir, 'lstm_multitask_eurusd.keras'))
            v['lstm_time_steps'] = joblib.load(os.path.join(variant_dir, 'lstm_time_steps.pkl'))
        except Exception as e:
            self.load_errors.append(f"[{name}] Multi-Task LSTM: {e}")

        v['pca_ready'] = v['lag_scaler'] is not None and v['lag_pca'] is not None
        v['scaler_ready'] = v['global_scaler'] is not None
        v['gbm_ready'] = v['pca_ready'] and v['scaler_ready'] and None not in (v['gbm_classifier'], v['gbm_regressor'])
        v['lstm_ready'] = v['pca_ready'] and v['scaler_ready'] and None not in (v['lstm_model'], v['lstm_time_steps'])
        return v

    def _variant_ready(self, name: str) -> bool:
        """A variant is servable when at least one of its model families
        (GBM committee member or LSTM) loaded completely."""
        v = self.variants.get(name)
        return bool(v) and (v['gbm_ready'] or v['lstm_ready'])

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

        Returns the RAW engineered feature window over the SUPERSET
        FEATURE_COLUMNS (27 cols) — predict() then selects each variant's own
        column subset and applies that variant's PCA/scaler, so one data fetch
        serves both variants.
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
        macro_df, macro_sources = fetch_macro_features(
            ohlcv_df.index.min(), ohlcv_df.index.max(), macro_cfg, base_dir=self.base_dir,
        )
        # merge_macro_features + compute_features neutralize any entirely-missing
        # macro column, so an empty frame here (every feed unreachable) degrades
        # gracefully rather than failing the prediction. The baseline variant
        # never consumes these columns at all, so it is immune either way.
        ohlcv_df = merge_macro_features(
            ohlcv_df, macro_df if macro_df is not None else pd.DataFrame(index=ohlcv_df.index)
        )
        macro_source = macro_sources.get('yield_differential', 'unavailable')

        engineered = compute_features(ohlcv_df).dropna(subset=FEATURE_COLUMNS)
        if len(engineered) < time_steps:
            raise RuntimeError(
                f"Insufficient bars after SMA_200/lag warm-up: got {len(engineered)} usable rows, need {time_steps}."
            )

        feature_window = engineered[FEATURE_COLUMNS].tail(time_steps)

        as_of_date = engineered.index[-1]
        last_row = ohlcv_df.loc[as_of_date]

        def _macro_val(col):
            # ffilled level for display; None if the feed was entirely absent
            v = last_row.get(col)
            return float(v) if v is not None and pd.notna(v) else None

        bar_used = {
            "date": as_of_date.date().isoformat(),
            "open": float(last_row['open']),
            "high": float(last_row['high']),
            "low": float(last_row['low']),
            "close": float(last_row['close']),
            "tick_volume": float(last_row['tick_volume']),
            "yield_differential": _macro_val('yield_differential'),
            "usd_index": _macro_val('usd_index'),
            "policy_rate_differential": _macro_val('policy_rate_differential'),
            "inflation_differential": _macro_val('inflation_differential'),
            "macro_source": macro_source,
            "macro_sources": macro_sources,
        }
        # The model's "next bar" is the next *trading* session, not the next
        # calendar day: FX closes Friday night, and the shift(-1) targets were
        # built over history that already skips weekends, so the row after a
        # Friday is a Monday. Roll Friday/Saturday forward to Monday so the
        # displayed forecast date matches what the model actually predicts.
        weekday = as_of_date.weekday()           # Mon=0 .. Fri=4, Sat=5, Sun=6
        days_ahead = {4: 3, 5: 2}.get(weekday, 1)  # Fri->Mon, Sat->Mon, else +1
        forecasting_date = (as_of_date + pd.Timedelta(days=days_ahead)).date().isoformat()

        return feature_window, data_source, bar_used, as_of_date.date().isoformat(), forecasting_date

    @staticmethod
    def _predict_gbm(v: dict, model_input_row):
        """
        Run one variant's GBM dual pipeline on a single flat feature row (no
        sliding window -- tree ensembles consume one observation at a time,
        unlike the LSTM). `model_input_row` must already be PCA-reduced with
        THAT variant's lag_pca and in its model_input_columns() order; it is
        scaled here with the variant's global_scaler (the same instance its
        LSTM uses) before either head sees it.
        """
        scaled = v['global_scaler'].transform(model_input_row.to_frame().T)
        prob_up = float(v['gbm_classifier'].predict_proba(scaled)[0, 1])
        pred_class = int(v['gbm_classifier'].predict(scaled)[0])
        # The regressor is trained on target_return in PERCENT units (see
        # src/features.py), so its output is already a percentage -- no *100.
        predicted_return = float(v['gbm_regressor'].predict(scaled)[0])
        return {
            "direction": "UP" if pred_class == 1 else "DOWN",
            "confidence": prob_up if pred_class == 1 else (1 - prob_up),
            "predicted_return_pct": predicted_return,
        }

    @staticmethod
    def _predict_lstm(v: dict, model_input_window):
        """
        Run one variant's Multi-Task LSTM on a `(time_steps, n_features)`
        sliding window (two Functional-API heads: `return_output` for the
        continuous % return, `direction_output` for the UP/DOWN probability).
        Scaled with the SAME per-variant global_scaler the variant's GBM uses.
        """
        # Pass the named DataFrame (columns already in model_input_columns()
        # order) so the scaler validates feature names instead of warning.
        scaled = v['global_scaler'].transform(model_input_window)
        window_3d = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
        predicted_return, prob_up = v['lstm_model'].predict(window_3d, verbose=0)
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

    def _predict_volatility(self, feature_window) -> dict:
        """
        Run the 5-seed volatility ensemble on the shared raw feature window:
        select the price-only column subset, apply the volatility family's own
        PCA + scaler (fit on [0:80%] at training time), and average the
        VOLATILITY head across the seed models. The return/direction heads of
        these multi-task models are training scaffolding only and are
        deliberately discarded here — the daily variants remain the sole
        source of return/direction predictions.

        The response block carries the validated GARCH/persistence context
        from vol_metrics.json so the UI can frame the number exactly as
        rigorously as it was earned (Production Methodology: label according
        to what the test actually found).
        """
        vol_window = feature_window[PRICE_FEATURE_COLUMNS]
        model_input = apply_lag_pca(vol_window, self.vol_lag_scaler, self.vol_lag_pca,
                                    lag_columns=LAG_COLUMNS)
        if len(model_input) < self.vol_time_steps:
            return {"error": "Not enough historical context for the volatility sliding window."}
        scaled = self.vol_global_scaler.transform(model_input.tail(self.vol_time_steps))
        window_3d = scaled.reshape(1, scaled.shape[0], scaled.shape[1])

        vol_preds = []
        for m in self.vol_models:
            _ret, _dir, vol = m.predict(window_3d, verbose=0)
            vol_preds.append(float(vol.ravel()[0]))
        predicted_vol_pct = sum(vol_preds) / len(vol_preds)

        block = {
            "predicted_vol_pct": predicted_vol_pct,
            "unit": "percent — |next-day log return| * 100 (single-day realized volatility proxy)",
            "model": f"{len(self.vol_models)}-seed multi-task LSTM ensemble, volatility head (price-only)",
        }
        metrics = self.vol_metrics or {}
        decision = metrics.get('validation_decision', {})
        if decision:
            block["vs_garch_baseline"] = {
                "validated": bool(decision.get('cleared_bar')),
                "arbiter": "validation slice [70%:80%], test block untouched",
                "ensemble_mae_pct": decision.get('mt_ensemble_mae'),
                "garch_mae_pct": decision.get('garch_mae'),
                "delta_mae_pct": decision.get('point_delta_mae'),
                "delta_mae_ci": [decision.get('ci_dmae_low'), decision.get('ci_dmae_high')],
                "ensemble_r2": decision.get('mt_ensemble_r2'),
                "garch_r2": decision.get('garch_r2'),
                "delta_r2_ci": [decision.get('ci_dr2_low'), decision.get('ci_dr2_high')],
                "alpha_bonferroni": decision.get('alpha_bar'),
            }
        persistence = metrics.get('validation_persistence_baseline', {})
        if persistence:
            block["vs_persistence_baseline"] = {
                "persistence_mae_pct": persistence.get('persistence_mae'),
                "persistence_r2": persistence.get('persistence_r2'),
            }
        if 'test_ensemble_mae' in metrics:
            block["test_report_one_shot"] = {
                "ensemble_mae_pct": metrics['test_ensemble_mae'],
                "ensemble_r2": metrics['test_ensemble_r2'],
                "garch_mae_pct": metrics['test_garch_mae'],
                "garch_r2": metrics['test_garch_r2'],
            }
        return block

    def _predict_h1(self, now=None):
        """
        Run the auxiliary H1->Daily ensemble on the latest COMPLETE trading day.

        Fetches recent H1 bars live-first with a staleness gate (refresh_h1_frame:
        cache served only if already current, else MT5 -> yfinance with cached
        history backfilled under thin pulls), engineers that day's flattened
        features + 24-hour tensor, and runs all four regressors. Each outputs a
        next-day % return; its direction is that return's sign. The RBF SVM
        consumes the scaled flat features; the scale-invariant tree models
        consume the raw ones; the LSTM consumes the per-hour-scaled tensor.
        Returns (per_model_dict, as_of_date_iso, data_source).

        Kept deliberately separate from the daily GBM/LSTM committee
        (compute_consensus): these are return-only regressors with no calibrated
        probability, so they carry their own agreement-based consensus
        (compute_h1_consensus) rather than polluting the probability-tuned daily
        one.
        """
        from .h1_features import build_h1_inference_sample

        h1_cfg = self.config.get('h1', {})
        cache = os.path.join(self.base_dir, h1_cfg.get('cache_path', 'results/eurusd_h1.csv'))
        flat_row, seq, as_of, h1_source = build_h1_inference_sample(cache_path=cache, now=now)

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
        return per_model, as_of.date().isoformat(), h1_source

    def _predict_ti_h1(self, now=None) -> dict:
        """
        Run the observational H1 TI-LSTM on the latest complete session's
        (1, 24, 8) technical-indicator tensor. The block is HARD-LABELED
        unvalidated: this model shipped by explicit owner override DESPITE a
        DROP verdict (no demonstrated edge; negative point ΔAUC vs the H1
        ensemble on the one-shot test comparison). Client code can key off
        `validated: false`; the evidence numbers ride along verbatim so no
        consumer has to trust a bare badge.
        """
        from .ti_lstm_h1_experimental import build_ti_inference_sample

        h1_cfg = self.config.get('h1', {})
        cache = os.path.join(self.base_dir, h1_cfg.get('cache_path', 'results/eurusd_h1.csv'))
        X, as_of, source = build_ti_inference_sample(cache_path=cache, now=now)

        n_feat = X.shape[2]
        X_s = self.ti_scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype('float32')
        pred_ret, prob_up = self.ti_model.predict(X_s, verbose=0)
        pred_ret = float(pred_ret.ravel()[0])
        prob_up = float(prob_up.ravel()[0])

        m = self.ti_metrics or {}
        return {
            "as_of_date": as_of.date().isoformat(),
            "data_source": source,
            "direction": "UP" if prob_up >= 0.5 else "DOWN",
            "probability_up": prob_up,
            "predicted_return_pct": pred_ret,
            "model": f"H1 technical-indicator LSTM ({m.get('architecture', '2x64')}, "
                     f"%B/MACD-13-34-8/SMA504-168/RSI24/CCI20/ADX14)",
            "validated": False,
            "status": "NOT VALIDATED — NO DEMONSTRATED EDGE. Shipped by explicit "
                      "owner override for transparent forward observation only; "
                      "it FAILED its own hypothesis bar (DROP).",
            "evidence": {
                "test_auc": m.get('test_auc'),
                "test_acc": m.get('test_acc'),
                "test_h1_ensemble_auc": m.get('test_h1_ensemble_auc'),
                "test_dauc_vs_ensemble": m.get('test_dauc_vs_ensemble'),
                "test_dauc_ci": m.get('test_dauc_ci'),
                "val_auc": m.get('val_auc'),
                "hypothesis_log": m.get('hypothesis_log'),
            },
        }

    def predict_h1_direction(self, now=None, frame=None) -> dict:
        """
        H_dir.1 next-H1-bar direction, on demand. One call, one prediction.

        THE COMPLETED-BAR RULE. The model is trained on *features from a
        COMPLETED bar -> direction of the NEXT bar*, so the base bar here MUST be
        the last fully closed hourly bar. `drop_incomplete_h1_bars` removes the
        currently-forming hour (and weekend bars), so the still-moving hour can
        never become the base. Every timestamp below is derived from that cleaned
        frame, never from the wall clock alone.

        BAR LABELLING. The feed labels a bar by its START, so the bar labelled
        14:00 covers 14:00-15:00 and CLOSES at 15:00. `as_of_bar_close` is
        therefore the base bar's label + 1h -- the instant it closed -- and the
        forecast bar is the hour that begins at that instant.

        HONESTY. The shipped model is a full-history refit and has NO
        out-of-sample confirmation of its own, so the response carries
        `validated_out_of_sample: false` and NEVER echoes H_dir.1's test-block
        numbers -- those belong to the [0:70%] model, not this one. The only
        evidence for this model is its own forward ledger, whose settled count
        rides along in `forward_observations`.

        Observational. Simulated ledger only. No order placement, no sizing, no
        stop-loss -- nothing here converts a prediction into an action.
        """
        from .h1_features import (
            DIRECTION_FEATURE_COLUMNS, apply_direction_standardizer,
            compute_h1_direction_features, MIN_BARS_FOR_DIRECTION_FEATURES,
        )
        from .live_data import (drop_incomplete_h1_bars, fetch_h1_market_data,
                                h1_feed_now_with_status)

        # `frame` is a test seam: pass a hand-built OHLC frame to drive the
        # completed-bar and minutes-remaining logic deterministically without a
        # live feed. Production always leaves it None.
        source = 'injected'
        if frame is None:
            # cache_path=None: never rewrite results/eurusd_h1.csv (a protected
            # file owned by the DAILY auxiliary predictor) as a serving side effect.
            frame, source = fetch_h1_market_data(bars=2000, cache_path=None)
        if frame is None or len(frame) == 0:
            fallback = os.path.join(self.base_dir, 'results/pooled_h1/EURUSD_h1.csv')
            if not os.path.exists(fallback):
                raise RuntimeError('No H1 data reachable (live chain failed, no cache).')
            frame = pd.read_csv(fallback, index_col=0, parse_dates=True)
            frame.index = (frame.index.tz_localize('UTC') if frame.index.tz is None
                           else frame.index.tz_convert('UTC'))
            source = 'cache'

        frame = frame.sort_index()
        # `now` is in the FEED'S clock (broker server time), inferred from the
        # feed itself. Neither utcnow nor local time matches those bar labels --
        # see live_data.infer_h1_feed_now for the measured evidence.
        clock_confirmed = True
        if now is None:
            now_ts, clock_confirmed = h1_feed_now_with_status(frame.index)
        else:
            now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is not None:
            now_ts = now_ts.tz_localize(None)

        closed = drop_incomplete_h1_bars(frame, now=now_ts)
        if closed is None or len(closed) < MIN_BARS_FOR_DIRECTION_FEATURES:
            raise RuntimeError(
                f'Not enough closed H1 bars for features: got '
                f'{0 if closed is None else len(closed)}, need '
                f'{MIN_BARS_FOR_DIRECTION_FEATURES}.')

        feats = compute_h1_direction_features(closed).dropna(
            subset=list(DIRECTION_FEATURE_COLUMNS))
        if not len(feats):
            raise RuntimeError('No complete H1 feature row available.')

        # Pin the column order against what the model was trained with, so a
        # reordering can never silently feed the booster a permuted matrix.
        trained_cols = list((self.h1_dir_meta or {}).get(
            'feature_columns', DIRECTION_FEATURE_COLUMNS))
        row = feats.iloc[[-1]][trained_cols]
        base_label = pd.Timestamp(feats.index[-1])
        if base_label.tzinfo is not None:
            base_label = base_label.tz_localize(None)

        X = apply_direction_standardizer(row.to_numpy(float), self.h1_dir_scaler)
        prob_up = float(self.h1_dir_model.predict_proba(X)[0, 1])

        bar_close = base_label + pd.Timedelta(hours=1)      # the instant it closed
        forecast_end = bar_close + pd.Timedelta(hours=1)
        status = self._h1_forecast_status(bar_close, forecast_end, now_ts)
        if not clock_confirmed:
            # Cold start inside the emit-lag window: the base bar may be an hour
            # off and there is no persisted baseline to check it against. A
            # direction is only actionable when status == 'open', so this is
            # reported plainly and is excluded from the ledger downstream.
            status = 'clock_unconfirmed'
        remaining = (0 if status != 'open'
                     else int(max(0, (forecast_end - now_ts).total_seconds() // 60)))

        meta = self.h1_dir_meta or {}
        return {
            "direction": "UP" if prob_up >= 0.5 else "DOWN",
            "probability": prob_up,
            "as_of_bar_close": bar_close.isoformat(),
            "as_of_close": float(closed['close'].loc[feats.index[-1]]),
            "forecast_bar_start": bar_close.isoformat(),
            "forecast_bar_end": forecast_end.isoformat(),
            "minutes_remaining": remaining,
            "forecast_bar_status": status,
            "data_source": source,
            "model_version": meta.get('model_version'),
            "trained_through": str(meta.get('train_end', ''))[:10],
            "validated_out_of_sample": False,
            "disclaimer": ("Trained on full history; not validated out of sample. "
                           "The forward ledger is the only evidence for this "
                           "model. Observational, simulated only, not a trading "
                           "instruction."),
        }

    @staticmethod
    def _h1_forecast_status(bar_close, forecast_end, now_ts) -> str:
        """
        'open'           -- the forecast hour is still forming; the call is live.
        'already_closed' -- `now` has reached forecast_bar_end, so the predicted
                            bar is already history. NEVER present that as
                            actionable: a stale feed, a holiday or simply a call
                            made an hour late all land here.
        'market_closed'  -- the forecast hour falls in the weekend gap, so the
                            bar will not print at all.
        """
        from .live_data import H1_WEEKLY_OPEN_HOUR
        weekend_gap = (bar_close.weekday() == 5
                       or (bar_close.weekday() == 6
                           and bar_close.hour < H1_WEEKLY_OPEN_HOUR))
        if weekend_gap:
            return 'market_closed'
        if now_ts >= forecast_end:
            return 'already_closed'
        return 'open'

    @staticmethod
    def compute_h1_consensus(per_model: dict) -> dict:
        """Aggregate the four return-only H1 regressors into one call, using the
        same VOTE-BASED design as the daily committee (compute_consensus):

        * Direction = the STRICT majority sign. An exact 50/50 vote has no
          majority, so it is labeled "MIXED / TIE" — mirroring the daily
          "MIXED / LOW CONFIDENCE" honesty — never an arbitrarily crowned side.
          (The original `up >= down` silently labeled a 2-2 split "UP" while
          the displayed mean return was negative: a vote-derived label next to
          a magnitude-derived number from a contradicting definition.)
        * predicted_return_pct = mean over the MAJORITY-side models only, so
          the number is sign-consistent with the label by construction (an
          H1 model's direction IS the sign of its return, so a full-panel mean
          can contradict a 3-1 vote whenever the minority's magnitude
          dominates). On a tie, the full-panel mean is returned as context —
          the MIXED label makes no directional claim for it to contradict.
        * confidence = fraction of models on the majority side (a genuine
          [0.5, 1.0] agreement measure, NOT a probability); 0.5 on a tie.
        * agreement=True only on a unanimous sign.
        """
        dirs = [p['direction'] for p in per_model.values()]
        n = len(dirs)
        up = dirs.count("UP")
        down = n - up

        if up == down:
            return {
                "direction": "MIXED / TIE",
                "agreement": False,
                "confidence": 0.5,
                "predicted_return_pct": sum(p['predicted_return_pct'] for p in per_model.values()) / n,
                "n_models": n,
            }

        direction = "UP" if up > down else "DOWN"
        majority = [p for p in per_model.values() if p['direction'] == direction]
        return {
            "direction": direction,
            "agreement": up == n or down == n,
            "confidence": max(up, down) / n,
            "predicted_return_pct": sum(p['predicted_return_pct'] for p in majority) / len(majority),
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

    def _predict_variant(self, name: str, feature_window) -> dict:
        """Run ONE variant's full committee (GBM + LSTM + consensus) over the
        shared raw feature window. Selects the variant's own column subset and
        applies its own PCA + scaler, so the two variants can never bleed into
        each other. Returns the variant's response block ({} plus error notes
        if nothing could run)."""
        v = self.variants[name]
        block = {}
        if not (v['pca_ready'] and v['scaler_ready']):
            block['error'] = (f"Variant '{name}' artifacts not loaded: "
                              f"{[e for e in self.load_errors if e.startswith(f'[{name}]')]}")
            return block

        variant_window = feature_window[v['feature_columns']]
        model_input_window = apply_lag_pca(variant_window, v['lag_scaler'], v['lag_pca'], lag_columns=LAG_COLUMNS)

        predictions = {}
        if v['gbm_ready']:
            predictions['gbm'] = self._predict_gbm(v, model_input_window.iloc[-1])
        if v['lstm_ready']:
            if v['lstm_time_steps'] is None or len(model_input_window) < v['lstm_time_steps']:
                block['lstm_error'] = "Not enough historical context for the LSTM sliding window."
            else:
                predictions['lstm'] = self._predict_lstm(v, model_input_window.tail(v['lstm_time_steps']))

        block.update(predictions)
        if predictions:
            block['consensus'] = self.compute_consensus(predictions)
        return block

    def predict(self) -> dict:
        """
        End-to-end, zero-input inference for t+1: resolve the latest live
        feature window ONCE (MT5/yfinance + FRED fallback chains), then run it
        through BOTH variants' committees and assemble a single response:

            { as_of_date, forecasting_date, data_source, bar_used,
              "baseline":   { gbm, lstm, consensus },
              "with_macro": { gbm, lstm, consensus },
              "variant_agreement": bool | None,
              "volatility_forecast": { predicted_vol_pct, vs_garch_baseline, ... },
              "h1": {...} }

        `variant_agreement` compares the two consensus directions (True/False
        when both variants produced one; None when either side is missing) —
        a False is direct, honest evidence the unproven macro block is actually
        changing the decision. Raises RuntimeError if no variant loaded at all.
        """
        if not self.models_ready:
            raise RuntimeError(f"Model artifacts not loaded. Errors: {self.load_errors}")

        # One shared window sized for the largest LSTM lookback across variants
        # (and the volatility ensemble, which shares the same convention).
        max_window = max(
            [v['lstm_time_steps'] for v in self.variants.values() if v['lstm_time_steps']]
            + ([self.vol_time_steps] if self.vol_time_steps else [])
            or [1]
        )
        feature_window, data_source, bar_used, as_of_date, forecasting_date = \
            self._resolve_latest_window(max_window)

        response = {
            "as_of_date": as_of_date,
            "forecasting_date": forecasting_date,
            "data_source": data_source,
            "bar_used": bar_used,
        }

        consensus_directions = {}
        for name in self.variant_names:
            block = self._predict_variant(name, feature_window)
            response[name] = block
            if 'consensus' in block:
                consensus_directions[name] = block['consensus']['direction']

        # True/False only when BOTH variants produced a consensus; None (unknown)
        # when either side is degraded -- an honest "can't compare" rather than a
        # fake agreement.
        if len(consensus_directions) == len(self.variant_names) and len(self.variant_names) > 1:
            response['variant_agreement'] = len(set(consensus_directions.values())) == 1
        else:
            response['variant_agreement'] = None

        # Next-day realized-volatility forecast (validated against GARCH(1,1)
        # on the validation arbiter — the only NN family in this project with a
        # CI-confirmed edge over its honest baseline). Failure degrades to a
        # volatility_error note, never breaking the daily committees.
        if self.vol_ready:
            try:
                response['volatility_forecast'] = self._predict_volatility(feature_window)
            except Exception as e:
                response['volatility_error'] = str(e)

        # Supplementary H1->Daily ensemble (separate intraday predictor). It has
        # its own data feed and its own agreement-based consensus, surfaced as a
        # self-contained block so it never perturbs the daily committees above.
        # Any failure here (no H1 feed, etc.) degrades to an h1_error note.
        if self.h1_ready:
            try:
                h1_per_model, h1_as_of, h1_source = self._predict_h1()
                response['h1'] = {
                    "as_of_date": h1_as_of,
                    "data_source": h1_source,
                    "predictions": h1_per_model,
                    "consensus": self.compute_h1_consensus(h1_per_model),
                }
            except Exception as e:
                response['h1_error'] = str(e)

        # Observational H1 TI-LSTM — kept a SEPARATE block from the validated
        # H1 ensemble above (different validation history, separately
        # attributable). Hard-labeled `validated: false` (owner-override ship,
        # DROP verdict); failure degrades to ti_h1_error, never breaking the
        # validated panels.
        if self.ti_h1_ready:
            try:
                response['ti_h1_forecast'] = self._predict_ti_h1()
            except Exception as e:
                response['ti_h1_error'] = str(e)

        # Best-effort: log this forecast for the /history prediction-vs-actual
        # table and the dual paper-trading ledgers. Logging must never break a
        # prediction, so swallow any error.
        try:
            from .tracking import log_prediction
            log_path = self.config.get('tracking', {}).get('log_path', 'results/prediction_log.csv')
            log_prediction(response, os.path.join(self.base_dir, log_path))
        except Exception:
            pass

        return response
