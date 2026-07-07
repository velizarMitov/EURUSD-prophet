import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Canonical 24-column feature order consumed by the trained GBM/LSTM artifacts.
# Shared by the notebook (training) and api.py (inference) so the two
# can never silently drift apart.
#
# tick_volume is deliberately excluded: MT5's "tick_volume" is a broker-specific
# tick count, not genuine traded volume, and the bundled 1971-2026 training
# history carries decades of degenerate placeholder values (e.g. 1) from its
# earliest synthetic era. That contaminated the fitted StandardScaler's
# mean/std for the column, which then caused the LSTM to extrapolate wildly
# on genuine live volumes (8+ standard deviations out). It is still loaded
# and surfaced to the UI for display purposes (see load_history/build_live_features
# callers' bar_used dicts), it just never reaches the models.
#
# yield_differential (US 10Y - DE 10Y bond yield, via src/macro_data.py) is a
# passthrough column like tick_volume: it must already be present on the
# DataFrame passed into compute_features()/add_advanced_features() -- see
# merge_macro_features() below -- since it is exogenous macro data, not
# something derivable from OHLCV alone. It is NOT fed to the model directly --
# see yield_differential_delta below.
#
# yield_differential_delta = yield_differential.diff(1), computed in
# compute_features() exactly like log_return is derived from close. The raw
# LEVEL is highly persistent/non-stationary (bond yields trend for months) and
# results/2C_fred_ablation.csv showed it net-HURTS the GBM direction classifier
# (Accuracy -0.0039, ROC-AUC -0.0021 vs without). A quick re-ablation
# (day-over-day change, mirroring why the model uses log_return instead of raw
# close) flips the sign: Accuracy +0.0029, ROC-AUC +0.0032 vs without. The raw
# level is still merged and kept (unchanged) for the dashboard's human-readable
# "US10Y-DE10Y Yield Differential" display -- only the MODEL feature changed.
# Three more exogenous macro features (via src/macro_data.py::fetch_macro_features),
# all merged as levels by merge_macro_features and turned into the model features
# below by compute_features -- same passthrough discipline as yield_differential:
#   * usd_index_return       -- log-return of the trade-weighted USD index
#                               (DTWEXBGS). RETURN, not level: the raw level is
#                               ~57% EUR-weighted and would be near-collinear with
#                               EUR/USD itself. DTWEXBGS only exists from 2006, so
#                               earlier euro-era rows get a flat 0 return rather
#                               than truncating history (see compute_features).
#   * policy_rate_differential -- DFF (effective fed funds) - ECBDFR (ECB deposit
#                               facility). DFF/ECBDFR chosen over DFEDTARU so the
#                               series reach back to the euro's 1999 start.
#   * inflation_differential -- US CPI YoY% (CPIAUCSL) - DE HICP YoY%
#                               (CP0000DEM086NEST; the older DEUCPIALLMINMEI is
#                               stale since 2025-03). YoY computed in macro_data.
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'log_return',
    'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200', 'volatility_20',
    'bar_dynamics', 'return_lag_1', 'dynamics_lag_1', 'return_lag_2',
    'dynamics_lag_2', 'return_lag_3', 'dynamics_lag_3', 'day_sin', 'day_cos',
    'month_sin', 'month_cos', 'ATR_14', 'BB_width', 'yield_differential_delta',
    'usd_index_return', 'policy_rate_differential', 'inflation_differential',
]

# The level/derived macro columns merge_macro_features left-joins onto the OHLCV
# index (ffilled). yield_differential and usd_index are intermediate levels that
# compute_features turns into their stationary model features
# (yield_differential_delta, usd_index_return); policy_rate_differential and
# inflation_differential are already differentials and pass straight through.
MACRO_MERGE_COLUMNS = ['yield_differential', 'usd_index', 'policy_rate_differential', 'inflation_differential']

# ---------------------------------------------------------------------------
# Dual model variants (see ARCHITECTURE_DOCS.md "Dual-Variant Architecture").
# The four FRED-derived model features are statistically UNPROVEN (every one is
# KEEP-provisional under the Bonferroni-corrected validation bar — Production
# Methodology), so the system trains and serves TWO complete model families
# side by side and lets the forward paper-trading ledgers arbitrate:
#   * 'baseline'   — price-only: FEATURE_COLUMNS minus every macro-derived
#                    feature (23 columns). No FRED dependency at inference.
#   * 'with_macro' — the full 27-column FEATURE_COLUMNS set.
# Both variants train on the IDENTICAL euro-era row set (the macro merge's
# 1999+ truncation), so their comparison isolates the feature set itself, not
# a training-span difference.
MACRO_FEATURE_COLUMNS = [
    'yield_differential_delta', 'usd_index_return',
    'policy_rate_differential', 'inflation_differential',
]
PRICE_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in MACRO_FEATURE_COLUMNS]
VARIANT_FEATURE_COLUMNS = {
    'baseline': PRICE_FEATURE_COLUMNS,
    'with_macro': FEATURE_COLUMNS,
}


def variant_feature_columns(variant: str) -> list:
    """The exact model feature set for a named variant ('baseline' price-only /
    'with_macro' full). Raises KeyError on an unknown variant name rather than
    silently serving the wrong column set."""
    return list(VARIANT_FEATURE_COLUMNS[variant])

# The 6 autoregressive lag columns are the most mutually correlated block in
# FEATURE_COLUMNS (each is a shifted copy of log_return/bar_dynamics) and are
# the dimensionality-reduction target described in the project's PCA section.
LAG_COLUMNS = [
    'return_lag_1', 'dynamics_lag_1',
    'return_lag_2', 'dynamics_lag_2',
    'return_lag_3', 'dynamics_lag_3',
]

TARGET_RETURN_COLUMN = 'target_return'
TARGET_DIRECTION_COLUMN = 'target_direction'
# Next-day realized volatility proxy: |next-day log return| in PERCENT.
# A genuinely different task from direction/return — volatility clustering is a
# well-established FX stylized fact, unlike next-day direction. Same shift(-1)
# convention and percent unit as target_return; evaluated against a GARCH(1,1)
# baseline in src/volatility.py (its own hypothesis family, see
# results/volatility_hypothesis_log.csv).
TARGET_VOLATILITY_COLUMN = 'target_volatility_pct'

DEFAULT_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'eurusd_features.csv'
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 24 FEATURE_COLUMNS from raw OHLCV (+ yield_differential, if
    already merged in via merge_macro_features -- the model consumes its
    diff(1), yield_differential_delta, derived here). Unlike add_advanced_features,
    this does NOT compute targets or drop rows, so the most recent bar (which
    has no future bar to derive a target from) survives — required for live
    inference on the latest available row.
    """
    data = df.copy()

    # 1. Log return (Stationarity)
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # 1b. Yield differential DELTA (stationarity, same rationale as log_return
    # above): the raw level is a slow-moving trend, not something a next-day
    # model should read directly. diff(1) uses only the current and immediately
    # preceding (already-ffilled) rows, so this introduces no look-ahead beyond
    # what merge_macro_features's own ffill already guarantees.
    # An ENTIRELY missing/all-NaN macro column means that feed was unreachable
    # (no FRED, no cache) -- default it to a neutral value so the pipeline
    # degrades instead of dropna-ing the whole window. A column that is only
    # PARTIALLY NaN (real leading gap, e.g. no ECB rate before 1999) is left
    # intact so dropna truncates training to the genuine euro era.
    if 'yield_differential' in data and not data['yield_differential'].isna().all():
        data['yield_differential_delta'] = data['yield_differential'].diff(1)
    else:
        data['yield_differential_delta'] = 0.0

    # 1c. USD trade-weighted index -> log-return (same stationarity move as
    # log_return on close). DTWEXBGS only exists from 2006; pre-2006 euro-era
    # rows (and any leading/gap NaN) get a flat 0 return via fillna, so the
    # feature never truncates the 1999+ history the other macro features cover.
    # fillna(0.0) injects no future information -- 0 == "no move / unknown".
    if 'usd_index' in data and not data['usd_index'].isna().all():
        _usd = data['usd_index'].replace(0, np.nan)
        data['usd_index_return'] = np.log(_usd / _usd.shift(1)).fillna(0.0)
    else:
        data['usd_index_return'] = 0.0

    # 1d. Policy-rate and inflation differentials pass through as levels (each is
    # already a differential of two rates / two YoY inflations, reasonably
    # stationary). Genuine leading NaN is kept (truncates to the euro era);
    # only an entirely-absent feed is neutralized to 0.
    for _col in ('policy_rate_differential', 'inflation_differential'):
        if _col not in data or data[_col].isna().all():
            data[_col] = 0.0

    # 2. Simple Moving Averages
    for period in [21, 50, 100, 200]:
        data[f'SMA_{period}'] = data['close'].rolling(period).mean()

    # 3. Rolling Volatility
    data['volatility_20'] = data['log_return'].rolling(20).std()

    # 4. Bar Dynamics
    # To prevent DivisionByZero logically inside pandas, replace 0 open with NaN temporarily
    safe_open = data['open'].replace(0, np.nan)
    data['bar_dynamics'] = (data['high'] - data['low']) / safe_open

    # 5. Autoregressive Lags
    for lag in range(1, 4):
        data[f'return_lag_{lag}'] = data['log_return'].shift(lag)
        data[f'dynamics_lag_{lag}'] = data['bar_dynamics'].shift(lag)

    # 6. Cyclical Encoding (day of week + month, both periodic -- sin/cos
    # preserves the wrap-around geometry, e.g. Friday->Monday, Dec->Jan)
    day_of_week = data.index.dayofweek
    data['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    data['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    month = data.index.month
    data['month_sin'] = np.sin(2 * np.pi * month / 12)
    data['month_cos'] = np.cos(2 * np.pi * month / 12)

    # 7. Average True Range (vectorized, 14-period EWM smoothing)
    prev_close = data['close'].shift(1)
    true_range = pd.concat([
        data['high'] - data['low'],
        (data['high'] - prev_close).abs(),
        (data['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    data['ATR_14'] = true_range.ewm(com=13, adjust=False).mean()

    # 8. Normalized Bollinger Band Width: (upper - lower) / mid = 4*std/mid
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['BB_width'] = (4 * bb_std) / bb_mid

    return data


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracted purely functional processing logic for production API and Unit Testing.
    """
    data = compute_features(df)

    # Multi-Task Targets: continuous next-period return (in PERCENT units) + its
    # derived sign. Percent — not the raw fraction — is the single canonical unit
    # for BOTH heads. Fractional returns (std ~0.006) give an MSE roughly five
    # orders of magnitude below the direction head's binary-crossentropy, which
    # starved the shared LSTM trunk of return-head gradient. Training natively in
    # percent keeps the two losses on the same scale AND removes the old
    # asymmetry where the GBM regressor was trained on fractions and rescaled by
    # *100 only at inference time. The positive scaling does not change the sign,
    # so target_direction is identical to the fractional formulation.
    data[TARGET_RETURN_COLUMN] = ((data['close'].shift(-1) - data['close']) / data['close']) * 100
    data[TARGET_DIRECTION_COLUMN] = (data[TARGET_RETURN_COLUMN] > 0).astype(int)

    # Realized-volatility target: |next-day LOG return| * 100. Log (not simple)
    # return matches the GARCH literature convention; the shift(-1) is the same
    # next-bar geometry as target_return, so it is NaN on the exact same final
    # row and introduces no additional look-ahead surface.
    data[TARGET_VOLATILITY_COLUMN] = np.abs(np.log(data['close'].shift(-1) / data['close'])) * 100

    # Drop warm-up / undefined-target rows, but ONLY on the columns the model
    # actually consumes (the 27 FEATURE_COLUMNS + the two targets). The merged
    # intermediate LEVELS (usd_index, yield_differential) are deliberately NOT in
    # the subset: usd_index (DTWEXBGS) is NaN before 2006, and dropping on it
    # would truncate the whole euro-era history even though usd_index_return is a
    # flat 0 there. policy_rate_differential / inflation_differential ARE features,
    # so their genuine leading NaN still truncates training to the real euro era.
    data.dropna(subset=FEATURE_COLUMNS + [TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN,
                                          TARGET_VOLATILITY_COLUMN], inplace=True)
    return data


def merge_macro_features(ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the (already UTC-indexed) macro columns from src/macro_data.py
    (MACRO_MERGE_COLUMNS: yield_differential, usd_index, policy_rate_differential,
    inflation_differential) onto the OHLCV index, then forward-fill each -- weekend
    FX bars, bond/rate holidays, and the monthly-CPI/daily-price frequency mismatch
    all inherit the last known value rather than NaN. ffill only ever carries a
    *past* value forward (never a future one backward), so this introduces zero
    look-ahead regardless of how the calendars are offset. Any macro column absent
    from `macro_df` (a feature entirely unreachable this run) is created as all-NaN
    here; compute_features then applies each feature's own neutral default.
    """
    original_index = ohlcv_df.index
    utc_index = (
        original_index.tz_localize('UTC') if original_index.tz is None
        else original_index.tz_convert('UTC')
    )
    merged = ohlcv_df.copy()
    merged.index = utc_index
    for col in MACRO_MERGE_COLUMNS:
        s = macro_df[col].dropna() if col in macro_df.columns else pd.Series(dtype=float)
        if len(s):
            # As-of alignment: reindex each macro series onto the union of its own
            # dates and the OHLCV dates, ffill, then select the OHLCV dates. Unlike
            # a plain left-join, this never loses a macro observation that lands on
            # a non-trading day (e.g. a month-start CPI print on a weekend) -- its
            # value still propagates onto every LATER daily bar. ffill only carries
            # a PAST value forward, never a future one backward, so no look-ahead.
            s = s.sort_index()
            merged[col] = s.reindex(s.index.union(utc_index)).ffill().reindex(utc_index)
        else:
            merged[col] = np.nan  # feature unavailable -> compute_features neutralizes
    merged.index = original_index
    return merged


def load_history(path: str = DEFAULT_HISTORY_PATH) -> pd.DataFrame:
    """Load the real historical OHLCV bars used as live feature context."""
    history = pd.read_csv(path, index_col='time', parse_dates=True)
    return history[['open', 'high', 'low', 'close', 'tick_volume']]


def build_live_features(history: pd.DataFrame, new_bar: dict, time_steps: int) -> pd.DataFrame:
    """
    Append `new_bar` (a dict with open/high/low/close/tick_volume/
    yield_differential) as the next chronological bar after `history` (which
    must carry the same columns, including a pre-merged yield_differential --
    see merge_macro_features), recompute the real rolling indicators over
    genuine price history (no hardcoded/mocked SMA or lag values), and return
    the last `time_steps` engineered rows ending at the new bar.
    """
    next_date = history.index[-1] + pd.Timedelta(days=1)
    new_row = pd.DataFrame([new_bar], index=pd.DatetimeIndex([next_date], name=history.index.name))
    combined = pd.concat([history, new_row])
    engineered = compute_features(combined)
    return engineered[FEATURE_COLUMNS].tail(time_steps)


def fit_lag_pca(df: pd.DataFrame, lag_columns=LAG_COLUMNS, variance_threshold: float = 0.95):
    """
    Fit a StandardScaler + PCA on the autoregressive lag columns only. Must be
    fit on a chronological TRAIN slice (never the full dataset) to avoid leakage.
    Returns (lag_scaler, lag_pca); lag_pca.n_components_ is chosen automatically
    to explain >= variance_threshold of the lag block's variance.
    """
    lag_scaler = StandardScaler()
    scaled_lags = lag_scaler.fit_transform(df[lag_columns])
    lag_pca = PCA(n_components=variance_threshold)
    lag_pca.fit(scaled_lags)
    return lag_scaler, lag_pca


def apply_lag_pca(df: pd.DataFrame, lag_scaler: StandardScaler, lag_pca: PCA, lag_columns=LAG_COLUMNS) -> pd.DataFrame:
    """
    Replace the raw lag columns with their principal components. All other
    columns are preserved in their original order; PCA components are
    appended as lag_pca_1..lag_pca_k. Used identically at training time and
    at live inference time so the model always sees the same input shape.
    """
    scaled_lags = lag_scaler.transform(df[lag_columns])
    components = lag_pca.transform(scaled_lags)

    reduced = df.drop(columns=lag_columns).copy()
    for i in range(components.shape[1]):
        reduced[f'lag_pca_{i + 1}'] = components[:, i]
    return reduced


def model_input_columns(lag_pca: PCA, base_columns=FEATURE_COLUMNS, lag_columns=LAG_COLUMNS) -> list:
    """The exact column order produced by apply_lag_pca, for a given fitted PCA."""
    remaining = [c for c in base_columns if c not in lag_columns]
    return remaining + [f'lag_pca_{i + 1}' for i in range(lag_pca.n_components_)]