"""
Next-day realized-volatility experiment — VALIDATION-ONLY arbiter.

A genuinely different prediction task from direction/return: the target is
`target_volatility_pct = |next-day log return| * 100` (src/features.py), where
real signal is far more plausible (volatility clustering is a well-established
FX stylized fact, unlike next-day direction).

Methodology (mirrors src/ablation.py — post-defense Production Methodology):
  * ONE price-only volatility model, NOT duplicated per baseline/with_macro
    variant. GARCH and the persistence baseline consume only past returns, so
    the dual-variant treatment the direction/return models need does not apply.
  * Mandatory baselines BEFORE any neural network:
      - GARCH(1,1) (`arch` package): parameters fit on the TRAIN block
        [0:70%] ONLY, then FIXED-parameter rolling one-step-ahead forecasts on
        the later blocks (each forecast uses only data available up to that
        point — no refit, no peeking; `ARCHModel.fix()` guarantees this).
      - Naive persistence: today's realized |log return| = tomorrow's forecast.
  * The dedicated LSTM fits on [0:70%] with an INNER early-stopping tail
    ([63%:70%]) so the validation arbiter [70%:80%] stays genuinely held out
    from EVERYTHING the experiment fits (unlike the production multi-task LSTM,
    which may early-stop on [70:80] because its arbiter is the test block).
  * All KEEP/SHIP decisions on the VALIDATION slice [70%:80%]; the TEST block
    [80%:100%] is NEVER indexed here.
  * This is its OWN hypothesis family (continuous-magnitude R2/MAE metrics,
    not direction accuracy/ROC-AUC): tracked in
    results/volatility_hypothesis_log.csv with its own Bonferroni bar —
    it does not dilute the 4-feature direction/return family count.

GARCH -> |return| point forecast: for r ~ N(0, sigma^2), E|r| = sigma *
sqrt(2/pi) (folded-normal mean, mu ~= 0). Both the GARCH sigma and the
adjusted E|r| forecast are reported; metrics use the adjusted forecast so the
comparison against the LSTM (which regresses |r| directly) is unit-honest.

Run standalone (offline-safe; macro merge only sets the euro-era row span so
the volatility rows are IDENTICAL to the rows every other model trains on):

    python -m src.volatility

Writes results/volatility_validation.csv and appends the hypothesis to
results/volatility_hypothesis_log.csv (idempotent by hypothesis name).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from src.macro_data import fetch_macro_features
from src.features import (
    load_history, merge_macro_features, add_advanced_features,
    PRICE_FEATURE_COLUMNS, LAG_COLUMNS, TARGET_VOLATILITY_COLUMN,
    TARGET_RETURN_COLUMN, TARGET_DIRECTION_COLUMN,
    fit_lag_pca, apply_lag_pca, model_input_columns,
)
from src.h1_features import _rsi
from src.fomc_calendar import add_fomc_features, FOMC_FEATURE_COLUMNS

# Candidate INPUT features for the volatility model ONLY (hypotheses 4+ of this
# family). Deliberately NOT added to src/features.py::FEATURE_COLUMNS — the
# direction/return capacity question is closed (Ch.11 train-vs-test diagnostic,
# ARCHITECTURE_DOCS.md §4.2.1); these are volatility-regime oscillators tested
# strictly against the validated 5-seed ensemble.
CANDIDATE_VOL_FEATURES = ['RSI_14', 'BB_percent_b']


def add_volatility_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the candidate volatility input features on the FULL history frame
    (call BEFORE add_advanced_features, like the macro merge) so the rolling
    warm-ups fall in the pre-euro era that dropna truncates anyway — the modeled
    euro-era rows are then fully defined, exactly like SMA_200/BB_width.

    * RSI_14 — the H1 module's exact `_rsi` formula (src/h1_features.py),
      applied to the DAILY close with the conventional 14-day window. Trailing
      window only -> the same no-look-ahead guarantee already proven for rsi_24.
    * BB_percent_b — %B = (close - lower) / (upper - lower) over the SAME
      20-day rolling mean/std that BB_width already uses (src/features.py:
      upper/lower = mid ± 2σ, so the denominator 4σ·mid⁻¹ IS BB_width — %B adds
      WHERE price sits inside the bands, which BB_width does not carry). A
      degenerate flat window (σ=0) maps to the neutral mid-band 0.5, mirroring
      _rsi's neutral-50 convention.
    """
    data = df.copy()
    data['RSI_14'] = _rsi(data['close'], 14)
    bb_mid = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    lower = bb_mid - 2 * bb_std
    pct_b = (data['close'] - lower) / (4 * bb_std)
    data['BB_percent_b'] = pct_b.mask(bb_std == 0, 0.5)
    return data

VALIDATION_CSV = 'results/volatility_validation.csv'

# Separate hypothesis family from results/feature_hypothesis_log.csv: different
# target (volatility, not direction/return), different metrics (R2/MAE on a
# continuous magnitude), different bar. Same Bonferroni discipline.
HYPOTHESIS_LOG = 'results/volatility_hypothesis_log.csv'
HYPOTHESIS_LOG_COLUMNS = [
    'n', 'date', 'hypothesis', 'arbiter',
    'point_delta_mae', 'ci_dmae_low', 'ci_dmae_high',
    'point_delta_r2', 'ci_dr2_low', 'ci_dr2_high',
    'alpha_bonferroni', 'cleared_bar', 'verdict', 'notes',
]
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 2000
ABS_MOMENT = float(np.sqrt(2.0 / np.pi))   # E|N(0,1)|


def _p(base_dir, rel):
    return os.path.join(base_dir, rel) if base_dir else rel


def load_volatility_hypothesis_log(base_dir=''):
    p = _p(base_dir, HYPOTHESIS_LOG)
    if not os.path.exists(p):
        return pd.DataFrame(columns=HYPOTHESIS_LOG_COLUMNS)
    return pd.read_csv(p)


def bonferroni_alpha(family_size, family_alpha=FAMILY_ALPHA):
    return family_alpha / max(1, family_size)


def register_volatility_hypothesis(result, base_dir='', notes='', alpha_override=None):
    """Append a newly tested volatility hypothesis (idempotent by name). The
    recorded bar reflects the family size AFTER the addition, exactly like
    src/ablation.py::register_hypothesis. `alpha_override` records a stricter
    pre-declared bar instead (e.g. when several hypotheses are declared up
    front in one pass, every one is judged at the FINAL family size's bar)."""
    log = load_volatility_hypothesis_log(base_dir)
    if result['hypothesis'] in set(log['hypothesis']):
        return log
    n = len(log) + 1
    row = {
        'n': n, 'date': pd.Timestamp.utcnow().date().isoformat(),
        'hypothesis': result['hypothesis'], 'arbiter': result['arbiter'],
        'point_delta_mae': result['point_delta_mae'],
        'ci_dmae_low': result['ci_dmae_low'], 'ci_dmae_high': result['ci_dmae_high'],
        'point_delta_r2': result['point_delta_r2'],
        'ci_dr2_low': result['ci_dr2_low'], 'ci_dr2_high': result['ci_dr2_high'],
        'alpha_bonferroni': round(bonferroni_alpha(n) if alpha_override is None
                                  else alpha_override, 4),
        'cleared_bar': result['cleared_bar'],
        'verdict': result['verdict'], 'notes': notes,
    }
    new = pd.DataFrame([row], columns=HYPOTHESIS_LOG_COLUMNS)
    out = new if log.empty else pd.concat([log, new], ignore_index=True)
    out.to_csv(_p(base_dir, HYPOTHESIS_LOG), index=False)
    return out


def build_volatility_matrix(config, base_dir='', extra_feature_columns=None):
    """Engineered frame + price-only post-PCA model matrix + split boundaries.

    The macro merge is performed ONLY to reproduce the euro-era row truncation
    every other model trains on (identical row set -> comparable metrics); the
    model consumes PRICE_FEATURE_COLUMNS, so no macro column ever reaches it.
    PCA and the scaler are fit on [0:70%] only — the validation arbiter block
    stays held out of every fit, as in src/ablation.py.

    `extra_feature_columns` (subset of CANDIDATE_VOL_FEATURES) appends candidate
    input columns to the model matrix for hypothesis testing — the row set is
    identical with or without them (they are never in the dropna subset), so a
    base-vs-challenger comparison isolates the feature itself.
    """
    ohlcv = load_history(_p(base_dir, config['data']['history_csv_path']))
    macro, _ = fetch_macro_features(
        ohlcv.index.min(), ohlcv.index.max(), config['macro'], base_dir=base_dir
    )
    merged = add_volatility_candidate_features(merge_macro_features(ohlcv.copy(), macro))
    feat = add_advanced_features(merged)

    extra = list(extra_feature_columns or [])
    if any(c in FOMC_FEATURE_COLUMNS for c in extra):
        # Pure calendar-geometry join (no windows) — safe post-dropna.
        feat = add_fomc_features(feat, base_dir=base_dir)
    assert not feat[CANDIDATE_VOL_FEATURES + extra].isna().any().any(), \
        "candidate volatility features must be fully defined on the modeled euro-era rows"

    n = len(feat)
    train_fraction = config['split']['train_fraction']   # 0.80
    val_fraction = config['split']['val_fraction']       # 0.10
    train_end = int(n * (train_fraction - val_fraction))  # 70% — fit boundary
    val_end = int(n * train_fraction)                     # 80% — arbiter end
    split = {'train_end': train_end, 'val_end': val_end, 'n': n}

    lag_scaler, lag_pca = fit_lag_pca(
        feat.iloc[:train_end], lag_columns=LAG_COLUMNS,
        variance_threshold=config['pca']['variance_threshold'],
    )
    red = apply_lag_pca(feat, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    cols = model_input_columns(lag_pca, base_columns=PRICE_FEATURE_COLUMNS + extra,
                               lag_columns=LAG_COLUMNS)

    scaler = StandardScaler().fit(red[cols].iloc[:train_end])
    X_scaled = scaler.transform(red[cols])
    return feat, X_scaled, split


def garch_forecasts(feat, split, config):
    """GARCH(1,1) one-step-ahead |return| forecasts, honestly out-of-sample.

    Parameters are estimated on the TRAIN block [0:70%] ONLY; the fitted
    parameter vector is then FIXED (`ARCHModel.fix`) and rolled forward over
    the full series, so the forecast aligned at row t uses only returns up to
    and including t — validation/test observations never touch the estimation.

    Returns a full-length array where entry t = E|r_{t+1}| forecast in percent
    (sigma_{t+1|t} * sqrt(2/pi)); NaN before the first forecastable row.
    """
    from arch import arch_model

    returns_pct = feat['log_return'] * 100.0
    train_end = split['train_end']

    am_train = arch_model(returns_pct.iloc[:train_end], mean='Constant',
                          vol='GARCH', p=1, q=1, dist='normal')
    res_train = am_train.fit(disp='off')

    am_full = arch_model(returns_pct, mean='Constant', vol='GARCH', p=1, q=1,
                         dist='normal')
    fixed = am_full.fix(res_train.params)
    # Row label t carries the h.1 variance forecast for t+1, i.e. exactly the
    # prediction for target_volatility_pct at row t. Start a little before the
    # validation block purely to keep the output frame small.
    fc = fixed.forecast(horizon=1, start=feat.index[0], reindex=True)
    sigma = np.sqrt(fc.variance['h.1'].to_numpy(dtype=float))

    pred = np.full(split['n'], np.nan)
    valid = ~np.isnan(sigma)
    pred[valid] = sigma[valid] * ABS_MOMENT
    return pred, res_train


def persistence_forecasts(feat):
    """Naive persistence: today's realized |log return| (percent) forecasts
    tomorrow's. Entry t predicts target_volatility_pct at row t."""
    return np.abs(feat['log_return'].to_numpy()) * 100.0


def make_sequences(X_scaled, y, time_steps):
    """Sliding windows over the FULL scaled matrix: the sequence whose target
    is row t covers rows [t-time_steps+1 : t] — identical end-at-target-row
    geometry as _train_pipeline.py's create_mt_sequences, but built globally so
    validation rows are never sacrificed as per-block warm-up and every model
    (GARCH / persistence / LSTM) can be compared on IDENTICAL target rows."""
    n = len(X_scaled)
    windows = np.stack([X_scaled[t - time_steps + 1: t + 1]
                        for t in range(time_steps - 1, n)])
    targets = y[time_steps - 1:]
    target_rows = np.arange(time_steps - 1, n)
    return windows.astype('float32'), targets, target_rows


def train_volatility_lstm(X_scaled, y, split, config, random_state):
    """Dedicated single-head volatility LSTM (price-only). Fit on [0:63%] with
    [63%:70%] as the inner early-stopping tail; the validation arbiter
    [70%:80%] is NOT used for early stopping (it is the decision block here,
    unlike in the production multi-task pipeline where the arbiter is the
    test block). Returns full-length predictions aligned to target rows."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    lstm_cfg = config['lstm']
    time_steps = lstm_cfg['time_steps']
    train_end = split['train_end']
    inner_end = int(train_end * 0.9)          # ~63% of the full set

    windows, targets, target_rows = make_sequences(X_scaled, y, time_steps)
    is_tr = target_rows < inner_end
    is_es = (target_rows >= inner_end) & (target_rows < train_end)

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    inputs = Input(shape=(time_steps, X_scaled.shape[1]), name='vol_window')
    x = LSTM(lstm_cfg['units'], name='vol_lstm_trunk')(inputs)
    x = Dropout(lstm_cfg['dropout'], name='vol_dropout')(x)
    # softplus: realized volatility is nonnegative by construction.
    out = Dense(1, activation='softplus', name='volatility_output', dtype='float32')(x)
    model = Model(inputs, out, name='volatility_lstm_eurusd')
    model.compile(optimizer=Adam(learning_rate=lstm_cfg['learning_rate']),
                  loss='mse', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=lstm_cfg['patience'],
                       restore_best_weights=True, verbose=0)
    model.fit(windows[is_tr], targets[is_tr],
              validation_data=(windows[is_es], targets[is_es]),
              epochs=lstm_cfg['epochs'], batch_size=lstm_cfg['batch_size'],
              callbacks=[es], verbose=0)

    pred_seq = model.predict(windows, verbose=0).ravel()
    pred = np.full(split['n'], np.nan)
    pred[target_rows] = pred_seq
    return pred, model


def train_multitask_volatility_lstm(X_scaled, y_vol, y_ret, y_dir, split, config,
                                    random_state):
    """Follow-up experiment (only justified once the dedicated model cleared its
    bar): the EXISTING production architecture's shared LSTM trunk with a THIRD
    head (volatility_output) alongside return/direction. Same fit [0:63%] /
    early-stop [63%:70%] blocks as the dedicated model so the two are directly
    comparable on the untouched validation arbiter. Don't assume sharing helps
    — this trains it and lets the val slice say."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    lstm_cfg = config['lstm']
    time_steps = lstm_cfg['time_steps']
    train_end = split['train_end']
    inner_end = int(train_end * 0.9)

    windows, targets_vol, target_rows = make_sequences(X_scaled, y_vol, time_steps)
    targets_ret = y_ret[time_steps - 1:]
    targets_dir = y_dir[time_steps - 1:]
    is_tr = target_rows < inner_end
    is_es = (target_rows >= inner_end) & (target_rows < train_end)

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    inputs = Input(shape=(time_steps, X_scaled.shape[1]), name='ohlcv_window')
    shared = LSTM(lstm_cfg['units'], name='shared_lstm_trunk')(inputs)
    shared = Dropout(lstm_cfg['dropout'], name='shared_dropout')(shared)
    return_output = Dense(1, activation='linear', name='return_output', dtype='float32')(shared)
    direction_output = Dense(1, activation='sigmoid', name='direction_output', dtype='float32')(shared)
    volatility_output = Dense(1, activation='softplus', name='volatility_output', dtype='float32')(shared)

    model = Model(inputs, [return_output, direction_output, volatility_output],
                  name='multitask3_lstm_eurusd_experiment')
    model.compile(
        optimizer=Adam(learning_rate=lstm_cfg['learning_rate']),
        loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy',
              'volatility_output': 'mse'},
        loss_weights={'return_output': 1.0, 'direction_output': 1.0, 'volatility_output': 1.0},
    )
    es = EarlyStopping(monitor='val_loss', patience=lstm_cfg['patience'],
                       restore_best_weights=True, verbose=0)
    model.fit(
        windows[is_tr],
        {'return_output': targets_ret[is_tr], 'direction_output': targets_dir[is_tr],
         'volatility_output': targets_vol[is_tr]},
        validation_data=(windows[is_es],
                         {'return_output': targets_ret[is_es], 'direction_output': targets_dir[is_es],
                          'volatility_output': targets_vol[is_es]}),
        epochs=lstm_cfg['epochs'], batch_size=lstm_cfg['batch_size'],
        callbacks=[es], verbose=0)

    pred_ret_seq, pred_dir_seq, pred_vol_seq = (p.ravel() for p in model.predict(windows, verbose=0))
    n = split['n']
    pred_vol = np.full(n, np.nan)
    pred_ret = np.full(n, np.nan)
    pred_dir = np.full(n, np.nan)
    pred_vol[target_rows] = pred_vol_seq
    pred_ret[target_rows] = pred_ret_seq
    pred_dir[target_rows] = pred_dir_seq
    return pred_vol, pred_ret, pred_dir, model


def _metrics(y, pred):
    return {'mae': mean_absolute_error(y, pred), 'r2': r2_score(y, pred)}


def bootstrap_delta(y, pred_challenger, pred_baseline, alpha,
                    n_boot=BOOTSTRAP_RESAMPLES, random_state=42):
    """Paired bootstrap of (MAE improvement, R2 improvement) of the challenger
    over the baseline on identical rows. Positive delta = challenger better
    (delta_mae = mae_baseline - mae_challenger; delta_r2 = r2_challenger -
    r2_baseline). CI percentiles follow the Bonferroni-corrected alpha so a
    tighter family bar automatically demands a wider interval."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    d_mae = np.empty(n_boot)
    d_r2 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        d_mae[i] = (mean_absolute_error(yt, pred_baseline[idx])
                    - mean_absolute_error(yt, pred_challenger[idx]))
        d_r2[i] = r2_score(yt, pred_challenger[idx]) - r2_score(yt, pred_baseline[idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        'ci_dmae': np.percentile(d_mae, [lo, hi]),
        'ci_dr2': np.percentile(d_r2, [lo, hi]),
        'frac_dmae_positive': float(np.mean(d_mae > 0)),
    }


def run(config_path='config.json', base_dir='', out_csv=VALIDATION_CSV,
        register=True, random_state=None):
    with open(_p(base_dir, config_path)) as f:
        config = json.load(f)
    random_state = config['random_state'] if random_state is None else random_state

    feat, X_scaled, split = build_volatility_matrix(config, base_dir=base_dir)
    y = feat[TARGET_VOLATILITY_COLUMN].to_numpy()
    train_end, val_end = split['train_end'], split['val_end']
    time_steps = config['lstm']['time_steps']

    # Identical evaluation rows for every model: the validation slice, minus
    # nothing (the global sliding windows reach back into train for context,
    # so no val row is lost to warm-up).
    val_rows = np.arange(train_end, val_end)

    log = load_volatility_hypothesis_log(base_dir)
    family_size = max(1, len(set(log['hypothesis']) | {'volatility_lstm_vs_garch11'}))
    alpha = bonferroni_alpha(family_size)

    print('=' * 78)
    print('NEXT-DAY REALIZED VOLATILITY — arbiter = VALIDATION slice [70%:80%]')
    print(f'  target: {TARGET_VOLATILITY_COLUMN} = |next-day log return| * 100')
    print(f'  rows: n={split["n"]:,}  fit [0:{train_end}]  val [{train_end}:{val_end}]  '
          f'(test block [{val_end}:{split["n"]}] NEVER indexed here)')
    print(f'  hypothesis family size: {family_size}  ->  Bonferroni alpha = {alpha:.4g}  '
          f'({100 * (1 - alpha):.1f}% CIs)')
    print('=' * 78)

    # ---- Step 1: baselines FIRST (the bar the NN has to clear) --------------
    pred_garch, garch_res = garch_forecasts(feat, split, config)
    pred_pers = persistence_forecasts(feat)

    y_val = y[val_rows]
    m_garch = _metrics(y_val, pred_garch[val_rows])
    m_pers = _metrics(y_val, pred_pers[val_rows])
    print('\n--- Baselines (validation slice, BEFORE any neural network) ---')
    print(f'  GARCH(1,1) params (train-only fit): '
          f'{ {k: round(v, 6) for k, v in garch_res.params.items()} }')
    print(f'  GARCH(1,1) E|r| forecast : MAE={m_garch["mae"]:.6f}%  R2={m_garch["r2"]:+.4f}')
    print(f'  Persistence |r_t|        : MAE={m_pers["mae"]:.6f}%  R2={m_pers["r2"]:+.4f}')

    # ---- Step 2: dedicated LSTM, judged on the identical rows ---------------
    print('\n--- Dedicated volatility LSTM (price-only, fit [0:63%], ES tail [63:70%]) ---')
    pred_lstm, _model = train_volatility_lstm(X_scaled, y, split, config, random_state)
    m_lstm = _metrics(y_val, pred_lstm[val_rows])
    print(f'  Volatility LSTM          : MAE={m_lstm["mae"]:.6f}%  R2={m_lstm["r2"]:+.4f}')

    point_dmae = m_garch['mae'] - m_lstm['mae']
    point_dr2 = m_lstm['r2'] - m_garch['r2']
    boot = bootstrap_delta(y_val, pred_lstm[val_rows], pred_garch[val_rows],
                           alpha=alpha, random_state=random_state)
    ci_dmae, ci_dr2 = boot['ci_dmae'], boot['ci_dr2']

    cleared = bool(ci_dmae[0] > 0 and ci_dr2[0] > 0)
    if cleared:
        verdict = (f'CLEARED (LSTM beats GARCH(1,1): both {100 * (1 - alpha):.1f}% CIs '
                   f'exclude 0 in its favor)')
    else:
        verdict = ('NOT CLEARED (LSTM improvement over GARCH(1,1) indistinguishable '
                   'from noise at the family bar — do not ship as a headline feature)')

    print('\n--- LSTM vs GARCH(1,1) on identical validation rows ---')
    print(f'  point delta MAE (garch - lstm) = {point_dmae:+.6f}%  '
          f'CI[{ci_dmae[0]:+.6f}, {ci_dmae[1]:+.6f}]  '
          f'frac(dMAE>0)={boot["frac_dmae_positive"]:.3f}')
    print(f'  point delta R2  (lstm - garch) = {point_dr2:+.4f}  '
          f'CI[{ci_dr2[0]:+.4f}, {ci_dr2[1]:+.4f}]')
    print(f'  VERDICT: {verdict}')

    result = {
        'hypothesis': 'volatility_lstm_vs_garch11',
        'arbiter': 'validation[70:80]',
        'n_val': len(val_rows),
        'garch_mae': round(m_garch['mae'], 6), 'garch_r2': round(m_garch['r2'], 4),
        'persistence_mae': round(m_pers['mae'], 6), 'persistence_r2': round(m_pers['r2'], 4),
        'lstm_mae': round(m_lstm['mae'], 6), 'lstm_r2': round(m_lstm['r2'], 4),
        'point_delta_mae': round(point_dmae, 6),
        'ci_dmae_low': round(ci_dmae[0], 6), 'ci_dmae_high': round(ci_dmae[1], 6),
        'point_delta_r2': round(point_dr2, 4),
        'ci_dr2_low': round(ci_dr2[0], 4), 'ci_dr2_high': round(ci_dr2[1], 4),
        'alpha_bar': alpha,
        'cleared_bar': cleared,
        'verdict': verdict,
    }
    if register:
        register_volatility_hypothesis(result, base_dir=base_dir,
                                       notes='dedicated price-only LSTM vs train-only-fit GARCH(1,1)')

    out_path = _p(base_dir, out_csv)
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    if register:
        print(f'Hypothesis logged: {_p(base_dir, HYPOTHESIS_LOG)}')
    return result


MULTITASK_CSV = 'results/volatility_multitask_experiment.csv'
VOL_MODEL_DIR = os.path.join('models', 'volatility')
VOL_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
VOL_MODEL_FILES = [f'volatility_lstm_seed{s}.keras' for s in VOL_ENSEMBLE_SEEDS]
VOL_ARTIFACTS = VOL_MODEL_FILES + ['global_scaler.pkl', 'lag_scaler.pkl',
                                   'lag_pca.pkl', 'lstm_time_steps.pkl',
                                   'vol_metrics.json']


def train_production_volatility_model(config, base_dir=''):
    """Train + persist the PRODUCTION volatility model under models/volatility/.

    Ships the SEED-ENSEMBLE of the 3-head multi-task architecture — exactly the
    object that cleared the pre-registered ship gate
    (results/volatility_seed_ensemble.csv: MT 5-seed ensemble vs train-only-fit
    GARCH(1,1) on the validation arbiter, at the family_size=3 Bonferroni bar).
    Seed ensembling is not a style choice: single-seed runs showed TF/oneDNN
    training nondeterminism of the same order as the deltas under test, so the
    validated candidate was pre-specified as the mean over VOL_ENSEMBLE_SEEDS.
    Only the volatility head is served; the return/direction heads exist for
    the multi-task training signal (which beat the dedicated single-head
    architecture head-to-head) and are ignored at inference — the daily
    variants remain the sole source of return/direction predictions.

    Production split (standard pipeline convention, unlike the experiment's
    stricter inner split): PCA + scaler fit on [0:80%], each seed's LSTM trains
    [0:70%] with [70%:80%] for early stopping, and the untouched [80%:100%]
    test block yields the ONE-SHOT final report persisted into
    vol_metrics.json — a report, never a search knob. GARCH(1,1) is refit on
    [0:80%] and fixed-parameter-rolled over the test block so the report
    carries the honest baseline comparison alongside the ensemble's numbers.

    Price-only by nature — ONE model family, no baseline/with_macro
    duplication. Returns the metrics dict written to vol_metrics.json.
    """
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from arch import arch_model
    import joblib

    lstm_cfg = config['lstm']
    time_steps = lstm_cfg['time_steps']
    out_dir = _p(base_dir, VOL_MODEL_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # Same engineered euro-era row set as every other model (macro merge only
    # sets the row span; the model consumes price-only columns).
    ohlcv = load_history(_p(base_dir, config['data']['history_csv_path']))
    macro, _src = fetch_macro_features(
        ohlcv.index.min(), ohlcv.index.max(), config['macro'], base_dir=base_dir
    )
    feat = add_advanced_features(merge_macro_features(ohlcv.copy(), macro))

    n = len(feat)
    train_fraction = config['split']['train_fraction']    # 0.80
    val_fraction = config['split']['val_fraction']        # 0.10
    train_end = int(n * train_fraction)                    # 80% — PCA/scaler fit end
    lstm_train_end = int(n * (train_fraction - val_fraction))  # 70%

    lag_scaler, lag_pca = fit_lag_pca(
        feat.iloc[:train_end], lag_columns=LAG_COLUMNS,
        variance_threshold=config['pca']['variance_threshold'],
    )
    red = apply_lag_pca(feat, lag_scaler, lag_pca, lag_columns=LAG_COLUMNS)
    cols = model_input_columns(lag_pca, base_columns=PRICE_FEATURE_COLUMNS,
                               lag_columns=LAG_COLUMNS)
    global_scaler = StandardScaler().fit(red[cols].iloc[:train_end])
    X_scaled = global_scaler.transform(red[cols])
    y_vol = feat[TARGET_VOLATILITY_COLUMN].to_numpy()
    y_ret = feat[TARGET_RETURN_COLUMN].to_numpy()
    y_dir = feat[TARGET_DIRECTION_COLUMN].to_numpy()

    windows, targets_vol, target_rows = make_sequences(X_scaled, y_vol, time_steps)
    targets_ret = y_ret[time_steps - 1:]
    targets_dir = y_dir[time_steps - 1:]
    is_tr = target_rows < lstm_train_end
    is_es = (target_rows >= lstm_train_end) & (target_rows < train_end)
    is_te = target_rows >= train_end

    seed_test_preds = []
    for seed in VOL_ENSEMBLE_SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        inputs = Input(shape=(time_steps, X_scaled.shape[1]), name='ohlcv_window')
        shared = LSTM(lstm_cfg['units'], name='shared_lstm_trunk')(inputs)
        shared = Dropout(lstm_cfg['dropout'], name='shared_dropout')(shared)
        return_output = Dense(1, activation='linear', name='return_output', dtype='float32')(shared)
        direction_output = Dense(1, activation='sigmoid', name='direction_output', dtype='float32')(shared)
        volatility_output = Dense(1, activation='softplus', name='volatility_output', dtype='float32')(shared)
        model = Model(inputs, [return_output, direction_output, volatility_output],
                      name=f'volatility_mt_lstm_seed{seed}')
        model.compile(
            optimizer=Adam(learning_rate=lstm_cfg['learning_rate']),
            loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy',
                  'volatility_output': 'mse'},
            loss_weights={'return_output': 1.0, 'direction_output': 1.0,
                          'volatility_output': 1.0},
        )
        es = EarlyStopping(monitor='val_loss', patience=lstm_cfg['patience'],
                           restore_best_weights=True, verbose=0)
        model.fit(
            windows[is_tr],
            {'return_output': targets_ret[is_tr], 'direction_output': targets_dir[is_tr],
             'volatility_output': targets_vol[is_tr]},
            validation_data=(windows[is_es],
                             {'return_output': targets_ret[is_es],
                              'direction_output': targets_dir[is_es],
                              'volatility_output': targets_vol[is_es]}),
            epochs=lstm_cfg['epochs'], batch_size=lstm_cfg['batch_size'],
            callbacks=[es], verbose=0)
        model.save(os.path.join(out_dir, f'volatility_lstm_seed{seed}.keras'))
        _r, _d, pv = model.predict(windows[is_te], verbose=0)
        seed_test_preds.append(pv.ravel())
        print(f'[volatility] seed {seed}: trained + saved '
              f'(test MAE={mean_absolute_error(targets_vol[is_te], pv.ravel()):.6f}%)')

    # ---- ONE-SHOT final test report (report, never a search knob) -----------
    y_te = targets_vol[is_te]
    pred_ens_te = np.mean(seed_test_preds, axis=0)

    returns_pct = feat['log_return'] * 100.0
    garch_res = arch_model(returns_pct.iloc[:train_end], mean='Constant',
                           vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
    fixed = arch_model(returns_pct, mean='Constant', vol='GARCH', p=1, q=1,
                       dist='normal').fix(garch_res.params)
    fc = fixed.forecast(horizon=1, start=feat.index[0], reindex=True)
    pred_garch_all = np.sqrt(fc.variance['h.1'].to_numpy(dtype=float)) * ABS_MOMENT
    pred_garch_te = pred_garch_all[target_rows[is_te]]
    pred_pers_te = (np.abs(feat['log_return'].to_numpy()) * 100.0)[target_rows[is_te]]

    metrics = {
        'model': '5-seed multi-task LSTM ensemble, volatility head (price-only)',
        'seeds': VOL_ENSEMBLE_SEEDS,
        'target': TARGET_VOLATILITY_COLUMN,
        'n_test': int(is_te.sum()),
        'test_ensemble_mae': round(float(mean_absolute_error(y_te, pred_ens_te)), 6),
        'test_ensemble_r2': round(float(r2_score(y_te, pred_ens_te)), 4),
        'test_garch_mae': round(float(mean_absolute_error(y_te, pred_garch_te)), 6),
        'test_garch_r2': round(float(r2_score(y_te, pred_garch_te)), 4),
        'test_persistence_mae': round(float(mean_absolute_error(y_te, pred_pers_te)), 6),
        'test_persistence_r2': round(float(r2_score(y_te, pred_pers_te)), 4),
        'note': ('one-shot held-out test report; the SHIP decision was made on '
                 'the validation arbiter (results/volatility_seed_ensemble.csv + '
                 'results/volatility_hypothesis_log.csv), never on this block'),
    }
    # Attach the validation-arbiter ship-gate evidence so serving can surface
    # the honest framing without re-deriving it.
    gate_csv = _p(base_dir, SEED_ENSEMBLE_CSV)
    if os.path.exists(gate_csv):
        v = pd.read_csv(gate_csv).iloc[0].to_dict()
        metrics['validation_decision'] = {
            k: (bool(v[k]) if k == 'cleared_bar' else v[k]) for k in (
                'garch_mae', 'garch_r2', 'mt_ensemble_mae', 'mt_ensemble_r2',
                'point_delta_mae', 'ci_dmae_low', 'ci_dmae_high',
                'point_delta_r2', 'ci_dr2_low', 'ci_dr2_high',
                'alpha_bar', 'cleared_bar', 'verdict') if k in v
        }
    val_csv = _p(base_dir, VALIDATION_CSV)
    if os.path.exists(val_csv):
        v = pd.read_csv(val_csv).iloc[0].to_dict()
        metrics['validation_persistence_baseline'] = {
            'persistence_mae': v.get('persistence_mae'),
            'persistence_r2': v.get('persistence_r2'),
        }

    print(f"[volatility] ONE-SHOT test report: "
          f"ensemble MAE={metrics['test_ensemble_mae']}% R2={metrics['test_ensemble_r2']}  |  "
          f"GARCH(1,1) MAE={metrics['test_garch_mae']}% R2={metrics['test_garch_r2']}  |  "
          f"persistence MAE={metrics['test_persistence_mae']}% R2={metrics['test_persistence_r2']}")

    joblib.dump(lag_scaler, os.path.join(out_dir, 'lag_scaler.pkl'))
    joblib.dump(lag_pca, os.path.join(out_dir, 'lag_pca.pkl'))
    joblib.dump(global_scaler, os.path.join(out_dir, 'global_scaler.pkl'))
    joblib.dump(time_steps, os.path.join(out_dir, 'lstm_time_steps.pkl'))
    with open(os.path.join(out_dir, 'vol_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[volatility] Saved under {out_dir}/: {', '.join(VOL_ARTIFACTS)}")
    return metrics


def run_multitask_experiment(config_path='config.json', base_dir='',
                             out_csv=MULTITASK_CSV, register=True, random_state=None):
    """Step-2 follow-up (dedicated model already cleared its bar): does adding
    volatility as a THIRD head on the shared return/direction trunk help or
    hurt, versus the dedicated single-head model? Registering this grows the
    volatility hypothesis family to 2, so BOTH comparisons here are judged at
    the tightened Bonferroni bar — including a re-check that the primary
    dedicated-vs-GARCH result still clears it."""
    with open(_p(base_dir, config_path)) as f:
        config = json.load(f)
    random_state = config['random_state'] if random_state is None else random_state

    feat, X_scaled, split = build_volatility_matrix(config, base_dir=base_dir)
    y_vol = feat[TARGET_VOLATILITY_COLUMN].to_numpy()
    y_ret = feat[TARGET_RETURN_COLUMN].to_numpy()
    y_dir = feat[TARGET_DIRECTION_COLUMN].to_numpy()
    train_end, val_end = split['train_end'], split['val_end']
    val_rows = np.arange(train_end, val_end)
    y_val = y_vol[val_rows]

    log = load_volatility_hypothesis_log(base_dir)
    hyps = set(log['hypothesis']) | {'volatility_lstm_vs_garch11',
                                     'volatility_multitask_head_vs_dedicated'}
    family_size = len(hyps)
    alpha = bonferroni_alpha(family_size)

    print('=' * 78)
    print('VOLATILITY MULTI-TASK EXPERIMENT — third head on the shared trunk')
    print(f'  family size now {family_size} -> Bonferroni alpha = {alpha:.4g} '
          f'({100 * (1 - alpha):.1f}% CIs for EVERY comparison below)')
    print('=' * 78)

    pred_garch, _ = garch_forecasts(feat, split, config)
    pred_dedicated, _m = train_volatility_lstm(X_scaled, y_vol, split, config, random_state)
    pred_mt_vol, pred_mt_ret, pred_mt_dir, _mt = train_multitask_volatility_lstm(
        X_scaled, y_vol, y_ret, y_dir, split, config, random_state)

    m_garch = _metrics(y_val, pred_garch[val_rows])
    m_ded = _metrics(y_val, pred_dedicated[val_rows])
    m_mt = _metrics(y_val, pred_mt_vol[val_rows])
    print(f'  GARCH(1,1)          : MAE={m_garch["mae"]:.6f}%  R2={m_garch["r2"]:+.4f}')
    print(f'  Dedicated vol LSTM  : MAE={m_ded["mae"]:.6f}%  R2={m_ded["r2"]:+.4f}')
    print(f'  Multi-task vol head : MAE={m_mt["mae"]:.6f}%  R2={m_mt["r2"]:+.4f}')

    # Shared-factors diagnostic (Ch. 7.7 justification check, volatility form):
    # if the trunk truly carries shared risk factors, the vol head should
    # co-move with the return head's MAGNITUDE. Plus the existing sign check.
    v = pred_mt_vol[val_rows]
    r = pred_mt_ret[val_rows]
    d = (pred_mt_dir[val_rows] >= 0.5).astype(int)
    mag_corr = float(np.corrcoef(v, np.abs(r))[0, 1]) if np.std(np.abs(r)) > 0 else float('nan')
    sign_agree = float((np.sign(r) == np.where(d == 1, 1, -1)).mean())
    print(f'  [shared-factors] corr(vol_head, |return_head|)={mag_corr:+.4f}  '
          f'return-sign/direction agreement={sign_agree:.4f}')

    # Multi-task head vs dedicated (positive = multi-task better).
    boot_mt = bootstrap_delta(y_val, pred_mt_vol[val_rows], pred_dedicated[val_rows],
                              alpha=alpha, random_state=random_state)
    dmae_mt = m_ded['mae'] - m_mt['mae']
    dr2_mt = m_mt['r2'] - m_ded['r2']
    mt_better = bool(boot_mt['ci_dmae'][0] > 0 and boot_mt['ci_dr2'][0] > 0)
    mt_worse = bool(boot_mt['ci_dmae'][1] < 0 and boot_mt['ci_dr2'][1] < 0)
    print(f'\n  MT-head vs dedicated: dMAE={dmae_mt:+.6f}% '
          f'CI[{boot_mt["ci_dmae"][0]:+.6f}, {boot_mt["ci_dmae"][1]:+.6f}]  '
          f'dR2={dr2_mt:+.4f} CI[{boot_mt["ci_dr2"][0]:+.4f}, {boot_mt["ci_dr2"][1]:+.4f}]')
    if mt_better:
        verdict_mt = 'MULTI-TASK WINS (sharing helps at the family bar) — ship the 3-head trunk'
    elif mt_worse:
        verdict_mt = 'DEDICATED WINS (sharing hurts at the family bar) — ship the dedicated model'
    else:
        verdict_mt = ('TIE (no CI-confirmed difference) — ship the DEDICATED model: '
                      'simpler, independently validated, no coupling to the near-chance direction task')
    print(f'  VERDICT: {verdict_mt}')

    # Primary hypothesis re-check at the TIGHTENED bar (family grew 1 -> 2).
    boot_primary = bootstrap_delta(y_val, pred_dedicated[val_rows], pred_garch[val_rows],
                                   alpha=alpha, random_state=random_state)
    primary_holds = bool(boot_primary['ci_dmae'][0] > 0 and boot_primary['ci_dr2'][0] > 0)
    print(f'\n  Primary (dedicated vs GARCH) at alpha={alpha:.4g}: '
          f'dMAE CI[{boot_primary["ci_dmae"][0]:+.6f}, {boot_primary["ci_dmae"][1]:+.6f}]  '
          f'dR2 CI[{boot_primary["ci_dr2"][0]:+.4f}, {boot_primary["ci_dr2"][1]:+.4f}]  '
          f'-> {"STILL CLEARS" if primary_holds else "NO LONGER CLEARS — do not ship"}')

    result = {
        'hypothesis': 'volatility_multitask_head_vs_dedicated',
        'arbiter': 'validation[70:80]',
        'n_val': len(val_rows),
        'garch_mae': round(m_garch['mae'], 6), 'garch_r2': round(m_garch['r2'], 4),
        'dedicated_mae': round(m_ded['mae'], 6), 'dedicated_r2': round(m_ded['r2'], 4),
        'multitask_mae': round(m_mt['mae'], 6), 'multitask_r2': round(m_mt['r2'], 4),
        'point_delta_mae': round(dmae_mt, 6),
        'ci_dmae_low': round(boot_mt['ci_dmae'][0], 6), 'ci_dmae_high': round(boot_mt['ci_dmae'][1], 6),
        'point_delta_r2': round(dr2_mt, 4),
        'ci_dr2_low': round(boot_mt['ci_dr2'][0], 4), 'ci_dr2_high': round(boot_mt['ci_dr2'][1], 4),
        'magnitude_corr': round(mag_corr, 4), 'sign_agreement': round(sign_agree, 4),
        'alpha_bar': alpha,
        'cleared_bar': mt_better,
        'primary_still_clears_tightened_bar': primary_holds,
        'verdict': verdict_mt,
    }
    if register:
        register_volatility_hypothesis(result, base_dir=base_dir,
                                       notes='3rd head on shared trunk vs dedicated single-head model')

    out_path = _p(base_dir, out_csv)
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    return result


SEED_ENSEMBLE_CSV = 'results/volatility_seed_ensemble.csv'
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]


def run_seed_ensemble_confirmation(config_path='config.json', base_dir='',
                                   out_csv=SEED_ENSEMBLE_CSV, register=True):
    """FINAL pre-registered confirmation test (hypothesis 3 of the family).

    Motivation: the single-seed runs exposed training nondeterminism (TF/oneDNN
    on CPU) of the same order as the deltas under test — the dedicated model's
    val MAE moved 0.190->0.197 across identical-seed runs, enough to flip the
    primary re-check at the tightened bar. The bootstrap CIs capture row-
    sampling noise only, NOT training noise, so any single-seed comparison is
    unreliable evidence.

    The production candidate is therefore pre-specified as the SEED-ENSEMBLE
    (mean prediction over ENSEMBLE_SEEDS) of the 3-head multi-task architecture
    (which beat the dedicated model head-to-head at the tightened bar), and
    THIS test — MT seed-ensemble vs GARCH(1,1) on the validation arbiter at
    the family_size=3 Bonferroni bar — is the single ship gate. Clears -> ship
    exactly that ensemble; fails -> nothing ships, the family is spent, and
    further volatility hypotheses need genuinely new data (the forward window).

    The dedicated seed-ensemble is trained too but reported as CONTEXT ONLY —
    it is not an alternative ship candidate (that would be another hypothesis).
    """
    with open(_p(base_dir, config_path)) as f:
        config = json.load(f)

    feat, X_scaled, split = build_volatility_matrix(config, base_dir=base_dir)
    y_vol = feat[TARGET_VOLATILITY_COLUMN].to_numpy()
    y_ret = feat[TARGET_RETURN_COLUMN].to_numpy()
    y_dir = feat[TARGET_DIRECTION_COLUMN].to_numpy()
    train_end, val_end = split['train_end'], split['val_end']
    val_rows = np.arange(train_end, val_end)
    y_val = y_vol[val_rows]

    log = load_volatility_hypothesis_log(base_dir)
    hyps = set(log['hypothesis']) | {'volatility_mt_seed_ensemble_vs_garch11'}
    family_size = len(hyps)
    alpha = bonferroni_alpha(family_size)

    print('=' * 78)
    print('VOLATILITY SEED-ENSEMBLE CONFIRMATION — the single pre-registered ship gate')
    print(f'  candidate: mean over seeds {ENSEMBLE_SEEDS} of the 3-head multi-task LSTM')
    print(f'  family size {family_size} -> Bonferroni alpha = {alpha:.4g} '
          f'({100 * (1 - alpha):.2f}% CIs)')
    print('=' * 78)

    pred_garch, _ = garch_forecasts(feat, split, config)
    m_garch = _metrics(y_val, pred_garch[val_rows])
    print(f'  GARCH(1,1) baseline      : MAE={m_garch["mae"]:.6f}%  R2={m_garch["r2"]:+.4f}')

    mt_preds, ded_preds = [], []
    mt_maes, ded_maes = [], []
    for seed in ENSEMBLE_SEEDS:
        pv, _pr, _pd, _m = train_multitask_volatility_lstm(
            X_scaled, y_vol, y_ret, y_dir, split, config, random_state=seed)
        mt_preds.append(pv[val_rows])
        mt_maes.append(mean_absolute_error(y_val, pv[val_rows]))
        pdd, _m2 = train_volatility_lstm(X_scaled, y_vol, split, config, random_state=seed)
        ded_preds.append(pdd[val_rows])
        ded_maes.append(mean_absolute_error(y_val, pdd[val_rows]))
        print(f'  seed {seed}: MT val MAE={mt_maes[-1]:.6f}%  dedicated val MAE={ded_maes[-1]:.6f}%')

    pred_mt_ens = np.mean(mt_preds, axis=0)
    pred_ded_ens = np.mean(ded_preds, axis=0)
    m_mt = _metrics(y_val, pred_mt_ens)
    m_ded = _metrics(y_val, pred_ded_ens)
    print(f'\n  per-seed MT MAE spread   : [{min(mt_maes):.6f}, {max(mt_maes):.6f}]  '
          f'(training noise the single-seed CIs could not see)')
    print(f'  MT seed-ensemble         : MAE={m_mt["mae"]:.6f}%  R2={m_mt["r2"]:+.4f}   <- SHIP CANDIDATE')
    print(f'  dedicated seed-ensemble  : MAE={m_ded["mae"]:.6f}%  R2={m_ded["r2"]:+.4f}   (context only)')

    boot = bootstrap_delta(y_val, pred_mt_ens, pred_garch[val_rows], alpha=alpha,
                           random_state=42)
    dmae = m_garch['mae'] - m_mt['mae']
    dr2 = m_mt['r2'] - m_garch['r2']
    cleared = bool(boot['ci_dmae'][0] > 0 and boot['ci_dr2'][0] > 0)
    verdict = ('CLEARED — ship the MT seed-ensemble with honest GARCH-relative framing'
               if cleared else
               'NOT CLEARED — nothing ships; family is spent on this arbiter')
    print(f'\n  MT-ensemble vs GARCH: dMAE={dmae:+.6f}% '
          f'CI[{boot["ci_dmae"][0]:+.6f}, {boot["ci_dmae"][1]:+.6f}]  '
          f'dR2={dr2:+.4f} CI[{boot["ci_dr2"][0]:+.4f}, {boot["ci_dr2"][1]:+.4f}]  '
          f'frac(dMAE>0)={boot["frac_dmae_positive"]:.3f}')
    print(f'  VERDICT: {verdict}')

    result = {
        'hypothesis': 'volatility_mt_seed_ensemble_vs_garch11',
        'arbiter': 'validation[70:80]',
        'n_val': len(val_rows),
        'seeds': ' '.join(str(s) for s in ENSEMBLE_SEEDS),
        'garch_mae': round(m_garch['mae'], 6), 'garch_r2': round(m_garch['r2'], 4),
        'mt_ensemble_mae': round(m_mt['mae'], 6), 'mt_ensemble_r2': round(m_mt['r2'], 4),
        'dedicated_ensemble_mae': round(m_ded['mae'], 6),
        'dedicated_ensemble_r2': round(m_ded['r2'], 4),
        'mt_seed_mae_min': round(min(mt_maes), 6), 'mt_seed_mae_max': round(max(mt_maes), 6),
        'point_delta_mae': round(dmae, 6),
        'ci_dmae_low': round(boot['ci_dmae'][0], 6), 'ci_dmae_high': round(boot['ci_dmae'][1], 6),
        'point_delta_r2': round(dr2, 4),
        'ci_dr2_low': round(boot['ci_dr2'][0], 4), 'ci_dr2_high': round(boot['ci_dr2'][1], 4),
        'alpha_bar': alpha,
        'cleared_bar': cleared,
        'verdict': verdict,
    }
    if register:
        register_volatility_hypothesis(
            result, base_dir=base_dir,
            notes='pre-registered ship gate: 5-seed MT ensemble vs train-only-fit GARCH(1,1); '
                  'seed ensembling motivated by observed TF/oneDNN training nondeterminism')
    out_path = _p(base_dir, out_csv)
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    return result


CANDIDATE_CSV = 'results/volatility_candidate_features.csv'


def _mt_seed_ensemble_val_preds(feat, X_scaled, split, config, seeds=ENSEMBLE_SEEDS):
    """Mean validation-row prediction of the 5-seed 3-head MT ensemble — the
    exact validated methodology (run_seed_ensemble_confirmation), reused for
    both the base and each challenger feature set. Returns (ensemble_val_pred,
    per_seed_val_maes)."""
    y_vol = feat[TARGET_VOLATILITY_COLUMN].to_numpy()
    y_ret = feat[TARGET_RETURN_COLUMN].to_numpy()
    y_dir = feat[TARGET_DIRECTION_COLUMN].to_numpy()
    val_rows = np.arange(split['train_end'], split['val_end'])
    y_val = y_vol[val_rows]
    preds, maes = [], []
    for seed in seeds:
        pv, _pr, _pd, _m = train_multitask_volatility_lstm(
            X_scaled, y_vol, y_ret, y_dir, split, config, random_state=seed)
        preds.append(pv[val_rows])
        maes.append(mean_absolute_error(y_val, pv[val_rows]))
    return np.mean(preds, axis=0), maes


def run_candidate_feature_tests(features=None, config_path='config.json', base_dir='',
                                out_csv=CANDIDATE_CSV, register=True):
    """Hypotheses 4+ of the volatility family: one candidate INPUT feature at a
    time (never bundled) added to the validated 5-seed MT ensemble's price-only
    input set, judged on the validation arbiter against a freshly trained
    same-seeds BASE ensemble (paired bootstrap on identical rows).

    Family-bar discipline: the docs marked this family SPENT on the validation
    arbiter after hypothesis 3; the owner re-opened it (2026-07-17) under the
    standard Bonferroni widening instead — BOTH candidates are declared up
    front, so EVERY comparison here is judged at the FINAL family size's bar
    (0.05/5 = 0.01), stricter than the 0.0167 the original ensemble cleared.
    Each candidate is compared against the SAME base ensemble, never against
    another candidate's result. The test block [80%:100%] is never indexed.

    A `features` entry may be a single column name (str) or a `(name, [cols])`
    tuple — a BUNDLE of columns that are several views of one underlying fact
    (e.g. the FOMC calendar trio) and therefore spend ONE hypothesis slot, not
    one per column.
    """
    features = list(CANDIDATE_VOL_FEATURES if features is None else features)
    candidates = {}   # hypothesis display name -> list of columns (order kept)
    for f in features:
        if isinstance(f, str):
            candidates[f] = [f]
        else:
            name, cols = f
            candidates[name] = list(cols)
    with open(_p(base_dir, config_path)) as f:
        config = json.load(f)

    log = load_volatility_hypothesis_log(base_dir)
    hyp_names = {name: f'volatility_input_{name.lower()}_vs_base_ensemble'
                 for name in candidates}
    family_size = len(set(log['hypothesis']) | set(hyp_names.values()))
    alpha = bonferroni_alpha(family_size)

    print('=' * 78)
    print('VOLATILITY CANDIDATE INPUT FEATURES — one hypothesis per candidate')
    print(f'  candidates (sequential, never cross-bundled): '
          f'{ {n: c for n, c in candidates.items()} }')
    print(f'  family size {family_size} -> Bonferroni alpha = {alpha:.4g} '
          f'({100 * (1 - alpha):.1f}% CIs for every comparison)')
    print('=' * 78)

    # BASE: the validated feature set, retrained with the same seeds in this
    # session so base-vs-challenger differences can only come from the feature.
    feat0, X0, split = build_volatility_matrix(config, base_dir=base_dir)
    y_vol = feat0[TARGET_VOLATILITY_COLUMN].to_numpy()
    val_rows = np.arange(split['train_end'], split['val_end'])
    y_val = y_vol[val_rows]

    pred_garch, _ = garch_forecasts(feat0, split, config)
    m_garch = _metrics(y_val, pred_garch[val_rows])

    print(f'\n--- BASE 5-seed MT ensemble (current price-only inputs, {X0.shape[1]} cols) ---')
    pred_base, base_maes = _mt_seed_ensemble_val_preds(feat0, X0, split, config)
    m_base = _metrics(y_val, pred_base)
    print(f'  per-seed val MAE spread: [{min(base_maes):.6f}, {max(base_maes):.6f}]')
    print(f'  BASE ensemble : MAE={m_base["mae"]:.6f}%  R2={m_base["r2"]:+.4f}   '
          f'(GARCH context: MAE={m_garch["mae"]:.6f}%  R2={m_garch["r2"]:+.4f})')

    results = []
    for feature, cols in candidates.items():
        print(f'\n--- CHALLENGER: base + {feature} ({cols}) ---')
        feat_c, X_c, split_c = build_volatility_matrix(
            config, base_dir=base_dir, extra_feature_columns=cols)
        assert split_c == split and len(feat_c) == len(feat0) and \
            (feat_c.index == feat0.index).all(), \
            'challenger row set must be identical to base (feature isolated)'
        assert X_c.shape[1] == X0.shape[1] + len(cols)

        pred_ch, ch_maes = _mt_seed_ensemble_val_preds(feat_c, X_c, split, config)
        m_ch = _metrics(y_val, pred_ch)
        print(f'  per-seed val MAE spread: [{min(ch_maes):.6f}, {max(ch_maes):.6f}]')
        print(f'  +{feature} ensemble : MAE={m_ch["mae"]:.6f}%  R2={m_ch["r2"]:+.4f}')

        boot = bootstrap_delta(y_val, pred_ch, pred_base, alpha=alpha, random_state=42)
        dmae = m_base['mae'] - m_ch['mae']
        dr2 = m_ch['r2'] - m_base['r2']
        cleared = bool(boot['ci_dmae'][0] > 0 and boot['ci_dr2'][0] > 0)
        verdict = (f'KEEP — +{feature} beats the validated base ensemble at the family bar'
                   if cleared else
                   f'DROP — +{feature} improvement indistinguishable from noise at the family bar')
        print(f'  vs base: dMAE={dmae:+.6f}% CI[{boot["ci_dmae"][0]:+.6f}, {boot["ci_dmae"][1]:+.6f}]  '
              f'dR2={dr2:+.4f} CI[{boot["ci_dr2"][0]:+.4f}, {boot["ci_dr2"][1]:+.4f}]  '
              f'frac(dMAE>0)={boot["frac_dmae_positive"]:.3f}')
        print(f'  VERDICT: {verdict}')

        result = {
            'hypothesis': hyp_names[feature],
            'arbiter': 'validation[70:80]',
            'n_val': len(val_rows),
            'seeds': ' '.join(str(s) for s in ENSEMBLE_SEEDS),
            'garch_mae': round(m_garch['mae'], 6), 'garch_r2': round(m_garch['r2'], 4),
            'base_ensemble_mae': round(m_base['mae'], 6),
            'base_ensemble_r2': round(m_base['r2'], 4),
            'challenger_ensemble_mae': round(m_ch['mae'], 6),
            'challenger_ensemble_r2': round(m_ch['r2'], 4),
            'challenger_seed_mae_min': round(min(ch_maes), 6),
            'challenger_seed_mae_max': round(max(ch_maes), 6),
            'point_delta_mae': round(dmae, 6),
            'ci_dmae_low': round(boot['ci_dmae'][0], 6), 'ci_dmae_high': round(boot['ci_dmae'][1], 6),
            'point_delta_r2': round(dr2, 4),
            'ci_dr2_low': round(boot['ci_dr2'][0], 4), 'ci_dr2_high': round(boot['ci_dr2'][1], 4),
            'alpha_bar': alpha,
            'cleared_bar': cleared,
            'verdict': verdict,
        }
        results.append(result)
        if register:
            register_volatility_hypothesis(
                result, base_dir=base_dir, alpha_override=alpha,
                notes=f'candidate input {feature} (cols: {",".join(cols)}) on the validated '
                      f'5-seed MT ensemble; family re-opened by owner 2026-07-17 (Bonferroni '
                      f'widening, candidates pre-declared -> judged at family alpha={alpha:.4g})')

    out_path = _p(base_dir, out_csv)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'multitask':
        run_multitask_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == 'ensemble':
        run_seed_ensemble_confirmation()
    elif len(sys.argv) > 1 and sys.argv[1] == 'candidates':
        if len(sys.argv) > 2 and sys.argv[2] == 'fomc':
            # The calendar trio is ONE bundled hypothesis (three views of the
            # same scheduled-meeting fact), never three Bonferroni slots.
            run_candidate_feature_tests(
                features=[('fomc_calendar_block', FOMC_FEATURE_COLUMNS)],
                out_csv='results/volatility_candidate_fomc.csv')
        else:
            run_candidate_feature_tests()
    else:
        run()
