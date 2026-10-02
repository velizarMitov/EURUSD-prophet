"""The study record (task 7.1; spec horizon-study "Fixed horizon grid",
"Configuration declared once and frozen across horizons", "One pre-declared
tuning rule per cell").

Written BEFORE the first fit. It fixes the horizon grid, every challenger's
configuration (starting from the production values in config.json, read-only),
the tuning grids and the rule that lets a tuned variant replace a frozen one,
the forward cap, the seeds and the development settings. Once a first fit has
been recorded, the record is locked: changing the grid or any configuration
raises StudyRecordLocked, and the only way forward is a NEW study id.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone

from .challengers.base import config_hash

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORD_PATH = os.path.join(REPO, 'results', 'horizon_study', 'study_record.json')

GRID = {'M15': [1, 2, 4, 8, 16, 26], 'H1': [1, 2, 4, 6, 12, 24, 48, 120], 'D1': [1, 2, 5]}
CAP_YEARS = 3.0
SEEDS = [42, 43, 44, 45, 46]
LOCKED_KEYS = ('grid', 'challengers', 'tuning', 'cap_years', 'seeds', 'development', 'session')

# Sub-hourly declarations, fixed before the first fit (design D16, D17). The
# label clock is EVIDENCED by m15_data.verify_clock, not taken on faith.
SESSION = {
    'label_tz': 'Europe/Berlin',
    'label_window': '14:30-22:00',
    'owner_window': '15:30-23:00 Europe/Sofia',
    'weekdays_only': True,
    'eligibility': 'the as-of bar AND the target bar both inside the session, same label date',
    'price': 'mid = bid close + spread/2, from the per-bar M1 spread',
    'parity_tolerance_pp': 1.0,
    'clock_evidence': 'the weekly open label is 23:00 year-round and 22:00 in the US/EU '
                      'daylight-saving mismatch weeks; only Europe/Berlin fits both',
}


class StudyRecordLocked(RuntimeError):
    pass


def _cfg():
    with open(os.path.join(REPO, 'config.json'), encoding='utf-8') as fh:
        return json.load(fh)


def default_challengers() -> dict:
    """Declared configurations. Production values are copied from config.json;
    every departure from production is listed under `departures`."""
    c = _cfg()
    gbm_fixed = {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.8}
    lstm = {k: c['lstm'][k] for k in ('time_steps', 'units', 'dropout', 'learning_rate',
                                      'epochs', 'batch_size', 'patience')}
    h1 = c['h1']
    out = {
        'daily_gbm_price': {'cadence': 'D1', 'kind': 'direction', 'feature_set': 'price',
                            'lag_pca_variance': c['pca']['variance_threshold'], 'gbm': gbm_fixed},
        'daily_gbm_macro': {'cadence': 'D1', 'kind': 'direction', 'feature_set': 'macro',
                            'lag_pca_variance': c['pca']['variance_threshold'], 'gbm': gbm_fixed},
        'daily_lstm_price': {'cadence': 'D1', 'kind': 'direction', 'feature_set': 'price',
                             'lag_pca_variance': c['pca']['variance_threshold'], 'lstm': lstm},
        'daily_lstm_macro': {'cadence': 'D1', 'kind': 'direction', 'feature_set': 'macro',
                             'lag_pca_variance': c['pca']['variance_threshold'], 'lstm': lstm},
        'h1_daily_ensemble': {'cadence': 'D1', 'kind': 'direction', 'gbm': gbm_fixed,
                              'rf': {'n_estimators': 400, 'max_depth': 8},
                              'svm': {'C': 1.0, 'gamma': 'scale'},
                              'lstm': {k: h1[k] for k in ('lstm_units', 'lstm_dropout', 'lstm_lr',
                                                          'lstm_epochs', 'lstm_batch', 'lstm_patience')}},
        'ti_lstm': {'cadence': 'D1', 'kind': 'direction', 'n_layers': 2, 'units': 64,
                    'dropout': 0.25, 'lr': 1e-3, 'epochs': 100, 'batch_size': 32, 'patience': 10},
        'h1_gbm': {'cadence': 'H1', 'kind': 'direction',
                   'gbm': {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                           'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0}},
        'm15_session_gbm': {'cadence': 'M15', 'kind': 'direction',
                            'gbm': {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
                                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0}},
        'm15_session_lstm': {'cadence': 'M15', 'kind': 'direction',
                             'lstm': {**lstm, 'time_steps': 16, 'epochs': 30,
                                      'batch_size': 512, 'patience': 4}},
        'kronos_direction': {'cadence': 'H1', 'kind': 'direction', 'model': 'mini',
                             'n_paths': 30, 'train_free': True, 'history_stride': 12},
        'vol_ensemble': {'cadence': 'D1', 'kind': 'volatility', 'feature_set': 'price',
                         'lstm': lstm, 'seeds': SEEDS},
        'kronos_volatility': {'cadence': 'H1', 'kind': 'volatility', 'model': 'mini',
                              'n_paths': 30, 'train_free': True, 'history_stride': 12,
                              'horizons': [24]},
    }
    departures = {
        'daily_gbm_*': 'fixed hyperparameters at the centre of the production param_grid '
                       '(production grid-searches every retrain); the grid itself is the '
                       'declared TUNING grid',
        'h1_daily_ensemble': 'members are classifiers (production members are return '
                             'regressors); RF/SVM at fixed values from the production grids',
        'ti_lstm': 'CPU path (production run() requires CUDA); architecture = production 2x64',
        'h1_gbm': 'device=cpu (production trains on CUDA); same hyperparameters',
        'm15_session_gbm': 'the h1_gbm hyperparameters unchanged, so the two cadences stay '
                           'comparable; features are the shared H1 direction builder applied '
                           'to M15 mid bars plus the session/spread/intrabar columns',
        'm15_session_lstm': 'CPU; time_steps 16 (production 20), epochs 30 (100) and batch 512 '
                            '(64), because the eligible row set is ~7x the daily one and each '
                            'cell is fitted twice for the bid-versus-mid parity check',
        'kronos_*': 'scored only on bars after its training cutoff (2024-06); clean window '
                    'starts 2024-07-01. Historically every 12th as-of bar (5.4 s per sampling on '
                    'GPU: every bar would take ~19 h); forward logging samples every bar',
        'kronos_volatility': 'evaluated at h = 24 H1 bars only (design D13)',
        'vol_ensemble': 'CPCV with N = 4, k = 2 (6 splits): a 5-seed refit costs ~150 s',
    }
    return {'challengers': out, 'departures': departures,
            'tuning': {'grids': {'daily_gbm_price': c['gbm']['param_grid'],
                                 'daily_gbm_macro': c['gbm']['param_grid'],
                                 'h1_gbm': {'max_depth': [3, 4, 6], 'learning_rate': [0.03, 0.05, 0.1]},
                                 'm15_session_gbm': {'max_depth': [3, 4, 6],
                                                     'learning_rate': [0.03, 0.05, 0.1]}},
                       'rule': {'min_net_improvement_pct': 10.0, 'max_pbo': 0.5,
                                'inner_folds': 4}}}


def new_record(study_id: str, data_ranges: dict | None = None,
               development: dict | None = None) -> dict:
    d = default_challengers()
    rec = {
        'study_id': study_id,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'first_fit_at': None,
        'grid': copy.deepcopy(GRID),
        'challengers': d['challengers'],
        'config_hashes': {k: config_hash(v) for k, v in d['challengers'].items()},
        'departures': d['departures'],
        'tuning': d['tuning'],
        'cap_years': CAP_YEARS,
        'seeds': list(SEEDS),
        'development': development or {
            'walk_forward_splits': 5, 'min_train_fraction': 0.5,
            'cpcv': {'default': {'n_groups': 6, 'k': 2}, 'vol_ensemble': {'n_groups': 4, 'k': 2},
                     'm15_session_lstm': {'n_groups': 4, 'k': 2}},
            'macro': 'production fetch_macro_features chain (FRED API -> public CSV -> cache); '
                     'the euro-era row set comes from its leading NaNs, exactly as in production',
            'h1_source': 'results/eurusd_h1.csv (production H1 cache)',
            'daily_source': 'config.json data.history_csv_path',
            'm15_source': 'results/curl/raw/EURUSD_M1.parquet aggregated to M15 mid bars; '
                          'results/eurusd_m15.csv has no spread column and is never read'},
        'session': copy.deepcopy(SESSION),
        'data_ranges': data_ranges or {},
        'data_sources': {},
        'label': 'DESCRIPTIVE - not a verdict; describes challengers, not production models',
    }
    return rec


def load(path: str = RECORD_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def save(record: dict, path: str = RECORD_PATH) -> None:
    """Write the record. Refused when a locked record with the same study id
    already exists and any locked field would change."""
    existing = load(path)
    if existing and existing.get('study_id') == record.get('study_id') and existing.get('first_fit_at'):
        changed = [k for k in LOCKED_KEYS if existing.get(k) != record.get(k)]
        if changed:
            raise StudyRecordLocked(
                f'study {record["study_id"]} has fitted models; locked field(s) {changed} '
                'cannot change. Start a new study id instead.')
    if existing and existing.get('study_id') != record.get('study_id') and existing.get('first_fit_at'):
        archive = path.replace('.json', f'.{existing["study_id"]}.json')
        os.replace(path, archive)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1, sort_keys=True)


def mark_first_fit(path: str = RECORD_PATH) -> dict:
    rec = load(path)
    if rec is None:
        raise FileNotFoundError('write the study record before the first fit')
    if not rec.get('first_fit_at'):
        rec['first_fit_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
    return rec


def record_data_source(name: str, fingerprint: dict, path: str = RECORD_PATH) -> dict:
    """Store a source file's digest and row count (task 14.2). `data_sources` is
    deliberately NOT locked: the M1 parquet is outside git, so a regenerated copy
    must be recordable -- but it must then be PROVEN identical, which is what
    `assert_data_source` does with the stored fingerprint."""
    rec = load(path)
    if rec is None:
        raise FileNotFoundError('write the study record before recording a data source')
    existing = rec.setdefault('data_sources', {}).get(name)
    if existing and rec.get('first_fit_at') and existing.get('sha256') != fingerprint.get('sha256'):
        raise StudyRecordLocked(
            f'{name} changed after the first fit: recorded sha256 '
            f'{str(existing.get("sha256"))[:12]}, now {str(fingerprint.get("sha256"))[:12]}. '
            'Start a new study id rather than re-fitting on different data.')
    rec['data_sources'][name] = dict(fingerprint)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec


def assert_data_source(name: str, verifier, path: str = RECORD_PATH) -> dict:
    """Raise unless the file on disk matches what the record was built from.
    `verifier(recorded) -> fingerprint` is m15_data.verify_source for M15."""
    rec = load(path)
    recorded = (rec or {}).get('data_sources', {}).get(name)
    if not recorded:
        raise StudyRecordLocked(f'the study record has no fingerprint for data source {name!r}')
    return verifier(recorded)


def assert_fit_allowed(challenger: str, config: dict, h: int, cadence: str,
                       path: str = RECORD_PATH) -> None:
    """Raise unless the record exists, lists this horizon for the cadence, and
    declares exactly this configuration."""
    rec = load(path)
    if rec is None:
        raise StudyRecordLocked('no study record: it must be written before any fit')
    if h not in rec['grid'][cadence]:
        raise StudyRecordLocked(f'h={h} is not in the declared {cadence} grid {rec["grid"][cadence]}')
    if rec['config_hashes'].get(challenger) != config_hash(config):
        raise StudyRecordLocked(f'{challenger}: configuration differs from the declared one')
