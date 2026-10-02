"""Study record (task 7.1)."""
import pytest

from src.forecast_eval import record as R


def test_record_declares_grid_and_every_challenger(tmp_path):
    rec = R.new_record('s1')
    assert rec['grid'] == {'M15': [1, 2, 4, 8, 16, 26],
                           'H1': [1, 2, 4, 6, 12, 24, 48, 120], 'D1': [1, 2, 5]}
    assert set(rec['challengers']) == {
        'daily_gbm_price', 'daily_gbm_macro', 'daily_lstm_price', 'daily_lstm_macro',
        'h1_daily_ensemble', 'ti_lstm', 'h1_gbm', 'kronos_direction', 'vol_ensemble',
        'kronos_volatility', 'm15_session_gbm', 'm15_session_lstm'}
    assert rec['cap_years'] == 3.0 and rec['first_fit_at'] is None
    assert rec['challengers']['daily_lstm_price']['lstm']['time_steps'] == 20   # from config.json


def test_the_session_is_declared_before_any_fit():
    """Design D16: the label clock and the session window are locked
    declarations, not something a run decides for itself."""
    rec = R.new_record('s1')
    s = rec['session']
    assert s['label_tz'] == 'Europe/Berlin' and s['label_window'] == '14:30-22:00'
    assert s['owner_window'] == '15:30-23:00 Europe/Sofia'
    assert s['parity_tolerance_pp'] == 1.0
    assert 'mid' in s['price'] and 'target bar' in s['eligibility']
    assert 'session' in R.LOCKED_KEYS


def test_the_m15_challengers_are_declared_and_frozen():
    rec = R.new_record('s1')
    gbm, lstm = rec['challengers']['m15_session_gbm'], rec['challengers']['m15_session_lstm']
    assert gbm['cadence'] == lstm['cadence'] == 'M15'
    assert gbm['gbm'] == rec['challengers']['h1_gbm']['gbm'], 'comparable across cadences'
    assert lstm['lstm']['time_steps'] == 16 and lstm['lstm']['units'] == 64
    assert 'm15_session_lstm' in rec['departures'] and 'm15_session_gbm' in rec['departures']
    assert rec['development']['cpcv']['m15_session_lstm'] == {'n_groups': 4, 'k': 2}
    assert 'm15_session_gbm' in rec['tuning']['grids']


def test_a_session_change_after_the_first_fit_is_refused(tmp_path):
    path = str(tmp_path / 'rec.json')
    R.save(R.new_record('s1'), path)
    R.mark_first_fit(path)
    locked = R.load(path)
    with pytest.raises(R.StudyRecordLocked, match='session'):
        R.save({**locked, 'session': {**locked['session'], 'label_window': '12:30-20:00'}}, path)


# ── 14.2 source provenance in the record ──────────────────────────────────

def test_data_source_fingerprint_is_recorded_and_enforced(tmp_path):
    path = str(tmp_path / 'rec.json')
    R.save(R.new_record('s1'), path)
    fp = {'path': 'results/curl/raw/EURUSD_M1.parquet', 'sha256': 'a' * 64, 'rows': 2_980_060}
    R.record_data_source('m15_m1', fp, path)
    assert R.load(path)['data_sources']['m15_m1']['rows'] == 2_980_060

    seen = {}
    R.assert_data_source('m15_m1', lambda rec: seen.setdefault('rec', rec), path)
    assert seen['rec']['sha256'] == 'a' * 64

    R.mark_first_fit(path)
    R.record_data_source('m15_m1', fp, path)                     # unchanged: allowed
    with pytest.raises(R.StudyRecordLocked, match='changed after the first fit'):
        R.record_data_source('m15_m1', {**fp, 'sha256': 'b' * 64}, path)


def test_assert_data_source_refuses_an_unrecorded_source(tmp_path):
    path = str(tmp_path / 'rec.json')
    R.save(R.new_record('s1'), path)
    with pytest.raises(R.StudyRecordLocked, match='no fingerprint'):
        R.assert_data_source('m15_m1', lambda rec: rec, path)


def test_grid_change_after_first_fit_is_refused(tmp_path):
    path = str(tmp_path / 'rec.json')
    rec = R.new_record('s1')
    R.save(rec, path)
    R.save({**rec, 'grid': {'H1': [1, 2], 'D1': [1]}}, path)       # allowed: nothing fitted yet
    R.save(rec, path)
    R.mark_first_fit(path)
    locked = R.load(path)
    with pytest.raises(R.StudyRecordLocked, match='grid'):
        R.save({**locked, 'grid': {'H1': [1, 3], 'D1': [1, 2, 5]}}, path)
    with pytest.raises(R.StudyRecordLocked, match='challengers'):
        changed = {**locked, 'challengers': {**locked['challengers'], 'ti_lstm': {'units': 999}}}
        R.save(changed, path)


def test_a_new_study_id_is_the_way_forward(tmp_path):
    path = str(tmp_path / 'rec.json')
    R.save(R.new_record('s1'), path)
    R.mark_first_fit(path)
    new = R.new_record('s2')
    new['grid'] = {'H1': [1, 3], 'D1': [1]}
    R.save(new, path)
    assert R.load(path)['study_id'] == 's2'
    assert (tmp_path / 'rec.s1.json').exists(), 'the locked record is archived, not overwritten'


def test_fit_gate(tmp_path):
    path = str(tmp_path / 'rec.json')
    with pytest.raises(R.StudyRecordLocked, match='no study record'):
        R.assert_fit_allowed('h1_gbm', {}, 1, 'H1', path)
    rec = R.new_record('s1')
    R.save(rec, path)
    cfg = rec['challengers']['h1_gbm']
    R.assert_fit_allowed('h1_gbm', cfg, 4, 'H1', path)
    with pytest.raises(R.StudyRecordLocked, match='not in the declared'):
        R.assert_fit_allowed('h1_gbm', cfg, 3, 'H1', path)
    with pytest.raises(R.StudyRecordLocked, match='differs'):
        R.assert_fit_allowed('h1_gbm', {**cfg, 'gbm': {}}, 4, 'H1', path)
