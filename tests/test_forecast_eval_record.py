"""Study record (task 7.1)."""
import pytest

from src.forecast_eval import record as R


def test_record_declares_grid_and_every_challenger(tmp_path):
    rec = R.new_record('s1')
    assert rec['grid'] == {'H1': [1, 2, 4, 6, 12, 24, 48, 120], 'D1': [1, 2, 5]}
    assert set(rec['challengers']) == {
        'daily_gbm_price', 'daily_gbm_macro', 'daily_lstm_price', 'daily_lstm_macro',
        'h1_daily_ensemble', 'ti_lstm', 'h1_gbm', 'kronos_direction', 'vol_ensemble',
        'kronos_volatility'}
    assert rec['cap_years'] == 3.0 and rec['first_fit_at'] is None
    assert rec['challengers']['daily_lstm_price']['lstm']['time_steps'] == 20   # from config.json


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
