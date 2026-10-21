"""Gold study stages (tasks 6.1, 7.1; spec "Validation is the only arbiter",
"The test block is scored exactly once", "Pre-registration precedes the first
validation score")."""
import copy
import csv

import numpy as np
import pandas as pd
import pytest

from src.gold import load_config
from src.gold import prereg as P
from src.gold import study as SD

CFG = load_config()


def _targets(n, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 1, n)
    return {'dir': (ret > 0).astype(float), 'ret': ret, 'vol': np.abs(ret)}


def test_scoring_refuses_without_a_committed_registration(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['prereg_md'] = str(tmp_path / 'none.md')
    cfg['paths']['prereg_json'] = str(tmp_path / 'none.json')
    with pytest.raises(SD.StageRefused):
        SD.check_stage_allowed('validation', cfg)


@pytest.mark.skipif(not P.scoring_allowed(CFG), reason='needs the committed gold registration')
def test_second_test_run_is_refused(tmp_path):
    cfg = copy.deepcopy(CFG)
    report = tmp_path / 'test_report.json'
    report.write_text('{}')
    cfg['paths']['test_report'] = str(report)
    with pytest.raises(SD.StageRefused, match='never scored twice'):
        SD.check_stage_allowed('test', cfg)


@pytest.mark.skipif(not P.scoring_allowed(CFG), reason='needs the committed gold registration')
def test_test_stage_refuses_uncommitted_validation_verdicts(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['paths']['test_report'] = str(tmp_path / 'test_report.json')
    cfg['paths']['validation_report'] = str(tmp_path / 'validation_report.md')
    with pytest.raises(SD.StageRefused, match='not committed'):
        SD.check_stage_allowed('test', cfg)


def test_registry_rows_are_appended_never_rewritten(tmp_path):
    path = tmp_path / 'reg.csv'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        csv.DictWriter(fh, fieldnames=P.REGISTRY_FIELDS).writeheader()
    SD.append_registry(str(path), [{'cell': 'a', 'family': 'direction', 'verdict': 'DROP'}])
    first = path.read_text(encoding='utf-8')
    SD.append_registry(str(path), [{'cell': 'b', 'family': 'direction', 'verdict': 'DROP'}])
    second = path.read_text(encoding='utf-8')
    assert second.startswith(first)
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    assert [r['n'] for r in rows] == ['1', '2'] and rows[0]['arbiter'] == 'validation[70:80]'


def test_direction_verdicts():
    n = 4000
    tg = _targets(n)
    rows, fit = np.arange(2000, n), np.arange(0, 2000)
    perfect = {'p_up': np.where(tg['dir'] == 1, 0.9, 0.1)}
    coin = {'p_up': np.random.default_rng(9).uniform(0, 1, n)}
    r = SD.score_direction('h1_gbm', 'H1', perfect, tg, rows, fit, 0.025, 'ADMITTED', 500)
    assert r['verdict'].startswith('CLEARED') and r['cleared_bar'] is True
    r = SD.score_direction('h1_gbm', 'H1', coin, tg, rows, fit, 0.025, 'ADMITTED', 500)
    assert r['verdict'].startswith('DROP')
    r = SD.score_direction('d1_gbm', 'D1', perfect, tg, rows, fit, 0.025, P.UNDERPOWERED, 500)
    assert r['verdict'] == P.UNDERPOWERED and r['cleared_bar'] == ''


def test_volatility_and_return_verdicts():
    n = 3000
    tg = _targets(n, seed=1)
    rows = np.arange(n)
    good = {'vol_pct': tg['vol'] * 0.9 + 0.05}
    flat = {'vol_pct': np.full(n, tg['vol'].mean())}
    assert SD.score_volatility('h1_lstm', 'H1', 'h1_garch_hod', good, flat, tg, rows, 0.025, 500)['cleared_bar']
    assert not SD.score_volatility('h1_lstm', 'H1', 'h1_garch_hod', flat, good, tg, rows, 0.025, 500)['cleared_bar']
    assert SD.score_return('d1_gbm', 'D1', {'ret_pct': tg['ret'] * 0.5}, tg, rows, 0.0125)['cleared_bar']
    assert not SD.score_return('d1_gbm', 'D1', {'ret_pct': np.zeros(n) + 1e-9}, tg, rows, 0.0125)['cleared_bar']


def test_a_failed_baseline_voids_only_its_comparison():
    assert SD.families_of('h1_garch_hod') == ['volatility']
    assert SD.families_of('h1_lstm') == ['direction', 'return', 'volatility']


def _synthetic(n, freq, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2001-01-01', periods=int(n * 1.5), freq=freq, tz='UTC')
    idx = idx[idx.dayofweek < 5][:n]
    c = 1500 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({'open': c, 'high': c * 1.002, 'low': c * 0.998, 'close': c}, index=idx)


def test_score_stage_runs_end_to_end_on_synthetic_data(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['seeds'] = [42]
    cfg['gbm']['grid'] = {'n_estimators': [30], 'max_depth': [2], 'learning_rate': [0.1]}
    cfg['gbm']['cv_splits'] = 2
    for cad in ('D1', 'H1'):
        cfg['lstm'][cad].update(epochs=1, units=4)
    cfg['bootstrap']['n_boot'] = 50
    cfg['paths']['trial_log'] = str(tmp_path / 'trials.csv')
    frames = {'D1': _synthetic(900, 'D', 1), 'H1': _synthetic(3000, 'h', 2)}
    reg = P.build_registration(cfg, frames, {c: {'first': ''} for c in frames}, {'n_weeks': 1, 'match_rate': 1.0})
    results, failures, trials, extras, preds = SD.score_stage('validation', frames, reg, cfg)
    assert not failures, failures
    fams = {(r['family'], r['cell']) for r in results}
    assert ('volatility', 'h1_lstm') in fams and ('return', 'd1_gbm') in fams and ('direction', 'h1_gbm') in fams
    assert len(results) == 4 + 4 + 2
    assert len(trials) == 2                               # one grid point per GBM cell
    assert set(preds) == {'D1', 'H1'} and 'h1_lstm__vol_pct' in preds['H1'].columns
    md = SD.render_report('validation', results, failures, extras, {'finished_at': 'now', 'elapsed_s': 1.0})
    assert 'Gross of cost' in md


def test_hour_concentration_flags_excess_from_the_high_spread_hours():
    idx = pd.date_range('2024-01-01', periods=240, freq='h', tz='UTC')
    rows = np.arange(240)
    correct = np.where(idx.hour < 6, 1.0, 0.5 + 0.0 * idx.hour).astype(float)
    correct[(idx.hour >= 6) & (np.arange(240) % 2 == 0)] = 1.0
    correct[(idx.hour >= 6) & (np.arange(240) % 2 == 1)] = 0.0
    hc = SD.hour_concentration(idx, rows, correct)
    assert hc['flag'] and hc['share_00_06'] > 0.5
