"""fx-ny-volatility, group 1: the family config."""
import copy

from src.fx_nyvol import GROSS_HEADER, config_digest, feature_list_digest, load_config, session_view
from src.fx_session import load_config as session_config

CFG = load_config()


def test_config_loads_with_the_registered_settings():
    assert CFG['pairs'] == ['EURUSD', 'GBPUSD'] and CFG['factor_pairs'] == ['EURUSD', 'GBPUSD', 'AUDUSD']
    s = CFG['session']
    assert s['target_hours'] == [15, 21] and s['data_hour'] == 14 and s['asia_hours'] == [0, 8]
    assert s['london_hours'] == [9, 13] and s['range_window_days'] == 20 and s['har_windows'] == [1, 5, 20]
    assert CFG['data']['max_spread_pips'] == 50 and CFG['data']['history_start'] == '2012-01-01'
    assert len(CFG['features']['columns']) == len(set(CFG['features']['columns'])) == 14
    assert CFG['features']['har_columns'] == CFG['features']['columns'][:3]
    f = CFG['folds']
    assert (f['first_eval_start'], f['fold_months'], f['val_fraction'], f['embargo_days']) == ('2013-01-01', 6, 0.15, 1)
    assert CFG['lightgbm']['objective'] == 'l2' and CFG['mlp']['allow_cpu'] is False and len(CFG['mlp']['seeds']) == 5
    assert 'Gross of spread and slippage' in GROSS_HEADER


def test_window_is_15_to_21_and_the_session_calendar_matches_fx_session():
    s, fs = CFG['session'], session_config()['session']
    assert (s['entry_hour'], s['exit_hour']) == (fs['entry_hour'], fs['exit_hour']) == (14, 21)
    assert CFG['data']['known_gaps'] == session_config()['data']['known_gaps']
    assert session_view(CFG)['paths']['data'] == 'results/fx_session/data'


def test_alpha_is_005_over_4():
    sc = CFG['scoring']
    assert sc['family_size'] == 4 and sc['alpha'] == 0.05 / 4 and sc['block_days'] == 5 and sc['resamples'] == 10000
    assert sc['hot_quantile'] == 0.75 and sc['hot_window_days'] == 60


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other['session']['target_hours'] = [15, 22]
    assert config_digest(other) != config_digest(CFG) and len(feature_list_digest(CFG)) == 64


def test_no_class_weights_or_search():
    flat = str(CFG).lower()
    for word in ('class_weight', 'is_unbalance', 'grid', 'binary'):
        assert word not in flat
