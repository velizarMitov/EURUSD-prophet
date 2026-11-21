"""fx-session-ny, group 1: the family config."""
import copy

from src.fx_session import config_digest, feature_list_digest, load_config

CFG = load_config()


def test_config_loads_with_the_registered_settings():
    assert CFG['pairs'] == ['EURUSD', 'GBPUSD'] and CFG['factor_pairs'][-1] == 'AUDUSD'
    s = CFG['session']
    assert (s['entry_hour'], s['exit_hour'], s['data_hour']) == (14, 21, 14)
    assert CFG['data']['history_start'] == '2012-01-01' and CFG['data']['latest_allowed_start'] == '2012-01-31'
    assert CFG['data']['known_gaps'] == {'GBPUSD': ['2014-03']} and CFG['data']['min_h1_per_d1'] == 12
    assert CFG['folds']['first_eval_start'] == '2013-01-01' and CFG['folds']['embargo_days'] == 1
    assert CFG['mlp']['allow_cpu'] is False and len(CFG['mlp']['seeds']) == 5
    assert len(CFG['features']['columns']) == len(set(CFG['features']['columns'])) == 20


def test_alpha_is_bonferroni_over_four_cells():
    sc = CFG['scoring']
    assert sc['family_size'] == 4 and sc['alpha'] == 0.05 / 4 and sc['block_days'] == 5


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other['session']['entry_hour'] = 15
    assert config_digest(other) != config_digest(CFG)
    assert len(feature_list_digest(CFG)) == 64


def test_no_class_weights_or_search_settings():
    flat = str(CFG).lower()
    assert 'class_weight' not in flat and 'is_unbalance' not in flat and 'grid' not in flat
