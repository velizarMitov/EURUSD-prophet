"""fx-rates-2y, group 1: the family config."""
import copy

from src.fx_rates import config_digest, feature_list_digest, load_config

CFG = load_config()


def test_config_loads_with_the_registered_settings():
    assert CFG['pairs'] == ['EURUSD', 'GBPUSD'] and CFG['foreign_2y'] == {'EURUSD': 'de2y', 'GBPUSD': 'uk2y'}
    assert CFG['target']['horizon_days'] == 5 and CFG['alignment']['lag_fx_days'] == 1
    assert CFG['alignment']['fill_limit_business_days'] == 3
    assert CFG['data']['fx_history_start'] == '2012-01-01'
    f = CFG['folds']
    assert f['first_eval_start'] == '2015-01-01' and f['purge_days'] == 5 and f['embargo_days'] == 5
    assert len(CFG['features']['columns']) == len(set(CFG['features']['columns'])) == 12
    assert CFG['mlp']['allow_cpu'] is False and len(CFG['mlp']['seeds']) == 5


def test_alpha_and_block():
    sc = CFG['scoring']
    assert sc['family_size'] == 4 and sc['alpha'] == 0.05 / 4 and sc['block_days'] >= 5


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other['target']['horizon_days'] = 10
    assert config_digest(other) != config_digest(CFG) and len(feature_list_digest(CFG)) == 64


def test_no_class_weights_or_search_settings():
    flat = str(CFG).lower()
    assert 'class_weight' not in flat and 'is_unbalance' not in flat and 'grid' not in flat
