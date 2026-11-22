"""fx-bb-reversion, group 1: the family config."""
import copy

from src.fx_bb import GROSS_HEADER, config_digest, feature_list_digest, load_config

CFG = load_config()


def test_config_loads_with_the_registered_settings():
    assert CFG['pairs'] == ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
    assert {p for p, s in CFG['sources'].items() if s == 'fx_session'} == {'EURUSD', 'GBPUSD', 'AUDUSD'}
    assert CFG['pip']['USDJPY'] == 0.01 and CFG['pip']['EURUSD'] == 0.0001
    d = CFG['data']
    assert d['start']['USDCHF'] == '2015-02-01'
    assert {v for p, v in d['start'].items() if p != 'USDCHF'} == {'2012-01-01'}
    assert d['known_gaps']['USDCAD'] == ['2013-05'] and d['min_h1_per_d1'] == 12 and d['max_spread_pips'] == 50
    assert d['label_tz'] == 'Europe/Berlin' and d['day_tz'] == 'America/New_York' and d['day_close_hour'] == 17
    assert CFG['reference_quorum'] == 4
    assert len(CFG['features']['columns']) == len(set(CFG['features']['columns'])) == 10
    f = CFG['folds']
    assert (f['first_eval_start'], f['fold_months'], f['val_fraction'], f['purge_days'], f['embargo_days']) == \
        ('2015-01-01', 12, 0.15, 5, 5)
    assert 'Gross of spread and slippage' in GROSS_HEADER


def test_band_horizon_and_cooldown_are_the_published_values():
    assert (CFG['bands']['length'], CFG['bands']['width'], CFG['horizon_days'], CFG['cooldown_days']) == (20, 2, 3, 3)
    assert CFG['bands']['ddof'] == 0 and CFG['atr_period'] == 20


def test_alpha_and_block():
    sc = CFG['scoring']
    assert sc['family_size'] == 2 and sc['alpha'] == 0.05 / 2 and sc['block_days'] == 5
    assert sc['resamples'] == 10000 and sc['power'] == 0.8 and sc['parity_tolerance_atr'] == 0.02


def test_lightgbm_is_the_small_sample_setting():
    lg = CFG['lightgbm']
    assert lg['objective'] == 'binary' and lg['num_leaves'] == 4 and lg['min_data_in_leaf'] == 20
    assert lg['learning_rate'] == 0.02 and lg['seed'] == 42 and lg['deterministic'] is True


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other['bands']['width'] = 2.5
    assert config_digest(other) != config_digest(CFG) and len(feature_list_digest(CFG)) == 64


def test_no_class_weights_mlp_or_search_settings():
    flat = str(CFG).lower()
    for word in ('class_weight', 'is_unbalance', 'scale_pos_weight', 'grid', 'mlp'):
        assert word not in flat
