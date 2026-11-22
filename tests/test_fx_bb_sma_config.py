"""fx-bb-sma-exit, group 1: the family config."""
import copy

from src.fx_bb import load_config as bb_config
from src.fx_bb_sma import COST_HEADER, bb_view, config_digest, feature_list_digest, load_config

CFG = load_config()
BB = bb_config()


def test_event_data_and_feature_blocks_equal_fx_bb():
    for key in ('pairs', 'reference_pair', 'sources', 'pip', 'data', 'bands', 'atr_period', 'reference_quorum',
                'features'):
        assert CFG[key] == BB[key], key
    assert CFG['paths_read'] == {'data': BB['paths']['data'], 'fx_session_data': BB['paths']['fx_session_data']}
    assert CFG['lightgbm'] == BB['lightgbm']
    v = bb_view(CFG)
    assert v['paths']['data'] == 'results/fx_bb/data' and v['bands'] == BB['bands']


def test_trade_rule_values():
    t = CFG['trade']
    assert t['time_stop_days'] == 10 and t['stop_multiple'] == 1.0 and t['tie_rule'] == 'stop_wins'
    assert t['take_profit'] == 'sma20_at_entry' and t['exclusion_window'] == [-19, 10]
    assert CFG['data']['start']['USDCHF'] == '2015-02-01'


def test_alpha_block_and_power_settings():
    sc = CFG['scoring']
    assert sc['family_size'] == 2 and sc['alpha'] == 0.05 / 2 and sc['block_days'] == 10
    assert sc['declared_sd_r'] == 1.0 and sc['parity_tolerance_r'] == 0.05 and sc['resamples'] == 10000
    f = CFG['folds']
    assert (f['first_eval_start'], f['fold_months'], f['purge_days'] + f['embargo_days'], f['gap_from']) == \
        ('2015-01-01', 12, 10, 'exit_date')
    assert 'recorded bar spread only' in COST_HEADER and 'not a trading result' in COST_HEADER


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other['trade']['time_stop_days'] = 15
    assert config_digest(other) != config_digest(CFG) and len(feature_list_digest(CFG)) == 64


def test_no_class_weights_or_search():
    flat = str(CFG).lower()
    for word in ('class_weight', 'is_unbalance', 'scale_pos_weight', 'grid', 'mlp'):
        assert word not in flat
