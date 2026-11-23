"""gold-bb-nyrange, task 1.1: the config and its pinned blocks."""
import copy

from src.fx_bb import load_config as fx_bb_config
from src.fx_bb_sma import load_config as fx_bb_sma_config
from src.fx_nyvol import load_config as fx_nyvol_config
from src.gold_bbnr import config_digest, load_config

CFG = load_config()


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(load_config())
    other = copy.deepcopy(CFG)
    other['bands']['width'] = 2.5
    assert config_digest(other) != config_digest(CFG)


def test_hyperparameters_equal_the_currency_families():
    bb, sma, ny = fx_bb_config(), fx_bb_sma_config(), fx_nyvol_config()
    assert CFG['lightgbm_bb'] == bb['lightgbm'] == sma['lightgbm']
    assert CFG['bands'] == bb['bands'] == sma['bands']
    assert CFG['atr_period'] == bb['atr_period'] and CFG['cooldown_days'] == bb['cooldown_days']
    assert CFG['horizon_days'] == bb['horizon_days']
    assert CFG['trade']['time_stop_days'] == sma['trade']['time_stop_days']
    assert CFG['trade']['stop_multiple'] == sma['trade']['stop_multiple']
    assert CFG['lightgbm_ny'] == ny['lightgbm'] and CFG['mlp'] == ny['mlp']
    for k in ('rsi_period', 'sma_slow', 'atr_slow', 'width_rank_days', 'momentum_days'):
        assert CFG['features_bb'][k] == bb['features'][k]
    for k in ('fold_months', 'val_fraction', 'purge_days', 'embargo_days'):
        assert CFG['folds_bb'][k] == bb['folds'][k]
    for k in ('fold_months', 'val_fraction', 'embargo_days'):
        assert CFG['folds_ny'][k] == ny['folds'][k]


def test_feature_lists():
    bbf = CFG['features_bb']['columns']
    assert len(bbf) == 11 and 'simultaneous_frac' not in bbf
    assert bbf[:8] == [c for c in fx_bb_config()['features']['columns'] if c not in ('simultaneous_frac', 'breach_side')]
    ny = CFG['features_ny']['columns']
    assert len(ny) == 14 and 'dollar_pre_range_ratio' in ny and 'cross_pre_range_ratio' not in ny


def test_window_alphas_and_paths():
    assert CFG['data']['window_start'] == '2015-01-02' and CFG['data']['window_end'] == '2026-11-20'
    sc = CFG['scoring']
    assert sc['bb']['alpha'] == sc['bb_sma']['alpha'] == 0.025
    assert abs(sc['nyvol']['alpha'] - 0.05 / 3) < 1e-15
    assert (sc['bb']['block_days'], sc['bb_sma']['block_days'], sc['nyvol']['block_days']) == (5, 10, 5)
    assert CFG['folds_bb']['min_train'] == 60
    assert CFG['paths']['results'] == 'results/gold_bbnr' and CFG['paths']['artifacts'] == 'research_models/gold_bbnr'
