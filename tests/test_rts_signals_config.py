"""rts-signals task 1.1: the family config loads and its digest is stable."""

import copy

from src.rts_signals import CELL, config_digest, load_config, pair_by_symbol, sub


def test_config_loads_with_six_pairs():
    cfg = load_config()
    assert set(cfg['pairs']) == {'POUND', 'EURO', 'AUD', 'YEN', 'SWISS', 'CAD'}
    assert {p['symbol'] for p in cfg['pairs'].values()} == {'GBPUSD', 'EURUSD', 'AUDUSD', 'USDJPY', 'USDCHF',
                                                             'USDCAD'}
    assert cfg['pairs']['YEN']['factor'] == 100 and cfg['pairs']['SWISS']['factor'] == 10000
    assert cfg['forward'] == {'min_trades': 300, 'deadline': '2027-06-30'}
    sc = cfg['scoring']
    assert (sc['alpha'], sc['block_days'], sc['resamples'], sc['seed']) == (0.05, 5, 10000, 20261202)
    assert cfg['export']['display_days'] == 20 and cfg['task']['minute'] == 20
    assert CELL == 'rts_spot_net_r'
    assert pair_by_symbol(cfg)['USDJPY']['name'] == 'YEN'


def test_paths_are_family_trees():
    cfg = load_config()
    assert cfg['paths']['results'] == 'results/rts_signals'
    assert cfg['paths']['artifacts'] == 'research_models/rts_signals'
    for k in ('inbox', 'drop', 'secrets', 'cache'):
        assert sub(cfg, k).replace('\\', '/').endswith(f'research_models/rts_signals/{cfg["paths"][k]}')


def test_digest_is_stable_and_sensitive():
    cfg = load_config()
    assert config_digest(cfg) == config_digest(load_config())
    c2 = copy.deepcopy(cfg)
    c2['forward']['min_trades'] = 301
    assert config_digest(c2) != config_digest(cfg)
