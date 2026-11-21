"""fx-fibonacci-mlp: the family config loads and holds the fixed, unsearched settings."""
import math

from src.fx_fib import config_digest, feature_list_digest, load_config

CFG = load_config()


def test_pairs_and_fixed_parameters():
    assert CFG['pairs'] == ['EURUSD', 'GBPUSD']
    assert CFG['swings']['k_atr_d1'] == 1.5 and CFG['swings']['k_atr_h1'] == 2.0
    assert CFG['labels']['m_atr_d1'] == 1.0 and CFG['labels']['horizon_bars'] == 24
    assert CFG['mlp']['seeds'] == [11, 22, 33, 44, 55] and CFG['mlp']['allow_cpu'] is False
    assert CFG['scoring'] == {'alpha': 0.0125, 'block_bars': 24, 'resamples': 10000, 'seed': 20261121}


def test_both_positive_spiral_constants_with_their_rings():
    phi = (1 + math.sqrt(5)) / 2
    v = {x['name']: x for x in CFG['spiral']['variants']}
    assert abs(v['spiral306']['k'] - 2 / math.pi * math.log(phi)) < 1e-6 and v['spiral306']['rings'] == [0, 1, 2]
    assert abs(v['spiral077']['k'] - math.log(phi) / (2 * math.pi)) < 1e-6 and v['spiral077']['rings'] == [2, 3, 4]


def test_alpha_is_the_family_bonferroni_bar():
    assert CFG['scoring']['alpha'] == 0.05 / 4


def test_feature_columns_are_unique_and_digests_are_stable():
    cols = CFG['features']['columns']
    assert len(cols) == len(set(cols)) == 42
    assert {'d1_swing_bars', 'h1_swing_bars'} <= set(cols)
    assert config_digest(CFG) == config_digest(load_config())
    assert len(feature_list_digest(CFG)) == 64


def test_config_digest_ignores_key_order():
    a = {'x': 1, 'y': {'b': 2, 'a': 1}}
    b = {'y': {'a': 1, 'b': 2}, 'x': 1}
    assert config_digest(a) == config_digest(b)
