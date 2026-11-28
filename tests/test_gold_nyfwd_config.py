"""gold-nyrange-forward, group 1: the family config."""
import copy

from src.gold_nyfwd import config_digest, load_config


def test_config_loads_with_the_registered_values():
    c = load_config()
    assert c['paths'] == {'results': 'results/gold_nyfwd', 'artifacts': 'research_models/gold_nyfwd'}
    assert c['mt5']['gold_symbol'] == 'GOLD' and c['mt5']['contract_prefix'] == 'USDInd'
    assert c['mt5']['roll_margin_trading_days'] == 2 and c['mt5']['h1_bars'] == 4000
    assert c['lag_check']['lags'] == [-3, -2, -1, 0, 1, 2, 3]
    assert c['forward']['forecast_after'] == '15:00' and c['forward']['settle_after'] == '22:00'
    assert c['forward']['min_days'] == 60 and c['forward']['deadline'] == '2027-04-30'
    assert c['scoring'] == {'alpha': 0.05, 'family_size': 1, 'block_days': 5, 'resamples': 10000, 'seed': 20261201}
    assert c['export'] == {'subdir': 'EURUSDProphet', 'file': 'gold_ny_range.tsv', 'display_days': 60}


def test_digest_is_stable_and_changes_with_the_config():
    a, b = load_config(), load_config()
    assert config_digest(a) == config_digest(b) and len(config_digest(a)) == 64
    c = copy.deepcopy(a)
    c['forward']['min_days'] = 61
    assert config_digest(c) != config_digest(a)
