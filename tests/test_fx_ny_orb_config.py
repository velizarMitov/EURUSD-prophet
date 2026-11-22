"""fx-ny-orb-hot, group 1: the family config."""
import copy

from src.fx_ny_orb import COST_HEADER, config_digest, load_config, minutes

CFG = load_config()


def test_rule_values_are_the_owners():
    r = CFG['rule']
    assert r['or_bars'] == ['14:00', '14:15', '14:30', '14:45'] and r['breakout_from'] == '15:00'
    assert (r['entry_before'], r['last_bar']) == ('19:00', '20:45')
    assert (r['or_width_max_f'], r['breakout_max_f'], r['stop_f'], r['target_f']) == (0.50, 0.25, 0.25, 0.60)
    assert r['slippage_bps'] == 0.4 and r['tie_rule'] == 'stop_wins'
    assert minutes('14:45') == 885


def test_window_forward_and_scoring():
    assert (CFG['data']['window_start'], CFG['data']['window_end']) == ('2018-08-08', '2026-08-07')
    assert CFG['frozen']['fit_end'] == '2026-11-20'
    assert (CFG['forward']['min_days'], CFG['forward']['deadline']) == (270, '2028-01-31')
    sc = CFG['scoring']
    assert sc['alpha'] == 0.05 / 2 and sc['family_size'] == 2 and sc['block_days'] == 5
    assert sc['resamples'] == 10000 and sc['declared_sd_r'] == 1.5
    assert sc['seed_historical'] != sc['seed_forward']
    assert CFG['export']['file'] == 'ny_orb.tsv' and CFG['export']['display_days'] == 60
    assert 'not a trading result' in COST_HEADER


def test_digest_is_stable_and_sensitive():
    assert config_digest(CFG) == config_digest(copy.deepcopy(CFG))
    other = copy.deepcopy(CFG)
    other['rule']['target_f'] = 0.70
    assert config_digest(other) != config_digest(CFG)


def test_no_search_settings():
    flat = str({k: v for k, v in CFG.items() if k != 'paths'}).lower()
    for word in ('grid', 'search', 'optim', 'class_weight', 'lot_size', 'risk_pct'):
        assert word not in flat
