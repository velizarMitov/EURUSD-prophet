"""Window-strategy settings (openspec change gold-window-strategy, task 1.1)."""
from src import gold_window as W

CFG = W.load_config()


def test_exactly_four_fixed_rules():
    assert CFG['family'] == 'gold_window'
    assert CFG['rules_order'] == ['h1_gated_direction', 'continuation', 'us_open_breakout', 'd1_gated_direction']
    assert set(CFG['rules']) == set(CFG['rules_order']) | {'gate_lookback_days'}


def test_window_is_the_owners_window_on_the_berlin_label_clock():
    # 15:30-23:00 Sofia = labels 14:30-22:00 Berlin; the last M15 bar closes at 22:00
    w = CFG['window']
    assert (w['first_m15_label'], w['last_m15_label']) == ('14:30', '21:45')
    assert w['h1_labels'] == list(range(14, 22)) and w['weekdays'] == [0, 1, 2, 3, 4]


def test_rule_parameters_are_fixed():
    r = CFG['rules']
    assert r['gate_lookback_days'] == 250
    assert r['h1_gated_direction'] == {'as_of_labels': [14, 15, 16, 17, 18, 19, 20], 'gate_quantile': 0.80}
    assert r['continuation'] == {'from_label': 21, 'to_label': 14, 'exit_label': 21, 'gate_quantile': 0.50}
    assert r['us_open_breakout'] == {'range_closes': ['14:45', '15:30'], 'entry_closes': ['15:45', '19:00'],
                                     'exit_label': 21, 'gate_quantile': 0.50}
    assert r['d1_gated_direction'] == {'entry_label': 14, 'exit_label': 21, 'gate_quantile': 0.50}


def test_bootstrap_costs_and_walkforward():
    assert CFG['stats'] == {'ci': 0.95, 'block': 20, 'n_boot': 2000, 'seed': 42}
    assert CFG['costs'] == {'floor_round_trip_pct': 0.0062, 'stress_round_trip_pct': 0.0122}
    assert CFG['walkforward'] == {'warmup_years': 3, 'models': ['h1_lstm', 'd1_lstm', 'd1_garch']}


def test_no_registry_registration_or_production_path():
    for key, rel in CFG['paths'].items():
        assert rel.startswith(('results/gold/window', 'research_models/gold/window')), (key, rel)
        assert 'hypothesis_log' not in rel and 'PRE_REGISTRATION' not in rel, (key, rel)
    flat = repr(CFG)
    assert 'alpha' not in flat and 'archive' not in flat and 'registry' not in flat
