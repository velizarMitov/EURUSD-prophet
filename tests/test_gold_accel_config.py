"""Config and isolation of the acceleration study (openspec change gold-acceleration-direction)."""
import os
import re

from src.gold_accel import REPO, load_config

PKG = os.path.join(REPO, 'src', 'gold_accel')


def _code(src):
    src = re.sub(r'(\'\'\'|""").*?\1', '', src, flags=re.S)
    return '\n'.join(l.split('#', 1)[0] for l in src.splitlines())


def _sources():
    for root, _dirs, files in os.walk(PKG):
        for f in files:
            if f.endswith('.py'):
                with open(os.path.join(root, f), encoding='utf-8') as fh:
                    yield f, _code(fh.read())


def test_features_and_models():
    cfg = load_config()
    f = cfg['features']
    assert f['h1'] == ['accel_1h', 'activity_rel', 'accel_expected_next'] and f['m15'] == ['accel_m15']
    assert f['profile_days'] == 60 and f['calendar'] == ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    m = cfg['models']
    assert set(m) == {'h1_gbm', 'h1_gbm_accel', 'h1_lstm', 'h1_lstm_accel', 'h1_accel', 'h1_gbm_accel_m15',
                      'h1_accel_m15'}
    assert m['h1_gbm_accel'] == {'base': 'h1_gbm', 'columns': 'base+h1', 'first_year': 2018, 'pair': 'h1_gbm'}
    assert m['h1_lstm_accel'] == {'base': 'h1_lstm', 'columns': 'base+h1', 'first_year': 2018, 'pair': 'h1_lstm'}
    assert m['h1_accel']['columns'] == 'h1+calendar'
    assert m['h1_gbm_accel_m15']['first_year'] == 2024 and m['h1_accel_m15']['first_year'] == 2024
    assert sorted(cfg['run_order']) == sorted(m)


def test_walkforward_stats_costs_window():
    cfg = load_config()
    assert cfg['walkforward'] == {'warmup_years': 3, 'm15_warmup_years': 1}
    assert cfg['stats'] == {'ci': 0.95, 'block_rows': 120, 'block_days': 20, 'n_boot': 2000, 'seed': 42}
    assert cfg['costs'] == {'floor_round_trip_pct': 0.0062, 'stress_round_trip_pct': 0.0122}
    assert cfg['window']['as_of_labels'] == [14, 15, 16, 17, 18, 19, 20]
    assert cfg['label'] == 'descriptive: every year has been seen; not evidence'


def test_paths_stay_in_the_accel_folders():
    p = load_config()['paths']
    for key in ('models_dir', 'inputs_dir'):
        assert p[key].startswith('research_models/gold/accel'), key
    for key in ('results', 'backtest', 'artifact_manifest'):
        assert p[key].startswith('results/gold/accel'), key


def test_no_production_write_target_registry_or_logger():
    banned = re.compile(r'src\.gold\.(versions|prereg\w*|forward_logger|run_task|retrain_task|mt5_export)'
                        r'|from\s+src\.gold\s+import\s+(versions|prereg\w*|forward_logger)')
    for f, src in _sources():
        # a path ('models/...' or a join component 'models',); cfg['models'] is a config key
        assert not re.search(r"""(?<!\[)['"]models(?:[/\\]|['"]\s*[,)])""", src), f'{f} names models/'
        assert not re.search(r'(?<![\w_])config\.json', src), f'{f} names config.json'
        assert 'gold_config.json' not in src, f'{f} names gold_config.json'
        assert not banned.search(src), f'{f}: {banned.search(src).group(0)}'


def test_no_trade_wording():
    for f, src in _sources():
        low = src.lower()
        for word in ('buy', 'sell', 'купи', 'продай', 'order_send'):
            assert word not in low, (f, word)
