"""Config and isolation of the gold analytics page (openspec change gold-market-analytics)."""
import os
import re

from src.gold_analytics import REPO, load_config

PKG = os.path.join(REPO, 'src', 'gold_analytics')


def _code(src):
    """Source without docstrings and comments: the contract is stated in prose
    there, and only executable text can break it."""
    src = re.sub(r'(\'\'\'|""").*?\1', '', src, flags=re.S)
    return '\n'.join(l.split('#', 1)[0] for l in src.splitlines())


def _sources():
    for root, _dirs, files in os.walk(PKG):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, encoding='utf-8') as fh:
                    yield os.path.relpath(path, REPO).replace('\\', '/'), _code(fh.read())


def test_sessions_and_zones():
    cfg = load_config()
    assert cfg['sessions'] == [
        {'key': 'asia', 'start': '09:00', 'end': '15:00', 'zone': 'Asia/Tokyo'},
        {'key': 'london', 'start': '08:00', 'end': '16:30', 'zone': 'Europe/London'},
        {'key': 'new_york', 'start': '08:20', 'end': '13:30', 'zone': 'America/New_York'},
    ]
    assert cfg['overlap'] == {'key': 'overlap', 'of': ['london', 'new_york']}
    assert cfg['label_zone'] == 'Europe/Berlin' and cfg['display_zone'] == 'Europe/Sofia'
    assert cfg['owner_window'] == {'start': '15:30', 'end': '23:00'}


def test_intervals_timeframes_periods():
    cfg = load_config()
    assert cfg['intervals'] == [1, 5, 15, 30, 60]
    assert cfg['timeframes'] == ['1m', '5m', '15m', '30m', '1h', '4h', 'D1', 'W1']
    assert cfg['periods'] == [5, 20, 60, 120, 250]
    assert cfg['regime'] == {'window': 20, 'lookback': 250, 'low_q': 0.33, 'high_q': 0.67}
    assert cfg['arrival_horizons_min'] == [15, 30, 60, 120]


def test_event_windows():
    ev = load_config()['events']
    assert ev['nfp'] == {'source': 'fred', 'release_id': 50, 'time': '08:30', 'zone': 'America/New_York'}
    assert ev['cpi'] == {'source': 'fred', 'release_id': 10, 'time': '08:30', 'zone': 'America/New_York'}
    assert ev['fomc'] == {'source': 'fomc_csv', 'time': '14:00', 'zone': 'America/New_York'}
    assert load_config()['event_window'] == {'before_min': 120, 'after_min': 180, 'step_min': 30}


def test_caches_live_under_the_analytics_folder():
    paths = load_config()['paths']
    assert paths['cache_dir'] == 'research_models/gold/analytics'
    for key in ('bars_m1', 'bars_m15', 'spread_5m', 'events', 'last_tick'):
        assert paths[key].startswith('research_models/gold/analytics/'), key
    # The read-only inputs from the repo.
    assert (paths['d1'], paths['h1'], paths['fomc']) == (
        'results/gold/gold_d1.csv', 'results/gold/gold_h1.csv', 'results/fomc_dates.csv')


def test_no_trading_call_and_no_volume_field():
    banned = re.compile(r'\b(order_send|order_check|positions_get|orders_get|tick_volume|real_volume)\b')
    for rel, src in _sources():
        m = banned.search(src)
        assert not m, f'{rel} names {m.group(0)}'


def test_imports_only_the_symbol_resolver_from_gold():
    pattern = re.compile(r'^\s*(?:from\s+(src\.gold(?:\.[\w.]+)?)\s+import\s+([^\n]+)|import\s+(src\.gold\b[\w.]*))', re.M)
    for rel, src in _sources():
        for m in pattern.finditer(src):
            assert m.group(3) is None, f'{rel}: {m.group(0).strip()}'
            assert m.group(1) == 'src.gold.data' and m.group(2).strip() == 'resolve_gold_symbol', \
                f'{rel}: {m.group(0).strip()}'


def test_writes_nothing_under_models_or_config():
    for rel, src in _sources():
        assert not re.search(r"""['"]models[/\\'"]""", src), f'{rel} names models/'
        assert not re.search(r'(?<![\w_])config\.json', src), f'{rel} names config.json'
