"""rts-signals task 3.1: rule numbers and digest, the New York -> Berlin label clock, the README."""

import copy

import pandas as pd

from src.rts_signals.rules import load_rules, ny_to_label, readme, rules_digest, session

U = lambda s: pd.Timestamp(s, tz='UTC')  # noqa: E731


def test_numbers():
    r = load_rules()
    assert r['rule0']['proportionality_points'] == 15
    assert (r['rule2']['trigger_points'], r['rule2']['stop_points'], r['rule2']['target_points']) == (40, 40, 60)
    assert r['rule2']['friday_spot'] is False
    assert (r['rule3']['fallback_stop_points'], r['rule3']['fallback_target_points'], r['rule3']['max_losses_spot']) \
        == (38, 44, 3)
    assert (r['rule4']['trigger_points'], r['rule4']['stop_points'], r['rule4']['target_points']) == (80, 44, 48)
    assert (r['rule5']['gap_points'], r['rule5']['extra_points']) == (40, 20)
    assert r['clock']['friday_spot_close_ny'] == '12:00'


def test_normal_november_week():
    assert ny_to_label('2026-11-24', '18:00') == U('2026-11-25 00:00')
    assert ny_to_label('2026-11-24', '16:00') == U('2026-11-24 22:00')


def test_late_october_mismatch_week():
    # EU back on winter time 2026-10-25, US only on 2026-11-01
    assert ny_to_label('2026-10-27', '18:00') == U('2026-10-27 23:00')


def test_sessions():
    s = session('2026-11-25')            # Wednesday -> Thursday session
    assert s['open'] == U('2026-11-26 00:00') and s['end'] == U('2026-11-26 22:00') and not s['friday_close']
    s = session('2026-11-19')            # Thursday -> Friday, noon close
    assert s['trade_date'] == pd.Timestamp('2026-11-20') and s['friday_close']
    assert s['end'] == U('2026-11-20 18:00')
    s = session('2026-11-20')            # Friday -> Monday
    assert s['trade_date'] == pd.Timestamp('2026-11-23') and s['end'] == U('2026-11-23 22:00')


def test_readme_has_every_section():
    t = readme()
    for k in ('Rule 0', 'Proportionality', 'Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5', 'NPR', 'Friday close',
              'hypothetical and gross of cost'):
        assert k in t, k
    assert 'Пример' in t and 'Example' in t


def test_digest_changes_with_a_number():
    r = load_rules()
    r2 = copy.deepcopy(r)
    r2['rule2']['target_points'] = 61
    assert rules_digest(r) == rules_digest(load_rules()) != rules_digest(r2)
