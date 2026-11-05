"""Release calendars (openspec change gold-market-analytics, task 2.3)."""
from datetime import datetime, timezone

import pytest

from src.gold_analytics import load_config
from src.gold_analytics.events import calendar

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = {**c['paths'], 'events': str(tmp_path / 'events.csv')}
    return c


def fake_fred(release_id):
    return {10: ['2026-01-13', '2026-03-11', '2026-07-14', '2026-10-28'],
            50: ['2026-01-09', '2026-03-06']}[release_id]


def _sofia(cal, event, day):
    ev = cal['events']
    return ev.loc[(ev['event'] == event) & (ev['date'] == day), 'sofia'].item()


def test_cpi_in_winter_summer_and_the_mismatch_weeks(cfg):
    cal = calendar(cfg, client=fake_fred, now=NOW)
    assert cal['fred'] == 'ok' and cal['cache_age_days'] == 0.0
    assert _sofia(cal, 'cpi', '2026-01-13') == '15:30'
    assert _sofia(cal, 'cpi', '2026-07-14') == '15:30'
    assert _sofia(cal, 'cpi', '2026-03-11') == '14:30'   # US on summer time, Europe not yet
    assert _sofia(cal, 'cpi', '2026-10-28') == '14:30'   # Europe back on winter time, the US not yet
    assert _sofia(cal, 'nfp', '2026-01-09') == '15:30'


def test_fomc_comes_from_the_repo_csv_at_14_new_york(cfg):
    cal = calendar(cfg, client=fake_fred, now=NOW)
    fomc = cal['events'][cal['events']['event'] == 'fomc']
    assert len(fomc) > 100
    assert _sofia(cal, 'fomc', '2026-01-28') == '21:00'


def test_fred_failure_uses_the_cache_and_reports_its_age(cfg):
    calendar(cfg, client=fake_fred, now=NOW)

    def down(_):
        raise OSError('FRED unreachable')

    later = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    cal = calendar(cfg, client=down, now=later)
    assert cal['fred'] == 'cache' and cal['cache_age_days'] == 3.0
    assert _sofia(cal, 'cpi', '2026-07-14') == '15:30'


def test_without_fred_or_cache_only_fomc_remains(cfg):
    def down(_):
        raise RuntimeError('FRED_API_KEY is not set')

    cal = calendar(cfg, client=down, now=NOW)
    assert cal['fred'] == 'missing' and cal['cache_age_days'] is None
    assert set(cal['events']['event']) == {'fomc'}


def test_a_fresh_cache_is_not_refetched(cfg):
    calendar(cfg, client=fake_fred, now=NOW)
    calls = []

    def counting(rid):
        calls.append(rid)
        return fake_fred(rid)

    cal = calendar(cfg, client=counting, now=datetime(2026, 11, 4, 18, 0, tzinfo=timezone.utc))
    assert calls == [] and cal['fred'] == 'ok'
