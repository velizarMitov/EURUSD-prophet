"""The spread cache refresh (openspec change gold-market-analytics, task 2.2)."""
import os
from datetime import date, datetime

import pandas as pd
import pytest

from src.gold_analytics import load_config
from src.gold_analytics.data import Source
from src.gold_analytics.refresh import (_guard, _write_gz, bucket, last_trading_days, read_spread,
                                        refresh)
from tests.gold_analytics_fakes import FakeMT5, rates

TODAY = date(2026, 11, 4)


def _no_fred(_):
    raise OSError('offline in tests')


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    cache = tmp_path / 'research_models' / 'gold' / 'analytics'
    c['paths'] = {**c['paths'], 'cache_dir': str(cache),
                  **{k: str(cache / c['paths'][k].split('/')[-1])
                     for k in ('bars_m1', 'bars_m15', 'spread_5m', 'events', 'last_tick')}}
    return c


def test_bucket_median_and_p90_match_hand_values():
    t = pd.to_datetime(['2026-11-03 14:00:01', '2026-11-03 14:01:00', '2026-11-03 14:02:00', '2026-11-03 14:04:59',
                        '2026-11-03 14:05:00'], utc=True)
    ticks = pd.DataFrame({'time': t, 'bid': [100.0] * 5, 'ask': [100.2, 100.4, 100.3, 101.0, 100.5]})
    b = bucket(ticks)
    first = b.iloc[0]
    # spreads 0.2, 0.4, 0.3, 1.0 -> median 0.35; p90 by linear interpolation = 0.4 + 0.7*0.6 = 0.82
    assert first['slot'] == pd.Timestamp('2026-11-03 14:00', tz='UTC') and first['ticks'] == 4
    assert first['median'] == pytest.approx(0.35) and first['p90'] == pytest.approx(0.82)
    assert b.iloc[1]['ticks'] == 1 and b.iloc[1]['median'] == pytest.approx(0.5)
    assert list(b.columns) == ['day', 'slot', 'median', 'p90', 'ticks']


def test_249_of_250_cached_pulls_exactly_one_day(cfg):
    days = last_trading_days(250, TODAY)
    missing = days[100]
    cached = pd.DataFrame({'day': [d.isoformat() for d in days if d != missing],
                           'slot': [pd.Timestamp(d, tz='UTC') + pd.Timedelta(hours=14) for d in days if d != missing],
                           'median': 0.3, 'p90': 0.5, 'ticks': 50})
    _write_gz(cached, cfg['paths']['spread_5m'])
    rows = [(pd.Timestamp(missing) + pd.Timedelta(hours=15, seconds=s), 4000.0, 4000.4) for s in range(0, 600, 30)]
    fake = FakeMT5(tick_rows=rows, bars={FakeMT5.TIMEFRAME_M1: rates([]), FakeMT5.TIMEFRAME_M15: rates([])})
    res = refresh(250, Source(cfg, fake), cfg, now=datetime(2026, 11, 4, 12, 0), log=lambda *_: None, events_client=_no_fred)
    assert res['spread'] == {'pulled': 1, 'cached': 249, 'days': 250}
    assert {a.date() for a, _ in fake.tick_calls} == {missing} and len(fake.tick_calls) == 24
    sp = read_spread(cfg)
    assert sp['day'].nunique() == 250
    assert sp.loc[sp['day'] == missing.isoformat(), 'ticks'].sum() == 20
    # A second run pulls nothing.
    fake.tick_calls.clear()
    res = refresh(250, Source(cfg, fake), cfg, now=datetime(2026, 11, 4, 12, 0), log=lambda *_: None, events_client=_no_fred)
    assert res['spread']['pulled'] == 0 and fake.tick_calls == []


def test_a_holiday_with_no_ticks_is_not_pulled_again(cfg):
    fake = FakeMT5(tick_rows=[], bars={})
    refresh(3, Source(cfg, fake), cfg, now=datetime(2026, 11, 4, 12, 0), log=lambda *_: None, events_client=_no_fred)
    fake.tick_calls.clear()
    res = refresh(3, Source(cfg, fake), cfg, now=datetime(2026, 11, 4, 12, 0), log=lambda *_: None, events_client=_no_fred)
    assert res['spread']['pulled'] == 0


def test_nothing_is_written_outside_the_analytics_folder(cfg, tmp_path):
    times = pd.date_range('2026-11-03 14:00', periods=30, freq='min', tz='UTC')
    fake = FakeMT5(tick_rows=[(times[0], 1.0, 1.2)],
                   bars={FakeMT5.TIMEFRAME_M1: rates(times), FakeMT5.TIMEFRAME_M15: rates(times[::15])})
    res = refresh(5, Source(cfg, fake), cfg, now=datetime(2026, 11, 4, 12, 0), log=lambda *_: None, events_client=_no_fred)
    assert res['bars']['M1']['rows'] == 30 and res['bars']['M15']['rows'] == 2
    root = os.path.normcase(cfg['paths']['cache_dir'])
    written = [os.path.join(r, f) for r, _, fs in os.walk(tmp_path) for f in fs]
    assert written and all(os.path.normcase(p).startswith(root) for p in written), written
    with pytest.raises(RuntimeError):
        _guard(str(tmp_path / 'elsewhere.csv'), cfg)


def test_the_last_trading_days_skip_weekends_and_today():
    d = last_trading_days(3, date(2026, 11, 2))  # a Monday
    assert d == [date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30)]
