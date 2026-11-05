"""TWAP, arrival and news-hour windows (openspec change gold-market-analytics, task 3.4)."""
import numpy as np
import pandas as pd
import pytest

from src.gold_analytics import load_config
from src.gold_analytics.benchmarks import against, arrival, twap
from src.gold_analytics.events import news_windows

CFG = load_config()


def _m15(start, end):
    t = pd.date_range(start, end, freq='15min', tz='UTC', inclusive='left')
    t = t[t.weekday < 5]
    c = np.full(len(t), 2000.0)
    return pd.DataFrame({'time': t, 'open': c, 'high': c + 1, 'low': c - 1, 'close': c})


def test_twap_is_the_mean_of_the_closes_in_the_interval():
    bars = pd.DataFrame({'time': pd.date_range('2026-11-03 14:00', periods=6, freq='min', tz='UTC'),
                         'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': [10.0, 11, 12, 13, 14, 15]})
    t = twap(bars, '2026-11-03 14:01+00:00', '2026-11-03 14:04+00:00')
    assert t['twap'] == pytest.approx(12.0) and t['n_bars'] == 3
    assert against(13.2, 12.0) == {'usd': pytest.approx(1.2), 'pct': pytest.approx(10.0)}
    assert twap(bars, '2026-11-04 00:00+00:00', '2026-11-05 00:00+00:00')['twap'] is None


def test_arrival_is_the_close_of_the_bar_ending_at_the_start():
    bars = pd.DataFrame({'time': pd.date_range('2026-11-03 14:00', periods=200, freq='min', tz='UTC'),
                         'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 100.0 + np.arange(200)})
    a = arrival(bars, pd.Timestamp('2026-11-03 15:00', tz='UTC'), 1, [15, 30, 60, 120])
    # the bar ending at 15:00 is labelled 14:59: the 60th bar, close 159
    assert a['arrival'] == 159.0
    assert [m['price'] for m in a['moves']] == [174.0, 189.0, 219.0, 279.0]
    assert a['moves'][0]['usd'] == 15.0 and a['moves'][0]['pct'] == pytest.approx(15 / 159 * 100)


def test_the_event_comparison_uses_the_same_weekdays_slots():
    bars = _m15('2026-02-02', '2026-03-02')  # four weeks, winter: releases at 14:30 label = 15:30 Sofia
    fridays = sorted({d for d in bars['time'].dt.normalize() if d.weekday() == 4})
    releases = fridays[:2]

    def bump(day, pct):
        m = bars['time'] == day + pd.Timedelta(hours=14, minutes=45)   # the last M15 bar of [14:30, 15:00)
        bars.loc[m, 'close'] = 2000 * (1 + pct / 100)

    for d in releases:
        bump(d, 0.5)
    for d in fridays[2:]:
        bump(d, 0.1)
    for d in bars['time'].dt.normalize().unique():
        if pd.Timestamp(d).weekday() == 3:
            bump(pd.Timestamp(d), 3.0)   # Thursdays: large, and must not reach the Friday baseline
    rel = pd.DataFrame({'event': 'nfp', 'date': [d.strftime('%Y-%m-%d') for d in releases],
                        'label': [d + pd.Timedelta(hours=14, minutes=30) for d in releases], 'sofia': '15:30'})
    out = news_windows(bars, rel, CFG)
    assert out['n_releases'] == 2 and out['n_comparison_days'] == 2
    first = next(r for r in out['offsets'] if r['offset_min'] == 0)
    assert first['release_abs_mean'] == pytest.approx(0.5)
    assert first['baseline_abs_mean'] == pytest.approx(0.1)          # Fridays only, never the 3 % Thursdays
    assert first['n_release'] == 2 and first['n_baseline'] == 4      # 2 releases x 2 other Fridays
    assert first['ratio'] == pytest.approx(5.0)
    assert out['first_hour']['length_min'] == 60
    assert [r['offset_min'] for r in out['offsets']] == list(range(-120, 180, 30))


def test_news_windows_with_spread():
    bars = _m15('2026-02-02', '2026-02-14')
    fri = pd.Timestamp('2026-02-06', tz='UTC')
    rel = pd.DataFrame({'event': 'nfp', 'date': ['2026-02-06'], 'label': [fri + pd.Timedelta(hours=14, minutes=30)],
                        'sofia': '15:30'})
    slots = pd.date_range('2026-02-02', '2026-02-14', freq='5min', tz='UTC', inclusive='left')
    sp = pd.DataFrame({'day': slots.strftime('%Y-%m-%d'), 'slot': slots, 'median': 0.3, 'p90': 0.5, 'ticks': 10})
    sp.loc[(sp['slot'] >= rel['label'][0]) & (sp['slot'] < rel['label'][0] + pd.Timedelta(minutes=30)), 'median'] = 1.2
    out = news_windows(bars, rel, CFG, sp)
    first = next(r for r in out['offsets'] if r['offset_min'] == 0)
    assert first['release_spread_median'] == pytest.approx(1.2) and first['baseline_spread_median'] == pytest.approx(0.3)
