"""The read-only MT5 source (openspec change gold-market-analytics, task 1.2)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from src.gold_analytics import load_config
from src.gold_analytics.data import Source, month_chunks, year_chunks
from tests.gold_analytics_fakes import FakeMT5, rates


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = {k: str(tmp_path / v.split('/')[-1]) for k, v in c['paths'].items()}
    return c


def _write_h1(cfg, times):
    pd.DataFrame({'time': pd.to_datetime(times, utc=True), 'open': 1.0, 'high': 2.0, 'low': 0.5,
                  'close': 1.5}).to_csv(cfg['paths']['h1'], index=False)


def test_terminal_closed_falls_back_to_the_cache_stale(cfg):
    _write_h1(cfg, ['2026-11-03 10:00', '2026-11-03 11:00'])
    src = Source(cfg, FakeMT5(up=False))
    f = src.bars('H1', '2026-11-01', '2026-11-05')
    assert f.stale is True
    assert len(f.df) == 2 and f.as_of == pd.Timestamp('2026-11-03 11:00', tz='UTC')
    t = src.tick()
    assert t['stale'] is True and t['bid'] == 1.5 and t['as_of'] == '2026-11-03T12:00:00+00:00'


def test_last_live_tick_is_the_stale_fallback(cfg):
    live = FakeMT5(tick=SimpleNamespace(bid=4000.1, ask=4000.5, time_msc=1_762_000_000_000))
    t = Source(cfg, live).tick()
    assert (t['bid'], t['ask'], t['stale']) == (4000.1, 4000.5, False)
    t2 = Source(cfg, FakeMT5(up=False)).tick()
    assert (t2['bid'], t2['ask'], t2['stale'], t2['as_of']) == (4000.1, 4000.5, True, t['time'])


def test_volume_fields_never_reach_the_frame(cfg):
    times = pd.date_range('2026-11-02 09:00', periods=5, freq='h', tz='UTC')
    fake = FakeMT5(bars={FakeMT5.TIMEFRAME_H1: rates(times)})
    f = Source(cfg, fake).bars('H1', '2026-11-02', '2026-11-03')
    assert f.stale is False and len(f.df) == 5
    assert list(f.df.columns) == ['time', 'open', 'high', 'low', 'close']
    assert f.df['close'].tolist() == [2001.0, 2002.0, 2003.0, 2004.0, 2005.0]


def test_live_bars_top_up_the_cache_from_its_newest_label(cfg):
    _write_h1(cfg, ['2026-11-02 09:00', '2026-11-02 10:00'])
    times = pd.date_range('2026-11-02 10:00', periods=3, freq='h', tz='UTC')
    fake = FakeMT5(bars={FakeMT5.TIMEFRAME_H1: rates(times)})
    f = Source(cfg, fake).bars('H1', '2026-11-02', '2026-11-03')
    assert len(f.df) == 4 and f.stale is False
    # the live 10:00 bar replaced the cached one
    assert f.df.loc[f.df['time'] == pd.Timestamp('2026-11-02 10:00', tz='UTC'), 'close'].item() == 2001.0
    assert fake.rate_calls[0][1] == pd.Timestamp('2026-11-02 10:00', tz='UTC').to_pydatetime()


def test_a_long_range_is_chunked_by_year(cfg):
    fake = FakeMT5(bars={FakeMT5.TIMEFRAME_D1: rates(pd.date_range('2022-06-01', '2025-03-01', freq='7D', tz='UTC'))})
    df = Source(cfg, fake).fetch_bars('D1', '2022-06-01', '2025-03-01')
    starts = [(a.year, a.month, a.day) for _, a, _ in fake.rate_calls]
    assert starts == [(2022, 6, 1), (2023, 1, 1), (2024, 1, 1), (2025, 1, 1)]
    assert df['time'].is_monotonic_increasing and df['time'].is_unique
    assert len(year_chunks('2024-01-01', '2024-01-01')) == 0


def test_ticks_keep_only_bid_and_ask(cfg):
    rows = [('2026-11-03 14:00:01', 4000.0, 4000.3), ('2026-11-03 15:10:00', 4001.0, 4001.5)]
    f = Source(cfg, FakeMT5(tick_rows=rows)).ticks('2026-11-03 14:00', '2026-11-03 16:00')
    assert list(f.df.columns) == ['time', 'bid', 'ask'] and len(f.df) == 2
    assert len(Source(cfg, FakeMT5(tick_rows=rows)).fetch_ticks('2026-11-03 14:00', '2026-11-03 16:00')) == 2
    closed = Source(cfg, FakeMT5(up=False)).ticks('2026-11-03 14:00', '2026-11-03 16:00')
    assert closed.stale is True and closed.df.empty


def test_unknown_timeframe_is_refused(cfg):
    with pytest.raises(ValueError):
        Source(cfg, FakeMT5()).bars('H4')


def test_m1_is_chunked_by_month(cfg):
    """A year of M1 exceeds the terminal's bar cap, and MT5 then returns nothing."""
    fake = FakeMT5(bars={FakeMT5.TIMEFRAME_M1: rates(pd.date_range('2026-10-30 23:58', periods=5, freq='min', tz='UTC'))})
    df = Source(cfg, fake).fetch_bars('M1', '2026-09-15', '2026-11-02')
    assert [(a.month, a.day) for _, a, _ in fake.rate_calls] == [(9, 15), (10, 1), (11, 1)]
    assert len(df) == 5
    assert [(a.month, b.month) for a, b in month_chunks('2026-12-10', '2027-01-05')] == [(12, 1), (1, 1)]
