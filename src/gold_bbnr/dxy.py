"""The US dollar index, fetched once into the family's own cache (spec gold-bb-nyrange/dollar-index; design D9).

    python -m src.gold_bbnr.dxy --mt5-path "C:\\...\\terminal64.exe"   hourly DXY CFD from MT5 -> data/dxy_h1.csv
    python -m src.gold_bbnr.dxy --yahoo                                daily DX-Y.NYB from Yahoo -> data/dxy_d1.csv

MT5 is used for MARKET DATA ONLY: initialize, symbol_select (Market Watch), copy_rates_range, shutdown.
Only `time` and OHLC are kept. The broker labels are its Berlin wall clock stored UTC-localised as-is, like
every other cache. The Yahoo request carries only the generic client name.

`load_h1` checks the clock every time: for each calendar year, the correlation of DXY and EURUSD H1 log
returns (fx_session's cache, read-only) must be most negative at lag 0 among lags -1, 0, +1.
Both caches are pinned by digest in the pre-registration; nothing refetches them afterwards.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import repo_path

OHLC = ['open', 'high', 'low', 'close']


class DxyClockMismatch(RuntimeError):
    pass


def data_path(cfg: dict, key: str) -> str:
    return os.path.join(repo_path(cfg['paths']['data']), cfg['dxy'][key])


def _write_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    out.index.name = out.index.name or 'time'
    tmp = path + '.tmp'
    out.to_csv(tmp, float_format='%.6f', lineterminator='\n')
    os.replace(tmp, path)
    return path


# -- fetch (run once, before the pre-registration) ------------------------------------------------

def fetch_mt5(cfg: dict, terminal_path: str, mt5=None) -> str:
    if mt5 is None:
        import MetaTrader5 as mt5
    d = cfg['dxy']
    if not mt5.initialize(path=terminal_path, timeout=30000):
        raise RuntimeError(f'MT5 initialize failed: {mt5.last_error()}')
    try:
        if not mt5.symbol_select(d['mt5_symbol'], True):
            raise RuntimeError(f"{d['mt5_symbol']} is not available in this terminal")
        a = datetime.fromisoformat(d['h1_start']).replace(tzinfo=timezone.utc)
        b = (pd.Timestamp(cfg['data']['window_end']) + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=timezone.utc)
        rates = mt5.copy_rates_range(d['mt5_symbol'], mt5.TIMEFRAME_H1, a, b)
    finally:
        mt5.shutdown()
    if rates is None or not len(rates):
        raise RuntimeError('no DXY H1 bars returned')
    df = pd.DataFrame({k: rates[k] for k in ['time'] + OHLC})
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('time').sort_index().astype(float)
    df = df[(df.index >= pd.Timestamp(d['h1_start'], tz='UTC')) &
            (df.index < pd.Timestamp(cfg['data']['window_end'], tz='UTC') + pd.Timedelta(days=1))]
    return _write_csv(df, data_path(cfg, 'h1_file'))


def yahoo_request(cfg: dict) -> tuple:
    """(url, params, headers) of the one Yahoo chart request. Headers carry only the generic client name."""
    d = cfg['dxy']
    p1 = int(pd.Timestamp(d['d1_start'], tz='UTC').timestamp())
    p2 = int((pd.Timestamp(cfg['data']['window_end'], tz='UTC') + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{d['yahoo_ticker']}"
    return url, {'period1': p1, 'period2': p2, 'interval': '1d'}, {'User-Agent': d['client_name']}


def parse_yahoo(js: dict, cfg: dict) -> pd.DataFrame:
    res = js['chart']['result'][0]
    q = res['indicators']['quote'][0]
    tz = res['meta'].get('exchangeTimezoneName', 'America/New_York')
    dates = pd.to_datetime(res['timestamp'], unit='s', utc=True).tz_convert(tz).tz_localize(None).normalize()
    df = pd.DataFrame({'open': q['open'], 'high': q['high'], 'low': q['low'], 'close': q['close']},
                      index=pd.DatetimeIndex(dates, name='date')).astype(float)
    df = df[df['close'].notna()]
    df = df[~df.index.duplicated(keep='last')].sort_index()
    end = pd.Timestamp(cfg['data']['window_end'])
    return df[(df.index >= pd.Timestamp(cfg['dxy']['d1_start'])) & (df.index <= end)]


def fetch_yahoo(cfg: dict, get=None) -> str:
    if get is None:
        import requests
        get = requests.get
    url, params, headers = yahoo_request(cfg)
    r = get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return _write_csv(parse_yahoo(r.json(), cfg), data_path(cfg, 'd1_file'))


# -- load -----------------------------------------------------------------------------------------

def lag_check(dxy: pd.DataFrame, eur: pd.DataFrame, lags=(-1, 0, 1)) -> dict:
    """{year: {lag: corr}}; raises DxyClockMismatch unless lag 0 is the most negative in every year."""
    j = pd.concat([np.log(dxy['close']).diff().rename('dx'), np.log(eur['close']).diff().rename('eu')], axis=1)
    j = j[(j.index >= dxy.index.min()) & (j.index <= dxy.index.max())]
    out, bad = {}, []
    for y, g in j.groupby(j.index.year):
        c = {lag: float(g['dx'].corr(g['eu'].shift(lag))) for lag in lags}
        out[int(y)] = c
        if g['dx'].notna().sum() < 500 or min(c, key=lambda k: c[k]) != 0:
            bad.append(int(y))
    if bad:
        raise DxyClockMismatch(f'DXY vs EURUSD H1 correlation is not most negative at lag 0 in {bad}: {out}')
    return out


def _eurusd(cfg: dict) -> pd.DataFrame:
    from ..fx_session.data_loader import load_h1
    return load_h1(cfg['dxy']['lag_check_pair'], {'paths': {'data': cfg['paths_read']['fx_session_data']}})


def load_h1(cfg: dict, path: str | None = None, eur: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or data_path(cfg, 'h1_file'))
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time').sort_index()[OHLC].astype(float)
    lag_check(df, eur if eur is not None else _eurusd(cfg), cfg['dxy']['lag_check_lags'])
    return df


def load_d1(cfg: dict, path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or data_path(cfg, 'd1_file'), parse_dates=['date'])
    return df.set_index('date').sort_index()[OHLC].astype(float)


def main(argv=None):
    from . import load_config
    ap = argparse.ArgumentParser(description='fetch the dollar index once (gold-bb-nyrange)')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--mt5-path')
    g.add_argument('--yahoo', action='store_true')
    a = ap.parse_args(argv)
    cfg = load_config()
    print(fetch_yahoo(cfg) if a.yahoo else fetch_mt5(cfg, a.mt5_path))


if __name__ == '__main__':
    main()
