"""The chart file for mql5/Indicators/ProphetGoldNYRange.mq5 (spec gold-nyrange-forward/chart-display; design D9).

    python -m src.gold_nyfwd.export      rebuild the file now

Rebuilds ONE file, %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\EURUSDProphet\\gold_ny_range.tsv (override the
folder with MT5_COMMON_FILES), through a temporary file and os.replace. It is the family's only write
outside the repository, and nothing in the family reads it back.

It holds the last `display_days` forecast days: the band of +-F/2 around the 14:00 close for the bars
labelled 15:00-22:00 Berlin, and after settlement the realised high and low. Every time is a Berlin bar
label ("YYYY.MM.DD HH:MM", the chart's own server clock) and every visible text is built here, in
Bulgarian and English. Size only: no direction, no trade word, no promise.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import load_config
from .live import read_log
from .registry import verdict

COMMON_ENV = 'MT5_COMMON_FILES'
FIELDS = ['date', 'status', 'band_start', 'band_end', 'close14', 'band_lo', 'band_hi', 'size', 'real_high',
          'real_low', 'label_bg', 'label_en']
STATUS_BG = {'ok': 'прогноза', 'unverified clock week': 'часовникът не е потвърден',
             'missing pre-session bar': 'липсва бар преди сесията', 'no dollar data': 'няма данни за долара',
             'warm-up': 'загряване'}


def export_path(cfg: dict, env=None) -> str:
    env = os.environ if env is None else env
    base = env.get(COMMON_ENV)
    if not base:
        appdata = env.get('APPDATA')
        if not appdata:
            raise RuntimeError(f'no APPDATA and no {COMMON_ENV}: cannot find the MT5 Common\\Files folder')
        base = os.path.join(appdata, 'MetaQuotes', 'Terminal', 'Common', 'Files')
    return os.path.join(base, cfg['export']['subdir'], cfg['export']['file'])


def lab(t) -> str:
    return pd.Timestamp(t).strftime('%Y.%m.%d %H:%M')


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, str) and not x):
        return ''
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ''
    return f'{v:.{nd}f}' if np.isfinite(v) else ''


def forward_count(cfg: dict, committed=None) -> int:
    st = read_log(cfg, 'settlements')
    if not len(st):
        return 0
    ok = st[st['status'] == 'ok']
    if committed is not None:
        ok = ok[ok['date'] > committed]
    return int(len(ok))


def committed_day(cfg: dict):
    """The pre-registration's commit date on the label clock, or None when it cannot be read."""
    try:
        from .prereg import gate
        from .feed import label_now
        return label_now(pd.Timestamp(gate(cfg)).tz_convert('UTC')).normalize()
    except Exception:
        return None


def texts(cfg: dict, n: int, v: str) -> dict:
    need = int(cfg['forward']['min_days'])
    if v:
        st = (f'Злато · размер на NY сесията · тест напред: {n} дни · присъда: {v}',
              f'Gold · NY session size · forward test: {n} days · verdict: {v}')
    else:
        st = (f'Злато · размер на NY сесията · тест напред: {n} / {need} дни · още няма присъда',
              f'Gold · NY session size · forward test: {n} / {need} days · no verdict yet')
    return {'status': st,
            'note': ('Само очакван размер, не посока. Брутно, на bid цени.',
                     'Expected size only, not direction. Gross, on bid prices.'),
            'other_symbol': ('Този индикатор е само за GOLD.', 'This indicator is for GOLD only.')}


def build(cfg: dict, now=None, committed=None) -> str:
    v = verdict(cfg)
    n = forward_count(cfg, committed)
    lines = ['\t'.join(['#meta', 'version=1', f'generated_utc={datetime.now(timezone.utc):%Y.%m.%d %H:%M}',
                        f'forward_days={n}', f"min_days={cfg['forward']['min_days']}",
                        f"verdict={v or 'none'}"])]
    for k, (bg, en) in texts(cfg, n, v).items():
        lines.append('\t'.join(['#text', k, bg, en]))
    lines.append('\t'.join(FIELDS))
    fc = read_log(cfg, 'forecasts')
    st = read_log(cfg, 'settlements')
    if len(fc):
        fc = fc.sort_values('date').tail(int(cfg['export']['display_days']))
        for _, f in fc.iterrows():
            d = f['date']
            row = {k: '' for k in FIELDS}
            row.update(date=d.strftime('%Y.%m.%d'), status=str(f['status']))
            if f['status'] == 'ok':
                F, c14 = float(f['F']), float(f['close14'])
                row.update(band_start=lab(d + pd.Timedelta(hours=15)), band_end=lab(d + pd.Timedelta(hours=22)),
                           close14=num(c14), band_lo=num(c14 - F / 2), band_hi=num(c14 + F / 2), size=num(F),
                           label_bg=f'Очакван размер на NY сесията: {F:.2f} ({F / c14:.2%}) · размер, не посока',
                           label_en=f'Expected NY session size: {F:.2f} ({F / c14:.2%}) · size, not direction')
                s = st[st['date'] == d] if len(st) else st
                if len(s):
                    s = s.iloc[0]
                    row.update(real_high=num(s.get('realised_high')), real_low=num(s.get('realised_low')))
                    if row['real_high'] and row['real_low']:
                        real = float(s['realised_high']) - float(s['realised_low'])
                        row['label_bg'] += f' · реален: {real:.2f}'
                        row['label_en'] += f' · realised: {real:.2f}'
            else:
                row.update(label_bg=f"Няма прогноза: {STATUS_BG.get(f['status'], f['status'])}",
                           label_en=f"No forecast: {f['status']}")
            lines.append('\t'.join(row[k] for k in FIELDS))
    return '\n'.join(lines) + '\n'


def write_atomic(path: str, text: str, replace=os.replace, sleep=time.sleep, tries: int = 5) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    for attempt in range(tries):
        try:
            replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            sleep(0.2)


def rebuild(cfg: dict | None = None, now=None, env=None, committed=None) -> str:
    cfg = cfg or load_config()
    p = export_path(cfg, env)
    write_atomic(p, build(cfg, now, committed if committed is not None else committed_day(cfg)))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='rebuild the gold-nyrange-forward chart file').parse_args(argv)
    print(rebuild())


if __name__ == '__main__':
    main()
