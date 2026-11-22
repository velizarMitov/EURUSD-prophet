"""The chart file for mql5/Indicators/ProphetNYOrb.mq5 (spec fx-ny-orb-hot/chart-display; design D8, D9).

    python -m src.fx_ny_orb.export      rebuild the file now

Rebuilds ONE file, %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\EURUSDProphet\\ny_orb.tsv (override the
folder with MT5_COMMON_FILES), through a temporary file and os.replace. Nothing in the repository reads
it back. It holds the last `display_days` forecast days plus today, per pair, with every price, every
time as a Berlin bar label ("YYYY.MM.DD HH:MM", the chart's own server clock) and every text shown, in
Bulgarian and English.

THE GATE: today's row carries a live signal (direction, entry, stop, target) only when the historical
cell orb_hot_net_r is KEEP, today is hot and the signal pass has logged today's entry. Otherwise today
shows only the opening range, the forecast band and the flag. Past settled days carry their simulated
trade, worded as a simulation. No text is an order instruction, a lot size or a promise of profit.
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
from .registry import CELL_HOT, hist_verdict

COMMON_ENV = 'MT5_COMMON_FILES'
FIELDS = ['date', 'pair', 'status', 'hot', 'or_start', 'or_end', 'or_high', 'or_low', 'line_end', 'band_start',
          'band_end', 'band_lo', 'band_hi', 'f_pips', 'real_high', 'real_low', 'kind', 'direction', 'entry_time',
          'entry', 'stop', 'target', 'exit_time', 'exit', 'net_r', 'label_bg', 'label_en', 'band_label_bg',
          'band_label_en']
REASON_BG = {'take profit': 'TP', 'stop loss': 'SL', 'forced exit': 'изход 21:00'}
REASON_EN = {'take profit': 'TP', 'stop loss': 'SL', 'forced exit': 'exit 21:00'}


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
    return '' if t is None or (isinstance(t, float) and np.isnan(t)) or pd.isna(t) else \
        pd.Timestamp(t).strftime('%Y.%m.%d %H:%M')


def num(x, nd=5) -> str:
    return '' if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)) or pd.isna(x) else f'{float(x):.{nd}f}'


def side_word(d) -> str:
    return 'long' if int(d) > 0 else 'short'


def records(cfg: dict) -> tuple:
    sig_ledger = read_log(cfg, 'signal_ledger')
    settled = sig_ledger[sig_ledger['status'] == 'settled'] if len(sig_ledger) else sig_ledger
    n_sig = len(settled)
    mean_r = float(pd.to_numeric(settled['r_net']).mean()) if n_sig else float('nan')
    st = read_log(cfg, 'settlements')
    n_fwd = int(st[st['status'] == 'ok'].groupby('pair').size().min()) if len(st) and (st['status'] == 'ok').any() else 0
    return n_sig, mean_r, n_fwd


def texts(cfg: dict, verdict: str, n_sig: int, mean_r: float, n_fwd: int) -> dict:
    v_bg = verdict or 'още не е оценена'
    v_en = verdict or 'not scored yet'
    rec_bg = f'{n_sig} сигнала, средно {mean_r:+.2f} R' if n_sig else 'още няма сигнали'
    rec_en = f'{n_sig} signals, mean {mean_r:+.2f} R' if n_sig else 'no signals yet'
    return {'status': (f"NY пробив · историческа проверка: {v_bg} · прогноза на рейнджа в forward проверка: "
                       f"{n_fwd} от ~{cfg['forward']['min_days']} дни · сигнали досега: {rec_bg}",
                       f"NY breakout · historical check: {v_en} · range forecast under forward test: "
                       f"{n_fwd} of ~{cfg['forward']['min_days']} days · signals so far: {rec_en}"),
            'gate_closed': ('Знак за вход се показва само след исторически KEEP на правилото.',
                            'An entry sign is shown only after a historical KEEP of the rule.'),
            'record_bg': (rec_bg, rec_en)}


def build(cfg: dict, now=None) -> str:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    today = now.normalize()
    verdict = hist_verdict(cfg, CELL_HOT)
    gate = verdict == 'KEEP'
    n_sig, mean_r, n_fwd = records(cfg)
    tx = texts(cfg, verdict, n_sig, mean_r, n_fwd)
    fc = read_log(cfg, 'forecasts')
    st = read_log(cfg, 'settlements')
    sg = read_log(cfg, 'signals')
    lines = ['\t'.join(['#meta', 'version=1', f"generated_utc={datetime.now(timezone.utc):%Y.%m.%d %H:%M}",
                        f"gate={'open' if gate else 'closed'}"])]
    for k, (bg, en) in tx.items():
        lines.append('\t'.join(['#text', k, bg, en]))
    lines.append('\t'.join(FIELDS))
    if len(fc):
        days = sorted(fc['date'].unique())
        keep = [d for d in days if d < today][-int(cfg['export']['display_days']):] + [d for d in days if d == today]
        for _, f in fc[fc['date'].isin(keep)].sort_values(['date', 'pair']).iterrows():
            d, p = f['date'], f['pair']
            row = {k: '' for k in FIELDS}
            row.update(date=d.strftime('%Y.%m.%d'), pair=p, status=f['status'], hot='1' if bool(f['hot']) else '0')
            if f['status'] == 'ok':
                pip = float(cfg['pip'][p])
                F, c14 = float(f['F']), float(f['close14'])
                hot = bool(f['hot'])
                row.update(or_start=lab(d + pd.Timedelta(hours=14)), or_end=lab(d + pd.Timedelta(hours=15)),
                           or_high=num(f['or_high']), or_low=num(f['or_low']), line_end=lab(d + pd.Timedelta(hours=21)),
                           band_start=lab(d + pd.Timedelta(hours=15)), band_end=lab(d + pd.Timedelta(hours=22)),
                           band_lo=num(c14 - F / 2), band_hi=num(c14 + F / 2), f_pips=num(F / pip, 1),
                           band_label_bg=f"Очакван размер: {F / pip:.0f} пипса · {'ГОРЕЩ ден' if hot else 'обикновен ден'} · "
                                         'размер, не посока',
                           band_label_en=f"Expected size: {F / pip:.0f} pips · {'HOT day' if hot else 'ordinary day'} · "
                                         'size, not direction')
            s = st[(st['date'] == d) & (st['pair'] == p)] if len(st) else st
            if len(s):
                s = s.iloc[0]
                row.update(real_high=num(s.get('real_high')), real_low=num(s.get('real_low')))
                if str(s.get('trade_reason', '')) in ('', 'nan') and pd.notna(s.get('entry_time')):
                    r = float(s['r_net'])
                    row.update(kind='sim', direction=str(int(s['direction'])), entry_time=lab(s['entry_time']),
                               entry=num(s['entry']), stop=num(s['stop']), target=num(s['target']),
                               exit_time=lab(s['exit_time']), exit=num(s['exit']), net_r=num(r, 2),
                               label_bg=f"Симулация: {side_word(s['direction'])}, {REASON_BG[s['exit_reason']]}, {r:+.2f} R",
                               label_en=f"Simulation: {side_word(s['direction'])}, {REASON_EN[s['exit_reason']]}, {r:+.2f} R")
            elif d == today and gate and bool(f['hot']) and f['status'] == 'ok':
                g = sg[(sg['date'] == d) & (sg['pair'] == p)] if len(sg) else sg
                if len(g):
                    g = g.iloc[0]
                    row.update(kind='live', direction=str(int(g['direction'])), entry_time=lab(g['entry_time']),
                               entry=num(g['entry']), stop=num(g['stop']), target=num(g['target']),
                               label_bg=f"Сигнал на правилото: {side_word(g['direction'])} · исторически {verdict} · "
                                        f"forward: {tx['record_bg'][0]}",
                               label_en=f"Rule signal: {side_word(g['direction'])} · historical {verdict} · "
                                        f"forward: {tx['record_bg'][1]}")
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


def rebuild(cfg: dict | None = None, now=None, env=None) -> str:
    cfg = cfg or load_config()
    p = export_path(cfg, env)
    write_atomic(p, build(cfg, now))
    return p


def main(argv=None):
    argparse.ArgumentParser(description='rebuild the fx-ny-orb-hot chart file').parse_args(argv)
    print(rebuild())


if __name__ == '__main__':
    main()
