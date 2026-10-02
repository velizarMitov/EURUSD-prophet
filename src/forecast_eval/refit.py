"""Monthly walk-forward refits during the forward window (task 10.1; design D12;
spec forward-arbiter "Walk-forward refits during the forward window").

On the first weekend of each month (market closed), every (model, horizon)
in the manifest is refitted with its FROZEN configuration -- read from the
version's own challenger.json -- on an expanding window that ends at the last
closed Friday bar. The new version is appended to the manifest; the forward
logger picks it up for any bar after its training end. A failed refit records
the failure and leaves the previous version live.

The refit never sees a bar after its own fit time: data is cut at the last
closed Friday bar before the refit runs, and the refit runs on the weekend.

Training data = the study's history plus newer MT5 bars appended after it, so
the forward window refits on the same series the study fitted on.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import features as F
from . import study as ST
from .challengers import make
from .forward_logger import FAIL_FIELDS, MANIFEST, OUT, append_rows, read

REFIT_FAILURES = 'refit_failures'


def first_weekend(now_utc) -> bool:
    t = pd.Timestamp(now_utc)
    return t.dayofweek >= 5 and t.day <= 7


def last_friday_close(now_utc) -> pd.Timestamp:
    """The newest bar time a weekend refit may use: Friday 23:59:59 server-clock
    labels, i.e. every bar of the week that just closed."""
    t = pd.Timestamp(now_utc).normalize()
    back = (t.dayofweek - 4) % 7
    return t - pd.Timedelta(days=back) + pd.Timedelta(hours=23, minutes=59, seconds=59)


def month_tag(now_utc) -> str:
    return pd.Timestamp(now_utc).strftime('%Y%m')


def due_cells(now_utc, manifest_path=MANIFEST) -> list[dict]:
    """Latest version per (model, horizon) that has no refit for this month yet."""
    if not first_weekend(now_utc):
        return []
    man = read(manifest_path)
    if man.empty:
        return []
    man = man.copy()
    man['train_end_ts'] = pd.to_datetime(man['train_end'], utc=True, format='mixed')
    tag = f'refit{month_tag(now_utc)}'
    out = []
    for (model, h), g in man.groupby(['model', 'horizon']):
        if g['version_id'].astype(str).str.contains(tag).any():
            continue
        out.append(g.sort_values('train_end_ts').iloc[-1].to_dict())
    return out


def history_sources(cutoff, extra=None, macro: bool = True, m15: bool = True) -> dict:
    """The study's history extended with NEWER MT5 bars from `extra`
    ({'h1': ..., 'd1': ..., 'm15': ..., 'point': ...}), cut at `cutoff`. Without
    the extension a refit would retrain on the same frozen history every month:
    the daily history CSV is a frozen input and ends where the study's data
    ended."""
    raw = F.load_daily_ohlcv()
    h1 = F.load_h1()
    if extra:
        if extra.get('h1') is not None and len(extra['h1']):
            newer = F.normalize_h1(extra['h1'])
            h1 = pd.concat([h1, newer[newer.index > h1.index[-1]]])
        if extra.get('d1') is not None and len(extra['d1']):
            d1 = extra['d1'][['open', 'high', 'low', 'close', 'tick_volume']].copy()
            idx = pd.DatetimeIndex(d1.index)
            d1.index = (idx.tz_convert('UTC') if idx.tz is not None else idx).tz_localize(None).normalize()
            raw = pd.concat([raw, d1[d1.index > raw.index[-1]]])
    cut = pd.Timestamp(cutoff)
    naive = cut.tz_localize(None) if cut.tzinfo else cut
    aware = cut if cut.tzinfo else cut.tz_localize('UTC')
    h1 = h1[h1.index <= aware]
    raw = raw[raw.index <= naive]
    macro_df = ST.load_macro_window(raw.index.min(), raw.index.max()) if macro else None
    out = {'h1': h1, 'daily': {fs: F.daily_features(raw, macro_df, fs) for fs in F.FEATURE_SETS},
           'm15': None}
    if m15:
        try:
            from . import m15_data as MD
            frame = MD.m15_bars()
            live = (extra or {}).get('m15')
            if live is not None and len(live):
                newer = MD.from_mt5_bars(live, (extra or {}).get('point') or frame['point'].iloc[-1])
                frame = pd.concat([frame, newer[newer.index > frame.index[-1]]])
            out['m15'] = frame[frame.index <= aware]
        except Exception as e:                           # noqa: BLE001 -- isolate the family
            out['m15_error'] = f'{type(e).__name__}: {e}'
    return out


def refit_cell(row: dict, sources: dict, now_utc, seed: int = 42) -> dict:
    with open(os.path.join(ST.REPO, row['path'], 'challenger.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    name, cfg, h = meta['name'], meta['config'], int(row['horizon'])
    model = make(name, cfg)
    inputs = model.build_inputs(ST.source_for(name, cfg, sources))
    tg = ST.make_targets(inputs.close, h)
    n = len(inputs)
    pos = np.flatnonzero(np.isfinite(tg['dir']) | np.isfinite(tg['vol']))
    pos = pos[pos < n - h]
    if cfg.get('cadence') == 'M15':
        from . import m15_data as MD
        MD.verify_clock(inputs.index)
        pos = np.intersect1d(pos, MD.eligible_positions(inputs.index, h))
    model.fit(inputs, pos, tg, seed, h)
    digest = F.feature_code_digest()['_combined']
    study_id = f"{row['study_id']}-refit{month_tag(now_utc)}"
    return ST.save_artifact(model, study_id, name, row['cadence'], h, row.get('variant', 'frozen'),
                            inputs, pos, digest)


def run_refit(now_utc=None, manifest_path=MANIFEST, out=OUT, sources=None) -> dict:
    now_utc = pd.Timestamp(now_utc or datetime.now(timezone.utc))
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize('UTC')
    cells = due_cells(now_utc, manifest_path)
    if not cells:
        return {'status': 'not due', 'at': str(now_utc)}
    cutoff = last_friday_close(now_utc)
    if sources is None:
        from .forward_logger import fetch_snapshot
        from .mt5_reader import MT5Reader, MT5Unavailable
        try:
            snap = fetch_snapshot(MT5Reader())
        except MT5Unavailable as e:
            append_rows(os.path.join(out, f'{REFIT_FAILURES}.csv'),
                        [{'model': '*', 'cadence': '*', 'horizon': '', 'as_of_bar': str(cutoff),
                          'error': f'no fresh MT5 bars, refit skipped: {e}',
                          'logged_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}],
                        FAIL_FIELDS)
            return {'status': 'skipped', 'reason': str(e)}
        sources = history_sources(cutoff, {'h1': snap.h1, 'd1': snap.d1, 'm15': snap.m15,
                                           'point': snap.meta.get('point')})
    done, failed = 0, 0
    for row in cells:
        try:
            new = refit_cell(row, sources, now_utc)
            if pd.Timestamp(new['train_end']) > cutoff:
                raise RuntimeError('refit saw a bar after its cutoff')
            ST._append_csv(manifest_path, new, ST.MANIFEST_FIELDS)
            done += 1
        except Exception as e:                           # noqa: BLE001 -- previous version stays live
            append_rows(os.path.join(out, f'{REFIT_FAILURES}.csv'),
                        [{'model': row['model'], 'cadence': row['cadence'], 'horizon': row['horizon'],
                          'as_of_bar': str(cutoff), 'error': f'{type(e).__name__}: {e}',
                          'logged_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}],
                        FAIL_FIELDS)
            failed += 1
    return {'status': 'refit', 'cutoff': str(cutoff), 'refitted': done, 'failed': failed}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Monthly walk-forward refit (first weekend only).')
    ap.parse_args(argv)
    print(json.dumps(run_refit(), default=str))


if __name__ == '__main__':
    main()
