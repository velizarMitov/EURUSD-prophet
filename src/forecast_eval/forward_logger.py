"""Forward logger (tasks 9.2-9.5; spec forward-arbiter; design D11).

Run once an hour by the Windows Task Scheduler (scripts/forecast_eval/). One run:

  1. take a lock (a second concurrent run exits immediately);
  2. read MT5 (read-only): closed H1 and D1 bars with each bar's spread, the
     symbol's swap rates, the broker SERVER name (never the login);
  3. for each cadence with a newly closed bar: write a GAP record for every
     closed bar that was missed since the last run, then predict the newest one
     -- every model, every horizon, from the latest manifest version trained
     strictly before that bar. Missed bars are never predicted afterwards;
  4. settle every earlier prediction whose exit bar has now closed, from MT5
     closes only.

Append-only CSVs under results/forward_eval/: predictions, settlements, gaps,
failures. A row, once written, is never changed. A model that fails (missing
artifact, changed feature code, unavailable dependency) gets a failure record
and every other model is still logged.

Settlement of daily-cadence forecasts uses MT5 D1 closes for every daily model,
including the two fed by H1 bars, so all daily rows are settled on one price.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import features as F
from . import targets as T
from .challengers import make
from .challengers.base import ChallengerUnavailable
from .mt5_reader import SOURCE, MT5Reader, MT5Unavailable, closed_bars, feed_now

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'results', 'forward_eval')
STATE = os.path.join(REPO, 'research_models', 'forward_eval')
MANIFEST = os.path.join(REPO, 'results', 'horizon_study', 'artifact_manifest.csv')
RECORD = os.path.join(REPO, 'results', 'horizon_study', 'study_record.json')
SYMBOL = 'EURUSD'
M15_BARS, H1_BARS, D1_BARS = 3000, 3000, 400
CADENCES = ('M15', 'H1', 'D1')
# 3000 M15 bars span about six weekends, so the live clock check has real
# evidence; the threshold tolerates one odd broker opening while still failing a
# systematic one-hour shift of the server clock, which would silently move every
# session rule (design D16).
LIVE_CLOCK_MATCH_MIN = 0.6

PRED_FIELDS = ['key', 'model', 'cadence', 'horizon', 'version_id', 'variant', 'as_of_bar',
               'target_offset_bars', 'p_up', 'ret_pct', 'vol_pct', 'price_source', 'server',
               'entry_close', 'spread_points', 'swap_long', 'swap_short', 'phase',
               'feature_digest', 'logged_at']
SETTLE_FIELDS = ['key', 'settled_at', 'exit_bar', 'exit_close', 'realised_dir', 'realised_ret_pct',
                 'realised_vol_pct', 'correct', 'scorable', 'exclusion_reason']
GAP_FIELDS = ['cadence', 'bar', 'reason', 'logged_at']
FAIL_FIELDS = ['model', 'cadence', 'horizon', 'as_of_bar', 'error', 'logged_at']


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _path(out, name):
    return os.path.join(out, f'{name}.csv')


def append_rows(path, rows, fields):
    """Append-only writer: the file is only ever opened in 'a' mode."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='raise')
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def read(path) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) else pd.DataFrame()


def pred_key(model, horizon, as_of) -> str:
    return f'{model}|h{int(horizon)}|{pd.Timestamp(as_of).isoformat()}'


# ── lock ────────────────────────────────────────────────────────────────────

class Locked(RuntimeError):
    pass


class RunLock:
    """Exclusive lock file; a lock older than `stale_s` is treated as abandoned."""

    def __init__(self, path, stale_s=3000):
        self.path, self.stale_s = path, stale_s

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path) and time.time() - os.path.getmtime(self.path) > self.stale_s:
            os.remove(self.path)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise Locked('another logger run holds the lock')
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *exc):
        os.close(self.fd)
        os.remove(self.path)


# ── market snapshot ─────────────────────────────────────────────────────────

class Snapshot:
    """Closed bars and metadata from one MT5 read."""

    def __init__(self, h1: pd.DataFrame, d1: pd.DataFrame, server: str, meta: dict,
                 source: str = SOURCE, m15: pd.DataFrame | None = None):
        self.h1, self.d1, self.server, self.meta, self.source = h1, d1, server, meta, source
        self.m15 = m15


def bars_for(snap: Snapshot, cadence: str) -> pd.DataFrame:
    frame = {'M15': snap.m15, 'H1': snap.h1, 'D1': snap.d1}[cadence]
    return frame if frame is not None else pd.DataFrame()


def fetch_snapshot(reader: MT5Reader, now_utc=None, state_path=None) -> Snapshot:
    with reader:
        m15_all = reader.bars(SYMBOL, 'M15', M15_BARS)
        h1_all = reader.bars(SYMBOL, 'H1', H1_BARS)
        d1_all = reader.bars(SYMBOL, 'D1', D1_BARS)
        server = reader.server()
        meta = reader.symbol_meta(SYMBOL)
    kw = {'state_path': state_path} if state_path else {}
    # The feed clock is inferred from the H1 series by the production helper; the
    # M15 closure rule needs it at minute resolution, which it has.
    now_feed = feed_now(h1_all.index, now_utc, **kw)
    return Snapshot(_utc(closed_bars(h1_all, 'H1', now_feed)), _utc(closed_bars(d1_all, 'D1', now_feed)),
                    server, meta, m15=_utc(closed_bars(m15_all, 'M15', now_feed)))


def _utc(df):
    idx = pd.DatetimeIndex(df.index)
    df = df.copy()
    df.index = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    return df


# ── manifest ────────────────────────────────────────────────────────────────

def latest_versions(as_of, cadence, manifest_path=MANIFEST) -> list[dict]:
    """One row per (model, horizon): the newest version trained strictly before
    `as_of`. Rows are the manifest rows as dicts."""
    man = read(manifest_path)
    if man.empty:
        return []
    man = man[man['cadence'] == cadence].copy()
    man['train_end_ts'] = pd.to_datetime(man['train_end'], utc=True, format='mixed')
    man = man[man['train_end_ts'] < pd.Timestamp(as_of)]
    man = man.sort_values('train_end_ts')
    return [g.iloc[-1].to_dict() for _, g in man.groupby(['model', 'horizon'])]


def load_version(row: dict):
    path = os.path.join(REPO, row['path'])
    # Digests first: a missing or tampered file of ANY kind -- weights or the
    # challenger.json that names them -- must read as exactly that.
    for rel, digest in json.loads(row['files_sha256']).items():
        full = os.path.join(REPO, rel)
        if not os.path.exists(full) or F.file_digest(full) != digest:
            raise FileNotFoundError(f'artifact missing or altered: {rel}')
    with open(os.path.join(path, 'challenger.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    return make(meta['name'], meta['config']).load(path)


def kronos_rows(as_of, record_path=RECORD) -> list[dict]:
    """Kronos is pinned and train-free: no manifest version, one row per cell
    declared in the study record."""
    if not os.path.exists(record_path):
        return []
    with open(record_path, encoding='utf-8') as fh:
        rec = json.load(fh)
    out = []
    for name, cfg in rec['challengers'].items():
        if cfg.get('train_free'):
            for h in cfg.get('horizons') or rec['grid'][cfg['cadence']]:
                out.append({'model': name, 'horizon': h, 'cadence': cfg['cadence'], 'config': cfg,
                            'version_id': f'pinned-{name}-{cfg.get("model", "mini")}',
                            'variant': 'pinned', 'feature_digest': ''})
    return out


# ── inputs ──────────────────────────────────────────────────────────────────

def source_frame(model_name: str, cadence: str, snap: Snapshot, as_of, macro_df=None):
    if cadence == 'M15':
        from . import m15_data as MD
        point = snap.meta.get('point')
        if not point:
            raise ValueError('the symbol metadata carries no point size, so no mid price can be '
                             'built; refusing to guess one')
        m15 = bars_for(snap, 'M15')
        return MD.from_mt5_bars(m15[m15.index <= pd.Timestamp(as_of)], point)
    if model_name.startswith(('daily_', 'vol_')):
        cfg_set = 'macro' if model_name.endswith('macro') else 'price'
        d1 = snap.d1[snap.d1.index <= as_of].copy()
        d1.index = d1.index.tz_localize(None)
        return F.daily_features(d1, macro_df, cfg_set)
    h1 = snap.h1
    if cadence == 'D1':
        h1 = h1[h1.index < pd.Timestamp(as_of) + pd.Timedelta(days=1)]
    else:
        h1 = h1[h1.index <= as_of]
    return h1[['open', 'high', 'low', 'close', 'tick_volume']]


def _pos(index, as_of):
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    hits = np.flatnonzero(idx == pd.Timestamp(as_of))
    if hits.size == 0:
        raise KeyError(f'as-of bar {as_of} not in the model inputs')
    return int(hits[0])


# ── one run ─────────────────────────────────────────────────────────────────

def newest_closed(snap: Snapshot, cadence: str):
    df = bars_for(snap, cadence)
    return df.index[-1] if len(df) else None


def in_session(stamp) -> bool:
    from . import m15_data as MD
    return bool(MD.session_mask(pd.DatetimeIndex([pd.Timestamp(stamp)]))[0])


def gaps_since(snap: Snapshot, cadence: str, last_logged, newest) -> list:
    """Bar closes missed since the last run. For M15 only IN-SESSION closes
    count: the logger never forecasts outside the session, so a bar it was never
    going to use is not a gap in its coverage (spec forward-arbiter "Sub-hourly
    logging follows the session")."""
    if last_logged is None:
        return []
    idx = bars_for(snap, cadence).index
    missed = idx[(idx > pd.Timestamp(last_logged)) & (idx < pd.Timestamp(newest))]
    if cadence == 'M15' and len(missed):
        from . import m15_data as MD
        missed = missed[MD.session_mask(missed)]
    return list(missed)


def last_logged_bar(out, cadence):
    stamps = []
    for name in ('predictions', 'gaps'):
        d = read(_path(out, name))
        if not d.empty:
            col = 'as_of_bar' if name == 'predictions' else 'bar'
            d = d[d['cadence'] == cadence]
            if not d.empty:
                stamps.append(pd.to_datetime(d[col], utc=True, format='mixed').max())
    return max(stamps) if stamps else None


def predict_cadence(snap: Snapshot, cadence: str, phase: str, out: str, manifest_path: str,
                    record_path: str, digest: str, macro_df=None) -> dict:
    newest = newest_closed(snap, cadence)
    if newest is None:
        return {'cadence': cadence, 'status': 'no closed bars'}
    preds = read(_path(out, 'predictions'))
    have = set(preds['key']) if not preds.empty else set()
    last = last_logged_bar(out, cadence)
    if last is not None and pd.Timestamp(last) >= newest:
        return {'cadence': cadence, 'status': 'nothing new', 'as_of': str(newest)}
    gaps = gaps_since(snap, cadence, last, newest)
    append_rows(_path(out, 'gaps'), [{'cadence': cadence, 'bar': str(b),
                                     'reason': 'bar closed while the logger was not running',
                                     'logged_at': _now_iso()} for b in gaps], GAP_FIELDS)
    if cadence == 'M15':
        from . import m15_data as MD
        try:
            MD.verify_clock(bars_for(snap, 'M15').index, min_match=LIVE_CLOCK_MATCH_MIN)
        except MD.ClockMismatch as e:
            append_rows(_path(out, 'failures'),
                        [{'model': '*', 'cadence': cadence, 'horizon': '', 'as_of_bar': str(newest),
                          'error': f'ClockMismatch: {e}', 'logged_at': _now_iso()}], FAIL_FIELDS)
            return {'cadence': cadence, 'status': 'clock mismatch', 'as_of': str(newest)}
        if not in_session(newest):
            # Outside the owner's session there is no forecast and no gap either.
            return {'cadence': cadence, 'status': 'outside the session', 'as_of': str(newest),
                    'gaps': len(gaps)}
    bars = bars_for(snap, cadence)
    entry_close = float(bars.loc[newest, 'close'])
    spread = float(bars.loc[newest, 'spread']) if 'spread' in bars else float('nan')
    rows, fails = [], []
    by_model: dict = {}
    for r in latest_versions(newest, cadence, manifest_path) + \
            [k for k in kronos_rows(newest, record_path) if k['cadence'] == cadence]:
        by_model.setdefault(r['model'], []).append(r)
    kronos_paths = None
    for model_name, versions in sorted(by_model.items()):
        for r in sorted(versions, key=lambda v: int(v['horizon'])):
            h = int(r['horizon'])
            key = pred_key(model_name, h, newest)
            if key in have:
                continue
            try:
                if r.get('variant') == 'pinned':
                    from .challengers.kronos import Kronos
                    model = Kronos(r['config'], kind='direction' if model_name.endswith('direction') else 'volatility')
                    frame = source_frame(model_name, cadence, snap, newest)
                    inputs = model.build_inputs(frame)
                    p = _pos(inputs.index, newest)
                    if kronos_paths is None:
                        longest = max(int(v['horizon']) for vs in by_model.values() for v in vs
                                      if v.get('variant') == 'pinned')
                        kronos_paths = model.sample(inputs, [p], longest)
                    pred = model.predict(inputs, h=h, paths=kronos_paths)
                else:
                    if r.get('feature_digest') and r['feature_digest'] != digest:
                        raise RuntimeError('feature code changed since this version was trained')
                    model = load_version(r)
                    frame = source_frame(model_name, cadence, snap, newest, macro_df)
                    inputs = model.build_inputs(frame)
                    p = _pos(inputs.index, newest)
                    pred = model.predict(inputs)
                rows.append({
                    'key': key, 'model': model_name, 'cadence': cadence, 'horizon': h,
                    'version_id': r['version_id'], 'variant': r.get('variant', ''),
                    'as_of_bar': str(newest), 'target_offset_bars': h,
                    'p_up': _at(pred, 'p_up', p), 'ret_pct': _at(pred, 'ret_pct', p),
                    'vol_pct': _at(pred, 'vol_pct', p), 'price_source': snap.source,
                    'server': snap.server, 'entry_close': entry_close, 'spread_points': spread,
                    'swap_long': snap.meta.get('swap_long'), 'swap_short': snap.meta.get('swap_short'),
                    'phase': phase, 'feature_digest': digest, 'logged_at': _now_iso()})
            except (ChallengerUnavailable, Exception) as e:      # noqa: BLE001 -- isolate per model
                fails.append({'model': model_name, 'cadence': cadence, 'horizon': h,
                              'as_of_bar': str(newest), 'error': f'{type(e).__name__}: {e}',
                              'logged_at': _now_iso()})
    append_rows(_path(out, 'predictions'), rows, PRED_FIELDS)
    append_rows(_path(out, 'failures'), fails, FAIL_FIELDS)
    return {'cadence': cadence, 'as_of': str(newest), 'predicted': len(rows), 'failed': len(fails),
            'gaps': len(gaps)}


def _at(pred, k, p):
    v = pred.get(k)
    if v is None:
        return ''
    x = float(np.asarray(v)[p])
    return x if np.isfinite(x) else ''


def _m15_mid_pair(bars, entry_i, exit_i, h, idx, snap):
    """(entry mid, exit mid, exclusion reasons) for a sub-hourly row.

    Sub-hourly cells are scored on the MID price, matching the study (spec
    forward-arbiter "Sub-hourly forward rows are scored on the mid price"). A
    missing spread or a point size is never guessed, and a trade that ends
    outside the session is excluded, exactly as the study's eligibility rule
    excludes it."""
    from . import m15_data as MD
    entry = float(bars['close'].iloc[entry_i])
    exit_ = float(bars['close'].iloc[exit_i])
    reasons = []
    point = snap.meta.get('point')
    sp_in = bars['spread'].iloc[entry_i] if 'spread' in bars else float('nan')
    sp_out = bars['spread'].iloc[exit_i] if 'spread' in bars else float('nan')
    if not point:
        reasons.append('no point size recorded, mid price unavailable')
    elif not (np.isfinite(sp_in) and np.isfinite(sp_out)):
        reasons.append('missing spread, mid price unavailable')
    else:
        entry = MD.mid_from(entry, sp_in, point)
        exit_ = MD.mid_from(exit_, sp_out, point)
    if not MD.eligible_mask(idx, h)[entry_i]:
        reasons.append('trade ends outside the session')
    return entry, exit_, reasons


def settle(snap: Snapshot, out: str, registered_server: str | None = None) -> int:
    preds = read(_path(out, 'predictions'))
    if preds.empty:
        return 0
    settled = read(_path(out, 'settlements'))
    done = set(settled['key']) if not settled.empty else set()
    rows = []
    for _, r in preds[~preds['key'].isin(done)].iterrows():
        bars = bars_for(snap, r['cadence'])
        as_of = pd.Timestamp(r['as_of_bar'])
        idx = bars.index
        where = np.flatnonzero(idx == as_of)
        if where.size == 0:
            continue                                     # outside the current window
        h = int(r['target_offset_bars'])
        exit_i = where[0] + h
        if exit_i >= len(idx):
            continue                                     # exit bar not closed yet
        entry, exit_ = float(bars['close'].iloc[where[0]]), float(bars['close'].iloc[exit_i])
        reasons = []
        if r['cadence'] == 'M15':
            entry, exit_, m15_reasons = _m15_mid_pair(bars, where[0], exit_i, h, idx, snap)
            reasons.extend(m15_reasons)
        s = pd.Series([entry, exit_])
        d, _ = T.direction_target(s, 1)
        if r['price_source'] != SOURCE:
            reasons.append(f"price source {r['price_source']}")
        if registered_server and r['server'] != registered_server:
            reasons.append(f"server {r['server']} is not the registered {registered_server}")
        if r['phase'] != 'scoring':
            reasons.append(f"phase {r['phase']}")
        rd = d.iloc[0]
        p_up = pd.to_numeric(r.get('p_up'), errors='coerce')
        correct = '' if (np.isnan(rd) or pd.isna(p_up)) else float((p_up >= 0.5) == (rd == 1.0))
        rows.append({'key': r['key'], 'settled_at': _now_iso(), 'exit_bar': str(idx[exit_i]),
                     'exit_close': exit_, 'realised_dir': '' if np.isnan(rd) else rd,
                     'realised_ret_pct': float(T.return_pct_target(s, 1).iloc[0]),
                     'realised_vol_pct': float(T.volatility_pct_target(s, 1).iloc[0]),
                     'correct': correct, 'scorable': not reasons,
                     'exclusion_reason': '; '.join(reasons)})
    append_rows(_path(out, 'settlements'), rows, SETTLE_FIELDS)
    return len(rows)


def run_once(reader=None, phase: str | None = None, out: str = OUT, manifest_path: str = MANIFEST,
             record_path: str = RECORD, now_utc=None, snapshot: Snapshot | None = None,
             macro_df=None, registered_server: str | None = None) -> dict:
    from . import prereg
    if phase is None:
        phase = 'scoring' if prereg.scoring_allowed() else 'dry_run'
    if registered_server is None and phase == 'scoring':
        registered_server = prereg.registered_server()
    with RunLock(os.path.join(STATE if out == OUT else out, 'logger.lock')):
        if snapshot is None:
            try:
                snapshot = fetch_snapshot(reader or MT5Reader(), now_utc)
            except MT5Unavailable as e:
                append_rows(_path(out, 'failures'), [{'model': '*', 'cadence': '*', 'horizon': '',
                                                      'as_of_bar': '', 'error': f'MT5Unavailable: {e}',
                                                      'logged_at': _now_iso()}], FAIL_FIELDS)
                return {'status': 'mt5 unavailable', 'error': str(e)}
        digest = F.feature_code_digest()['_combined']
        if macro_df is None and not read(manifest_path).empty and \
                read(manifest_path)['model'].astype(str).str.endswith('macro').any():
            macro_df = _macro(snapshot)
        res = [predict_cadence(snapshot, c, phase, out, manifest_path, record_path, digest, macro_df)
               for c in CADENCES]
        n = settle(snapshot, out, registered_server)
    return {'phase': phase, 'cadences': res, 'settled': n}


def _macro(snap: Snapshot):
    """Production macro chain on a sandboxed cache copy (see study.load_sources)."""
    try:
        from .study import load_macro_window
        return load_macro_window(snap.d1.index.min(), snap.d1.index.max())
    except Exception:                                    # noqa: BLE001 -- macro models then fail alone
        return None


def refresh_view(out: str = OUT) -> str | None:
    """Rewrite the operator view after a run. A view failure must never affect
    logging, so it is reported and swallowed (task 15.2)."""
    try:
        from .report import write_dashboard
        return write_dashboard(out)
    except Exception as e:                               # noqa: BLE001 -- logging comes first
        append_rows(_path(out, 'failures'),
                    [{'model': 'dashboard', 'cadence': '*', 'horizon': '', 'as_of_bar': '',
                      'error': f'{type(e).__name__}: {e}', 'logged_at': _now_iso()}], FAIL_FIELDS)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description='Forward logger: one predict/gap/settle pass.')
    ap.add_argument('--phase', choices=['dry_run', 'scoring'])
    ap.add_argument('--no-view', action='store_true', help='skip rewriting dashboard.html')
    a = ap.parse_args(argv)
    try:
        result = run_once(phase=a.phase)
        if not a.no_view:
            result['view'] = refresh_view()
        print(json.dumps(result, default=str))
    except Locked as e:
        print(json.dumps({'status': 'skipped', 'reason': str(e)}))


if __name__ == '__main__':
    main()
