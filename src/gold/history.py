"""Read-only gold history: every forecast next to what gold actually did
(openspec change app-shell-and-gold-history, spec gold-forecasting/gold-history).

Three sources share one long row schema, so the page draws them with one table
and one chart:

  live        the forward logger's predictions, each paired with the settlement
              that has the SAME key and no other; gaps listed as missed bars
  validation  the study's saved predictions on [70:80 %], from fits on [0:70 %]
  test        the one-shot test predictions on [80:100 %], from fits on [0:80 %]

Reads recorded files only. Runs no model, fetches no market data, writes
nothing and imports no model library, like src/gold/summary.py.

Summaries are DESCRIPTIVE: one model, one digest, one source, settled rows only.
They never say KEEP or DROP. The verdicts live in the registries and, for
forward data, in forward_report after the forward pre-registration.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from . import load_config, repo_path
from .summary import CELLS

SOURCES = ('live', 'validation', 'test')
CADENCES = ('H1', 'D1')
LAYOUTS = ('long', 'wide')
MAX_LIMIT = 1000
MIN_SUMMARY_ROWS = 30
WILSON_Z = 1.959963984540054
DESCRIPTIVE = 'descriptive — not a verdict'
VOL_BASELINE = {'h1_lstm': 'h1_garch_hod', 'd1_lstm': 'd1_garch'}

NOTICES = {
    'live': {
        'en': ('Live forecasts from the frozen models, logged hourly as each bar closed and settled '
               'when the next bar closed. Gross of cost.'),
        'bg': ('Живи прогнози от замразените модели, записвани всеки час при затваряне на бара и '
               'проверявани при затварянето на следващия. Без разходи.'),
    },
    'validation': {
        'en': ('Past research: the study\'s validation block [70–80 %], from models fit on the first 70 %. '
               'Scored once by the study; not live forecasting.'),
        'bg': ('Минало изследване: валидационният блок [70–80 %] на проучването, от модели, обучени на '
               'първите 70 %. Оценен веднъж; не е жива прогноза.'),
    },
    'test': {
        'en': ('Past research: the one-shot test predictions [80–100 %], from models fit on the first 80 %. '
               'Recorded for reference and not used to choose anything; not live forecasting.'),
        'bg': ('Минало изследване: еднократният тест [80–100 %], от модели, обучени на първите 80 %. '
               'Записан за справка и не е използван за избор на модел; не е жива прогноза.'),
    },
}

STUDY_DIGEST = {'validation': 'study validation fit [0:70 %]', 'test': 'study test fit [0:80 %]'}

COLUMNS = ['source', 'cell', 'cadence', 'model_digest', 'as_of_bar', 'as_of_close',
           'p_up', 'predicted_dir', 'probability', 'ret_pct', 'implied_price', 'vol_pct',
           'exit_bar', 'actual_price', 'realised_dir', 'realised_ret_pct', 'realised_vol_pct',
           'status', 'hit', 'price_error', 'move_error', 'phase', 'logged_at', 'reason',
           'range80_lo', 'range80_hi', 'range80_inside', 'range95_lo', 'range95_hi', 'range95_inside']

_CACHE: dict = {}


def _stamp(paths):
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((p, None, None))
    return tuple(out)


def _cached(name, paths, build):
    """`build()`'s result, rebuilt only when one of `paths` changes on disk."""
    key = (name, _stamp(paths))
    hit = _CACHE.get(name)
    if hit is not None and hit[0] == key:
        return hit[1]
    value = build()
    _CACHE[name] = (key, value)
    return value


def _read(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def _derive(df):
    """Predicted direction, probability, implied price and errors, the way
    src/gold/summary.py reads a row: up only when p_up > 0.5."""
    p = df['p_up']
    df['predicted_dir'] = np.where(p.isna(), None, np.where(p > 0.5, 'up', 'down'))
    df['probability'] = np.where(p > 0.5, p, 1.0 - p)
    df.loc[p.isna(), 'probability'] = np.nan
    df['implied_price'] = (df['as_of_close'] * np.exp(df['ret_pct'] / 100.0)).round(2)
    settled = df['status'] == 'settled'
    real_up = df['realised_dir'] == 1
    hit = pd.Series(None, index=df.index, dtype=object)
    scored = settled & p.notna() & df['realised_dir'].notna()
    hit[scored] = (df.loc[scored, 'predicted_dir'] == 'up') == real_up[scored]
    df['hit'] = hit
    df['price_error'] = np.where(settled, df['implied_price'] - df['actual_price'], np.nan)
    df['move_error'] = np.where(settled, df['vol_pct'] - df['realised_vol_pct'], np.nan)
    return df


def _empty():
    return pd.DataFrame(columns=COLUMNS)


def _live_frame(cfg):
    fwd = repo_path(cfg['paths']['forward'])
    pred = _read(os.path.join(fwd, 'predictions.csv'))
    sett = _read(os.path.join(fwd, 'settlements.csv'))
    gaps = _read(os.path.join(fwd, 'gaps.csv'))
    if pred.empty and gaps.empty:
        return _empty()
    rows = pd.DataFrame()
    if not pred.empty:
        s = (sett.drop_duplicates('key', keep='first').set_index('key')
             if not sett.empty else pd.DataFrame(columns=['exit_bar', 'exit_close', 'realised_dir',
                                                          'realised_ret_pct', 'realised_vol_pct', 'eligible']))
        rows = pd.DataFrame({
            'source': 'live', 'cell': pred['cell'], 'cadence': pred['cadence'],
            'model_digest': pred.get('model_digest', ''), 'as_of_bar': pred['as_of_bar'],
            'as_of_close': _num(pred['as_of_close']), 'p_up': _num(pred['p_up']),
            'ret_pct': _num(pred['ret_pct']), 'vol_pct': _num(pred['vol_pct']),
            'phase': pred.get('phase', ''), 'logged_at': pred.get('logged_at', ''), 'reason': None,
        })
        joined = s.reindex(pred['key'])                 # same key, nothing else
        rows['exit_bar'] = joined['exit_bar'].to_numpy()
        rows['actual_price'] = _num(joined['exit_close']).to_numpy()
        rows['realised_dir'] = _num(joined['realised_dir']).to_numpy()
        rows['realised_ret_pct'] = _num(joined['realised_ret_pct']).to_numpy()
        rows['realised_vol_pct'] = _num(joined['realised_vol_pct']).to_numpy()
        has = joined['exit_bar'].notna().to_numpy()
        eligible = joined['eligible'].astype(str).str.lower().eq('true').to_numpy()
        rows['status'] = np.where(~has, 'pending', np.where(eligible, 'settled', 'not_scored'))
        rows = _derive(rows)
    if not gaps.empty:
        missed = []
        for _, g in gaps.iterrows():
            for cell, cad in CELLS:
                if cad == g['cadence']:
                    missed.append({'source': 'live', 'cell': cell, 'cadence': cad, 'model_digest': '',
                                   'as_of_bar': g['as_of_bar'], 'status': 'missed',
                                   'reason': g.get('reason', ''), 'logged_at': g.get('logged_at', '')})
        rows = pd.concat([rows, pd.DataFrame(missed)], ignore_index=True) if len(rows) else pd.DataFrame(missed)
    rows = rows.reindex(columns=COLUMNS)
    return _attach_ranges(rows, fwd)


def _attach_ranges(rows: pd.DataFrame, fwd: str) -> pd.DataFrame:
    """The round-two live ranges (change gold-forecasting-v2) for each forecast
    key, with whether the exit close fell inside; blank where none was logged."""
    rng = _read(os.path.join(fwd, 'ranges.csv'))
    if rng.empty or rows.empty:
        return rows
    keys = rows['cell'] + '|' + rows['cadence'] + '|' + rows['as_of_bar']
    rs = _read(os.path.join(fwd, 'range_settlements.csv'))
    inside = {} if rs.empty else {(k, str(lv)): v for k, lv, v in zip(rs['key'], rs['level'], rs['inside'])}
    for level, tag in (('0.8', 'range80'), ('0.95', 'range95')):
        part = rng[rng['level'].astype(str) == level].drop_duplicates('key').set_index('key')
        rows[f'{tag}_lo'] = _num(part['lo'].reindex(keys)).to_numpy()
        rows[f'{tag}_hi'] = _num(part['hi'].reindex(keys)).to_numpy()
        flag = [inside.get((k, level)) for k in keys]
        rows[f'{tag}_inside'] = [None if f is None else str(f).lower() == 'true' for f in flag]
    return rows


def live(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    fwd = repo_path(cfg['paths']['forward'])
    paths = [os.path.join(fwd, f) for f in ('predictions.csv', 'settlements.csv', 'gaps.csv',
                                            'ranges.csv', 'range_settlements.csv')]
    return _cached(('live', fwd), paths, lambda: _live_frame(cfg))


def _study_path(cfg, stage, cadence):
    return os.path.join(repo_path(cfg['paths']['results']), f'{stage}_predictions_{cadence.lower()}.csv')


def _study_frame(cfg, stage, cadence):
    wide_ = _read(_study_path(cfg, stage, cadence))
    if wide_.empty:
        return _empty()
    close = _num(wide_['close'])
    y_dir, y_ret, y_vol = _num(wide_['y_dir']), _num(wide_['y_ret']), _num(wide_['y_vol'])
    eligible = wide_['eligible'].astype(str).str.lower().eq('true') & y_ret.notna()
    parts = []
    for cell, cad in CELLS:
        if cad != cadence:
            continue
        col = lambda name: _num(wide_[f'{cell}__{name}']) if f'{cell}__{name}' in wide_ else pd.Series(np.nan, index=wide_.index)
        part = pd.DataFrame({
            'source': stage, 'cell': cell, 'cadence': cadence, 'model_digest': STUDY_DIGEST[stage],
            'as_of_bar': wide_['time'], 'as_of_close': close,
            'p_up': col('p_up'), 'ret_pct': col('ret_pct'), 'vol_pct': col('vol_pct'),
            'exit_bar': None, 'actual_price': (close * np.exp(y_ret / 100.0)).round(2),
            'realised_dir': y_dir, 'realised_ret_pct': y_ret, 'realised_vol_pct': y_vol,
            'status': np.where(eligible, 'settled', 'not_scored'),
            'phase': stage, 'logged_at': None, 'reason': None,
        })
        parts.append(_derive(part))
    return pd.concat(parts, ignore_index=True).reindex(columns=COLUMNS) if parts else _empty()


def study(cfg=None, stage='test', cadence='H1') -> pd.DataFrame:
    cfg = cfg or load_config()
    path = _study_path(cfg, stage, cadence)
    return _cached(('study', path), [path], lambda: _study_frame(cfg, stage, cadence))


def wilson(hits: int, n: int, z: float = WILSON_Z):
    """The Wilson score interval for a proportion; (None, None) when n = 0."""
    if n <= 0:
        return None, None
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def _mae(s):
    s = s.dropna().abs()
    return float(s.mean()) if len(s) else None


def summaries(rows: pd.DataFrame) -> list[dict]:
    """One descriptive summary per (cell, model digest, source), over settled rows."""
    out = []
    settled = rows[rows['status'] == 'settled']
    for (source, cell, digest), g in settled.groupby(['source', 'cell', 'model_digest'], sort=True, dropna=False):
        item = {'source': source, 'cell': cell, 'model_digest': digest or None, 'n': int(len(g)),
                'kind': DESCRIPTIVE}
        if len(g) < MIN_SUMMARY_ROWS:
            item['state'] = 'too_few'
            out.append(item)
            continue
        item['state'] = 'ok'
        scored = g[g['hit'].notna()]
        if len(scored):
            hits = int(scored['hit'].astype(bool).sum())
            lo, hi = wilson(hits, len(scored))
            item.update({'n_direction': int(len(scored)), 'hits': hits, 'hit_rate': hits / len(scored),
                         'hit_low': lo, 'hit_high': hi,
                         'vs_coin': 'above_coin' if lo > 0.5 else ('below_coin' if hi < 0.5 else 'coin_range')})
        if g['ret_pct'].notna().any():
            item['ret_mae'] = _mae(g['ret_pct'] - g['realised_ret_pct'])
        if g['vol_pct'].notna().any():
            item['vol_mae'] = _mae(g['move_error'])
            base = VOL_BASELINE.get(cell)
            if base:
                b = settled[(settled['cell'] == base) & (settled['source'] == source)]
                shared = sorted(set(g['as_of_bar']) & set(b['as_of_bar']))
                if shared:
                    mine = g[g['as_of_bar'].isin(shared)]
                    theirs = b[b['as_of_bar'].isin(shared)].drop_duplicates('as_of_bar')
                    item.update({'vol_baseline': base, 'n_shared': len(shared),
                                 'vol_mae_shared': _mae(mine['move_error']),
                                 'vol_mae_baseline': _mae(theirs['move_error'])})
        out.append(item)
    return out


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, str) and v == '':
        return None
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict(orient='records')]


REALISED = ('exit_bar', 'actual_price', 'realised_dir', 'realised_ret_pct', 'realised_vol_pct')
PER_MODEL = ('model_digest', 'p_up', 'predicted_dir', 'probability', 'ret_pct', 'implied_price',
             'vol_pct', 'status', 'hit', 'price_error', 'move_error', 'phase', 'reason')


def wide(rows: pd.DataFrame) -> list[dict]:
    """One line per as-of bar: every model's forecast side by side, next to the
    one realised outcome (taken from a settled row of that bar)."""
    out = []
    for bar, g in rows.groupby('as_of_bar', sort=False):
        done = g[g['status'].isin(['settled', 'not_scored'])]
        src = done.iloc[0] if len(done) else g.iloc[0]
        line = {'as_of_bar': bar, 'as_of_close': _clean(g['as_of_close'].dropna().iloc[0]) if g['as_of_close'].notna().any() else None,
                'realised': {k: _clean(src[k]) for k in REALISED} if len(done) else None,
                'forecasts': {r['cell']: {k: _clean(r[k]) for k in PER_MODEL} for _, r in g.iterrows()}}
        out.append(line)
    return out


def query(source='live', cadence='H1', cell=None, start=None, end=None, layout='long',
          offset=0, limit=200, cfg=None) -> dict:
    """The payload of GET /api/gold/history. Raises ValueError, naming the
    allowed values, on any bad parameter."""
    cfg = cfg or load_config()
    cells = [c for c, cad in CELLS if cad == cadence]
    allowed = {'source': list(SOURCES), 'cadence': list(CADENCES), 'cell': cells,
               'layout': list(LAYOUTS), 'limit': [1, MAX_LIMIT]}
    if source not in SOURCES:
        raise ValueError(f'unknown source {source!r}; allowed: {allowed}')
    if cadence not in CADENCES:
        raise ValueError(f'unknown cadence {cadence!r}; allowed: {allowed}')
    if cell is not None and cell not in cells:
        raise ValueError(f'unknown cell {cell!r} for {cadence}; allowed: {allowed}')
    if layout not in LAYOUTS:
        raise ValueError(f'unknown layout {layout!r}; allowed: {allowed}')
    if not (1 <= int(limit) <= MAX_LIMIT) or int(offset) < 0:
        raise ValueError(f'limit must be 1..{MAX_LIMIT} and offset >= 0; allowed: {allowed}')
    bounds = []
    for name, value in (('start', start), ('end', end)):
        if value in (None, ''):
            bounds.append(None)
            continue
        try:
            t = pd.Timestamp(value)
        except (ValueError, TypeError):
            raise ValueError(f'{name} {value!r} is not a date; allowed: {allowed}') from None
        bounds.append(t.tz_localize('UTC') if t.tzinfo is None else t)

    rows = live(cfg) if source == 'live' else study(cfg, source, cadence)
    rows = rows[rows['cadence'] == cadence]
    if bounds[0] is not None or bounds[1] is not None:
        t = pd.to_datetime(rows['as_of_bar'], utc=True, errors='coerce')
        mask = pd.Series(True, index=rows.index)
        if bounds[0] is not None:
            mask &= t >= bounds[0]
        if bounds[1] is not None:
            mask &= t < bounds[1] + pd.Timedelta(days=1)
        rows = rows[mask]
    rows = rows.assign(_t=pd.to_datetime(rows['as_of_bar'], utc=True, errors='coerce'))
    rows = rows.sort_values(['_t', 'cell'], ascending=[False, True], kind='mergesort').drop(columns='_t')

    # A chosen cell narrows the long table and the summaries; its GARCH baseline
    # stays in the summary input so the shared-bar comparison can be made. The
    # side-by-side layout always shows every model of the cadence.
    summ = summaries(rows if cell is None else rows[rows['cell'].isin([cell, VOL_BASELINE.get(cell)])])
    if cell is not None:
        summ = [s for s in summ if s['cell'] == cell]
    shown = rows if cell is None else rows[rows['cell'] == cell]
    offset, limit = int(offset), int(limit)
    if layout == 'wide':
        bars = rows['as_of_bar'].drop_duplicates()          # already newest first
        total = len(bars)
        page = wide(rows[rows['as_of_bar'].isin(set(bars.iloc[offset:offset + limit]))])
    else:
        total, page = len(shown), records(shown.iloc[offset:offset + limit])
    return {'source': source, 'cadence': cadence, 'cell': cell, 'layout': layout,
            'offset': offset, 'limit': limit, 'total': total,
            'next_offset': offset + limit if offset + limit < total else None,
            'source_notice': NOTICES[source], 'clock': 'Europe/Berlin bar labels',
            'cost_basis': 'gross of cost', 'kind': DESCRIPTIVE,
            'summaries': summ, 'rows': page, 'allowed': allowed}
