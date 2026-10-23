"""Gold macro drivers (openspec change gold-forecasting-v2, spec
gold-forecasting/macro-drivers).

Daily inputs that the gold literature names as drivers, each known at the gold
D1 bar's close:

  FRED  DFII10 (10-year TIPS real yield), DTWEXBGS (broad dollar index),
        VIXCLS (VIX) -- the value dated on or before the PREVIOUS US business
        day, forward-filled only (a FRED daily value is published after its date)
  MT5   SILVER and Brent D1 bars -- the bar labelled on or before the gold bar
        (same broker clock), forward-filled only

Features are scale-free: changes, log returns, a VIX level (stationary by
construction) and a 60-day z-score of the gold/silver log ratio. A raw price
level is refused by the gold scale-free guard.

FRED comes through the shared src.macro_data.fetch_fred_feature (API -> public
CSV -> cache); MT5 bars are cached append-only like the gold bars. Every cache
lives under results/gold/macro/. A source that cannot supply a series raises
MacroUnavailable naming it -- never a zero.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import repo_path
from .features import assert_scale_free

MACRO_COLUMNS = ['real_yield_chg_1', 'real_yield_chg_5', 'usd_ret_1', 'usd_ret_5', 'vix_level', 'vix_chg_5',
                 'silver_ret_1', 'silver_ret_5', 'brent_ret_1', 'brent_ret_5', 'gold_silver_z60']


class MacroUnavailable(RuntimeError):
    """A macro series could not be supplied (named in the message)."""


# ── data ────────────────────────────────────────────────────────────────────

def cache_dir(cfg2) -> str:
    return repo_path(cfg2['paths']['macro_cache'])


def fred_series(cfg2, name: str, sid: str, start, end, allow_fetch: bool = True) -> pd.Series:
    """One FRED level series by date (UTC midnight), from the shared fetcher or
    the cache. MacroUnavailable when neither has it."""
    from src.macro_data import _read_cache, fetch_fred_feature
    path = os.path.join(cache_dir(cfg2), f'{name}.csv')
    if allow_fetch:
        frame, _src = fetch_fred_feature({name: sid}, start, end, lambda raw: raw[[name]].dropna(), path)
    else:
        frame = _read_cache(path)
    if frame is None or name not in frame or frame[name].dropna().empty:
        raise MacroUnavailable(f'FRED {sid} ({name}) is unreachable and not cached')
    s = frame[name].dropna()
    s.index = pd.DatetimeIndex(s.index).normalize()
    return s[~s.index.duplicated(keep='last')].sort_index()


def _resolve(names, aliases):
    have = {n.upper(): n for n in names}
    for a in aliases:
        if a.upper() in have:
            return have[a.upper()]
    return None


def fetch_mt5_d1(aliases, count: int, mt5=None):
    """(frame, symbol) of closed D1 bars from the terminal, or None."""
    from . import data as D
    if mt5 is None:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            return None
    if not mt5.initialize():
        return None
    try:
        symbol = _resolve([s.name for s in (mt5.symbols_get() or [])], aliases)
        if symbol is None:
            return None
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, int(count))
        tick = mt5.symbol_info_tick(symbol)
    finally:
        mt5.shutdown()
    if rates is None or len(rates) == 0:
        return None
    return D._closed_only(D._rates_to_frame(rates), 'D1', getattr(tick, 'time', None)), symbol


def mt5_close(cfg2, name: str, allow_fetch: bool = True, mt5=None) -> pd.Series:
    """Daily closes of an MT5 instrument, cached append-only."""
    from . import data as D
    path = os.path.join(cache_dir(cfg2), f'{name}_d1.csv')
    if allow_fetch:
        got = fetch_mt5_d1(cfg2['study']['macro']['mt5'][name], cfg2['study']['macro']['bars'], mt5)
        if got is not None:
            D.append_cache(path, got[0])
    df = D.read_cache(path)
    if df is None or df.empty:
        raise MacroUnavailable(f'MT5 {"/".join(cfg2["study"]["macro"]["mt5"][name])} ({name}) is unreachable and not cached')
    return df['close']


def load_inputs(cfg2, start, end, allow_fetch: bool = True, mt5=None) -> dict:
    fred = {n: fred_series(cfg2, n, sid, start, end, allow_fetch) for n, sid in cfg2['study']['macro']['fred'].items()}
    bars = {n: mt5_close(cfg2, n, allow_fetch, mt5) for n in cfg2['study']['macro']['mt5']}
    return {'fred': fred, 'mt5': bars}


# ── alignment (no look-ahead) ───────────────────────────────────────────────

def align_fred(series: pd.Series, gold_index: pd.DatetimeIndex) -> pd.Series:
    """For the gold D1 bar labelled day d: the newest value dated on or before
    the previous business day (Monday -> Friday). Forward-fill only."""
    keys = pd.DatetimeIndex(gold_index).normalize() - pd.offsets.BDay(1)
    s = series.sort_index()
    pos = s.index.searchsorted(keys, side='right') - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=gold_index)


def align_bars(close: pd.Series, gold_index: pd.DatetimeIndex) -> pd.Series:
    """The newest MT5 close labelled on or before the gold bar. Forward-fill only."""
    c = close.sort_index()
    c.index = pd.DatetimeIndex(c.index).normalize()
    c = c[~c.index.duplicated(keep='last')]
    keys = pd.DatetimeIndex(gold_index).normalize()
    pos = c.index.searchsorted(keys, side='right') - 1
    vals = np.where(pos >= 0, c.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=gold_index)


def macro_features(gold: pd.DataFrame, inputs: dict, z_window: int = 60) -> pd.DataFrame:
    """The scale-free macro block on the gold D1 index."""
    idx = gold.index
    ry = align_fred(inputs['fred']['dfii10'], idx)
    usd = align_fred(inputs['fred']['dtwexbgs'], idx)
    vix = align_fred(inputs['fred']['vixcls'], idx)
    silver = align_bars(inputs['mt5']['silver'], idx)
    brent = align_bars(inputs['mt5']['brent'], idx)
    lr = lambda s, k: np.log(s / s.shift(k)) * 100.0
    ratio = np.log(gold['close'] / silver)
    out = pd.DataFrame({
        'real_yield_chg_1': ry.diff(1), 'real_yield_chg_5': ry.diff(5),
        'usd_ret_1': lr(usd, 1), 'usd_ret_5': lr(usd, 5),
        'vix_level': vix, 'vix_chg_5': vix.diff(5),
        'silver_ret_1': lr(silver, 1), 'silver_ret_5': lr(silver, 5),
        'brent_ret_1': lr(brent, 1), 'brent_ret_5': lr(brent, 5),
        'gold_silver_z60': (ratio - ratio.rolling(z_window).mean()) / ratio.rolling(z_window).std(),
    }, index=idx)[MACRO_COLUMNS]
    assert_scale_free(out.columns)
    return out


def features_digest(feats: pd.DataFrame) -> str:
    import hashlib
    return hashlib.sha256(feats.to_csv(float_format='%.6f').encode()).hexdigest()


# ── pinning (registration) ──────────────────────────────────────────────────

def pinned_path(cfg2) -> str:
    return os.path.join(cache_dir(cfg2), 'pinned_features_d1.csv')


def pin_features(cfg2, d1: pd.DataFrame, allow_fetch: bool = True) -> dict:
    """Fetch (or read) every macro series, build the block on the pinned gold D1
    rows and write it to a file whose digest the registration records. FRED
    revises history, so the study reads THIS file, never a re-fetch."""
    start = d1.index[0] - pd.Timedelta(days=30)
    feats = macro_features(d1, load_inputs(cfg2, start, d1.index[-1], allow_fetch),
                           cfg2['study']['macro']['zscore_window'])
    path = pinned_path(cfg2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    feats.to_csv(path, float_format='%.10g')
    complete = feats.notna().all(axis=1)
    return {'features_file': cfg2['paths']['macro_cache'] + '/pinned_features_d1.csv',
            'features_sha256': features_digest(pd.read_csv(path, index_col=0)),
            'complete_rows': int(complete.sum()),
            'first_complete': str(feats.index[complete.to_numpy()][0]) if complete.any() else None}


def load_pinned_features(cfg2, expected_sha: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    path = pinned_path(cfg2)
    if not os.path.exists(path):
        raise MacroUnavailable('the pinned macro feature file is missing')
    raw = pd.read_csv(path, index_col=0)
    if features_digest(raw) != expected_sha:
        raise MacroUnavailable('the pinned macro feature file changed after registration')
    raw.index = pd.to_datetime(raw.index, utc=True)
    out = raw.reindex(index)
    assert_scale_free(out.columns)
    return out[MACRO_COLUMNS]


# ── scoring (round-two validation) ──────────────────────────────────────────

def with_macro(inp, mac: np.ndarray):
    """(base, both): the same rows marked complete in each -- feature-complete
    INCLUDING every macro input -- so the two variants fit and score alike."""
    from src.forecast_eval.challengers.base import Inputs
    complete = inp.extra['complete'] & ~np.isnan(mac).any(axis=1)
    both = Inputs(index=inp.index, close=inp.close, X=np.hstack([inp.X, mac]),
                  extra={**inp.extra, 'complete': complete})
    base = Inputs(index=inp.index, close=inp.close, X=inp.X, extra={**inp.extra, 'complete': complete})
    return base, both


def _row(cell, hypothesis, n, mm, mb, delta, lo, hi, p, alpha, cleared, verdict, notes):
    return {'cell': cell, 'family': 'macro', 'hypothesis': hypothesis, 'n_rows': int(n), 'metric_model': mm,
            'metric_baseline': mb, 'point_delta': delta, 'ci_low': lo, 'ci_high': hi, 'p_value': p,
            'alpha_bonferroni': alpha, 'cleared_bar': cleared, 'verdict': verdict, 'notes': notes}


def score_from_predictions(cfg2, fam, preds, inp_m, tg, pos, n_boot) -> list:
    """Score the macro cells from fitted predictions {(model, 'base'|'macro'): pred}."""
    from src.forecast_eval import stats as ST
    from . import challengers as C
    from . import ranges as R
    alpha = fam['alpha']
    bl = ST.block_length(1, 'D1')
    note = 'fitted and scored on rows where every macro input exists; gross of cost'
    have = lambda name: (name, 'base') in preds and (name, 'macro') in preds
    out = []
    if have('d1_lstm'):
        b, m = preds[('d1_lstm', 'base')], preds[('d1_lstm', 'macro')]
        rows = C.usable(inp_m, pos['score'], tg, ('vol',))
        y = tg['vol'][rows]
        eb, em = np.abs(y - b['vol_pct'][rows]), np.abs(y - m['vol_pct'][rows])
        lo, hi, d = ST.block_bootstrap_delta(eb, em, bl, alpha, n_boot)
        out.append(_row('d1_lstm_macro:vol', 'gold_d1_lstm_volatility_with_vs_without_macro', len(rows),
                        float(em.mean()), float(eb.mean()), d, lo, hi, '', alpha, bool(lo > 0),
                        'CLEARED (macro lowers the move error: CI excludes 0)' if lo > 0
                        else 'DROP (CI includes 0 or favours no macro)',
                        f'MAE with macro {em.mean():.5f} % vs without {eb.mean():.5f} %; block {bl}; {note}'))
        st = cfg2['study']['range']
        sc = pos['score']
        y_v = np.where(inp_m.extra['eligible'] & inp_m.extra['complete'], tg['ret'], np.nan)[sc]
        res = {v: R.run_aci(y_v, R.sigma_from_abs(pr['vol_pct'][sc]), st['scored_level'], st['gamma'],
                            st['window']['D1']) for v, pr in (('base', b), ('macro', m))}
        w = st['window']['D1']
        ok = np.zeros(len(sc), dtype=bool)
        ok[w:] = True
        ok &= np.isfinite(y_v) & np.isfinite(res['base'][0]) & np.isfinite(res['macro'][0])
        a = 1.0 - st['scored_level']
        wb = R.winkler(res['base'][0][ok], res['base'][1][ok], y_v[ok], a)
        wm = R.winkler(res['macro'][0][ok], res['macro'][1][ok], y_v[ok], a)
        if ok.sum() >= 2 * bl:
            lo, hi, d = ST.block_bootstrap_delta(wb, wm, bl, alpha, n_boot)
            verdict = ('CLEARED (macro sharpens the range: CI excludes 0)' if lo > 0
                       else 'DROP (CI includes 0 or favours no macro)')
        else:
            lo = hi = d = float('nan')
            verdict = 'DROP (too few scored rows after the warm-up)'
        out.append(_row('d1_range_macro', 'gold_d1_range_with_vs_without_macro', ok.sum(),
                        float(wm.mean()) if ok.any() else float('nan'),
                        float(wb.mean()) if ok.any() else float('nan'), d, lo, hi, '', alpha,
                        bool(np.isfinite(lo) and lo > 0), verdict,
                        f'Winkler@{st["scored_level"]:.0%}; warm-up {w} rows; block {bl}; {note}'))
    for name in ('d1_lstm', 'd1_gbm'):
        if not have(name):
            continue
        b, m = preds[(name, 'base')], preds[(name, 'macro')]
        rows = C.usable(inp_m, pos['score'], tg, ('ret',))
        y, fm, fb = tg['ret'][rows], m['ret_pct'][rows], b['ret_pct'][rows]
        ok = ~np.isnan(fm) & ~np.isnan(fb)
        cw = ST.clark_west(y[ok], fm[ok], fb[ok], 1)
        cleared = bool(np.isfinite(cw['p_value']) and cw['p_value'] < alpha)
        out.append(_row(f'{name}_macro:ret', f'gold_{name}_return_with_vs_without_macro', ok.sum(),
                        float(np.mean((y[ok] - fm[ok]) ** 2)), float(np.mean((y[ok] - fb[ok]) ** 2)),
                        '', '', '', cw['p_value'], alpha, cleared,
                        'CLEARED (Clark-West p < alpha)' if cleared else 'DROP (Clark-West p >= alpha)',
                        f'MSPE with vs without macro; CW stat {cw["stat"]:.3f}; {note}'))
    return out


def score_macro_cells(cfg, cfg2, reg2, frames) -> tuple[list, list]:
    """Each daily model fitted with and without the macro block on identical
    rows, scored on identical validation rows. Nothing is appended to round
    one's trial log."""
    from . import challengers as C
    from . import features as F
    from . import splits as S
    from . import targets as G
    fam = reg2['families']['macro']
    d1 = frames['D1']
    try:
        mac = load_pinned_features(cfg2, fam['pinned']['features_sha256'], d1.index).to_numpy(float)
    except MacroUnavailable as e:
        return [], [{'cell': c, 'error': str(e)} for c in fam['cells']]
    inp_b, inp_m = with_macro(C.build_inputs(d1, 'D1'), mac)
    tg = G.build_targets(d1['close'])
    pos = S.stage_positions(len(d1), 'validation', cfg)
    cols = F.D1_FEATURE_COLUMNS + MACRO_COLUMNS
    preds, failures = {}, []
    for name in ('d1_lstm', 'd1_gbm'):
        for variant, inp, over in (('base', inp_b, None), ('macro', inp_m, {'columns': cols})):
            try:
                m = C.make(name, cfg, over)
                m.fit(inp, pos['fit'], tg, seed=cfg['seeds'][0])
                preds[(name, variant)] = m.predict(inp)
            except Exception as e:  # noqa: BLE001 -- recorded, never hidden
                failures.append({'cell': f'{name}_macro', 'error': f'{variant}: {type(e).__name__}: {e}'})
            finally:
                try:
                    import keras
                    keras.backend.clear_session()
                except Exception:
                    pass
    results = score_from_predictions(cfg2, fam, preds, inp_m, tg, pos, cfg['bootstrap']['n_boot'])
    return results, failures
