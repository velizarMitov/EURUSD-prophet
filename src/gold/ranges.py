"""Gold price ranges (openspec change gold-forecasting-v2, spec
gold-forecasting/price-ranges).

A range is "between X and Y" at a coverage level, for the next hour or day:

    sigma  = E|r| x sqrt(pi/2)        (the frozen LSTM forecasts E|r| in percent)
    range  = close x exp(+/- q_t x sigma / 100)

centred on the as-of close: return has no proven skill, so no drift is assumed.
q_t comes from adaptive conformal inference (Gibbs & Candes 2021): the target
miss rate alpha_t is nudged after every settled bar, alpha_t+1 = alpha_t +
gamma (alpha - miss_t), and q_t is the (1 - alpha_t) quantile of the last W
standardised errors |z| = |r| / sigma. Only bars settled BEFORE bar t enter
bar t's quantile: the forecast is made, then the bar settles, then the state
moves.

Scoring works in return-percent space, so it is scale-free: the Winkler
interval score is the width plus 2/alpha times any miss distance.

Reads saved predictions only; runs no model.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from src.forecast_eval import stats as ST

SQRT_PI_2 = math.sqrt(math.pi / 2.0)


def sigma_from_abs(e_abs):
    """Normal-scale sigma from a forecast of E|r| (E|r| = sigma x sqrt(2/pi))."""
    return np.asarray(e_abs, dtype=float) * SQRT_PI_2


def normal_q(level: float) -> float:
    """Two-sided standard-normal quantile for a central `level` interval."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2.0))


class ACI:
    """Adaptive conformal calibration for one coverage level."""

    def __init__(self, level: float, gamma: float, window: int, seed_z=()):
        self.level = level
        self.target = 1.0 - level
        self.alpha_t = self.target
        self.gamma = gamma
        self.buf = deque((abs(float(z)) for z in seed_z if np.isfinite(z)), maxlen=window)

    def q(self) -> float | None:
        if not self.buf:
            return None
        a = min(max(self.alpha_t, 0.0), 1.0)
        if a <= 0.0:
            return float(max(self.buf))
        return float(np.quantile(np.fromiter(self.buf, float), 1.0 - a))

    def bounds_pct(self, sigma_pct: float):
        """(lo, hi) in return percent around zero, or (None, None) before any data."""
        q = self.q()
        if q is None or not np.isfinite(sigma_pct):
            return None, None
        return -q * sigma_pct, q * sigma_pct

    def step(self, z_abs: float, covered: bool) -> None:
        """Move the state with a bar that has just settled."""
        if not np.isfinite(z_abs):
            return
        self.alpha_t += self.gamma * (self.target - (0.0 if covered else 1.0))
        self.buf.append(abs(float(z_abs)))

    def state(self) -> dict:
        return {'level': self.level, 'alpha_t': self.alpha_t, 'q': self.q(), 'n': len(self.buf)}


def price_bounds(close: float, lo_pct: float, hi_pct: float):
    return close * math.exp(lo_pct / 100.0), close * math.exp(hi_pct / 100.0)


def winkler(lo, hi, y, alpha: float) -> np.ndarray:
    """Interval score: width, plus 2/alpha times how far y falls outside."""
    lo, hi, y = (np.asarray(v, dtype=float) for v in (lo, hi, y))
    return (hi - lo) + (2.0 / alpha) * np.clip(lo - y, 0, None) + (2.0 / alpha) * np.clip(y - hi, 0, None)


def run_aci(y_ret, sigma, level: float, gamma: float, window: int, seed_z=()):
    """Walk a series in time order: range for bar t from the state before t,
    then settle t. Returns lo, hi (return %), covered, q per bar."""
    aci = ACI(level, gamma, window, seed_z)
    n = len(y_ret)
    lo, hi, q = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    covered = np.full(n, np.nan)
    for i in range(n):
        s, y = sigma[i], y_ret[i]
        a, b = aci.bounds_pct(s)
        if a is not None:
            lo[i], hi[i], q[i] = a, b, aci.q()
        if np.isfinite(y) and np.isfinite(s) and s > 0:
            inside = a is not None and a <= y <= b
            if a is not None:
                covered[i] = float(inside)
            aci.step(y / s, inside)
    return lo, hi, covered, q


def static_range(sigma, level: float):
    q = normal_q(level)
    return -q * np.asarray(sigma, float), q * np.asarray(sigma, float)


def score_range_cell(cell: str, cad: str, pred: pd.DataFrame, model: str, baseline: str,
                     settings: dict, alpha: float, n_boot: int) -> dict:
    """Score one range cell on saved study predictions (`pred` is a
    *_predictions_{d1,h1}.csv frame). The first `window` rows warm up the
    calibration and are not scored; the baseline is scored on the same rows."""
    level = settings['scored_level']
    w = settings['window'][cad]
    y = pred['y_ret'].to_numpy(float)
    elig = pred['eligible'].astype(str).str.lower().eq('true').to_numpy()
    s_m = sigma_from_abs(pred[f'{model}__vol_pct'].to_numpy(float))
    s_b = sigma_from_abs(pred[f'{baseline}__vol_pct'].to_numpy(float))
    y_cal = np.where(elig, y, np.nan)                 # ineligible bars never move the calibration
    out = {}
    for lv in settings['levels']:
        lo, hi, cov, q = run_aci(y_cal, s_m, lv, settings['gamma'], w)
        out[lv] = (lo, hi, cov)
    lo_m, hi_m, cov_m = out[level]
    lo_b, hi_b = static_range(s_b, level)
    lo_g, hi_g = static_range(s_m, level)
    scored = np.zeros(len(y), dtype=bool)
    scored[w:] = True
    scored &= elig & np.isfinite(y) & np.isfinite(lo_m) & np.isfinite(lo_b)
    a = 1.0 - level
    wm = winkler(lo_m[scored], hi_m[scored], y[scored], a)
    wb = winkler(lo_b[scored], hi_b[scored], y[scored], a)
    wg = winkler(lo_g[scored], hi_g[scored], y[scored], a)
    bl = ST.block_length(1, cad)
    lo_ci, hi_ci, d = ST.block_bootstrap_delta(wb, wm, bl, alpha, n_boot)
    coverage = {lv: float(np.nanmean(out[lv][2][scored])) for lv in settings['levels']}
    cov_b = float(np.mean((y[scored] >= lo_b[scored]) & (y[scored] <= hi_b[scored])))
    tol = settings['coverage_tolerance']
    cov_ok = abs(coverage[level] - level) <= tol
    cleared = bool(lo_ci > 0 and cov_ok)
    if cleared:
        verdict = f'CLEARED (sharper than {baseline}: CI of Winkler_baseline - Winkler_model excludes 0; coverage in tolerance)'
    elif lo_ci > 0:
        verdict = f'DROP (sharper than {baseline} but {level:.0%} coverage {coverage[level]:.1%} is outside +/-{tol:.0%})'
    else:
        verdict = f'DROP (CI includes 0 or favours {baseline})'
    return {'cell': cell, 'family': 'range', 'hypothesis': f'gold_{cell}_aci_{model}_vs_static_{baseline}',
            'n_rows': int(scored.sum()), 'metric_model': float(wm.mean()), 'metric_baseline': float(wb.mean()),
            'point_delta': d, 'ci_low': lo_ci, 'ci_high': hi_ci, 'p_value': '', 'alpha_bonferroni': alpha,
            'cleared_bar': cleared, 'verdict': verdict,
            'notes': (f'Winkler@{level:.0%} model {wm.mean():.5f} vs {baseline} {wb.mean():.5f} '
                      f'vs static-Gaussian {model} {wg.mean():.5f} (reported, not decisive); coverage '
                      + ', '.join(f'{lv:.0%}: {c:.1%}' for lv, c in coverage.items())
                      + f'; {baseline} coverage {cov_b:.1%}; warm-up {w} rows; block {bl}; gross of cost'),
            '_coverage': coverage}


# ── live ranges (logger) ────────────────────────────────────────────────────

RANGE_FIELDS = ['key', 'cell', 'cadence', 'as_of_bar', 'level', 'lo', 'hi', 'lo_pct', 'hi_pct', 'q',
                'alpha_t', 'phase', 'logged_at']
RANGE_SETTLE_FIELDS = ['key', 'level', 'exit_close', 'inside', 'settled_at']


def _seed_z(cfg, cell: str, cadence: str, window: int) -> np.ndarray:
    """|z| of the newest `window` eligible rows of the saved study predictions.
    They come from the test-stage fit of the same cell, not from the final refit
    (no saved predictions of the final refit exist); ACI corrects within W bars."""
    import os
    from . import repo_path
    path = os.path.join(repo_path(cfg['paths']['results']), f'test_predictions_{cadence.lower()}.csv')
    if not os.path.exists(path):
        return np.array([])
    p = pd.read_csv(path)
    ok = p['eligible'].astype(str).str.lower().eq('true') & p['y_ret'].notna() & p[f'{cell}__vol_pct'].notna()
    p = p[ok].tail(window)
    return (p['y_ret'] / sigma_from_abs(p[f'{cell}__vol_pct'])).abs().to_numpy()


def live_aci(cfg, cfg2, cell: str, cadence: str, level: float, before, preds: list, setts: dict) -> ACI:
    """The calibration as it stood just before bar `before`: the seed, then every
    live bar of this cell that settled with an as-of bar earlier than `before`,
    replayed in time order."""
    st = cfg2['study']['range']
    w = st['window'][cadence]
    aci = ACI(level, st['gamma'], w, _seed_z(cfg, cell, cadence, w))
    rows = sorted((r for r in preds if r['cell'] == cell and pd.Timestamp(r['as_of_bar']) < before
                   and r['key'] in setts), key=lambda r: pd.Timestamp(r['as_of_bar']))
    for r in rows:
        s = setts[r['key']]
        if str(s.get('eligible', '')).lower() != 'true':
            continue
        try:
            sig = float(sigma_from_abs(float(r['vol_pct'])))
            y = float(s['realised_ret_pct'])
        except (TypeError, ValueError):
            continue
        a, b = aci.bounds_pct(sig)
        aci.step(y / sig, a is not None and a <= y <= b)
    return aci


def forecast_rows(cfg, cfg2, new_preds: list, all_preds: list, setts: dict, now_iso: str) -> list:
    """Range rows for the newly logged LSTM forecasts, at every configured level."""
    models = {spec['model']: spec['cadence'] for spec in cfg2['study']['range']['cells'].values()}
    out = []
    for p in new_preds:
        if p['cell'] not in models or p.get('vol_pct') in ('', None):
            continue
        sig = float(sigma_from_abs(float(p['vol_pct'])))
        close = float(p['as_of_close'])
        for level in cfg2['study']['range']['levels']:
            aci = live_aci(cfg, cfg2, p['cell'], p['cadence'], level, pd.Timestamp(p['as_of_bar']), all_preds, setts)
            a, b = aci.bounds_pct(sig)
            if a is None:
                continue
            lo, hi = price_bounds(close, a, b)
            out.append({'key': p['key'], 'cell': p['cell'], 'cadence': p['cadence'], 'as_of_bar': p['as_of_bar'],
                        'level': level, 'lo': round(lo, 2), 'hi': round(hi, 2), 'lo_pct': a, 'hi_pct': b,
                        'q': aci.q(), 'alpha_t': aci.alpha_t, 'phase': p.get('phase', ''), 'logged_at': now_iso})
    return out


def settle_rows(range_rows: list, settled_keys: set, setts: dict, now_iso: str) -> list:
    """For every range whose forecast has settled and that has no settlement yet:
    did the exit close fall inside it?"""
    out = []
    for r in range_rows:
        k = (r['key'], str(r['level']))
        if k in settled_keys or r['key'] not in setts:
            continue
        s = setts[r['key']]
        try:
            exit_close = float(s['exit_close'])
            inside = float(r['lo']) <= exit_close <= float(r['hi'])
        except (TypeError, ValueError):
            continue
        out.append({'key': r['key'], 'level': r['level'], 'exit_close': exit_close,
                    'inside': inside, 'settled_at': now_iso})
    return out
