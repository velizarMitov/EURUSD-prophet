"""One-shot scoring of both families (spec fx-ny-orb-hot/evaluation, range-forward; design D6).

    python -m src.fx_ny_orb.score --family historical   orb_hot_net_r, orb_hot_minus_cold
    python -m src.fx_ny_orb.score --family forward      EURUSD_harx_vs_har, GBPUSD_harx_vs_har

Historical: refuses until the pre-registration is committed and unchanged (prereg.gate); only then are
outcomes computed (saved to hist_trades.csv.gz). Per business day the hot and non-hot net R are summed
(S) and counted (N); a circular 5-day block bootstrap draws the same blocks for all four series and
recomputes the hot mean and the hot-minus-cold difference. A resample with no hot (or no cold) trade
is dropped and counted; more than 1 % dropped refuses. KEEP when the alpha-quantile is above zero.

Forward: per pair, the per-day squared-error difference (HAR-X minus HAR) on settled forward days
after the pre-registration commit; scored once after `min_days` days, or at the deadline (then
UNDERPOWERED -- NO DECISION when fewer). fx_fib's test_cell, KEEP when the upper bound is below zero.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..fx_bb.score import day_sums
from . import config_digest, load_config
from .registry import (CELL_DIFF, CELL_HOT, FWD, FWD_FIELDS, HIST, HIST_FIELDS, PREREG_JSON, AlreadyScored,  # noqa: F401
                       StageOrderError, append, guard, path, read_rows, results_dir)

NOTE = ('Net of the recorded spread plus 0.4 bps slippage, not a trading result. HAR-X was chosen after its range '
        'MSE was seen (fx-ny-volatility, descriptive); the breakout outcomes were not computed before the committed '
        'pre-registration. Earlier fx families fx-fibonacci, fx-session-ny, fx-rates-2y, fx-bb-reversion and '
        'fx-bb-sma-exit were DROP.')
HIST_FILE = 'hist_trades.csv.gz'


class TooManyDropped(RuntimeError):
    pass


def _sha(p):
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def pair_block_means(S_hot, N_hot, S_cold, N_cold, block: int, resamples: int, seed: int) -> tuple:
    """(hot means, hot-minus-cold differences) under one circular block bootstrap; the same draws as
    fx_bb.score.ratio_block_means, so with no cold trades the hot means reproduce it exactly.
    NaN where a resample drew no hot (or, for the difference, no cold) trade."""
    series = [np.asarray(x, dtype=float) for x in (S_hot, N_hot, S_cold, N_cold)]
    n = len(series[0])
    block = max(1, min(block, n))
    k, r = divmod(n, block)

    def sums(x):
        ext = np.r_[x, x[:block]]
        c = np.r_[0.0, np.cumsum(ext)]
        return c[np.arange(n) + block] - c[np.arange(n)], c[np.arange(n) + r] - c[np.arange(n)]

    blocks = [sums(x) for x in series]
    rng = np.random.default_rng(seed)
    hot, diff = np.empty(resamples), np.empty(resamples)
    for i0 in range(0, resamples, 500):
        m = min(500, resamples - i0)
        idx = rng.integers(0, n, size=(m, k))
        tot = [f[idx].sum(axis=1) for f, _ in blocks]
        if r:
            j = rng.integers(0, n, size=m)
            tot = [t + p[j] for t, (_, p) in zip(tot, blocks)]
        sh, nh, sc, nc = tot
        with np.errstate(invalid='ignore', divide='ignore'):
            h = np.where(nh > 0, sh / nh, np.nan)
            c = np.where(nc > 0, sc / nc, np.nan)
        hot[i0:i0 + m] = h
        diff[i0:i0 + m] = h - c
    return hot, diff


def _bound(boots: np.ndarray, alpha: float, max_drop: float) -> dict:
    ok = boots[np.isfinite(boots)]
    dropped = 1 - len(ok) / len(boots)
    if dropped > max_drop:
        raise TooManyDropped(f'{dropped:.2%} of resamples drew no trade on one side; more than {max_drop:.0%}')
    se = float(ok.std(ddof=1))
    z = NormalDist().inv_cdf
    bound = float(np.quantile(ok, alpha))
    return {'bound': bound, 'boot_se': se, 'mde_80': float((z(1 - alpha) + z(0.80)) * se), 'dropped': dropped,
            'verdict': 'KEEP' if bound > 0 else 'DROP'}


def simulate_history(cfg: dict, assembled: tuple | None = None) -> pd.DataFrame:
    """Every simulable day of the window with its decision and, for trades, its result."""
    from . import rule as R
    from .history import assemble
    days, day_bars = assembled if assembled is not None else assemble(cfg)
    rows = []
    for _, d in days.iterrows():
        base = d.to_dict()
        if d['day_reason']:
            rows.append({**base, 'reason': d['day_reason']})
            continue
        t = R.simulate_day(day_bars[d['pair']][d['date']], d['F'], bool(d['hot']), float(cfg['pip'][d['pair']]), cfg)
        rows.append({**base, **t})
    return pd.DataFrame(rows)


def score_historical(cfg: dict | None = None, now_iso: str | None = None, assembled: tuple | None = None,
                     gate=None) -> list:
    from .prereg import gate as prereg_gate
    cfg = cfg or load_config()
    guard(cfg)
    (gate or prereg_gate)(cfg)                     # refuses before any outcome is computed
    now_iso = now_iso or datetime.now(timezone.utc).isoformat(timespec='seconds')
    have = {r['hypothesis'] for r in read_rows(cfg, HIST)}
    if have & {CELL_HOT, CELL_DIFF}:
        raise AlreadyScored('the historical cells are already scored; each cell is scored once')
    sim = simulate_history(cfg, assembled)
    os.makedirs(results_dir(cfg), exist_ok=True)
    sim.to_csv(path(cfg, HIST_FILE), index=False, compression='gzip')
    tr = sim[sim['reason'] == '']
    hot, cold = tr[tr['hot'].astype(bool)], tr[~tr['hot'].astype(bool)]
    first, last = cfg['data']['window_start'], cfg['data']['window_end']
    naive = lambda s: pd.DatetimeIndex(s).tz_localize(None) if pd.DatetimeIndex(s).tz is not None else pd.DatetimeIndex(s)  # noqa: E731
    Sh, Nh = day_sums(naive(hot['date']), hot['r_net'], first, last)
    Sc, Nc = day_sums(naive(cold['date']), cold['r_net'], first, last)
    sc = cfg['scoring']
    hb, db = pair_block_means(Sh, Nh, Sc, Nc, sc['block_days'], sc['resamples'], sc['seed_historical'])
    hm, cm = float(hot['r_net'].mean()), float(cold['r_net'].mean())
    common = {'date': now_iso[:10], 'family': 'historical', 'n_trades_hot': len(hot), 'n_trades_cold': len(cold),
              'n_dates_hot': int(hot['date'].nunique()), 'window_start': first, 'window_end': last, 'hot_mean': hm,
              'cold_mean': cm, 'bound_quantile': sc['alpha'], 'alpha': sc['alpha'], 'block_days': sc['block_days'],
              'resamples': sc['resamples'], 'seed': sc['seed_historical'], 'config_digest': config_digest(cfg),
              'prereg_sha256': _sha(path(cfg, PREREG_JSON)), 'notes': NOTE}
    rows = []
    for n, (cell, stat, boots, metric) in enumerate(((CELL_HOT, hm, hb, 'mean net R, hot days'),
                                                     (CELL_DIFF, hm - cm, db, 'mean net R, hot minus non-hot')), 1):
        b = _bound(boots, sc['alpha'], float(sc['max_dropped_resamples']))
        rows.append({**common, 'n': n, 'hypothesis': cell, 'metric': metric, 'statistic': stat,
                     'bound': b['bound'], 'boot_se': b['boot_se'], 'mde_80': b['mde_80'],
                     'dropped_resamples': b['dropped'], 'verdict': b['verdict']})
    append(cfg, HIST, HIST_FIELDS, rows)
    return rows


def score_forward(cfg: dict | None = None, today: pd.Timestamp | None = None, gate=None) -> list:
    from ..fx_fib.score import test_cell
    from .prereg import gate as prereg_gate
    from .frozen import manifest_sha
    from .live import read_log
    cfg = cfg or load_config()
    guard(cfg)
    committed = pd.Timestamp((gate or prereg_gate)(cfg)).tz_convert('UTC').normalize()
    today = pd.Timestamp(today or datetime.now(timezone.utc)).normalize()
    if today.tzinfo is None:
        today = today.tz_localize('UTC')
    have = {r['hypothesis'] for r in read_rows(cfg, FWD)}
    cells = [f'{p}_harx_vs_har' for p in cfg['pairs']]
    if have & set(cells):
        raise AlreadyScored('the forward cells are already scored; each cell is scored once')
    st = read_log(cfg, 'settlements')
    st = st[(st['status'] == 'ok') & (st['date'] > committed)] if len(st) else st
    fw, sc = cfg['forward'], cfg['scoring']
    deadline = pd.Timestamp(fw['deadline'], tz='UTC')
    counts = {p: int((st['pair'] == p).sum()) if len(st) else 0 for p in cfg['pairs']}
    if today < deadline and min(counts.values()) < int(fw['min_days']):
        raise StageOrderError(f'forward scoring waits for {fw["min_days"]} settled days per pair (have {counts}) '
                              f'or the deadline {fw["deadline"]}')
    rows = []
    for n, pair in enumerate(cfg['pairs'], 1):
        d = st[st['pair'] == pair].sort_values('date') if len(st) else st
        base = {'n': n, 'date': str(today.date()), 'hypothesis': f'{pair}_harx_vs_har', 'pair': pair,
                'n_days': len(d), 'alpha': sc['alpha'], 'block_days': sc['block_days'], 'resamples': sc['resamples'],
                'seed': sc['seed_forward'], 'config_digest': config_digest(cfg),
                'frozen_manifest_sha256': manifest_sha(cfg)}
        if len(d) < int(fw['min_days']):
            rows.append({**base, 'verdict': 'UNDERPOWERED — NO DECISION',
                         'notes': f'{len(d)} settled forward days at the deadline, fewer than {fw["min_days"]}. Gross.'})
            continue
        y = d['y'].to_numpy(float)
        se_x, se_h = (y - d['p_harx'].to_numpy(float)) ** 2, (y - d['p_har'].to_numpy(float)) ** 2
        res = test_cell(se_x - se_h, sc['alpha'], sc['block_days'], sc['resamples'], sc['seed_forward'])
        rows.append({**base, 'first_day': str(d['date'].min().date()), 'last_day': str(d['date'].max().date()),
                     'mse_harx': float(se_x.mean()), 'mse_har': float(se_h.mean()),
                     'rel_mse_reduction': float(1 - se_x.mean() / se_h.mean()),
                     **{k: res[k] for k in ('mean_diff', 'ci_upper', 'boot_se', 'mde_80', 'verdict')},
                     'notes': 'Frozen OLS HAR-X vs HAR on live forward days after the pre-registration commit; '
                              'gross predictive result, not a trading result.'})
    append(cfg, FWD, FWD_FIELDS, rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description='score one fx-ny-orb-hot family, once')
    ap.add_argument('--family', choices=['historical', 'forward'], required=True)
    a = ap.parse_args(argv)
    for r in (score_historical() if a.family == 'historical' else score_forward()):
        print({k: r.get(k) for k in ('hypothesis', 'statistic', 'hot_mean', 'cold_mean', 'bound', 'mde_80',
                                     'n_trades_hot', 'n_days', 'rel_mse_reduction', 'ci_upper', 'verdict') if k in r})


if __name__ == '__main__':
    main()
