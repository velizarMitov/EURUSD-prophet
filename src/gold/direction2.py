"""Gold direction, second attempt (openspec change gold-forecasting-v2, spec
gold-forecasting/direction-research).

Three cells, each admitted only by the round-one power rule:

  h1_selective  the round-one GBM classifier, forecasting only when |p - 0.5| >= tau;
                tau is chosen on inner walk-forward folds of the TRAINING block as the
                largest value that keeps coverage >= the registered floor, never on
                validation. Abstained bars are never scored.
  h1_barrier    triple-barrier labels: which of +/- m x EWMA volatility is touched
                first by the closes of the next k hours; a time-out is an abstention.
                The EWMA is causal, so no fitted model enters a label.
  h4_dir        non-overlapping 4-hour windows starting on label hours 0, 4, ... 20;
                as-of is the window's last bar, the target is the next window's close.

The label builders here are pure functions of the OHLC data, so the registration
can count each cell's eligible validation rows before any model is fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


def ewma_sigma(close: pd.Series, span: int) -> pd.Series:
    """Causal EWMA standard deviation of hourly log returns, as a fraction."""
    r = np.log(close).diff()
    return np.sqrt((r ** 2).ewm(span=span, adjust=False, min_periods=span).mean())


def barrier_labels(df: pd.DataFrame, k: int, m: float, span: int) -> pd.DataFrame:
    """Per as-of bar t: label 1 (upper first), 0 (lower first), NaN (time-out or
    ineligible). Uses the closes of the next k bars, which must be the next k
    consecutive hours. Columns: label, sigma, eligible."""
    close = df['close']
    sigma = ewma_sigma(close, span)
    idx = df.index
    n = len(df)
    c = close.to_numpy()
    s = sigma.to_numpy()
    t_ns = idx.asi8
    label = np.full(n, np.nan)
    eligible = np.zeros(n, dtype=bool)
    k_ns = k * HOUR.value
    for i in range(n - k):
        if not np.isfinite(s[i]) or t_ns[i + k] - t_ns[i] != k_ns:
            continue
        eligible[i] = True
        up, dn = c[i] * np.exp(m * s[i]), c[i] * np.exp(-m * s[i])
        path = c[i + 1:i + k + 1]
        hit_up = np.nonzero(path >= up)[0]
        hit_dn = np.nonzero(path <= dn)[0]
        first_up = hit_up[0] if hit_up.size else k
        first_dn = hit_dn[0] if hit_dn.size else k
        if first_up < first_dn:
            label[i] = 1.0
        elif first_dn < first_up:
            label[i] = 0.0
    return pd.DataFrame({'label': label, 'sigma': s, 'eligible': eligible}, index=idx)


def barrier_purge(n_fit_end: int, k: int, positions: np.ndarray) -> np.ndarray:
    """Training rows whose k-hour label window stays inside the training block."""
    return positions[positions + k < n_fit_end]


def h4_windows(df: pd.DataFrame, start_hours=(0, 4, 8, 12, 16, 20)) -> pd.DataFrame:
    """One row per complete 4-hour window: as-of = its last H1 bar, the window's
    close, the next window's close, the 4-hour log return and its direction, and
    whether the next window is complete and starts right after this one."""
    idx = df.index
    pos = {t: i for i, t in enumerate(idx)}
    rows = []
    starts = [t for t in idx if t.hour in start_hours and t.minute == 0]
    for t0 in starts:
        bars = [t0 + j * HOUR for j in range(4)]
        if not all(b in pos for b in bars):
            continue
        rows.append({'start': t0, 'asof': bars[-1], 'asof_pos': pos[bars[-1]],
                     'close': float(df['close'].iloc[pos[bars[-1]]])})
    w = pd.DataFrame(rows)
    if w.empty:
        return pd.DataFrame(columns=['start', 'asof', 'asof_pos', 'close', 'next_close', 'ret_pct', 'dir', 'eligible'])
    nxt_start = w['start'].shift(-1)
    w['next_close'] = w['close'].shift(-1)
    w['eligible'] = (nxt_start - w['start']) == pd.Timedelta(hours=4)
    w['ret_pct'] = np.log(w['next_close'] / w['close']) * 100.0
    w['dir'] = np.where(w['ret_pct'] > 0, 1.0, np.where(w['ret_pct'] < 0, 0.0, np.nan))
    w.loc[~w['eligible'], ['ret_pct', 'dir']] = np.nan
    return w.reset_index(drop=True)


def choose_tau(p_up: np.ndarray, floor: float) -> float:
    """The largest confidence threshold on |p - 0.5| whose coverage on these
    (training-fold) predictions is still >= floor."""
    conf = np.sort(np.abs(np.asarray(p_up, dtype=float) - 0.5))[::-1]
    conf = conf[np.isfinite(conf)]
    if conf.size == 0:
        return 0.0
    keep = int(np.ceil(floor * conf.size))
    return float(conf[min(max(keep, 1), conf.size) - 1])


def selective(p_up: np.ndarray, tau: float) -> np.ndarray:
    """1 / 0 where the model forecasts, NaN where it abstains."""
    p = np.asarray(p_up, dtype=float)
    out = np.where(p > 0.5, 1.0, 0.0)
    out[~(np.abs(p - 0.5) >= tau) | ~np.isfinite(p)] = np.nan
    return out


def inner_tau(make_model, inputs, fit_pos: np.ndarray, targets: dict, floor: float, splits: int,
              seed: int) -> tuple[float, list]:
    """tau from walk-forward folds INSIDE the training block: fit on folds
    1..k, predict fold k+1, pool those out-of-sample probabilities, take the
    largest tau that keeps their coverage >= floor. Validation rows are never
    passed in, so they cannot shape tau. Returns (tau, positions predicted)."""
    folds = np.array_split(np.asarray(fit_pos), splits)
    pooled, seen = [], []
    for k in range(1, len(folds)):
        train = np.concatenate(folds[:k])
        m = make_model()
        m.fit(inputs, train, targets, seed=seed)
        p = m.predict(inputs)['p_up'][folds[k]]
        pooled.append(p)
        seen.append(folds[k])
    p = np.concatenate(pooled) if pooled else np.array([])
    return choose_tau(p[np.isfinite(p)], floor), list(np.concatenate(seen)) if seen else []


# ── scoring (round-two validation) ──────────────────────────────────────────

def _underpowered(cell: str, n: int, alpha: float) -> dict:
    from . import prereg2 as P2
    return {'cell': cell, 'family': 'direction2', 'hypothesis': f'gold_{cell}_direction_round_two',
            'n_rows': n, 'metric_model': '', 'metric_baseline': '', 'point_delta': '', 'ci_low': '',
            'ci_high': '', 'p_value': '', 'alpha_bonferroni': alpha, 'cleared_bar': '',
            'verdict': P2.UNDERPOWERED, 'notes': 'not scored: below the registered power requirement'}


def _as_direction2(r: dict, cell: str, extra: str) -> dict:
    out = dict(r, cell=cell, family='direction2', hypothesis=f'gold_{cell}_direction_round_two')
    out['notes'] = f'{extra}; {r["notes"]}'
    return out


def score_direction2_cells(cfg, cfg2, reg2, frames, make=None) -> tuple[list, list]:
    """Admitted cells are fitted on [0:70] and scored on their covered
    validation rows; UNDERPOWERED cells are recorded without a score."""
    from . import challengers as C
    from . import splits as S
    from . import study as ST
    from . import targets as G
    make = make or (lambda name: C.make(name, cfg))
    fam = reg2['families']['direction2']
    alpha, n_boot = fam['alpha'], cfg['bootstrap']['n_boot']
    status = {r['cell']: r['status'] for r in fam['power_table']}
    n_reg = {r['cell']: r['n_validation'] for r in fam['power_table']}
    results, failures = [], []
    for cell in fam['cells']:
        if status.get(cell) != 'ADMITTED':
            results.append(_underpowered(cell, n_reg.get(cell, 0), alpha))
    h1 = frames['H1']
    inp = C.build_inputs(h1, 'H1')
    tg = G.build_targets(h1['close'])
    pos = S.stage_positions(len(h1), 'validation', cfg)
    seed = cfg['seeds'][0]
    st = cfg2['study']['direction2']

    if status.get('h1_selective') == 'ADMITTED':
        try:
            tau, _ = inner_tau(lambda: make('h1_gbm'), inp, pos['fit'], tg, fam['selective_floor'],
                               st['selective']['inner_splits'], seed)
            m = make('h1_gbm')
            m.fit(inp, pos['fit'], tg, seed=seed)
            p = m.predict(inp)['p_up']
            sel = selective(p, tau)
            rows = C.usable(inp, pos['score'], tg, ('dir',))
            covered = rows[np.isfinite(sel[rows])]
            fit_rows = C.usable(inp, pos['fit'], tg, ('dir',))
            r = ST.score_direction('h1_selective', 'H1', {'p_up': p}, tg, covered, fit_rows, alpha, 'ADMITTED', n_boot)
            results.append(_as_direction2(r, 'h1_selective', f'tau {tau:.4f} from inner training folds; '
                                          f'covered {len(covered)} of {len(rows)} validation rows '
                                          f'({len(covered) / max(len(rows), 1):.1%}); abstentions not scored'))
        except Exception as e:  # noqa: BLE001 -- recorded, never hidden
            failures.append({'cell': 'h1_selective', 'error': f'{type(e).__name__}: {e}'})

    if status.get('h1_barrier') == 'ADMITTED':
        try:
            b = st['barrier']
            lab = barrier_labels(h1, b['k_hours'], b['m'], b['ewma_span'])['label'].to_numpy()
            tg_b = dict(tg, dir=lab)
            fit = barrier_purge(pos['fit_end'], b['k_hours'], pos['fit'])
            m = make('h1_gbm')
            m.fit(inp, fit, tg_b, seed=seed)
            p = m.predict(inp)['p_up']
            rows = C.usable(inp, pos['score'], tg_b, ('dir',))
            fit_rows = C.usable(inp, fit, tg_b, ('dir',))
            r = ST.score_direction('h1_barrier', 'H1', {'p_up': p}, tg_b, rows, fit_rows, alpha, 'ADMITTED', n_boot)
            results.append(_as_direction2(r, 'h1_barrier', f'k {b["k_hours"]} h, m {b["m"]} x EWMA({b["ewma_span"]}); '
                                          f'{len(fit_rows)} purged training labels; time-outs not scored'))
        except Exception as e:  # noqa: BLE001
            failures.append({'cell': 'h1_barrier', 'error': f'{type(e).__name__}: {e}'})
    return results, failures
