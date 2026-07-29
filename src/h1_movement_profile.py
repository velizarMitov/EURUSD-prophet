"""
EURUSD H1 — MOVEMENT BY NEW YORK HOUR AND SESSION (descriptive statistics).

Answers one question: in New York time, which hours and which sessions carry the
most price movement across the full MT5 EURUSD history?

DESCRIPTIVE ONLY. No model, no hypothesis, no verdict, no alpha, no P&L, no
dealing cost, no trading simulation. Nothing here is registered anywhere and no
results/*hypothesis_log.csv is created, appended to or modified.

Reads the already-built and verified results/pooled_h1/EURUSD_h1_newyork.csv
(from src/h1_newyork_time.py) and never rebuilds it, refetches it or modifies
it. Writes ONLY results/h1_movement_by_ny_hour.csv and
results/h1_movement_by_ny_session.csv.

THE TWO SCOPES — different quantities, different data, on purpose
----------------------------------------------------------------
  SCOPE A — MAGNITUDE (direction-free: |move|, bar range).
    FULL history, all 70,000 bars. Intraday FX volatility seasonality is a
    robust structural property, not a searchable signal, so measuring it on the
    full sample contaminates nothing.

  SCOPE B — DIRECTION (signed: up-rate, signed pips).
    RESTRICTED to the first 85% of the series. A signed per-hour drift IS a
    tradeable signal; measuring it on the reserved test block would spend the
    only clean slice this project has left for hour-of-day questions.

Every output column carries its scope in its name (`_scopeA_full` /
`_scopeB_first85`). The two are never mixed in an unlabelled column.

CLOSE-TO-CLOSE IS MEASURED ON CONTIGUOUS BARS ONLY
--------------------------------------------------
The next-bar return is computed only where the next bar really is the next
hour. Roughly 600 bars sit at a week or holiday boundary, and almost all of
them are the Friday 16:00 NY bar, whose naive `shift(-1)` "next bar" is the
Sunday-evening reopen — a weekend gap, not an hourly move. Including those
would inflate the 16:00 NY hour by mixing a two-day gap into a one-hour
statistic. Every excluded bar is counted and reported.

This differs from the H_dir programs, which used a plain `shift(-1)` for their
TARGET. That is a deliberate difference of purpose (a label vs a measurement of
how far price travels in an hour), it is reported rather than silent, and it
does not touch those programs. Intrabar range is unaffected either way.

CONTEXT THE RANKING DOES NOT CONTAIN
------------------------------------
  * The hours with the largest movement are not automatically the best to
    trade. Larger moves cut both ways -- the adverse excursion is larger too.
  * Dealing spreads are TIGHTEST in the deepest-liquidity hours (the London/NY
    overlap) and WIDEST in thin hours and around the roll. A thin hour with a
    large mean move may still be the worse choice once the dealing cost is
    paid, and this report does not measure dealing cost.
  * Magnitude (scope A) says where there is something to capture. It says
    nothing about whether it is capturable. Those are different questions.
"""

import os

import numpy as np
import pandas as pd

# Imported UNCHANGED -- the session definition and the pip size have one home.
from src.h1_newyork_time import (
    NY_HISTORY_CSV, NY_SESSIONS, SESSION_ORDER, PIP, ny_session,
)

DIRECTION_SCOPE_FRACTION = 0.85       # scope B ends here; [85%:100%] is RESERVED
EARLY_YEARS = 3                       # stability: earliest N vs latest N years
LATE_YEARS = 3

HOUR_CSV = 'results/h1_movement_by_ny_hour.csv'
SESSION_CSV = 'results/h1_movement_by_ny_session.csv'

MAGNITUDE_SUFFIX = '_scopeA_full'
DIRECTION_SUFFIX = '_scopeB_first85'


class TestBlockTouchedError(RuntimeError):
    """Raised if a SIGNED (directional) statistic would read a bar beyond the
    85% boundary. Magnitude may use the whole series; direction may not."""


# ───────────────────────── data + derived series ──────────────────────────────

def load_newyork_frame(path: str = NY_HISTORY_CSV) -> pd.DataFrame:
    """
    Read the already-verified New York history. Read-only: this module never
    rebuilds, refetches or rewrites it.
    """
    df = pd.read_csv(path, index_col=0)
    # The NY index alternates -05:00 and -04:00 across DST, so it must be parsed
    # through UTC and converted back -- a plain parse yields mixed-offset objects.
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index, utc=True)).tz_convert('America/New_York')
    df.index.name = 'ny_timestamp'
    return df.sort_index()


def contiguous_next_bar_mask(ny_df: pd.DataFrame) -> np.ndarray:
    """
    True where the FOLLOWING bar is exactly one hour later, so a close-to-close
    move really is an hour of trading rather than a weekend or holiday gap.
    The final bar is False (it has no successor).
    """
    idx = pd.DatetimeIndex(ny_df.index)
    mask = np.zeros(len(idx), dtype=bool)
    if len(idx) > 1:
        mask[:-1] = np.diff(idx.to_numpy()) == np.timedelta64(1, 'h')
    return mask


def derive_movement_columns(ny_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the movement measures. Prices are read, never written.

      signed_pips      (close[t+1] - close[t]) / PIP   -- NaN across gaps
      abs_pips         |signed_pips|
      abs_return_pct   |close[t+1]/close[t] - 1| * 100
      bar_range_pips   (high - low) / PIP              -- intrabar, no successor
                                                          needed, so never NaN
    """
    out = ny_df.copy()
    close = out['close']
    contiguous = contiguous_next_bar_mask(out)

    nxt = close.shift(-1).where(contiguous)
    out['signed_pips'] = (nxt - close) / PIP
    out['abs_pips'] = out['signed_pips'].abs()
    out['abs_return_pct'] = ((nxt / close) - 1.0).abs() * 100.0
    out['bar_range_pips'] = (out['high'] - out['low']) / PIP
    out['is_contiguous'] = contiguous
    return out


def direction_scope_bound(ny_df: pd.DataFrame,
                          fraction: float = DIRECTION_SCOPE_FRACTION):
    """
    Last positional index and timestamp scope B may read: the first `fraction`
    of the series. Everything at or beyond it is the reserved test block.
    """
    n = len(ny_df)
    end_pos = int(n * fraction)
    return end_pos, pd.DatetimeIndex(ny_df.index)[end_pos - 1]


def assert_direction_scope(index_used, bound_ts):
    """Hard guard for every SIGNED statistic: nothing beyond the 85% boundary."""
    idx = pd.DatetimeIndex(index_used)
    if len(idx) and idx.max() > bound_ts:
        raise TestBlockTouchedError(
            f"signed statistic would read {idx.max()}, beyond the scope-B bound "
            f"{bound_ts} -- the reserved test block must not be indexed."
        )
    return True


# ───────────────────────── aggregation blocks ─────────────────────────────────

def _magnitude_block(frame: pd.DataFrame) -> dict:
    """Scope A: direction-free movement. Bar range uses every bar; close-to-close
    uses the contiguous ones only."""
    a = frame['abs_pips'].dropna()
    r = frame['abs_return_pct'].dropna()
    rng = frame['bar_range_pips'].dropna()
    return {
        f'n_bars{MAGNITUDE_SUFFIX}': int(len(frame)),
        f'n_close_to_close{MAGNITUDE_SUFFIX}': int(len(a)),
        f'n_gap_excluded{MAGNITUDE_SUFFIX}': int(len(frame) - len(a)),
        f'mean_abs_return_pct{MAGNITUDE_SUFFIX}': float(r.mean()) if len(r) else np.nan,
        f'median_abs_return_pct{MAGNITUDE_SUFFIX}': float(r.median()) if len(r) else np.nan,
        f'mean_abs_pips{MAGNITUDE_SUFFIX}': float(a.mean()) if len(a) else np.nan,
        f'median_abs_pips{MAGNITUDE_SUFFIX}': float(a.median()) if len(a) else np.nan,
        f'p75_abs_pips{MAGNITUDE_SUFFIX}': float(a.quantile(0.75)) if len(a) else np.nan,
        f'p90_abs_pips{MAGNITUDE_SUFFIX}': float(a.quantile(0.90)) if len(a) else np.nan,
        f'mean_bar_range_pips{MAGNITUDE_SUFFIX}': float(rng.mean()) if len(rng) else np.nan,
        f'median_bar_range_pips{MAGNITUDE_SUFFIX}': float(rng.median()) if len(rng) else np.nan,
    }


def _direction_block(frame: pd.DataFrame) -> dict:
    """Scope B: signed drift, on the first 85% only. The caller is responsible
    for having already restricted `frame`; assert_direction_scope enforces it."""
    s = frame['signed_pips'].dropna()
    return {
        f'n_bars{DIRECTION_SUFFIX}': int(len(frame)),
        f'n_close_to_close{DIRECTION_SUFFIX}': int(len(s)),
        f'up_rate_pct{DIRECTION_SUFFIX}': float((s > 0).mean() * 100.0) if len(s) else np.nan,
        f'mean_signed_pips{DIRECTION_SUFFIX}': float(s.mean()) if len(s) else np.nan,
        f'median_signed_pips{DIRECTION_SUFFIX}': float(s.median()) if len(s) else np.nan,
    }


def _grouped(full: pd.DataFrame, scope_b: pd.DataFrame, key: str, values,
             label_name: str) -> pd.DataFrame:
    """One row per group: scope-A magnitude from `full`, scope-B direction from
    `scope_b`, side by side but never conflated."""
    rows = []
    for v in values:
        mag = _magnitude_block(full[full[key] == v])
        dirn = _direction_block(scope_b[scope_b[key] == v])
        rows.append({label_name: v, **mag, **dirn})
    return pd.DataFrame(rows)


# ───────────────────────── the three tables ───────────────────────────────────

def hour_table(full: pd.DataFrame, scope_b: pd.DataFrame) -> pd.DataFrame:
    """Per NY hour 0-23, both scopes, ranked by scope-A mean_abs_pips."""
    t = _grouped(full, scope_b, 'ny_hour', range(24), 'ny_hour')
    t['session'] = [ny_session(h) for h in t['ny_hour']]
    t['rank_by_mean_abs_pips'] = (
        t[f'mean_abs_pips{MAGNITUDE_SUFFIX}'].rank(ascending=False).astype(int))
    return t.sort_values('rank_by_mean_abs_pips').reset_index(drop=True)


def session_table(full: pd.DataFrame, scope_b: pd.DataFrame) -> pd.DataFrame:
    """Per pre-registered NY session, both scopes, ranked by scope-A mean."""
    t = _grouped(full, scope_b, 'session', list(SESSION_ORDER), 'session')
    t['ny_hours'] = [f"{NY_SESSIONS[s][0]:02d}:00-{NY_SESSIONS[s][1]:02d}:00"
                     for s in t['session']]
    t['rank_by_mean_abs_pips'] = (
        t[f'mean_abs_pips{MAGNITUDE_SUFFIX}'].rank(ascending=False).astype(int))
    return t.sort_values('rank_by_mean_abs_pips').reset_index(drop=True)


def session_by_weekday_table(full: pd.DataFrame) -> pd.DataFrame:
    """
    Session x FX week day, MAGNITUDE ONLY (scope A). Friday's NY afternoon does
    not behave like Monday's, and a weekly average hides that.
    """
    rows = []
    for sess in SESSION_ORDER:
        for day in sorted(full['fx_week_day'].unique()):
            sub = full[(full['session'] == sess) & (full['fx_week_day'] == day)]
            if not len(sub):
                continue
            rows.append({'session': sess, 'fx_week_day': int(day),
                         **_magnitude_block(sub)})
    return pd.DataFrame(rows)


# ───────────────────────── stability across years ─────────────────────────────

def yearly_hour_stability(full: pd.DataFrame):
    """
    Scope-A mean_abs_pips per NY hour, per calendar year, with the per-year rank
    of every hour (1 = most movement). A ranking computed once over 11 years is
    useless if it drifts, so this is what decides whether the headline table is
    actionable.
    """
    df = full.dropna(subset=['abs_pips']).copy()
    df['year'] = pd.DatetimeIndex(df.index).year
    means = df.groupby(['year', 'ny_hour'])['abs_pips'].mean().unstack('ny_hour')
    ranks = means.rank(axis=1, ascending=False)

    years = list(means.index)
    early, late = years[:EARLY_YEARS], years[-LATE_YEARS:]
    early_rank = means.loc[early].mean(axis=0).rank(ascending=False)
    late_rank = means.loc[late].mean(axis=0).rank(ascending=False)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(early_rank.to_numpy(), late_rank.to_numpy())

    summary = pd.DataFrame({
        'ny_hour': means.columns,
        'mean_abs_pips_overall': [full.loc[full['ny_hour'] == h, 'abs_pips'].mean()
                                  for h in means.columns],
        'rank_min_across_years': ranks.min(axis=0).astype(int).to_numpy(),
        'rank_max_across_years': ranks.max(axis=0).astype(int).to_numpy(),
        'rank_early_years': early_rank.to_numpy(),
        'rank_late_years': late_rank.to_numpy(),
    })
    summary['rank_span'] = (summary['rank_max_across_years']
                            - summary['rank_min_across_years'])

    overall_rank = summary.set_index('ny_hour')['mean_abs_pips_overall'].rank(
        ascending=False)
    top3_overall = set(overall_rank.nsmallest(3).index)
    top3_by_year = {int(y): set(ranks.loc[y].nsmallest(3).index) for y in years}
    n_years_top3_identical = sum(1 for y in years if top3_by_year[y] == top3_overall)

    return {
        'means_by_year': means, 'ranks_by_year': ranks, 'summary': summary,
        'years': years, 'early_years': early, 'late_years': late,
        'spearman_rho': float(rho), 'spearman_p': float(pval),
        'top3_overall': sorted(int(h) for h in top3_overall),
        'top3_by_year': {y: sorted(int(h) for h in s) for y, s in top3_by_year.items()},
        'n_years_top3_identical': int(n_years_top3_identical),
        'max_rank_span_in_top3': int(
            summary[summary['ny_hour'].isin(top3_overall)]['rank_span'].max()),
    }


def mismatch_sensitivity(full: pd.DataFrame, threshold_pct: float = 5.0):
    """
    Session-level scope-A mean_abs_pips with and without the DST-mismatch bars,
    and whether dropping them moves any session by more than `threshold_pct`.
    """
    rows = []
    worst = 0.0
    for sess in SESSION_ORDER:
        inc = full[full['session'] == sess]['abs_pips'].dropna()
        exc = full[(full['session'] == sess) & (~full['is_dst_mismatch'])]['abs_pips'].dropna()
        m_inc, m_exc = float(inc.mean()), float(exc.mean())
        change = 100.0 * (m_exc - m_inc) / m_inc if m_inc else np.nan
        worst = max(worst, abs(change))
        rows.append({'session': sess,
                     'mean_abs_pips_including_mismatch': m_inc,
                     'mean_abs_pips_excluding_mismatch': m_exc,
                     'pct_change_when_excluded': change,
                     'n_mismatch_bars': int(len(inc) - len(exc))})
    table = pd.DataFrame(rows)
    return {'table': table, 'max_abs_pct_change': float(worst),
            'material': bool(worst > threshold_pct),
            'threshold_pct': threshold_pct}


# ───────────────────────── orchestration ──────────────────────────────────────

def run(path: str = NY_HISTORY_CSV, write: bool = True,
        hour_path: str = HOUR_CSV, session_path: str = SESSION_CSV):
    """
    Full descriptive pass. Scope A reads every bar; scope B is hard-bounded at
    the 85% mark and asserted. Writes only the two movement CSVs.
    """
    ny = load_newyork_frame(path)
    full = derive_movement_columns(ny)
    full['session'] = [ny_session(h) for h in full['ny_hour']]

    end_pos, bound_ts = direction_scope_bound(full)
    scope_b = full.iloc[:end_pos]
    assert_direction_scope(scope_b.index, bound_ts)

    hours = hour_table(full, scope_b)
    sessions = session_table(full, scope_b)
    by_weekday = session_by_weekday_table(full)
    stability = yearly_hour_stability(full)
    mismatch = mismatch_sensitivity(full)

    if write:
        os.makedirs(os.path.dirname(hour_path) or '.', exist_ok=True)
        hours.to_csv(hour_path, index=False)

        # One session CSV holding the session, day-of-week, yearly-stability and
        # DST-sensitivity blocks, each tagged by `block` so they never merge.
        blocks = [
            sessions.assign(block='session'),
            by_weekday.assign(block='session_x_fx_week_day'),
            stability['summary'].assign(block='yearly_rank_stability'),
            stability['means_by_year'].reset_index().melt(
                id_vars='year', var_name='ny_hour',
                value_name=f'mean_abs_pips{MAGNITUDE_SUFFIX}'
            ).assign(block='mean_abs_pips_by_year'),
            stability['ranks_by_year'].reset_index().melt(
                id_vars='year', var_name='ny_hour', value_name='rank_within_year'
            ).assign(block='rank_by_year'),
            mismatch['table'].assign(block='dst_mismatch_sensitivity'),
        ]
        os.makedirs(os.path.dirname(session_path) or '.', exist_ok=True)
        pd.concat(blocks, ignore_index=True).to_csv(session_path, index=False)

    return {'full': full, 'scope_b': scope_b, 'hours': hours,
            'sessions': sessions, 'by_weekday': by_weekday,
            'stability': stability, 'mismatch': mismatch,
            'n_bars_scope_a': int(len(full)),
            'n_bars_scope_b': int(len(scope_b)),
            'scope_b_bound_ts': bound_ts,
            'n_gap_excluded': int((~full['is_contiguous']).sum())}


def _print_report(r):
    """The ranked answer first -- it is the whole point."""
    h, s = r['hours'], r['sessions']
    st, mm = r['stability'], r['mismatch']

    print('=' * 84)
    print('EURUSD H1 — MOVEMENT BY NEW YORK HOUR   (descriptive; no model, no alpha)')
    print('=' * 84)
    print(f"  scope A (magnitude, direction-free) : ALL {r['n_bars_scope_a']} bars")
    print(f"  scope B (signed direction)          : first 85% = "
          f"{r['n_bars_scope_b']} bars, up to {r['scope_b_bound_ts']}")
    print(f"  close-to-close excludes             : {r['n_gap_excluded']} "
          "week/holiday-gap bars (not one-hour moves)")

    print('\n' + '-' * 84)
    print('RANKED — 24 NY HOURS BY MEAN ABSOLUTE MOVE  (scope A, full history)')
    print('-' * 84)
    print(f"  {'#':<3}{'NY hour':<11}{'session':<12}{'mean':>8}{'median':>9}"
          f"{'p75':>8}{'p90':>8}{'range':>9}{'  |  up%':>10}{'signed':>9}")
    for _, row in h.iterrows():
        print(f"  {row['rank_by_mean_abs_pips']:<3}"
              f"{int(row['ny_hour']):02d}:00-{(int(row['ny_hour']) + 1) % 24:02d}:00"
              f"{'':<1}{row['session']:<12}"
              f"{row[f'mean_abs_pips{MAGNITUDE_SUFFIX}']:>8.2f}"
              f"{row[f'median_abs_pips{MAGNITUDE_SUFFIX}']:>9.2f}"
              f"{row[f'p75_abs_pips{MAGNITUDE_SUFFIX}']:>8.2f}"
              f"{row[f'p90_abs_pips{MAGNITUDE_SUFFIX}']:>8.2f}"
              f"{row[f'mean_bar_range_pips{MAGNITUDE_SUFFIX}']:>9.2f}"
              f"{row[f'up_rate_pct{DIRECTION_SUFFIX}']:>10.2f}"
              f"{row[f'mean_signed_pips{DIRECTION_SUFFIX}']:>+9.3f}")
    print("   (mean/median/p75/p90/range = pips, scope A | up% and signed = scope B)")

    print('\n' + '-' * 84)
    print('RANKED — THE FIVE NY SESSIONS')
    print('-' * 84)
    print(f"  {'#':<3}{'session':<12}{'NY hours':<14}{'n':>8}{'mean':>8}{'median':>9}"
          f"{'p90':>8}{'range':>9}{'  |  up%':>10}{'signed':>9}")
    for _, row in s.iterrows():
        print(f"  {row['rank_by_mean_abs_pips']:<3}{row['session']:<12}"
              f"{row['ny_hours']:<14}{int(row[f'n_bars{MAGNITUDE_SUFFIX}']):>8}"
              f"{row[f'mean_abs_pips{MAGNITUDE_SUFFIX}']:>8.2f}"
              f"{row[f'median_abs_pips{MAGNITUDE_SUFFIX}']:>9.2f}"
              f"{row[f'p90_abs_pips{MAGNITUDE_SUFFIX}']:>8.2f}"
              f"{row[f'mean_bar_range_pips{MAGNITUDE_SUFFIX}']:>9.2f}"
              f"{row[f'up_rate_pct{DIRECTION_SUFFIX}']:>10.2f}"
              f"{row[f'mean_signed_pips{DIRECTION_SUFFIX}']:>+9.3f}")

    print('\n' + '-' * 84)
    print('SESSION x FX WEEK DAY — mean abs pips (scope A only)')
    print('-' * 84)
    piv = r['by_weekday'].pivot(index='session', columns='fx_week_day',
                                values=f'mean_abs_pips{MAGNITUDE_SUFFIX}')
    piv = piv.reindex(SESSION_ORDER)
    print(f"  {'session':<12}" + ''.join(f"{'day ' + str(c):>10}" for c in piv.columns))
    for sess, row in piv.iterrows():
        print(f"  {sess:<12}" + ''.join(f"{v:>10.2f}" for v in row))

    print('\n' + '-' * 84)
    print('STABILITY OF THE RANKING ACROSS YEARS')
    print('-' * 84)
    print(f"  years covered              : {st['years'][0]}-{st['years'][-1]}")
    print(f"  top-3 hours overall        : "
          + ', '.join(f"{x:02d}:00 NY" for x in st['top3_overall']))
    print(f"  years whose own top-3 is that SAME set : "
          f"{st['n_years_top3_identical']}/{len(st['years'])}")
    print(f"  widest rank span among those 3 hours   : "
          f"{st['max_rank_span_in_top3']} places")
    print(f"  Spearman rank corr, {st['early_years'][0]}-{st['early_years'][-1]}"
          f" vs {st['late_years'][0]}-{st['late_years'][-1]} : "
          f"rho = {st['spearman_rho']:.4f}  (p = {st['spearman_p']:.3g})")
    print(f"\n  {'NY hour':<10}{'mean pips':>11}{'rank min':>10}{'rank max':>10}"
          f"{'span':>7}{'early':>8}{'late':>7}")
    for _, row in st['summary'].sort_values('mean_abs_pips_overall',
                                            ascending=False).iterrows():
        print(f"  {int(row['ny_hour']):02d}:00     "
              f"{row['mean_abs_pips_overall']:>11.2f}"
              f"{int(row['rank_min_across_years']):>10}"
              f"{int(row['rank_max_across_years']):>10}"
              f"{int(row['rank_span']):>7}{row['rank_early_years']:>8.1f}"
              f"{row['rank_late_years']:>7.1f}")

    print('\n' + '-' * 84)
    print('DST-MISMATCH BARS — session means with and without them')
    print('-' * 84)
    print(f"  {'session':<12}{'incl':>9}{'excl':>9}{'change':>10}{'n dropped':>12}")
    for _, row in mm['table'].iterrows():
        print(f"  {row['session']:<12}{row['mean_abs_pips_including_mismatch']:>9.3f}"
              f"{row['mean_abs_pips_excluding_mismatch']:>9.3f}"
              f"{row['pct_change_when_excluded']:>+9.2f}%"
              f"{row['n_mismatch_bars']:>12}")
    print(f"\n  largest absolute change: {mm['max_abs_pct_change']:.2f}% "
          f"(threshold {mm['threshold_pct']:.0f}%) -> "
          + ('MATERIAL — the mismatch weeks change the picture'
             if mm['material'] else
             'the mismatch weeks do NOT materially affect the picture'))

    print('\n' + '-' * 84)
    print('CONTEXT THE RANKING DOES NOT CONTAIN')
    print('-' * 84)
    print('  * Largest movement is not automatically best to trade: larger moves')
    print('    cut both ways, and the adverse excursion is larger too.')
    print('  * Dealing spreads are TIGHTEST in the deepest-liquidity hours (the')
    print('    London/NY overlap) and WIDEST in thin hours and around the roll. A')
    print('    thin hour with a large mean move may still be the worse choice once')
    print('    the dealing cost is paid — and this report does not measure it.')
    print('  * Magnitude says where there is something to capture. It says nothing')
    print('    about whether it is capturable. Those are different questions.')


if __name__ == '__main__':
    _print_report(run())
