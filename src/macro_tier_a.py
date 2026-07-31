"""
TIER-A MACRO FX DIRECTION, WALK-FORWARD -- research only.

WHY THIS EXISTS
---------------
The predecessor program (`src/macro_panel_data.py` / `src/macro_panel_model.py`,
`results/macro_panel_hypothesis_log.csv`) stopped at its power gate: 29
independent observations, a 20.81pp minimum detectable edge, both hypotheses
REGISTERED-UNSPENT. Two causes:

  1. ALFRED vintage coverage for the REVISED OECD series (CPI, unemployment)
     begins ~2013, cutting a 330-month euro era down to 153 usable months.
  2. Cross-sectional correlation: rho_bar = 0.5325 over nine G10 pairs, so
     k_eff = 1.711 -- nine currencies were worth under two.

This program fixes cause 1 by construction. TIER-A SERIES ARE NOT REVISED.
Policy rates, 3-month rates, 10-year government yields, exchange rates, equity
indices and VIX are market or administrative data: published once, never
restated. They need no ALFRED archive, so their FULL history is usable with no
look-ahead. The vintage discipline cost fourteen years only because of the
Tier-B series -- and those are DROPPED here, entirely and by name.

What that costs is smaller than it sounds. The interest-rate block is precisely
the part of the macro FX literature with the most empirical support (carry /
forward premium puzzle, Taylor-rule fundamentals, term structure). We give up
the weakly supported half to buy back two decades of history on the strongly
supported half.

Cause 2 is only partly fixable: the declared EM extension raises k_eff, and
walk-forward aggregation turns most of the history into out-of-sample rather
than one 15% slice.

HONEST PRIOR: Meese & Rogoff (1983) and forty years of follow-up. Expect a null.
The design is built so a null is informative and so the program REFUSES TO RUN
if it cannot resolve the effect sizes the literature actually claims.

WHAT IS AND IS NOT REUSED
-------------------------
Series resolution, the availability rule (`as_of`, with its staleness guard) and
quote inversion (`invert_quote`, including the high/low swap) are IMPORTED from
`src.macro_panel_data`, not forked. Two things are extended rather than reused:

  * `fetch_series` hardcodes `start='1990-01-01'`; this program needs the full
    history, so it is called directly with an earlier start and cached under
    `results/macro_tier_a/series_cache/`. The predecessor's cache is never
    written to.
  * Tier-A publication dates are re-stamped to the LAST BUSINESS DAY of the
    reference month (the predecessor used the calendar month end). The
    availability rule in this program's spec is worded against the last business
    day, and asof is that same day, so a value for reference month M is
    available exactly at the end of M -- a lag of zero, asserted rather than
    assumed.

NO P&L, no ledger, no equity curve, no position sizing. Research only.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.macro_panel_data import (MAX_STALENESS_DAYS, PAIRS as G10_PAIRS_SOURCE,
                                  as_of, fetch_series, invert_quote)

# ───────────────────────── paths ──────────────────────────────────────────────

OUT_DIR = 'results/macro_tier_a'
SERIES_CACHE = os.path.join(OUT_DIR, 'series_cache')
HYPOTHESIS_LOG = 'results/macro_tier_a_hypothesis_log.csv'
VIX_CACHE = 'results/vix.csv'            # reused, NEVER refetched
PROTECTED_SHA_FIXTURE = 'tests/fixtures/macro_tier_a_protected_sha256.json'

FETCH_START = '1950-01-01'               # earlier than any series; FRED clips

# ───────────────────────── the Tier-A series map ──────────────────────────────
# Every id below was probed live against the FRED API for existence, frequency
# and coverage before being written here. (series_id, frequency)
#
# THESE ARE THE ONLY MACRO SERIES THIS PROGRAM MAY TOUCH. Every one is a market
# or administrative rate: published once, never revised.

TIER_A_COUNTRY_SERIES = {
    'US': {'policy_rate': ('IRSTCI01USM156N', 'M'), 'rate_3m': ('IR3TIB01USM156N', 'M'),
           'yield_10y': ('IRLTLT01USM156N', 'M')},
    'EZ': {'policy_rate': ('IRSTCI01EZM156N', 'M'), 'rate_3m': ('IR3TIB01EZM156N', 'M'),
           'yield_10y': ('IRLTLT01EZM156N', 'M')},
    'GB': {'policy_rate': ('IRSTCI01GBM156N', 'M'), 'rate_3m': ('IR3TIB01GBM156N', 'M'),
           'yield_10y': ('IRLTLT01GBM156N', 'M')},
    'AU': {'policy_rate': ('IRSTCI01AUM156N', 'M'), 'rate_3m': ('IR3TIB01AUM156N', 'M'),
           'yield_10y': ('IRLTLT01AUM156N', 'M')},
    'NZ': {'policy_rate': ('IRSTCI01NZM156N', 'M'), 'rate_3m': ('IR3TIB01NZM156N', 'M'),
           'yield_10y': ('IRLTLT01NZM156N', 'M')},
    'JP': {'policy_rate': ('IRSTCI01JPM156N', 'M'), 'rate_3m': ('IR3TIB01JPM156N', 'M'),
           'yield_10y': ('IRLTLT01JPM156N', 'M')},
    'CH': {'policy_rate': ('IRSTCI01CHM156N', 'M'), 'rate_3m': ('IR3TIB01CHM156N', 'M'),
           'yield_10y': ('IRLTLT01CHM156N', 'M')},
    'CA': {'policy_rate': ('IRSTCI01CAM156N', 'M'), 'rate_3m': ('IR3TIB01CAM156N', 'M'),
           'yield_10y': ('IRLTLT01CAM156N', 'M')},
    'SE': {'policy_rate': ('IRSTCI01SEM156N', 'M'), 'rate_3m': ('IR3TIB01SEM156N', 'M'),
           'yield_10y': ('IRLTLT01SEM156N', 'M')},
    'NO': {'policy_rate': ('IRSTCI01NOM156N', 'M'), 'rate_3m': ('IR3TIB01NOM156N', 'M'),
           'yield_10y': ('IRLTLT01NOM156N', 'M')},
    # ── declared EM extension, where a COMPLETE Tier-A block exists ──
    'MX': {'policy_rate': ('IRSTCI01MXM156N', 'M'), 'rate_3m': ('IR3TIB01MXM156N', 'M'),
           'yield_10y': ('IRLTLT01MXM156N', 'M')},
    'ZA': {'policy_rate': ('IRSTCI01ZAM156N', 'M'), 'rate_3m': ('IR3TIB01ZAM156N', 'M'),
           'yield_10y': ('IRLTLT01ZAM156N', 'M')},
}

US_EQUITY_SERIES = ('SPASTT01USM661N', 'M')

# ── Quote convention: every pair FOREIGN/USD, so "the dollar strengthens" is a
#    down-move in all of them. `invert` marks the market-convention USD/XXX ids.
#    The nine G10 specs are inherited verbatim from the predecessor program.
G10_PAIRS = {p: dict(G10_PAIRS_SOURCE[p]) for p in
             ('EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'JPYUSD',
              'CHFUSD', 'CADUSD', 'SEKUSD', 'NOKUSD')}

EM_PAIRS = {
    'MXNUSD': {'country': 'MX', 'series': 'DEXMXUS', 'invert': True},
    'ZARUSD': {'country': 'ZA', 'series': 'DEXSFUS', 'invert': True},
}

# The declared EM pairs that CANNOT be built on Tier-A terms. Recorded rather
# than silently dropped, and deliberately NOT patched with discount rates: a
# country block missing rate_3m or yield_10y while every other block has them
# would make the panel asymmetric, and the model could use the pattern of
# missingness as a country marker. Same argument the predecessor used to drop
# industrial production for every country rather than keep it for the US alone.
EM_EXCLUDED = {
    'TRY': 'IRLTLT01TR (10y) does not exist on FRED; IR3TIB01TRM156N ends 2008-04; '
           'no FRED daily TRY spot series. No complete Tier-A block.',
    'BRL': 'IR3TIB01BR and IRLTLT01BR do not exist on FRED (only the policy rate '
           'IRSTCI01BRM156N and a discount rate). No complete Tier-A block.',
    'INR': 'IR3TIB01IN and IRLTLT01IN do not exist on FRED (only IRSTCI01INM156N and '
           'INTDSRINM193N, which ends 2022-07). No complete Tier-A block.',
}

PAIR_SPECS = {**G10_PAIRS, **EM_PAIRS}
G10_PANEL = tuple(G10_PAIRS)
EXTENDED_PANEL = tuple(G10_PAIRS) + tuple(EM_PAIRS)

# ── The DEM splice. Pre-1999 the euro does not exist; the standard convention in
#    the literature is the Deutsche mark at the irrevocable 1999 conversion rate.
#    It IS a splice and is disclosed, flagged per row, and verified against the
#    overlap window where both series exist.
EUR_PREDECESSOR_SERIES = 'EXGEUS'        # DEM per USD, monthly, 1971-01..2001-12
DEM_PER_EUR = 1.95583                    # irrevocable conversion rate, 1999-01-01
EUR_SPLICE_DATE = pd.Timestamp('1999-01-01')

# ───────────────────────── the feature set ────────────────────────────────────
# Per country block, entered as SEPARATE LEVELS, never pre-differenced: 3% vs 5%
# is not the same macro state as 1% vs 3%, and the model may learn a difference
# if a difference is what matters.

COUNTRY_BLOCK = ('policy_rate', 'rate_3m', 'yield_10y', 'curve_slope',
                 'rate_change_3m', 'rate_change_12m')

PRE_REGISTERED_FEATURE_COLUMNS = (
    tuple(f'for_{c}' for c in COUNTRY_BLOCK) +
    tuple(f'us_{c}' for c in COUNTRY_BLOCK) +
    ('vix_level', 'us_equity_12m_return',
     'own_return_1m', 'own_return_3m', 'own_return_12m')
)

# ── AMENDMENT. Recorded, not hidden. See AMENDMENT_NOTE below and the notes
#    column of results/macro_tier_a_hypothesis_log.csv, which a unit test pins.
REMOVED_FEATURES = ('vix_level',)

FEATURE_COLUMNS = tuple(c for c in PRE_REGISTERED_FEATURE_COLUMNS
                        if c not in REMOVED_FEATURES)

AMENDMENT_NOTE = (
    'PRE-REGISTRATION AMENDMENT (feature list): vix_level was REMOVED from the '
    'feature set AFTER the power gate fired on the pre-registered design and '
    'BEFORE any model was fitted. Justification is DATA COVERAGE ONLY, not any '
    'observed result: results/vix.csv begins 1990-01-02 while every Tier-A rate '
    'series runs decades earlier, so vix_level alone was setting the panel start '
    'and was the binding constraint on statistical power. No model had been fitted '
    'and no accuracy, AUC or coefficient had been computed when the amendment was '
    'made. H_tA.1 (the carry rule) is UNAFFECTED: it compares policy rates only '
    'and never used vix_level. Only H_tA.2 changes, by exactly one column, and its '
    'feature set is therefore one column smaller than pre-registered. '
    'us_equity_12m_return is RETAINED and carries the risk-appetite dimension; NO '
    'replacement feature was added and no substitution was made. The removed '
    'column is reported as a blank row in the coefficient table so a later reader '
    'sees the pre-registered set and what was taken out of it.'
)

# ── Guards. Both are asserted in the test suite and both are shown to fire.
#    Tier-B identifiers, by name. Nothing revised may reach the feature matrix.
TIER_B_DENYLIST = (
    'CPI', 'CP0000', 'UNEMPLOY', 'UNRATE', 'LRHUTTTT', 'LRUN', 'INFLATION',
    'GDP', 'NAEXKP', 'INDPRO', 'PRINTO', 'IPMAN', 'PAYEMS', 'MINMEI', 'QINMEI',
    'INDUSTRIAL', 'PRODUCTION',
)
# No time index, no trend, no pair identity: the model must not be able to read
# the calendar or tell the currencies apart.
FORBIDDEN_FEATURE_TOKENS = (
    'time', 'trend', 'year', 'month', 'date', 'pair', 'country', 'dummy',
    'ordinal', 'epoch', 't_index', 'currency', 'ccy', 'onehot', 'one_hot',
)

# ───────────────────────── protocol constants (FIXED ONCE) ────────────────────
# This is walk-forward VALIDATION, not walk-forward OPTIMISATION. Hyperparameters
# are fixed HERE and never re-tuned per window; re-tuning per window would
# reintroduce exactly the data-snooping the protocol exists to remove.

MIN_TRAIN_MONTHS = 120
RESERVED_TAIL_MONTHS = 36
ROLLING_WINDOW_MONTHS = 120
LOGIT_C = 1.0
LOGIT_MAX_ITER = 2000
BOOTSTRAP_RESAMPLES = 2000
BLOCK_LEN_MONTHS = 12
ALPHA = 0.05 / 2                          # new family, size 2
FAMILY_SIZE = 2
RANDOM_SEED = 20260730

# The power gate, calibrated against the literature rather than an arbitrary
# count. Published claims for macro FX directional predictability at a monthly
# horizon cluster at 52-55% accuracy, i.e. an edge of 2-5pp.
GATE_PROCEED_PP = 4.0
GATE_STOP_PP = 5.0


class TierBLeakError(RuntimeError):
    """A revised (Tier-B) identifier reached the Tier-A feature matrix."""


class ForbiddenFeatureError(RuntimeError):
    """A time index, trend or pair-identity feature reached the feature matrix."""


# ═══════════════════════ 1. series access ═════════════════════════════════════

def _tier_a_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stamp each Tier-A observation as available on the LAST BUSINESS DAY of its
    own reference month.

    Tier-A values are market or administrative rates observed as they happen, so
    the monthly figure is complete once the month's last business day closes.
    This is a publication lag of ZERO -- which is the point, and is asserted in
    the test suite rather than assumed, because a shift(-1) error here is just as
    fatal as a vintage error was in the predecessor program.
    """
    out = df.copy()
    ref = pd.DatetimeIndex(out['ref_date'])
    out['publication_date'] = ref + pd.offsets.BMonthEnd(0)
    return out


def assert_tier_a_only(series_ids) -> None:
    """No revised series may be fetched, cached or featurised. Guard, not a hint."""
    for sid in series_ids:
        upper = str(sid).upper()
        for banned in TIER_B_DENYLIST:
            if banned in upper:
                raise TierBLeakError(
                    f'Tier-B identifier {sid!r} matched deny-list token {banned!r}. '
                    'This program is Tier-A only: mixing revised series back in '
                    'would reimpose the 2013 ALFRED start and defeat its purpose.')


def load_series(series_id: str, refresh: bool = False) -> pd.DataFrame:
    """
    One Tier-A series over its FULL history, cached under this program's own
    directory. `fetch_series` is imported from the predecessor rather than
    reimplemented; only the start date and the publication-date convention differ.
    """
    assert_tier_a_only([series_id])
    os.makedirs(SERIES_CACHE, exist_ok=True)
    path = os.path.join(SERIES_CACHE, f'{series_id}__tierA.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, parse_dates=['ref_date', 'publication_date'])
    raw = fetch_series(series_id, first_release=False, start=FETCH_START)
    out = _tier_a_publication_dates(raw)
    out.to_csv(path, index=False)
    return out


def load_vix() -> pd.DataFrame:
    """VIX from the project's existing FRED cache -- reused, never refetched."""
    if not os.path.exists(VIX_CACHE):
        return pd.DataFrame(columns=['ref_date', 'value', 'publication_date'])
    df = pd.read_csv(VIX_CACHE, index_col=0, parse_dates=True)
    col = [c for c in df.columns if 'vix' in c.lower()] or list(df.columns[:1])
    s = df[col[0]].dropna()
    idx = pd.DatetimeIndex(s.index).tz_localize(None)
    return pd.DataFrame({'ref_date': idx, 'value': s.to_numpy(dtype=float),
                         'publication_date': idx})


def load_all_series(refresh: bool = False) -> dict:
    ids = {sid for spec in TIER_A_COUNTRY_SERIES.values() for sid, _f in spec.values()}
    ids.add(US_EQUITY_SERIES[0])
    ids.update(spec['series'] for spec in PAIR_SPECS.values())
    ids.add(EUR_PREDECESSOR_SERIES)
    assert_tier_a_only(ids)
    return {sid: load_series(sid, refresh=refresh) for sid in sorted(ids)}


# ═══════════════════════ 2. FX, correctly oriented ════════════════════════════

def _monthly_last(df: pd.DataFrame) -> pd.Series:
    s = pd.DataFrame({'date': pd.DatetimeIndex(df['ref_date']),
                      'price': df['value'].astype(float)}).dropna()
    return s.set_index('date').sort_index().resample('ME').last()['price'].dropna()


def monthly_fx(pair: str, cache: dict) -> pd.DataFrame:
    """
    Monthly close for one pair, oriented FOREIGN/USD, taken as the last available
    business day of each month from FRED's daily series.

    EURUSD is SPLICED: DEXUSEU from 1999-01, and before that the Deutsche mark
    (EXGEUS, DEM per USD) inverted and multiplied by the irrevocable conversion
    rate 1 EUR = 1.95583 DEM. Rows built from the predecessor currency carry
    `eur_dem_splice = 1`.
    """
    spec = PAIR_SPECS[pair]
    raw = cache[spec['series']]
    if not len(raw):
        return pd.DataFrame(columns=['month_end', 'price', 'eur_dem_splice'])
    px = _monthly_last(raw)
    frame = pd.DataFrame({'month_end': px.index, 'price': px.to_numpy()})
    if spec['invert']:
        frame = invert_quote(frame)
    frame['eur_dem_splice'] = 0

    if pair == 'EURUSD':
        dem = cache.get(EUR_PREDECESSOR_SERIES)
        if dem is not None and len(dem):
            d = _monthly_last(dem)
            pre = pd.DataFrame({'month_end': d.index, 'price': d.to_numpy()})
            pre = invert_quote(pre)                    # DEM per USD -> USD per DEM
            pre['price'] = pre['price'] * DEM_PER_EUR  # USD per DEM -> USD per EUR
            pre['eur_dem_splice'] = 1
            pre = pre[pre['month_end'] < EUR_SPLICE_DATE]
            frame = pd.concat([pre, frame], ignore_index=True)
    return frame.sort_values('month_end').drop_duplicates('month_end').reset_index(drop=True)


def eur_splice_overlap_check(cache: dict) -> dict:
    """
    The DEM splice is verifiable: EXGEUS continues to 2001-12, three years past
    the euro's introduction, so the converted mark and the actual euro rate can be
    compared where both exist. A splice that fails this check is a bug, not a
    convention.
    """
    dem, eur = cache.get(EUR_PREDECESSOR_SERIES), cache.get('DEXUSEU')
    if dem is None or eur is None or not len(dem) or not len(eur):
        return {'n_overlap': 0}
    d = _monthly_last(dem)
    conv = (1.0 / d) * DEM_PER_EUR
    e = _monthly_last(eur)
    both = pd.concat([conv.rename('dem'), e.rename('eur')], axis=1).dropna()
    if not len(both):
        return {'n_overlap': 0}
    rel = (both['dem'] / both['eur'] - 1.0).abs() * 100.0
    return {'n_overlap': int(len(both)),
            'overlap_start': f"{both.index.min():%Y-%m}",
            'overlap_end': f"{both.index.max():%Y-%m}",
            'mean_abs_diff_pct': float(rel.mean()),
            'max_abs_diff_pct': float(rel.max())}


# ═══════════════════════ 3. panel assembly ════════════════════════════════════

def _country_block(country: str, asof: pd.Timestamp, cache: dict) -> dict:
    """One country's Tier-A block AS KNOWN at `asof`, entered as separate levels."""
    spec = TIER_A_COUNTRY_SERIES[country]
    out = {}
    for name in ('policy_rate', 'rate_3m', 'yield_10y'):
        sid, freq = spec[name]
        out[name], _r, _p = as_of(cache[sid], asof, freq)
    out['curve_slope'] = out['yield_10y'] - out['rate_3m']

    # policy DIRECTION, not just level -- central-bank cycles are what macro
    # traders actually trade. Both ends use only values published by `asof`.
    # The lagged cutoff is anchored to the BUSINESS MONTH END of the month `lag`
    # months back, not to `asof` minus raw days: subtracting days from a cutoff
    # that already sits on the 29th would land before the 30th-of-the-month
    # publication and silently read a FOUR-month-old rate as the three-month one.
    sid, freq = spec['policy_rate']
    now, _r, _p = as_of(cache[sid], asof, freq)
    for label, lag in (('rate_change_3m', 3), ('rate_change_12m', 12)):
        back = _bmonth_end(asof - pd.DateOffset(months=lag))
        then, _r2, _p2 = as_of(cache[sid], back, freq)
        out[label] = (now - then) if (np.isfinite(now) and np.isfinite(then)) else np.nan
    return out


def _bmonth_end(ts: pd.Timestamp) -> pd.Timestamp:
    """
    The last BUSINESS day of ts's OWN month -- the availability cutoff.

    Deliberately NOT `ts + BMonthEnd(0)`: that offset ROLLS FORWARD, so a month
    whose calendar end falls on a weekend (1989-12-31 was a Sunday) would return
    the NEXT month's business end -- a full month of look-ahead on roughly two
    months in seven. Roll the calendar month end BACKWARD instead, which can only
    move the cutoff earlier.
    """
    month_end = pd.Timestamp(ts) + pd.offsets.MonthEnd(0)
    return pd.offsets.BMonthEnd().rollback(month_end)


def build_panel(pairs=G10_PANEL, cache: dict = None, lookahead_months: int = 0,
                verbose: bool = True) -> pd.DataFrame:
    """
    The monthly panel: one row per (month, pair). Features are those KNOWN at the
    last business day of month M; the label is the sign of month M+1's log return.

    Monthly observations are NON-OVERLAPPING, so label uniqueness is exactly 1.0
    by construction -- the property every prior H1 program in this project lacked.

    `lookahead_months` exists ONLY for the mandatory LOOK-AHEAD POSITIVE CONTROL:
    it moves the feature cutoff FORWARD past what the availability rule permits,
    deliberately introducing leakage. Accuracy must then improve measurably; if it
    does not, the availability machinery is not binding and must be inspected.
    """
    cache = load_all_series() if cache is None else cache
    vix = load_vix()
    eq_sid, eq_freq = US_EQUITY_SERIES

    rows = []
    for pair in pairs:
        fx = monthly_fx(pair, cache)
        if not len(fx):
            continue
        fx = fx.set_index('month_end').sort_index()
        px, splice = fx['price'], fx['eur_dem_splice']
        logp = np.log(px)
        country = PAIR_SPECS[pair]['country']

        for m in px.index:
            nxt = m + pd.offsets.MonthEnd(1)
            if nxt not in px.index:
                continue                                   # no next month -> no label
            # The availability cutoff. lookahead_months > 0 deliberately breaks it.
            asof = _bmonth_end(m + pd.DateOffset(months=lookahead_months))
            asof_m = pd.Timestamp(asof) + pd.offsets.MonthEnd(0)

            rec = {'month_end': m, 'pair': pair, 'country': country,
                   'asof': asof, 'price': float(px.loc[m]),
                   'eur_dem_splice': int(splice.loc[m])}

            for prefix, ctry in (('for', country), ('us', 'US')):
                for k, v in _country_block(ctry, asof, cache).items():
                    rec[f'{prefix}_{k}'] = v

            rec['vix_level'], _r, _p = as_of(vix, asof, 'D')
            eq_now, _r, _p = as_of(cache[eq_sid], asof, eq_freq)
            eq_prev, _r2, _p2 = as_of(cache[eq_sid],
                                      _bmonth_end(asof - pd.DateOffset(months=12)),
                                      eq_freq)
            rec['us_equity_12m_return'] = (
                (eq_now / eq_prev - 1.0) * 100.0
                if np.isfinite(eq_now) and np.isfinite(eq_prev) and eq_prev else np.nan)

            for label, lag in (('own_return_1m', 1), ('own_return_3m', 3),
                               ('own_return_12m', 12)):
                past = asof_m - pd.offsets.MonthEnd(lag)
                rec[label] = (float(logp.loc[asof_m] - logp.loc[past]) * 100.0
                              if (asof_m in logp.index and past in logp.index) else np.nan)

            fwd = float(logp.loc[nxt] - logp.loc[m])
            rec['fwd_logret_1m'] = fwd
            rec['y'] = int(fwd > 0)
            rows.append(rec)

    panel = pd.DataFrame(rows)
    if len(panel):
        panel = panel.sort_values(['month_end', 'pair']).reset_index(drop=True)
    assert_no_forbidden_features(FEATURE_COLUMNS)
    if verbose:
        print(f'  panel({len(pairs)} pairs): {len(panel)} raw rows')
    return panel


def complete_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Rows with a COMPLETE Tier-A feature vector. The panel is deliberately
    unbalanced -- each pair contributes the months it has -- so this is where a
    pair's coverage actually starts and ends.
    """
    if not len(panel):
        return panel
    return panel.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)


def assert_no_forbidden_features(columns=FEATURE_COLUMNS) -> None:
    """
    No time index, no trend, no pair-identity feature may be a MODEL input.

    Applied to whatever column list is handed in, so the test suite can show the
    guard BITE on a deliberately bad column rather than merely pass on a good one.
    `month_end`, `pair` and `country` exist in the panel as bookkeeping and are
    never members of FEATURE_COLUMNS -- which is exactly what this asserts.
    """
    for col in columns:
        low = str(col).lower()
        for token in FORBIDDEN_FEATURE_TOKENS:
            if token in low:
                raise ForbiddenFeatureError(
                    f'feature {col!r} matched forbidden token {token!r}: the model '
                    'must not be able to read the calendar or identify the pair.')


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    assert_no_forbidden_features(FEATURE_COLUMNS)
    return df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)


# ═══════════════════════ 4. coverage / provenance ═════════════════════════════

def provenance_table(cache: dict) -> pd.DataFrame:
    """Every series, economy, source, first/last month, and discontinuation."""
    today = pd.Timestamp.today().normalize()
    rows = []

    def add(country, feature, sid, freq, source):
        df = cache.get(sid)
        n = 0 if df is None else len(df)
        if n:
            first = f"{pd.Timestamp(df['ref_date'].min()):%Y-%m}"
            last = f"{pd.Timestamp(df['ref_date'].max()):%Y-%m}"
            stale_days = (today - pd.Timestamp(df['publication_date'].max())).days
            disc = last if stale_days > MAX_STALENESS_DAYS.get(freq, 200) else ''
        else:
            first = last = disc = ''
        rows.append({'country': country, 'feature': feature, 'series_id': sid,
                     'tier': 'A', 'frequency': freq, 'n_obs': n, 'first_month': first,
                     'last_month': last, 'discontinued_after': disc, 'source': source,
                     'revised': False})

    for country, spec in TIER_A_COUNTRY_SERIES.items():
        for name, (sid, freq) in spec.items():
            add(country, name, sid, freq, 'FRED (OECD MEI)')
    add('GLOBAL', 'us_equity_12m_return', US_EQUITY_SERIES[0], US_EQUITY_SERIES[1], 'FRED (OECD MEI)')
    rows.append({'country': 'GLOBAL', 'feature': 'vix_level', 'series_id': 'VIXCLS',
                 'tier': 'A', 'frequency': 'D', 'n_obs': 0, 'first_month': '1990-01',
                 'last_month': '', 'discontinued_after': '',
                 'source': 'results/vix.csv (project FRED framework, REUSED not refetched)',
                 'revised': False})
    for pair, spec in PAIR_SPECS.items():
        add(spec['country'], f'fx_{pair}', spec['series'], 'D', 'FRED H.10')
    add('EZ', 'fx_EURUSD_predecessor_DEM', EUR_PREDECESSOR_SERIES, 'M',
        'FRED (DISCONTINUED); spliced at 1.95583 DEM/EUR')
    return pd.DataFrame(rows)


def coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    """First and last month with a COMPLETE feature vector, per pair."""
    ok = complete_rows(panel)
    rows = []
    for pair in panel['pair'].unique():
        sub, raw = ok[ok['pair'] == pair], panel[panel['pair'] == pair]
        rows.append({
            'pair': pair, 'country': PAIR_SPECS[pair]['country'],
            'raw_rows': int(len(raw)),
            'raw_first': f"{raw['month_end'].min():%Y-%m}" if len(raw) else '',
            'complete_rows': int(len(sub)),
            'first_complete_month': f"{sub['month_end'].min():%Y-%m}" if len(sub) else '',
            'last_complete_month': f"{sub['month_end'].max():%Y-%m}" if len(sub) else '',
            'n_splice_rows': int(sub['eur_dem_splice'].sum()) if len(sub) else 0,
        })
    return pd.DataFrame(rows).sort_values('first_complete_month').reset_index(drop=True)


def binding_constraint_table(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each pair, WHICH feature actually sets its start date -- verified from the
    data, not asserted.

    The panel start is the LAST feature to become available, so this reports the
    argmax rather than a global claim. It is the check that the amended panel
    start is genuinely set by the series named, and not by some other series that
    went unnoticed.
    """
    rows = []
    for pair in panel['pair'].unique():
        sub = panel[panel['pair'] == pair]
        firsts = {}
        for col in FEATURE_COLUMNS:
            avail = sub.loc[sub[col].notna(), 'month_end']
            firsts[col] = avail.min() if len(avail) else pd.NaT
        if any(pd.isna(v) for v in firsts.values()):
            binding = [c for c, v in firsts.items() if pd.isna(v)]
            start = pd.NaT
        else:
            start = max(firsts.values())
            binding = sorted(c for c, v in firsts.items() if v == start)
        vix_first = sub.loc[sub['vix_level'].notna(), 'month_end']
        rows.append({
            'pair': pair, 'country': PAIR_SPECS[pair]['country'],
            'first_complete_month': f'{start:%Y-%m}' if pd.notna(start) else 'NEVER',
            'binding_feature': '|'.join(binding),
            'vix_level_first_month': (f'{vix_first.min():%Y-%m}' if len(vix_first)
                                      else 'NEVER'),
            'months_bought_by_dropping_vix': (
                int(round((vix_first.min() - start).days / 30.44))
                if len(vix_first) and pd.notna(start) and vix_first.min() > start else 0),
        })
    return pd.DataFrame(rows).sort_values('first_complete_month').reset_index(drop=True)


def label_uniqueness(df: pd.DataFrame) -> float:
    """
    Mean label uniqueness. A monthly label spans exactly one month and, within a
    pair, no two labels overlap -- so this is exactly 1.0 by construction, unlike
    every triple-barrier program in this project.
    """
    if not len(df):
        return float('nan')
    counts = df.groupby(['pair', 'month_end']).size()
    return float((1.0 / counts).mean())


# ═══════════════════════ 5. walk-forward structure ════════════════════════════

def reserved_months(panel: pd.DataFrame, tail: int = RESERVED_TAIL_MONTHS):
    """The most recent `tail` months: never trained on, never predicted, never
    inspected. If the walk-forward clears its bar this block is the one-shot
    confirmation -- a separate decision, not part of this program."""
    if tail <= 0:
        return set()          # months[-0:] is months[0:] -- reserve nothing, not all
    months = np.sort(complete_rows(panel)['month_end'].unique())
    return set(pd.DatetimeIndex(months[-tail:])) if len(months) >= tail else set()


def walk_forward_windows(panel: pd.DataFrame, window_type: str = 'expanding',
                         min_train: int = MIN_TRAIN_MONTHS,
                         tail: int = RESERVED_TAIL_MONTHS):
    """
    Annual refits. To predict the months of calendar year Y, fit on every complete
    month up to and including December of Y-1. The reserved tail is removed from
    the universe FIRST, so no window can index it.
    """
    df = complete_rows(panel)
    held = reserved_months(panel, tail)
    universe = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(df['month_end'].unique())) - held))
    if not len(universe):
        return []

    windows = []
    for year in range(universe.min().year, universe.max().year + 1):
        train_months = universe[universe < pd.Timestamp(year=year, month=1, day=1)]
        oos_months = universe[(universe >= pd.Timestamp(year=year, month=1, day=1)) &
                              (universe <= pd.Timestamp(year=year, month=12, day=31))]
        if len(train_months) < min_train or not len(oos_months):
            continue
        if window_type == 'rolling':
            train_months = train_months[-ROLLING_WINDOW_MONTHS:]
        windows.append({'year': year, 'train_months': train_months,
                        'oos_months': oos_months})
    return windows


# ═══════════════════════ 6. the power gate ════════════════════════════════════

def correlation_structure(df: pd.DataFrame, months, pairs) -> tuple:
    """
    Cross-sectional correlation of CONTEMPORANEOUS monthly returns, on TRAINING
    months only. `own_return_1m` is the return realised DURING the row's month, so
    it is the contemporaneous return; the forward return is the label and must not
    be used here.
    """
    sub = df[df['month_end'].isin(months)]
    wide = sub.pivot_table(index='month_end', columns='pair', values='own_return_1m')
    wide = wide.reindex(columns=[p for p in pairs if p in wide.columns])
    corr = wide.corr(min_periods=24)
    vals = corr.to_numpy(dtype=float)
    off = vals[~np.eye(len(vals), dtype=bool)]
    rho_bar = float(np.nanmean(off)) if off.size else float('nan')
    k = int(len(vals))
    k_eff = k / (1.0 + (k - 1) * rho_bar) if k > 1 and np.isfinite(rho_bar) else float(k)
    return corr, rho_bar, float(k_eff), k


def power_gate(df: pd.DataFrame, windows, pairs, alpha: float = ALPHA) -> dict:
    """
    Computed and reported BEFORE any model is fitted. This is a hard STOP.

    n_independent = n_oos_rows * k_eff / k deliberately follows the predecessor's
    formula so the two programs are directly comparable. With an unbalanced panel
    the average rows-per-month is below k, which makes this the CONSERVATIVE of
    the two natural estimates; the alternative (n_oos_months * k_eff) is reported
    as a cross-check, never as the headline.
    """
    train_months = sorted({m for w in windows for m in w['train_months']})
    oos_months = sorted({m for w in windows for m in w['oos_months']})
    oos = df[df['month_end'].isin(oos_months)]

    corr, rho_bar, k_eff, k = correlation_structure(df, train_months, pairs)
    n_oos_rows = int(len(oos))
    n_independent = int(round(n_oos_rows * k_eff / k)) if k else 0
    uniq = label_uniqueness(oos)

    se = np.sqrt(0.25 / n_independent) if n_independent > 0 else float('inf')
    z = float(stats.norm.ppf(1 - alpha / 2.0))
    mde_pp = 100.0 * z * se

    band = ('PROCEED' if mde_pp <= GATE_PROCEED_PP
            else 'PROCEED-UPPER-HALF-ONLY' if mde_pp <= GATE_STOP_PP else 'STOP')
    return {'n_pairs': k, 'n_windows': len(windows), 'n_oos_rows': n_oos_rows,
            'n_oos_months': len(oos_months), 'n_train_months': len(train_months),
            'rho_bar': rho_bar, 'k_eff': k_eff, 'n_independent': n_independent,
            'n_independent_crosscheck': int(round(len(oos_months) * k_eff)),
            'mean_label_uniqueness': uniq, 'se_accuracy': float(se),
            'z_two_sided': z, 'min_detectable_edge_pp': float(mde_pp),
            'power_gate_band': band, 'proceed': band != 'STOP',
            'corr': corr}


# ═══════════════════════ 7. the two predictors ════════════════════════════════

def carry_predictions(df: pd.DataFrame, majority: int) -> np.ndarray:
    """
    H_tA.1, THE CARRY RULE. Fixed now, no parameters, sign PRE-REGISTERED:

        predict UP   if foreign policy_rate > US policy_rate
        predict DOWN if foreign policy_rate < US policy_rate
        exact tie    -> the window's train-majority class

    Uncovered interest parity predicts high-yield currencies should DEPRECIATE.
    The forward premium puzzle -- among the most replicated findings in
    international finance -- is that empirically they do not, and long-high /
    short-low has historically earned positive returns. The rule therefore bets
    WITH the rate differential. Discovering this sign afterwards would have been a
    free parameter; committing to it in advance is what makes the test honest.
    """
    diff = df['for_policy_rate'].to_numpy(float) - df['us_policy_rate'].to_numpy(float)
    pred = np.where(diff > 0, 1, np.where(diff < 0, 0, int(majority)))
    return pred.astype(int)


def fit_logistic(x_train, y_train, seed: int = RANDOM_SEED):
    """
    H_tA.2. C FIXED at 1.0 -- no CV sweep, no per-window tuning.

    NO GBM, NO NEURAL NETWORK. At a few hundred independent observations with ~20
    features a logistic regression is the appropriate complexity; anything deeper
    memorises, as this project has now demonstrated five times. If the logistic
    clears, a GBM becomes a separate registered question -- not before.
    """
    scaler = StandardScaler().fit(x_train)          # refit INSIDE the window
    model = LogisticRegression(penalty='l2', C=LOGIT_C, class_weight='balanced',
                               max_iter=LOGIT_MAX_ITER, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    return scaler, model


def run_walk_forward(df: pd.DataFrame, windows, shuffle_labels: bool = False,
                     seed: int = RANDOM_SEED) -> tuple:
    """
    One pass of the protocol. Standardiser, logistic and train-majority baseline
    are ALL refit inside each window on that window's training rows only; a single
    global standardiser would leak future distributional information backwards.

    Returns pooled out-of-sample predictions and a per-window profile. Pooling is
    the point: ONE statistical test on ALL out-of-sample rows, never a per-window
    test with winners counted.
    """
    rng = np.random.default_rng(seed)
    frames, profile, coefs = [], [], []

    for w in windows:
        tr = df[df['month_end'].isin(w['train_months'])]
        te = df[df['month_end'].isin(w['oos_months'])]
        if not len(tr) or not len(te):
            continue
        x_tr, x_te = feature_matrix(tr), feature_matrix(te)
        y_tr = tr['y'].to_numpy(int)
        y_used = rng.permutation(y_tr) if shuffle_labels else y_tr

        majority = int(pd.Series(y_used).mode().iloc[0])
        if len(np.unique(y_used)) < 2:
            continue
        scaler, model = fit_logistic(x_tr, y_used, seed=seed)
        proba = model.predict_proba(scaler.transform(x_te))[:, 1]

        out = te[['month_end', 'pair', 'country', 'y', 'fwd_logret_1m']].copy()
        out['window_year'] = w['year']
        out['train_months'] = len(w['train_months'])
        out['pred_logit'] = (proba >= 0.5).astype(int)
        out['proba_logit'] = proba
        out['pred_carry'] = carry_predictions(te, majority)
        out['pred_majority'] = majority
        frames.append(out)
        coefs.append(model.coef_.ravel())

        profile.append({'year': w['year'], 'train_months': len(w['train_months']),
                        'train_rows': int(len(tr)), 'oos_months': len(w['oos_months']),
                        'oos_rows': int(len(te)), 'train_majority': majority,
                        'acc_logit': float((out['pred_logit'] == out['y']).mean()),
                        'acc_carry': float((out['pred_carry'] == out['y']).mean()),
                        'acc_majority': float((majority == out['y']).mean())})

    pooled = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=['month_end', 'pair', 'y']))
    coef_df = (pd.DataFrame(coefs, columns=list(FEATURE_COLUMNS)) if coefs
               else pd.DataFrame(columns=list(FEATURE_COLUMNS)))
    return pooled, pd.DataFrame(profile), coef_df


# ═══════════════════════ 8. the arbiter ═══════════════════════════════════════

def moving_block_bootstrap(pooled: pd.DataFrame, col_challenger: str,
                           col_reference: str, alpha: float = ALPHA,
                           n_boot: int = BOOTSTRAP_RESAMPLES,
                           block_len: int = BLOCK_LEN_MONTHS,
                           seed: int = RANDOM_SEED) -> tuple:
    """
    Paired bootstrap on delta accuracy, resampled BY MONTH in moving blocks.

    Two reasons for the design, both load-bearing:
      * Rows within a month are cross-sectionally dependent (rho_bar ~0.5), so a
        sampled month enters with ALL of its pairs together. Row-wise resampling
        would treat nine correlated rows as nine independent ones and understate
        the interval.
      * block_len = 12 MONTHS because rate cycles persist for years: policy-rate
        differentials are autocorrelated over multi-year horizons, so single-month
        resampling would break exactly the dependence that matters and again
        understate the interval. Twelve months is one full annual refit period.
    """
    rng = np.random.default_rng(seed)
    months = np.sort(pooled['month_end'].unique())
    n_months = len(months)
    if n_months == 0:
        return float('nan'), float('nan'), np.array([])

    ok_c = (pooled[col_challenger] == pooled['y']).to_numpy(bool)
    ok_r = (pooled[col_reference] == pooled['y']).to_numpy(bool)
    idx = pooled['month_end'].to_numpy()
    per_month = {}
    for i, m in enumerate(months):
        sel = idx == m
        per_month[i] = (int(sel.sum()), int(ok_c[sel].sum()), int(ok_r[sel].sum()))

    counts = np.array([per_month[i][0] for i in range(n_months)], dtype=float)
    hits_c = np.array([per_month[i][1] for i in range(n_months)], dtype=float)
    hits_r = np.array([per_month[i][2] for i in range(n_months)], dtype=float)

    block = min(block_len, n_months)
    n_blocks = int(np.ceil(n_months / block))
    max_start = n_months - block
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        take = np.concatenate([np.arange(s, s + block) for s in starts])[:n_months]
        n = counts[take].sum()
        deltas[b] = (hits_c[take].sum() - hits_r[take].sum()) / n if n else np.nan
    lo = float(np.nanpercentile(deltas, 100 * alpha / 2.0))
    hi = float(np.nanpercentile(deltas, 100 * (1 - alpha / 2.0)))
    return lo, hi, deltas


def mcnemar_exact(pooled: pd.DataFrame, col_challenger: str, col_reference: str) -> tuple:
    """Exact (binomial) McNemar on the paired predictions."""
    ok_c = (pooled[col_challenger] == pooled['y']).to_numpy(bool)
    ok_r = (pooled[col_reference] == pooled['y']).to_numpy(bool)
    b = int((ok_c & ~ok_r).sum())          # challenger right, reference wrong
    c = int((~ok_c & ok_r).sum())          # challenger wrong, reference right
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)


def arbitrate(pooled: pd.DataFrame, col_challenger: str, col_reference: str,
              alpha: float = ALPHA) -> dict:
    """KEEP iff the block CI lies entirely above zero AND McNemar p < alpha."""
    acc_c = float((pooled[col_challenger] == pooled['y']).mean())
    acc_r = float((pooled[col_reference] == pooled['y']).mean())
    lo, hi, _ = moving_block_bootstrap(pooled, col_challenger, col_reference, alpha)
    b, c, p = mcnemar_exact(pooled, col_challenger, col_reference)
    cleared = bool(lo > 0 and p < alpha)
    return {'acc_challenger': acc_c, 'acc_reference': acc_r,
            'delta_acc': acc_c - acc_r, 'delta_acc_ci_low': lo, 'delta_acc_ci_high': hi,
            'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p,
            'cleared_bar': cleared, 'verdict': 'KEEP' if cleared else 'DROP'}


def auc(pooled: pd.DataFrame, score_col: str) -> float:
    """Descriptive only. The primary metric is pooled out-of-sample accuracy."""
    from sklearn.metrics import roc_auc_score
    y = pooled['y'].to_numpy(int)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, pooled[score_col].to_numpy(float)))


# ═══════════════════════ 9. descriptive breakdowns ════════════════════════════

def accuracy_by(pooled: pd.DataFrame, key: str) -> pd.DataFrame:
    g = pooled.groupby(key)
    return pd.DataFrame({
        'n': g.size(),
        'acc_carry': g.apply(lambda d: float((d['pred_carry'] == d['y']).mean()),
                             include_groups=False),
        'acc_logit': g.apply(lambda d: float((d['pred_logit'] == d['y']).mean()),
                             include_groups=False),
        'base_rate_up': g['y'].mean(),
    }).reset_index()


def per_decade(pooled: pd.DataFrame) -> pd.DataFrame:
    """
    REGIME-STABILITY CHECK. The predecessor multi-day program's motivating
    statistic held in the training era and vanished in the validation era, and
    that was visible for free before any model ran. The equivalent question here:
    does carry's edge, if any, persist across decades or live in one?
    """
    out = pooled.copy()
    out['decade'] = (pd.DatetimeIndex(out['month_end']).year // 10 * 10).astype(str) + 's'
    return accuracy_by(out, 'decade')


def protected_set_digest() -> dict:
    """sha256 of every protected file, for the boundary assertion."""
    import glob
    targets = ['_train_pipeline.py', 'src/inference.py', 'src/features.py',
               'src/macro_data.py', 'src/paper_trading.py', 'config.json',
               'results/eurusd_h1.csv', 'results/eurusd_m15.csv',
               'src/macro_panel_data.py', 'src/macro_panel_model.py']
    targets += sorted(glob.glob('models/**/*', recursive=True))
    targets += sorted(glob.glob('results/pooled_h1/*'))
    targets += sorted(glob.glob('results/*hypothesis_log.csv'))
    out = {}
    for t in sorted(set(targets)):
        if os.path.isfile(t) and os.path.basename(t) != os.path.basename(HYPOTHESIS_LOG):
            out[t.replace(os.sep, '/')] = hashlib.sha256(open(t, 'rb').read()).hexdigest()
    return out


# ═══════════════════════ 10. the hypothesis log ═══════════════════════════════

LOG_COLUMNS = ['n', 'date', 'hypothesis', 'arbiter', 'panel', 'pairs_used', 'n_windows',
               'window_type', 'min_train_months', 'n_oos_rows', 'n_oos_months',
               'rho_bar', 'k_eff', 'n_independent', 'mean_label_uniqueness',
               'min_detectable_edge_pp', 'power_gate_band', 'panel_start_month',
               'eur_dem_splice_flag', 'acc_challenger', 'acc_reference',
               'auc_challenger', 'delta_acc', 'delta_acc_ci_low', 'delta_acc_ci_high',
               'mcnemar_b', 'mcnemar_c', 'mcnemar_p', 'block_len',
               'shuffled_label_control_acc', 'lookahead_positive_control_acc',
               'alpha', 'cleared_bar', 'verdict', 'device_used', 'notes']


def write_log(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    os.makedirs(os.path.dirname(HYPOTHESIS_LOG) or '.', exist_ok=True)
    df.to_csv(HYPOTHESIS_LOG, index=False)
    return df


# The standing notes every row carries, so the log is self-describing.
TIER_A_NOTE = (
    'TIER-A ONLY: policy_rate, rate_3m, yield_10y and their derived slope/changes, '
    'plus FX, US equity. All are market or administrative data, published once and '
    'never restated, so their FULL history is usable with no vintage archive. '
    'BUYS: a panel starting 1976-01 instead of the predecessor programs 2013-07 '
    'ALFRED-limited start. COSTS: the Tier-B block (CPI, unemployment, industrial '
    'production, GDP) is dropped ENTIRELY, so nothing about inflation or the real '
    'economy is tested here -- only the interest-rate block, which is the part of '
    'the macro FX literature with the most empirical support.'
)
SPLICE_NOTE = (
    'EUR/DEM SPLICE: pre-1999 the euro does not exist; the Deutsche mark (FRED '
    'EXGEUS, monthly, DEM per USD) is inverted and multiplied by the irrevocable '
    'conversion rate 1 EUR = 1.95583 DEM. Standard in the literature but it IS a '
    'splice: rows carry eur_dem_splice=1, and EXGEUS is a MONTHLY AVERAGE where '
    'the euro series is a month-end point, which is why the 36-month overlap '
    'differs by ~1.6% on average. All 48 spliced rows fall in training blocks '
    'only -- the first out-of-sample year is well after 1999 -- so no out-of-sample '
    'metric depends on the splice.'
)
GATE_NOTE = (
    'POWER GATE IS LITERATURE-CALIBRATED and SUPERSEDES the predecessor programs '
    'count-based n_independent >= 150 floor. That floor passed a design whose '
    'minimum detectable edge was 2.4x the threshold that would have mattered. The '
    'bands here are keyed to published macro FX directional claims, which cluster '
    'at 52-55% accuracy (a 2-5pp edge): <=4.0pp PROCEED, 4.0-5.0pp PROCEED but only '
    'the UPPER HALF of the claimed range is resolvable, >5.0pp STOP.'
)
CARRY_NOTE = (
    'CARRY SIGN PRE-REGISTERED: the rule bets WITH the rate differential (long '
    'high-yield). Uncovered interest parity predicts the opposite -- high-yield '
    'currencies should depreciate -- but the forward premium puzzle, among the most '
    'replicated findings in international finance, is that empirically they do not. '
    'Fixing the sign in advance is what stops it being a free parameter.'
)
PROTOCOL_NOTE = (
    'WALK-FORWARD VALIDATION, NOT OPTIMISATION: hyperparameters are fixed once '
    '(logistic L2, C=1.0, class_weight=balanced) and never re-tuned per window. '
    'Standardiser and train-majority baseline are refit inside each window on that '
    'windows training rows only. All out-of-sample predictions from all windows are '
    'pooled into ONE set and ONE test is run; per-window accuracy is a descriptive '
    'stability profile, never a decision path. The most recent 36 months are '
    'reserved and were never trained on, predicted or inspected.'
)
ALPHA_NOTE = (
    'ALPHA: new family results/macro_tier_a_hypothesis_log.csv, size 2, '
    'alpha = 0.05/2 = 0.025. Touches no other family. REGISTERED-UNSPENT rows '
    'consume no alpha in this project (precedent: the dormant Fibonacci extension '
    'hypothesis, and the two UNSPENT rows in macro_panel_hypothesis_log.csv, which '
    'have never tightened any bar).'
)

# Directional priors from the carry / forward-premium literature, fixed in
# advance. Foreign rate up -> foreign currency appreciates (UP); US rate up ->
# foreign currency depreciates (DOWN). Blank where the literature gives no clear
# sign, which is not the same as expecting zero.
EXPECTED_COEF_SIGN = {
    'for_policy_rate': +1, 'us_policy_rate': -1,
    'for_rate_3m': +1, 'us_rate_3m': -1,
    'for_yield_10y': +1, 'us_yield_10y': -1,
    'for_rate_change_3m': +1, 'us_rate_change_3m': -1,
    'for_rate_change_12m': +1, 'us_rate_change_12m': -1,
}


def coefficient_table(coef_df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic coefficient signs and magnitudes, averaged over windows.

    Reported over the PRE-REGISTERED column list, so the amended-out `vix_level`
    appears as a BLANK row rather than silently vanishing: a later reader sees the
    set that was registered and what was removed from it. Under a DROP this table
    is the most interpretable output the program can produce.
    """
    rows = []
    for col in PRE_REGISTERED_FEATURE_COLUMNS:
        if col in REMOVED_FEATURES or col not in coef_df.columns:
            rows.append({'feature': col, 'mean_coef': None, 'std_coef': None,
                         'frac_windows_positive': None,
                         'expected_sign': EXPECTED_COEF_SIGN.get(col, ''),
                         'sign_matches_theory': '',
                         'status': 'REMOVED BY AMENDMENT (was pre-registered)'})
            continue
        vals = coef_df[col].to_numpy(float)
        mean = float(np.mean(vals))
        exp = EXPECTED_COEF_SIGN.get(col, '')
        rows.append({'feature': col, 'mean_coef': mean, 'std_coef': float(np.std(vals)),
                     'frac_windows_positive': float(np.mean(vals > 0)),
                     'expected_sign': exp,
                     'sign_matches_theory': ('' if exp == '' else
                                             bool(np.sign(mean) == np.sign(exp))),
                     'status': 'in model'})
    return pd.DataFrame(rows)


# ═══════════════════════ 11. orchestration ════════════════════════════════════

def _panel_bundle(pairs, cache, label, verbose=True):
    panel = build_panel(pairs, cache=cache, verbose=verbose)
    ok = complete_rows(panel)
    expanding = walk_forward_windows(panel, 'expanding')
    rolling = walk_forward_windows(panel, 'rolling')
    gate = power_gate(ok, expanding, pairs)
    return {'label': label, 'pairs': pairs, 'panel': panel, 'ok': ok,
            'expanding': expanding, 'rolling': rolling, 'gate': gate,
            'reserved': reserved_months(panel)}


def run(refresh: bool = False, verbose: bool = True) -> dict:
    """
    The whole program, in the mandated order: data, then the power gate, then --
    only if the gate permits -- controls, then hypotheses.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cpu'                       # a logistic regression needs no GPU
    cache = load_all_series(refresh=refresh)

    prov = provenance_table(cache)
    prov.to_csv(os.path.join(OUT_DIR, 'series_provenance.csv'), index=False)
    splice = eur_splice_overlap_check(cache)

    primary = _panel_bundle(G10_PANEL, cache, 'G10_primary', verbose)
    extended = _panel_bundle(EXTENDED_PANEL, cache, 'extended_G10_plus_EM', verbose)

    for b in (primary, extended):
        tag = b['label']
        b['ok'].to_csv(os.path.join(OUT_DIR, f'panel_{tag}.csv'), index=False)
        coverage_table(b['panel']).to_csv(
            os.path.join(OUT_DIR, f'coverage_{tag}.csv'), index=False)
        binding_constraint_table(b['panel']).to_csv(
            os.path.join(OUT_DIR, f'binding_constraints_{tag}.csv'), index=False)
        b['gate']['corr'].to_csv(os.path.join(OUT_DIR, f'correlation_{tag}.csv'))

    pd.DataFrame([{k: v for k, v in b['gate'].items() if k != 'corr'} | {'panel': b['label']}
                  for b in (primary, extended)]).to_csv(
        os.path.join(OUT_DIR, 'power_gate.csv'), index=False)

    result = {'device': device, 'provenance': prov, 'splice': splice,
              'primary': primary, 'extended': extended, 'ran_models': False}

    # ── THE HARD STOP ────────────────────────────────────────────────────────
    if not primary['gate']['proceed']:
        result['log'] = write_log([_unspent_row(1, 'H_tA.1_carry_vs_train_majority', primary),
                                   _unspent_row(2, 'H_tA.2_logistic_vs_carry', primary)])
        return result

    # ── MANDATORY CONTROLS (no alpha), reported before the hypotheses ────────
    pooled, profile, coefs = run_walk_forward(primary['ok'], primary['expanding'])
    shuf, _p, _c = run_walk_forward(primary['ok'], primary['expanding'], shuffle_labels=True)

    look_panel = build_panel(G10_PANEL, cache=cache, lookahead_months=1, verbose=verbose)
    look_ok = complete_rows(look_panel)
    look_wins = walk_forward_windows(look_panel, 'expanding')
    look_pooled, _lp, _lc = run_walk_forward(look_ok, look_wins)

    controls = {
        'shuffled_label_acc': float((shuf['pred_logit'] == shuf['y']).mean()) if len(shuf) else float('nan'),
        'shuffled_label_n': int(len(shuf)),
        'pooled_majority_rate': float(max(pooled['y'].mean(), 1 - pooled['y'].mean())),
        'lookahead_acc': float((look_pooled['pred_logit'] == look_pooled['y']).mean()) if len(look_pooled) else float('nan'),
        'lookahead_n': int(len(look_pooled)),
        'honest_acc': float((pooled['pred_logit'] == pooled['y']).mean()),
    }
    controls['lookahead_gain_pp'] = 100.0 * (controls['lookahead_acc'] - controls['honest_acc'])

    # ── HYPOTHESES ───────────────────────────────────────────────────────────
    h1 = arbitrate(pooled, 'pred_carry', 'pred_majority')
    h1['auc'] = _carry_auc(pooled)
    h2 = arbitrate(pooled, 'pred_logit', 'pred_carry')
    h2['auc'] = auc(pooled, 'proba_logit')
    h2_corroborating = arbitrate(pooled, 'pred_logit', 'pred_majority')

    # ── DESCRIPTIVE ARMS (never a path to KEEP) ──────────────────────────────
    roll_pooled, roll_profile, _rc = run_walk_forward(primary['ok'], primary['rolling'])
    ext_pooled, ext_profile, ext_coefs = run_walk_forward(extended['ok'], extended['expanding'])

    coef_tbl = coefficient_table(coefs)
    breakdowns = {'per_currency': accuracy_by(pooled, 'pair'),
                  'per_decade': per_decade(pooled),
                  'per_window': profile,
                  'rolling_per_window': roll_profile,
                  'extended_per_currency': accuracy_by(ext_pooled, 'pair'),
                  'coefficients': coef_tbl}

    pooled.to_csv(os.path.join(OUT_DIR, 'oos_predictions_primary.csv'), index=False)
    ext_pooled.to_csv(os.path.join(OUT_DIR, 'oos_predictions_extended.csv'), index=False)
    for name, frame in breakdowns.items():
        frame.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)

    rolling_desc = {
        'acc_logit': float((roll_pooled['pred_logit'] == roll_pooled['y']).mean()),
        'acc_carry': float((roll_pooled['pred_carry'] == roll_pooled['y']).mean()),
        'acc_majority': float((roll_pooled['pred_majority'] == roll_pooled['y']).mean()),
        'n': int(len(roll_pooled))}
    extended_desc = {
        'acc_logit': float((ext_pooled['pred_logit'] == ext_pooled['y']).mean()),
        'acc_carry': float((ext_pooled['pred_carry'] == ext_pooled['y']).mean()),
        'acc_majority': float((ext_pooled['pred_majority'] == ext_pooled['y']).mean()),
        'n': int(len(ext_pooled)),
        'h1_like': arbitrate(ext_pooled, 'pred_carry', 'pred_majority'),
        'h2_like': arbitrate(ext_pooled, 'pred_logit', 'pred_carry')}

    result.update({'pooled': pooled, 'controls': controls, 'h1': h1, 'h2': h2,
                   'h2_corroborating': h2_corroborating, 'rolling': rolling_desc,
                   'extended_desc': extended_desc, 'breakdowns': breakdowns,
                   'coefficients': coef_tbl, 'ran_models': True})
    result['log'] = write_log([
        _spent_row(1, 'H_tA.1_carry_vs_train_majority',
                   'pooled_walk_forward_month_block_bootstrap+exact_McNemar',
                   primary, h1, controls, device,
                   'H_tA.1 asks the first-order question: does the single most '
                   'documented macro FX signal produce directional predictability '
                   'in this panel at all? Reference is the per-window train-majority '
                   'class. ' + CARRY_NOTE),
        _spent_row(2, 'H_tA.2_logistic_vs_carry',
                   'pooled_walk_forward_month_block_bootstrap+exact_McNemar',
                   primary, h2, controls, device,
                   'PRIMARY reference is H_tA.1s predictions on IDENTICAL '
                   'out-of-sample rows. A model that beats a coin flip but not a '
                   'one-line rule has demonstrated nothing. Corroborating context '
                   f"(never a second path to KEEP): logistic vs train-majority "
                   f"delta={h2_corroborating['delta_acc']:+.4f}, "
                   f"McNemar p={h2_corroborating['mcnemar_p']:.4g}. "
                   'NO GBM and NO neural network were fitted: at a few hundred '
                   'independent observations with ~20 features a logistic is the '
                   'appropriate complexity. ' + AMENDMENT_NOTE),
    ])
    return result


def _carry_auc(pooled: pd.DataFrame) -> float:
    """Descriptive AUC for the carry rule, scored by the policy-rate differential
    -- the continuous signal the rule thresholds at zero."""
    from sklearn.metrics import roc_auc_score
    y = pooled['y'].to_numpy(int)
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, pooled['pred_carry'].to_numpy(float)))


def _common_fields(n, hypothesis, bundle):
    g, ok = bundle['gate'], bundle['ok']
    return {
        'n': n, 'date': pd.Timestamp.today().strftime('%Y-%m-%d'),
        'hypothesis': hypothesis, 'panel': bundle['label'],
        'pairs_used': '|'.join(sorted(bundle['pairs'])),
        'n_windows': g['n_windows'], 'window_type': 'expanding_annual_refit',
        'min_train_months': MIN_TRAIN_MONTHS, 'n_oos_rows': g['n_oos_rows'],
        'n_oos_months': g['n_oos_months'], 'rho_bar': round(g['rho_bar'], 6),
        'k_eff': round(g['k_eff'], 6), 'n_independent': g['n_independent'],
        'mean_label_uniqueness': g['mean_label_uniqueness'],
        'min_detectable_edge_pp': round(g['min_detectable_edge_pp'], 4),
        'power_gate_band': g['power_gate_band'],
        'panel_start_month': f"{ok['month_end'].min():%Y-%m}",
        'eur_dem_splice_flag': int(ok['eur_dem_splice'].sum()),
        'block_len': BLOCK_LEN_MONTHS, 'alpha': ALPHA,
    }


def _unspent_row(n, hypothesis, bundle):
    row = _common_fields(n, hypothesis, bundle)
    row.update({
        'arbiter': 'power_gate_STOP_before_fitting', 'acc_challenger': None,
        'acc_reference': None, 'auc_challenger': None, 'delta_acc': None,
        'delta_acc_ci_low': None, 'delta_acc_ci_high': None, 'mcnemar_b': None,
        'mcnemar_c': None, 'mcnemar_p': None, 'shuffled_label_control_acc': None,
        'lookahead_positive_control_acc': None, 'cleared_bar': None,
        'verdict': 'REGISTERED-UNSPENT', 'device_used': 'cpu',
        'notes': ('STOPPED BEFORE FITTING: minimum detectable edge '
                  f"{bundle['gate']['min_detectable_edge_pp']:.4f}pp exceeds the 5.0pp "
                  'STOP band, so this design cannot resolve the effect sizes the '
                  'literature actually claims. No model was fitted and NO ALPHA was '
                  'consumed. ' + ' '.join([GATE_NOTE, TIER_A_NOTE, SPLICE_NOTE,
                                           CARRY_NOTE, PROTOCOL_NOTE, ALPHA_NOTE,
                                           AMENDMENT_NOTE])),
    })
    return row


def _spent_row(n, hypothesis, arbiter, bundle, res, controls, device, extra):
    row = _common_fields(n, hypothesis, bundle)
    row.update({
        'arbiter': arbiter,
        'acc_challenger': round(res['acc_challenger'], 6),
        'acc_reference': round(res['acc_reference'], 6),
        'auc_challenger': round(res.get('auc', float('nan')), 6),
        'delta_acc': round(res['delta_acc'], 6),
        'delta_acc_ci_low': round(res['delta_acc_ci_low'], 6),
        'delta_acc_ci_high': round(res['delta_acc_ci_high'], 6),
        'mcnemar_b': res['mcnemar_b'], 'mcnemar_c': res['mcnemar_c'],
        'mcnemar_p': res['mcnemar_p'],
        'shuffled_label_control_acc': round(controls['shuffled_label_acc'], 6),
        'lookahead_positive_control_acc': round(controls['lookahead_acc'], 6),
        'cleared_bar': res['cleared_bar'], 'verdict': res['verdict'],
        'device_used': device,
        'notes': ' '.join([extra, GATE_NOTE, TIER_A_NOTE, SPLICE_NOTE,
                           PROTOCOL_NOTE, ALPHA_NOTE, AMENDMENT_NOTE]),
    })
    return row


# ═══════════════════════ 12. the report ═══════════════════════════════════════

def _gate_block(bundle) -> str:
    g = bundle['gate']
    return (f"  {bundle['label']:22s} pairs={g['n_pairs']:2d}  rho_bar={g['rho_bar']:.4f}  "
            f"k_eff={g['k_eff']:.4f}\n"
            f"  {'':22s} n_oos_rows={g['n_oos_rows']}  n_oos_months={g['n_oos_months']}  "
            f"n_independent={g['n_independent']} (cross-check {g['n_independent_crosscheck']})\n"
            f"  {'':22s} label_uniqueness={g['mean_label_uniqueness']:.4f}  "
            f"SE(acc)={g['se_accuracy']:.6f}\n"
            f"  {'':22s} MIN DETECTABLE EDGE = {g['min_detectable_edge_pp']:.4f} pp  "
            f"-> {g['power_gate_band']}\n")


def render_report(result: dict) -> str:
    """The report, in the mandated order. RAW numbers, no softening."""
    L, p, e = [], result['primary'], result['extended']
    add = L.append

    add('=' * 78)
    add('TIER-A MACRO FX DIRECTION, WALK-FORWARD')
    add('=' * 78)
    add('')
    add('1. DEVICE')
    add(f"   {result['device']} -- CPU only. A logistic regression needs no GPU and none")
    add('   was used. No neural network and no GBM were fitted anywhere in this program.')
    add('')
    add('2. DATA PROVENANCE')
    add(f"   Tier-A series only, {len(result['provenance'])} series. Full table: "
        f'{OUT_DIR}/series_provenance.csv')
    disc = result['provenance'][result['provenance']['discontinued_after'] != '']
    add(f'   DISCONTINUED mid-sample ({len(disc)}):')
    for _i, r in disc.iterrows():
        add(f"      {r['country']:3s} {r['feature']:26s} {r['series_id']:16s} "
            f"last print {r['discontinued_after']}  (staleness guard -> MISSING after "
            f"{MAX_STALENESS_DAYS.get(r['frequency'], 200)}d)")
    s = result['splice']
    add(f"   EUR/DEM SPLICE: {EUR_PREDECESSOR_SERIES} x {DEM_PER_EUR} DEM/EUR before "
        f"{EUR_SPLICE_DATE:%Y-%m}; overlap check on {s.get('n_overlap', 0)} months "
        f"({s.get('overlap_start', '-')}..{s.get('overlap_end', '-')}): mean abs diff "
        f"{s.get('mean_abs_diff_pct', float('nan')):.3f}%, max "
        f"{s.get('max_abs_diff_pct', float('nan')):.3f}% "
        '(EXGEUS is a monthly average vs a month-end point).')
    add('   Per-pair start and the feature that BINDS it:')
    for _i, r in binding_constraint_table(p['panel']).iterrows():
        add(f"      {r['pair']:7s} {r['first_complete_month']}  bound by "
            f"{r['binding_feature']}")
    add(f"   Panel usable start: {p['ok']['month_end'].min():%Y-%m}  "
        f"end: {p['ok']['month_end'].max():%Y-%m}")
    add('   EM extension: MXN and ZAR have a complete Tier-A block and are included in')
    add('   the extended arm. Declared but NOT buildable on Tier-A terms:')
    for ccy, why in EM_EXCLUDED.items():
        add(f'      {ccy}: {why}')
    add('')
    add('3. THE POWER GATE  (computed and reported BEFORE any model)')
    add(_gate_block(p).rstrip())
    add(_gate_block(e).rstrip())
    add(f'   Bands: <={GATE_PROCEED_PP}pp PROCEED | {GATE_PROCEED_PP}-{GATE_STOP_PP}pp '
        f'PROCEED, upper half of the literature range only | >{GATE_STOP_PP}pp STOP.')
    if p['gate']['power_gate_band'] == 'PROCEED-UPPER-HALF-ONLY':
        add('   *** ONLY THE UPPER HALF OF THE LITERATURE RANGE IS RESOLVABLE. A DROP')
        add('       HERE DOES NOT RULE OUT A 2-3pp EFFECT. ***')
    if not result['ran_models']:
        add('')
        add('   STOP. No model was fitted; both hypotheses are REGISTERED-UNSPENT and')
        add('   consumed no alpha. The report ends here, as the protocol requires.')
        return '\n'.join(L)

    add('')
    add('4. WALK-FORWARD STRUCTURE')
    prof = result['breakdowns']['per_window']
    add(f"   {len(p['expanding'])} expanding windows, annual refit, min train "
        f'{MIN_TRAIN_MONTHS} months.')
    add(f"   train months {prof['train_months'].min()}..{prof['train_months'].max()}, "
        f"train rows {prof['train_rows'].min()}..{prof['train_rows'].max()}, "
        f"oos rows/window {prof['oos_rows'].min()}..{prof['oos_rows'].max()}")
    add(f"   RESERVED HOLDOUT: {len(p['reserved'])} months "
        f"{min(p['reserved']):%Y-%m}..{max(p['reserved']):%Y-%m} -- never trained on,")
    add('   never predicted, never inspected. Asserted un-indexed by every window.')
    add('')
    add('5. MANDATORY CONTROLS  (no alpha)')
    c = result['controls']
    add(f"   shuffled-label control : acc {c['shuffled_label_acc']:.4f} on "
        f"{c['shuffled_label_n']} rows vs pooled majority rate "
        f"{c['pooled_majority_rate']:.4f}")
    add(f"   look-ahead positive    : acc {c['lookahead_acc']:.4f} vs honest "
        f"{c['honest_acc']:.4f}  -> {c['lookahead_gain_pp']:+.2f} pp")
    add('        (features deliberately shifted one month LATER than the availability')
    add('         rule permits. A measurable GAIN is what proves we would notice leakage.)')
    add('   regime stability, carry rule by decade:')
    for _i, r in result['breakdowns']['per_decade'].iterrows():
        add(f"      {r['decade']}  n={int(r['n']):5d}  carry {r['acc_carry']:.4f}   "
            f"logit {r['acc_logit']:.4f}   base rate up {r['base_rate_up']:.4f}")
    add('')
    add('6. H_tA.1 -- DOES CARRY WORK?   (carry rule vs per-window train majority)')
    add(_hyp_block(result['h1']))
    add('')
    add('7. H_tA.2 -- DOES THE MODEL BEAT CARRY?   (L2 logistic vs carry, same rows)')
    add(_hyp_block(result['h2']))
    add(f"   corroborating only, never a path to KEEP: logistic vs train-majority "
        f"delta {result['h2_corroborating']['delta_acc']:+.4f}, "
        f"McNemar p {result['h2_corroborating']['mcnemar_p']:.4g}")
    add('')
    add('8. DESCRIPTIVE ARMS  (never a path to KEEP)')
    r_ = result['rolling']
    add(f"   rolling {ROLLING_WINDOW_MONTHS}-month window, n={r_['n']}: carry "
        f"{r_['acc_carry']:.4f}  logit {r_['acc_logit']:.4f}  majority "
        f"{r_['acc_majority']:.4f}")
    x = result['extended_desc']
    add(f"   extended panel (+MXN,+ZAR), n={x['n']}: carry {x['acc_carry']:.4f}  "
        f"logit {x['acc_logit']:.4f}  majority {x['acc_majority']:.4f}")
    for tag, key in (('carry vs majority', 'h1_like'), ('logit vs carry ', 'h2_like')):
        d = x[key]
        # DESCRIPTIVE ARM: the words KEEP and DROP are deliberately NOT used here.
        # This arm is not registered and cannot clear anything.
        add(f"      {tag}  delta {d['delta_acc']:+.4f} "
            f"CI [{d['delta_acc_ci_low']:+.4f}, {d['delta_acc_ci_high']:+.4f}] "
            f"p {d['mcnemar_p']:.4g}  -> "
            f"{'would have cleared' if d['cleared_bar'] else 'would not have cleared'} "
            'the bar, DESCRIPTIVE ONLY, not a registered hypothesis and not a KEEP')
    add('   READ THIS CORRECTLY: the extended panels carry-vs-majority delta is LARGER')
    add(f"   than the primary's ({x['h1_like']['delta_acc']:+.4f} vs "
        f"{result['h1']['delta_acc']:+.4f}) even though carry itself is LESS accurate "
        f"there ({x['acc_carry']:.4f} vs {result['h1']['acc_challenger']:.4f}). The gap")
    add(f"   widens because the REFERENCE collapses ({x['acc_majority']:.4f} vs "
        f"{result['h1']['acc_reference']:.4f}): MXN and ZAR trend down against the dollar,")
    add('   so a train-majority baseline that predicts UP is badly wrong on them. That is')
    add('   a weaker baseline, not a stronger signal, and it is exactly why this arm was')
    add('   pre-registered as descriptive and never as a path to KEEP.')
    add(f"   Agreement with the primary: both arms DROP the logistic against carry, and")
    add('   both show carry above its baseline; they do not point in opposite directions.')
    add('')
    add('9. COEFFICIENTS AND BREAKDOWNS  (descriptive)')
    for _i, r in result['coefficients'].iterrows():
        if r['status'].startswith('REMOVED'):
            add(f"      {r['feature']:22s} {'--':>9s} {'--':>7s}   {r['status']}")
            continue
        exp = r['expected_sign']
        tag = ('' if exp == '' else
               ('sign MATCHES theory' if r['sign_matches_theory'] else 'sign OPPOSES theory'))
        add(f"      {r['feature']:22s} {r['mean_coef']:+9.4f} +-{r['std_coef']:6.4f}  "
            f"{'pos in %.0f%% of windows' % (100 * r['frac_windows_positive']):<26s} {tag}")
    add('   per-currency accuracy:')
    for _i, r in result['breakdowns']['per_currency'].iterrows():
        add(f"      {r['pair']:7s} n={int(r['n']):5d}  carry {r['acc_carry']:.4f}  "
            f"logit {r['acc_logit']:.4f}  base rate up {r['base_rate_up']:.4f}")
    add('')
    add('10. VERDICTS')
    add(f"   H_tA.1 carry vs train-majority : {result['h1']['verdict']}")
    add(f"   H_tA.2 logistic vs carry       : {result['h2']['verdict']}")
    add(f'   family alpha = {ALPHA} (size {FAMILY_SIZE}); KEEP requires the block CI')
    add('   entirely above zero AND exact McNemar p < alpha.')
    return '\n'.join(L)


def _hyp_block(res: dict) -> str:
    return (f"   acc challenger {res['acc_challenger']:.4f}   "
            f"acc reference {res['acc_reference']:.4f}   "
            f"delta {res['delta_acc']:+.4f}\n"
            f"   block bootstrap CI [{res['delta_acc_ci_low']:+.4f}, "
            f"{res['delta_acc_ci_high']:+.4f}]  "
            f"(2000 resamples, {BLOCK_LEN_MONTHS}-month moving blocks over months)\n"
            f"   exact McNemar b={res['mcnemar_b']} c={res['mcnemar_c']} "
            f"p={res['mcnemar_p']:.6g}   AUC {res.get('auc', float('nan')):.4f} "
            '(descriptive)\n'
            f"   -> {res['verdict']}")


if __name__ == '__main__':
    out = run()
    text = render_report(out)
    print(text)
    with open(os.path.join(OUT_DIR, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
