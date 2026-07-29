"""
GLOBAL MACRO FX PANEL — data layer (research-only).

Builds a monthly, vintage-correct panel of nine G10 currencies against the
dollar, with each country's macro block entered as SEPARATE LEVELS rather than
as differentials.

THE VINTAGE PROBLEM, and how it is handled here
-----------------------------------------------
FRED's default series return the LATEST REVISED value. Using them is look-ahead
twice over: the value for month M is not knowable during month M (PUBLICATION
LAG), and the number known at the time is not today's number (REVISION).

Every series is classified into a tier and handled explicitly:

  TIER A -- not revised, known in real time (policy rates, 3-month rates,
      10-year yields, exchange rates, equity indices). The underlying market
      price is observable as it happens, so the monthly value is available at
      the end of its own reference month.

  TIER B -- revised, but ALFRED vintages available (CPI, unemployment). Fetched
      with `output_type=4` ("initial release only"), which returns the FIRST
      PRINT together with its ACTUAL publication date. The availability rule is
      therefore enforced against a real publication calendar, not an assumed
      lag.

  TIER C -- revised, no usable vintage coverage. Handled with a conservative
      per-series publication lag and FLAGGED in the output. The Tier-C fraction
      is reported.

AVAILABILITY DATING RULE (enforced by `as_of`)
----------------------------------------------
The feature vector used to predict month M+1 may contain ONLY values whose
PUBLICATION DATE is on or before the last day of month M. Not the reference
period -- the publication date. A quarterly CPI print from two months ago is
legitimately available; a monthly CPI for month M published on the 15th of M+1
is not. This one rule handles monthly, quarterly and irregular series
identically, which is why no series-specific lag bookkeeping is needed for
Tier A/B.

STALENESS GUARD: a value is only carried forward while it is plausibly current.
Past `max_staleness_days` for its frequency the feature goes MISSING rather than
pretending a years-old print is today's macro state. Several OECD series were
discontinued mid-sample; without this guard they would silently forward-fill to
the present.

A PRE-REGISTERED FEATURE THAT COULD NOT BE BUILT
------------------------------------------------
`industrial_production_yoy_pct` is NOT in the output. FRED's OECD
`PRINTO01*` family was retired and no consistent cross-country monthly
industrial-production series survives on FRED; only the US `INDPRO` remains.
Including it for the US block alone would make the US and foreign blocks
ASYMMETRIC, which breaks the entire separate-levels design -- the model would
see a US variable with no foreign counterpart and could use its presence as a
country marker. Dropping it for every country is the symmetric choice. This is a
deviation from the pre-registered feature list, made for data availability and
reported rather than hidden.

NO P&L, no position sizing, no backtest. This module only assembles data.
"""

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = 'results/macro_panel'
PANEL_CSV = os.path.join(CACHE_DIR, 'panel_monthly.csv')
PROVENANCE_CSV = os.path.join(CACHE_DIR, 'series_provenance.csv')
REVISIONS_CSV = os.path.join(CACHE_DIR, 'revision_check.csv')

FRED_BASE = 'https://api.stlouisfed.org/fred/series/observations'
PANEL_START = '1999-01-01'          # the euro era; DEXUSEU begins 1999-01

TIER_A, TIER_B, TIER_C = 'A', 'B', 'C'

# Staleness limits: past this many days since the last PUBLICATION, the feature
# is missing rather than forward-filled from a discontinued series.
MAX_STALENESS_DAYS = {'M': 200, 'Q': 320, 'D': 20}

# ── Quote convention: every pair FOREIGN/USD, so "dollar strengthens" is a
#    down-move in all nine. `invert` marks the market-convention USD/XXX series.
PAIRS = {
    'EURUSD': {'country': 'EZ', 'series': 'DEXUSEU', 'invert': False},
    'GBPUSD': {'country': 'GB', 'series': 'DEXUSUK', 'invert': False},
    'AUDUSD': {'country': 'AU', 'series': 'DEXUSAL', 'invert': False},
    'NZDUSD': {'country': 'NZ', 'series': 'DEXUSNZ', 'invert': False},
    'JPYUSD': {'country': 'JP', 'series': 'DEXJPUS', 'invert': True},
    'CHFUSD': {'country': 'CH', 'series': 'DEXSZUS', 'invert': True},
    'CADUSD': {'country': 'CA', 'series': 'DEXCAUS', 'invert': True},
    'SEKUSD': {'country': 'SE', 'series': 'DEXSDUS', 'invert': True},
    'NOKUSD': {'country': 'NO', 'series': 'DEXNOUS', 'invert': True},
}
PAIR_ORDER = tuple(PAIRS)

# ── The pinned series map. Every id below was probed live against the FRED API
#    for existence, frequency and coverage before being written here.
#    (series_id, tier, frequency, is_index_for_yoy)
COUNTRY_SERIES = {
    'US': {'policy_rate': ('IRSTCI01USM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01USM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01USM156N', TIER_A, 'M'),
           'cpi_index': ('CPIAUCSL', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTUSM156S', TIER_B, 'M')},
    'EZ': {'policy_rate': ('IRSTCI01EZM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01EZM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01EZM156N', TIER_A, 'M'),
           'cpi_index': ('CP0000EZ19M086NEST', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTEZM156S', TIER_B, 'M')},
    'GB': {'policy_rate': ('IRSTCI01GBM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01GBM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01GBM156N', TIER_A, 'M'),
           'cpi_index': ('GBRCPIALLMINMEI', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTGBM156S', TIER_B, 'M')},
    'AU': {'policy_rate': ('IRSTCI01AUM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01AUM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01AUM156N', TIER_A, 'M'),
           'cpi_index': ('AUSCPIALLQINMEI', TIER_B, 'Q'),
           'unemployment_rate': ('LRHUTTTTAUM156S', TIER_B, 'M')},
    'NZ': {'policy_rate': ('IRSTCI01NZM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01NZM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01NZM156N', TIER_A, 'M'),
           'cpi_index': ('NZLCPIALLQINMEI', TIER_B, 'Q'),
           'unemployment_rate': ('LRHUTTTTNZQ156S', TIER_B, 'Q')},
    'JP': {'policy_rate': ('IRSTCI01JPM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01JPM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01JPM156N', TIER_A, 'M'),
           'cpi_index': ('JPNCPIALLMINMEI', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTJPM156S', TIER_B, 'M')},
    'CH': {'policy_rate': ('IRSTCI01CHM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01CHM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01CHM156N', TIER_A, 'M'),
           'cpi_index': ('CHECPIALLMINMEI', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTCHQ156S', TIER_B, 'Q')},
    'CA': {'policy_rate': ('IRSTCI01CAM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01CAM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01CAM156N', TIER_A, 'M'),
           'cpi_index': ('CANCPIALLMINMEI', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTCAM156S', TIER_B, 'M')},
    'SE': {'policy_rate': ('IRSTCI01SEM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01SEM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01SEM156N', TIER_A, 'M'),
           'cpi_index': ('CP0000SEM086NEST', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTSEM156S', TIER_B, 'M')},
    'NO': {'policy_rate': ('IRSTCI01NOM156N', TIER_A, 'M'),
           'rate_3m': ('IR3TIB01NOM156N', TIER_A, 'M'),
           'yield_10y': ('IRLTLT01NOM156N', TIER_A, 'M'),
           'cpi_index': ('CP0000NOM086NEST', TIER_B, 'M'),
           'unemployment_rate': ('LRHUTTTTNOM156S', TIER_B, 'M')},
}

US_EQUITY_SERIES = ('SPASTT01USM661N', TIER_A, 'M')
VIX_CACHE = 'results/vix.csv'          # maintained by the project's FRED framework

# The three Tier-B series the mandatory revision-size check runs on.
REVISION_CHECK_SERIES = ('CPIAUCSL', 'LRHUTTTTUSM156S', 'CANCPIALLMINMEI')


class MissingApiKeyError(RuntimeError):
    """ALFRED vintages require a FRED API key. Without one the first-print
    discipline is impossible and this program must STOP rather than silently
    fall back to revised values."""


# ───────────────────── FRED / ALFRED access ───────────────────────────────────

def _api_key():
    key = os.getenv('FRED_API_KEY')
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv('.env')
            key = os.getenv('FRED_API_KEY')
        except Exception:
            key = None
    if not key or key == 'YOUR_FRED_API_KEY_HERE':
        raise MissingApiKeyError(
            'FRED_API_KEY is required: Tier-B series must be fetched as ALFRED '
            'first prints, and the public CSV endpoint serves only revised '
            'values. STOP rather than silently using revised data.')
    return key


def _get(url, retries=3, pause=1.0):
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.load(r)
        except Exception as exc:                       # transient network issues
            last = exc
            time.sleep(pause * (attempt + 1))
    raise last


def fetch_series(series_id: str, first_release: bool, start: str = '1990-01-01'):
    """
    One FRED/ALFRED series as (reference_date, value, publication_date).

    `first_release=True` uses ALFRED output_type=4 ("initial release only"),
    which returns the FIRST PRINT of each observation together with the real
    date it became public -- so the availability rule is enforced against an
    actual publication calendar rather than an assumed lag.

    `first_release=False` is for TIER A, where the value is not revised and the
    underlying market price is observable in real time; publication is taken as
    the end of the reference period.
    """
    params = {'series_id': series_id, 'api_key': _api_key(), 'file_type': 'json',
              'observation_start': start}
    if first_release:
        params.update({'realtime_start': '1776-07-04', 'realtime_end': '9999-12-31',
                       'output_type': '4'})
    data = _get(f'{FRED_BASE}?{urllib.parse.urlencode(params)}')
    rows = []
    for o in data.get('observations', []):
        if o['value'] in ('.', '', None):
            continue
        ref = pd.Timestamp(o['date'])
        pub = pd.Timestamp(o['realtime_start']) if first_release else None
        rows.append({'ref_date': ref, 'value': float(o['value']),
                     'publication_date': pub})
    df = pd.DataFrame(rows)
    if len(df) and not first_release:
        # TIER A: available at the end of its own reference period.
        df['publication_date'] = (df['ref_date'] + pd.offsets.MonthEnd(0))
    return df.sort_values('ref_date').reset_index(drop=True)


def load_or_fetch(series_id: str, first_release: bool, cache_dir: str = CACHE_DIR,
                  refresh: bool = False) -> pd.DataFrame:
    """Cache each series on disk so a re-run does not re-hit the API."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = 'firstprint' if first_release else 'level'
    path = os.path.join(cache_dir, f'{series_id}__{tag}.csv')
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, parse_dates=['ref_date', 'publication_date'])
        return df
    df = fetch_series(series_id, first_release)
    df.to_csv(path, index=False)
    return df


# ───────────────────── the availability rule ──────────────────────────────────

def as_of(series: pd.DataFrame, asof: pd.Timestamp, freq: str = 'M'):
    """
    The value of `series` KNOWN at `asof`: the most recent observation whose
    PUBLICATION DATE is on or before `asof`.

    Returns (value, ref_date, publication_date), or (nan, NaT, NaT) when nothing
    has been published yet or the newest print is staler than the frequency's
    limit. Dating by reference period instead of publication date would let a
    figure for month M be used during month M -- the exact error this guards.
    """
    if series is None or not len(series):
        return np.nan, pd.NaT, pd.NaT
    pub = pd.DatetimeIndex(series['publication_date'])
    known = series[pub <= asof]
    if not len(known):
        return np.nan, pd.NaT, pd.NaT
    row = known.iloc[-1]
    limit = MAX_STALENESS_DAYS.get(freq, 200)
    if (asof - pd.Timestamp(row['publication_date'])).days > limit:
        return np.nan, pd.NaT, pd.NaT
    return float(row['value']), pd.Timestamp(row['ref_date']), pd.Timestamp(row['publication_date'])


def as_of_yoy(series: pd.DataFrame, asof: pd.Timestamp, freq: str = 'M'):
    """
    Year-on-year percent change of an INDEX series, computed only from prints
    available at `asof`. Both the current and the year-ago observation must
    already have been published.
    """
    value, ref, _pub = as_of(series, asof, freq)
    if not np.isfinite(value) or pd.isna(ref):
        return np.nan
    pub = pd.DatetimeIndex(series['publication_date'])
    known = series[pub <= asof]
    target = ref - pd.DateOffset(years=1)
    prior = known[np.abs((pd.DatetimeIndex(known['ref_date']) - target).days) <= 20]
    if not len(prior):
        return np.nan
    base = float(prior.iloc[-1]['value'])
    return (value / base - 1.0) * 100.0 if base else np.nan


# ───────────────────── FX prices, correctly oriented ──────────────────────────

def invert_quote(frame: pd.DataFrame) -> pd.DataFrame:
    """
    USD/XXX -> XXX/USD. Reciprocal on every price column, and where a frame
    carries a high and a low they SWAP: 1/low_old is the new high and 1/high_old
    the new low. This project has made this mistake's cousin before, so it is
    asserted in a unit test.
    """
    out = frame.copy()
    for col in ('open', 'close', 'price'):
        if col in out:
            out[col] = 1.0 / frame[col]
    if 'high' in out and 'low' in out:
        out['high'] = 1.0 / frame['low']
        out['low'] = 1.0 / frame['high']
    return out


def monthly_fx(pair: str, refresh: bool = False) -> pd.DataFrame:
    """
    Monthly close for one pair, oriented FOREIGN/USD, taken as the LAST
    BUSINESS DAY of each month from FRED's H.10 daily series. Using FRED for all
    nine keeps the source consistent and needs no MT5 session.
    """
    spec = PAIRS[pair]
    raw = load_or_fetch(spec['series'], first_release=False, refresh=refresh)
    if not len(raw):
        return pd.DataFrame(columns=['month_end', 'price'])
    s = pd.DataFrame({'date': pd.DatetimeIndex(raw['ref_date']),
                      'price': raw['value'].astype(float)}).dropna()
    s = s.set_index('date').sort_index()
    monthly = s.resample('ME').last().dropna()
    out = pd.DataFrame({'month_end': monthly.index, 'price': monthly['price'].to_numpy()})
    if spec['invert']:
        out = invert_quote(out)
    return out.reset_index(drop=True)


# ───────────────────── panel assembly ─────────────────────────────────────────

def _country_block(country: str, asof: pd.Timestamp, cache: dict,
                   tier_b_asof: pd.Timestamp = None) -> dict:
    """
    One country's macro block AS KNOWN at `asof`, entered as separate levels.

    `tier_b_asof` exists ONLY for the mandatory look-ahead POSITIVE CONTROL: set
    it later than `asof` to deliberately leak revised-period information into the
    Tier-B features. Accuracy should then IMPROVE; if it does not, the
    availability dating is not binding and must be inspected.
    """
    spec = COUNTRY_SERIES[country]
    tb = asof if tier_b_asof is None else tier_b_asof
    out = {}
    for name in ('policy_rate', 'rate_3m', 'yield_10y'):
        sid, _tier, freq = spec[name]
        out[name], _r, _p = as_of(cache[sid], asof, freq)
    out['curve_slope'] = out['yield_10y'] - out['rate_3m']

    sid, _t, freq = spec['cpi_index']
    out['cpi_yoy_pct'] = as_of_yoy(cache[sid], tb, freq)

    sid, _t, freq = spec['unemployment_rate']
    out['unemployment_rate'], _r, _p = as_of(cache[sid], tb, freq)

    # policy DIRECTION, not just level -- central-bank cycles are what macro
    # traders actually trade. Both ends use only values published by `asof`.
    sid, _t, freq = spec['policy_rate']
    now, _r, _p = as_of(cache[sid], asof, freq)
    then, _r2, _p2 = as_of(cache[sid], asof - pd.DateOffset(months=3), freq)
    out['rate_change_3m'] = (now - then) if (np.isfinite(now) and np.isfinite(then)) else np.nan
    return out


def _load_all_series(refresh: bool = False) -> dict:
    """Every series once, cached by id. Tier B is fetched as FIRST PRINTS."""
    cache = {}
    for spec in COUNTRY_SERIES.values():
        for sid, tier, _freq in spec.values():
            if sid not in cache:
                cache[sid] = load_or_fetch(sid, first_release=(tier == TIER_B),
                                           refresh=refresh)
    sid, tier, _f = US_EQUITY_SERIES
    cache[sid] = load_or_fetch(sid, first_release=False, refresh=refresh)
    return cache


def _vix_monthly() -> pd.DataFrame:
    """VIX from the project's existing FRED cache -- reused, never refetched."""
    if not os.path.exists(VIX_CACHE):
        return pd.DataFrame(columns=['ref_date', 'value', 'publication_date'])
    df = pd.read_csv(VIX_CACHE, index_col=0, parse_dates=True)
    col = [c for c in df.columns if 'vix' in c.lower()] or list(df.columns[:1])
    s = df[col[0]].dropna()
    return pd.DataFrame({'ref_date': pd.DatetimeIndex(s.index).tz_localize(None),
                         'value': s.to_numpy(dtype=float),
                         'publication_date': pd.DatetimeIndex(s.index).tz_localize(None)})


def build_panel(refresh: bool = False, start: str = PANEL_START,
                write: bool = True, verbose: bool = True,
                lookahead_tier_b_months: int = 0) -> pd.DataFrame:
    """
    The monthly panel: one row per (month, pair), features known at the month's
    last day, target = sign of the NEXT month's log return.

    Monthly observations are NON-OVERLAPPING, so label uniqueness is exactly 1.0
    by construction -- the property every prior H1 program in this project
    lacked.
    """
    cache = _load_all_series(refresh=refresh)
    vix = _vix_monthly()
    eq_sid, _t, eq_freq = US_EQUITY_SERIES

    fx = {p: monthly_fx(p, refresh=refresh) for p in PAIR_ORDER}
    months = pd.date_range(start, pd.Timestamp.today().normalize(), freq='ME')

    rows = []
    for pair in PAIR_ORDER:
        px = fx[pair].set_index('month_end')['price'].sort_index()
        if not len(px):
            continue
        logp = np.log(px)
        for m in months:
            if m not in px.index:
                continue
            nxt = m + pd.offsets.MonthEnd(1)
            if nxt not in px.index:
                continue                       # no next month -> no label
            rec = {'month_end': m, 'pair': pair,
                   'country': PAIRS[pair]['country'], 'price': float(px.loc[m])}

            tb = (m + pd.DateOffset(months=lookahead_tier_b_months)
                  if lookahead_tier_b_months else None)
            for prefix, ctry in (('for', PAIRS[pair]['country']), ('us', 'US')):
                for k, v in _country_block(ctry, m, cache, tier_b_asof=tb).items():
                    rec[f'{prefix}_{k}'] = v

            vix_v, _r, _p = as_of(vix, m, 'D')
            rec['vix_level'] = vix_v
            eq_now, eq_ref, _p = as_of(cache[eq_sid], m, eq_freq)
            eq_prev, _r2, _p2 = as_of(cache[eq_sid], m - pd.DateOffset(months=1), eq_freq)
            rec['us_equity_1m_return'] = (
                (eq_now / eq_prev - 1.0) * 100.0
                if np.isfinite(eq_now) and np.isfinite(eq_prev) and eq_prev else np.nan)

            for k, lag in (('own_return_1m', 1), ('own_return_3m', 3), ('own_return_12m', 12)):
                past = m - pd.offsets.MonthEnd(lag)
                rec[k] = (float(logp.loc[m] - logp.loc[past]) * 100.0
                          if past in logp.index else np.nan)

            fwd = float(logp.loc[nxt] - logp.loc[m])
            rec['fwd_logret_1m'] = fwd
            rec['y_1m'] = int(fwd > 0)

            nxt3 = m + pd.offsets.MonthEnd(3)
            if nxt3 in logp.index:
                f3 = float(logp.loc[nxt3] - logp.loc[m])
                rec['fwd_logret_3m'] = f3
                rec['y_3m'] = int(f3 > 0)
            else:
                rec['fwd_logret_3m'] = np.nan
                rec['y_3m'] = np.nan
            rows.append(rec)

    panel = pd.DataFrame(rows).sort_values(['month_end', 'pair']).reset_index(drop=True)
    if write and len(panel) and not lookahead_tier_b_months:
        os.makedirs(CACHE_DIR, exist_ok=True)
        panel.to_csv(PANEL_CSV, index=False)
    if verbose:
        print(f"panel: {len(panel)} rows, {panel['month_end'].nunique()} months, "
              f"{panel['pair'].nunique()} pairs")
    return panel


# ───────────────────── provenance + revision check ────────────────────────────

def provenance_table(write: bool = True) -> pd.DataFrame:
    """Which series, which country, which tier, what coverage."""
    rows = []
    for country, spec in COUNTRY_SERIES.items():
        for name, (sid, tier, freq) in spec.items():
            try:
                df = load_or_fetch(sid, first_release=(tier == TIER_B))
                cov = (f"{pd.Timestamp(df['ref_date'].min()):%Y-%m}.."
                       f"{pd.Timestamp(df['ref_date'].max()):%Y-%m}") if len(df) else 'EMPTY'
                n = len(df)
            except Exception as exc:
                cov, n = f'FAIL {type(exc).__name__}', 0
            rows.append({'country': country, 'feature': name, 'series_id': sid,
                         'tier': tier, 'frequency': freq, 'n_obs': n,
                         'coverage': cov, 'source': 'FRED/ALFRED',
                         'vintage_adjusted': tier == TIER_B})
    sid, tier, freq = US_EQUITY_SERIES
    rows.append({'country': 'GLOBAL', 'feature': 'us_equity_1m_return',
                 'series_id': sid, 'tier': tier, 'frequency': freq, 'n_obs': 0,
                 'coverage': '', 'source': 'FRED', 'vintage_adjusted': False})
    rows.append({'country': 'GLOBAL', 'feature': 'vix_level', 'series_id': 'VIXCLS',
                 'tier': TIER_A, 'frequency': 'D', 'n_obs': 0, 'coverage': '',
                 'source': 'results/vix.csv (project FRED framework, reused)',
                 'vintage_adjusted': False})
    for pair, spec in PAIRS.items():
        rows.append({'country': spec['country'], 'feature': f'fx_{pair}',
                     'series_id': spec['series'], 'tier': TIER_A, 'frequency': 'D',
                     'n_obs': 0, 'coverage': '', 'source': 'FRED H.10',
                     'vintage_adjusted': False})
    out = pd.DataFrame(rows)
    if write:
        os.makedirs(CACHE_DIR, exist_ok=True)
        out.to_csv(PROVENANCE_CSV, index=False)
    return out


def tier_c_fraction() -> float:
    """Fraction of country-block features handled WITHOUT vintage correction.
    Every macro feature here is Tier A or Tier B, so this is 0.0 -- reported
    rather than assumed."""
    tiers = [t for spec in COUNTRY_SERIES.values() for (_s, t, _f) in spec.values()]
    return float(sum(1 for t in tiers if t == TIER_C) / len(tiers))


def revision_check(series_ids=REVISION_CHECK_SERIES, write: bool = True) -> pd.DataFrame:
    """
    MANDATORY revision-size check: first print versus today's revised value, for
    at least three Tier-B series including US CPI.

    If revisions turn out to be trivial the vintage discipline mattered little
    here; if they are large, that is itself a finding about every naive macro-ML
    study that used revised data.
    """
    rows = []
    for sid in series_ids:
        first = load_or_fetch(sid, first_release=True)
        latest = load_or_fetch(sid, first_release=False)
        if not len(first) or not len(latest):
            continue
        merged = first.merge(latest[['ref_date', 'value']], on='ref_date',
                             suffixes=('_first', '_revised')).dropna()
        merged = merged[merged['ref_date'] >= pd.Timestamp(PANEL_START)]
        if not len(merged):
            continue
        diff = (merged['value_revised'] - merged['value_first']).abs()
        rel = 100.0 * diff / merged['value_first'].abs().replace(0, np.nan)
        rows.append({
            'series_id': sid, 'n_compared': int(len(merged)),
            'mean_abs_revision': float(diff.mean()),
            'max_abs_revision': float(diff.max()),
            'mean_abs_revision_pct': float(rel.mean()),
            'max_abs_revision_pct': float(rel.max()),
            'n_identical': int((diff == 0).sum()),
        })
    out = pd.DataFrame(rows)
    if write and len(out):
        os.makedirs(CACHE_DIR, exist_ok=True)
        out.to_csv(REVISIONS_CSV, index=False)
    return out


if __name__ == '__main__':
    print(provenance_table().to_string(index=False))
    print()
    print(revision_check().to_string(index=False))
    print()
    build_panel()
