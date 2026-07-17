"""
FOMC meeting-day calendar features (candidate INPUT features, both families).

Source of truth: `results/fomc_dates.csv` — a SMALL, MOSTLY-STATIC reference
file (~8 dates/year) committed to the repo, NOT a live-fetch-per-request feed
like the FRED macro series. It carries the statement day (the LAST day of each
scheduled meeting — the rate decision / uncertainty-resolution moment) for
every scheduled FOMC meeting 1998→the last year the Fed has published.

    >>> python -m src.fomc_calendar        # rebuild the CSV from the Fed site

**MANUAL ANNUAL REFRESH REQUIRED**: the Fed publishes the next year's schedule
roughly 1–2 years ahead on fomccalendars.htm; re-run the builder once a year
(or whenever the calendar page adds a year) and commit the refreshed CSV.

Provenance (probed live 2026-07-17):
  * https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm — the
    official calendar, year panels 2021..2027 (current + future meetings).
  * https://www.federalreserve.gov/monetarypolicy/fomchistorical{YYYY}.htm —
    official historical year pages, used for 1998..2020 (pages appear ~5 years
    after the fact; the builder takes whatever exists and reports gaps).
  * REJECTED source: ALFRED's release-dates download for the "FOMC Press
    Release" release (rid=101). Probed live: it returns ~3.7k NEAR-DAILY dates
    because (per its own header) it lists "the dates when any series from this
    release was revised" — data-revision timestamps, not meeting days.

Scheduled meetings ONLY: headings/rows qualified "(unscheduled)",
"(cancelled)" or "(notation vote)" are excluded. Unscheduled emergency actions
(e.g. 2020-03-03 / 2020-03-15) were NOT knowable in advance, so a
`days_to_next_fomc` countdown that included them would inject future knowledge
into pre-announcement rows — exactly the look-ahead this project forbids.
(Known approximation: the cancelled 2020-03-17/18 meeting WAS in the ex-ante
calendar until it was scrapped; today's ex-post pages cannot reproduce that,
one meeting in ~28 years.) Scheduled dates themselves are public knowledge
months-to-years ahead, so consuming them at row t is look-ahead-free by
construction — there is no publish-lag surface (unlike COT / FRED series).

The three model features (bundled as ONE hypothesis per family — three views
of the same calendar fact, not independent signals):
  * is_fomc_day          — 1 iff the row's date IS a statement day
  * days_to_next_fomc    — calendar days to the next statement day (0 on it)
  * days_since_last_fomc — calendar days since the last statement day (0 on it)
"""
import os
import re

import numpy as np
import pandas as pd

FOMC_DATES_CSV = 'results/fomc_dates.csv'
FOMC_FEATURE_COLUMNS = ['is_fomc_day', 'days_to_next_fomc', 'days_since_last_fomc']

_CALENDARS_URL = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
_HISTORICAL_URL = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm'

_MONTHS = {m: i + 1 for i, m in enumerate([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])}
# The Fed pages abbreviate months in cross-month headings ("Jan/Feb 31-1
# Meeting - 2017") and month cells; accept the standard abbreviations too.
_MONTHS.update({m[:3]: v for m, v in list(_MONTHS.items())})
_MONTHS['Sept'] = 9

# "January 28-29 Meeting - 2003" / "March 18 Meeting - 2003" /
# "April/May 30-1 Meeting - 2019" / "June 30-July 1 Meeting - 1998" (the two
# cross-month spellings the pages use). A parenthetical qualifier between the
# days and the word "Meeting" — "(unscheduled)", "(cancelled)" — or a missing
# "Meeting" ("(notation vote)", "Conference Call") makes the regex NOT match,
# which is exactly the scheduled-only filter described in the module docstring.
_HISTORICAL_HEADING_RE = re.compile(
    r'<h5[^>]*>\s*([A-Z][a-z]+(?:/[A-Z][a-z]+)?)\s+(\d{1,2})'
    r'(?:-(?:([A-Z][a-z]+)\s+)?(\d{1,2}))?\s+Meeting\s*-\s*(\d{4})\s*</h5>')

_YEAR_PANEL_RE = re.compile(r'(\d{4})\s+FOMC\s+Meetings')
_MEETING_CELL_RE = re.compile(
    r'fomc-meeting__(month|date)[^>]*>(.*?)</div>', re.DOTALL)
_TAG_RE = re.compile(r'<[^>]+>')


def _statement_day(months: str, d1: str, d2, year: int, month2=None):
    """Last day of the meeting range = the statement/decision day.
    'April/May 30-1' -> May 1; 'June 30' + month2='July' d2=1 -> Jul 1;
    'January 28-29' -> Jan 29; 'March 18' -> Mar 18."""
    parts = months.split('/')
    if not d2:   # re.findall yields '' (not None) for an unmatched group
        return pd.Timestamp(year=year, month=_MONTHS[parts[0]], day=int(d1))
    month_last = _MONTHS[month2] if month2 else _MONTHS[parts[-1]]
    return pd.Timestamp(year=year, month=month_last, day=int(d2))


def _parse_historical_page(html: str):
    return [_statement_day(m, d1, d2, int(y), month2=m2 or None)
            for m, d1, m2, d2, y in _HISTORICAL_HEADING_RE.findall(html)]


def _parse_calendars_page(html: str):
    """The official calendar page: year panels ('2026 FOMC Meetings') each with
    alternating fomc-meeting__month / fomc-meeting__date cells. Date cells with
    a qualifier — '(cancelled)', '(notation vote)', '(unscheduled)' — are the
    non-scheduled-meeting rows and are EXCLUDED (same filter as the historical
    headings), but their dates are returned separately so the statement-link
    cross-check can account for them. The '*' (SEP marker) is cosmetic.
    Returns (scheduled_dates, excluded_dates)."""
    dates, excluded = [], []
    panels = list(_YEAR_PANEL_RE.finditer(html))
    for i, m in enumerate(panels):
        year = int(m.group(1))
        chunk = html[m.end(): panels[i + 1].start() if i + 1 < len(panels) else len(html)]
        month_txt = None
        for kind, raw in _MEETING_CELL_RE.findall(chunk):
            text = _TAG_RE.sub(' ', raw).strip()
            if kind == 'month':
                month_txt = text.split()[0] if text else None
                continue
            dm = re.search(r'(\d{1,2})(?:\s*-\s*(\d{1,2}))?', text)
            if month_txt is None or not dm or month_txt.split('/')[0] not in _MONTHS:
                month_txt = None
                continue
            day = _statement_day(month_txt, dm.group(1), dm.group(2), year)
            (excluded if '(' in text else dates).append(day)
            month_txt = None
    return dates, excluded


def build_fomc_dates_csv(out_path=FOMC_DATES_CSV, base_dir='',
                         hist_years=range(1998, 2021), timeout=30):
    """Fetch + parse the official Fed pages and (re)write the reference CSV.
    Cross-checks the calendars page against its own statement-press-release
    links (monetaryYYYYMMDDa.htm) so a parser drift fails loudly, not silently."""
    import requests

    rows = []
    for year in hist_years:
        r = requests.get(_HISTORICAL_URL.format(year=year), timeout=timeout)
        got = _parse_historical_page(r.text) if r.ok else []
        print(f'fomchistorical{year}: HTTP {r.status_code}, {len(got)} scheduled meetings')
        rows += [(d, f'fomchistorical{year}') for d in got]

    r = requests.get(_CALENDARS_URL, timeout=timeout)
    r.raise_for_status()
    cal, cal_excluded = _parse_calendars_page(r.text)
    print(f'fomccalendars: HTTP {r.status_code}, {len(cal)} scheduled meetings '
          f'({min(cal).year}..{max(cal).year}); excluded non-scheduled rows: '
          f'{[str(d.date()) for d in cal_excluded]}')
    rows += [(d, 'fomccalendars') for d in cal]

    # Consistency check: every statement link on the calendars page must land
    # on a parsed row (scheduled OR deliberately excluded) — a parser drift
    # fails loudly instead of silently dropping meetings.
    linked = {pd.Timestamp(s) for s in re.findall(r'monetary(\d{8})a\.htm', r.text)}
    missing = sorted(linked - set(cal) - set(cal_excluded))
    assert not missing, f'calendar parser missed statement-linked dates: {missing}'

    out = (pd.DataFrame(rows, columns=['date', 'source'])
           .drop_duplicates(subset='date').sort_values('date').reset_index(drop=True))
    out['date'] = out['date'].dt.date
    path = os.path.join(base_dir, out_path) if base_dir else out_path
    out.to_csv(path, index=False)
    print(f'Saved {len(out)} FOMC statement days '
          f'({out["date"].iloc[0]} .. {out["date"].iloc[-1]}) -> {path}')
    return out


def load_fomc_dates(path=FOMC_DATES_CSV, base_dir='') -> pd.DatetimeIndex:
    p = os.path.join(base_dir, path) if base_dir else path
    df = pd.read_csv(p, parse_dates=['date'])
    return pd.DatetimeIndex(df['date']).normalize().sort_values().unique()


def add_fomc_features(df: pd.DataFrame, fomc_dates=None, base_dir='') -> pd.DataFrame:
    """Join the three calendar features onto a daily-indexed frame. Pure
    calendar geometry (searchsorted against the sorted statement days) — no
    rolling windows, so there is no warm-up and the join is exact on every row.
    Rows outside the calendar's span get NaN (callers must assert coverage
    rather than silently learn from a truncated calendar)."""
    dates = load_fomc_dates(base_dir=base_dir) if fomc_dates is None else \
        pd.DatetimeIndex(pd.to_datetime(list(fomc_dates))).normalize().sort_values().unique()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    days = idx.normalize().values
    arr = pd.DatetimeIndex(dates).values

    pos_next = np.searchsorted(arr, days, side='left')     # first fomc >= d
    pos_prev = np.searchsorted(arr, days, side='right') - 1  # last fomc <= d
    n = len(days)
    to_next = np.full(n, np.nan)
    since_last = np.full(n, np.nan)
    ok_n = pos_next < len(arr)
    to_next[ok_n] = (arr[pos_next[ok_n]] - days[ok_n]) / np.timedelta64(1, 'D')
    ok_p = pos_prev >= 0
    since_last[ok_p] = (days[ok_p] - arr[pos_prev[ok_p]]) / np.timedelta64(1, 'D')

    out = df.copy()
    out['is_fomc_day'] = np.isin(days, arr).astype(int)
    out['days_to_next_fomc'] = to_next
    out['days_since_last_fomc'] = since_last
    return out


if __name__ == '__main__':
    build_fomc_dates_csv()
