"""Labels, instants and Sofia display (design D4, spec gold-analytics/market-data
"Labels follow the repo clock and are shown in Sofia time").

A label is Europe/Berlin wall time stored as UTC as-is. Turning it into a real
instant localises that wall time in Berlin with daylight saving; the fall-back
hour (02:00-03:00 on the last Sunday of October) occurs twice, and the first
occurrence in a series is read as summer time, the second as winter time.
Every time shown to the owner is Europe/Sofia, which is Berlin + 1 h all year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_ZONE = 'Europe/Berlin'
DISPLAY_ZONE = 'Europe/Sofia'


def _as_index(labels) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(labels, utc=True) if not isinstance(labels, pd.DatetimeIndex) else labels)
    return idx.tz_convert('UTC') if idx.tz is not None else idx.tz_localize('UTC')


def ns(times) -> np.ndarray:
    """Label times as int64 nanoseconds, for fast searchsorted (tz-aware columns
    otherwise become object arrays)."""
    return _as_index(times).as_unit('ns').asi8


def label_to_instant(labels, zone: str = LABEL_ZONE) -> pd.DatetimeIndex:
    """Label wall times -> tz-aware instants in `zone`."""
    wall = _as_index(labels).tz_localize(None)
    # Order of appearance resolves the doubled fall-back hour: first = DST.
    seen = pd.Series(0, index=range(len(wall))).groupby(np.asarray(wall)).cumcount().to_numpy()
    return wall.tz_localize(zone, ambiguous=seen == 0, nonexistent='shift_forward')


def instant_to_label(instants, zone: str = LABEL_ZONE) -> pd.DatetimeIndex:
    """Tz-aware instants -> labels (Berlin wall time stored as UTC)."""
    idx = pd.DatetimeIndex(instants)
    return idx.tz_convert(zone).tz_localize(None).tz_localize('UTC')


def to_display(labels, zone: str = DISPLAY_ZONE) -> pd.DatetimeIndex:
    """Label wall times -> the owner's wall clock."""
    return label_to_instant(labels).tz_convert(zone)


def label_scalar_to_display(label, zone: str = DISPLAY_ZONE) -> pd.Timestamp:
    return to_display([label], zone)[0]


def display_hhmm(labels, zone: str = DISPLAY_ZONE) -> list[str]:
    return [t.strftime('%H:%M') for t in to_display(labels, zone)]


def sofia_iso(label) -> str | None:
    """An ISO string in Sofia time for one label, or None."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    return label_scalar_to_display(label).isoformat()


def display_minute_of_day(labels, zone: str = DISPLAY_ZONE) -> np.ndarray:
    """Minutes after the owner's midnight for each label."""
    d = to_display(labels, zone)
    return (d.hour * 60 + d.minute).to_numpy()


def display_day(labels, zone: str = DISPLAY_ZONE) -> pd.Index:
    """The owner's calendar date for each label."""
    return pd.Index(to_display(labels, zone).date)
