"""Synthetic H1 bars for the fx-fibonacci tests: real broker-clock weeks, random prices."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.forecast_eval.m15_data import expected_open_label


def synthetic_weeks(first_sunday: date, n_weeks: int, shift_h: int = 0, seed: int = 0,
                    vol: float = 0.001) -> pd.DataFrame:
    """H1 OHLC whose weekly opens sit on the expected Berlin label hour (+ shift_h);
    the week's last bar is labelled Friday 22:00."""
    stamps = []
    for w in range(n_weeks):
        sun = first_sunday + timedelta(weeks=w)
        t = pd.Timestamp(sun, tz='UTC') + pd.Timedelta(hours=expected_open_label(sun) + shift_h)
        end = pd.Timestamp(sun + timedelta(days=5), tz='UTC') + pd.Timedelta(hours=23)
        while t < end:
            stamps.append(t)
            t += pd.Timedelta(hours=1)
    idx = pd.DatetimeIndex(stamps, name='time')
    rng = np.random.default_rng(seed)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, vol, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, vol / 2, len(idx)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, vol / 2, len(idx)))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=idx)
