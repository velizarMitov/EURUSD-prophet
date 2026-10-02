"""Challenger models, one module per model type (design D1). Common interface in base.py.

make(name, config) builds a challenger from its study-record name.
"""

from __future__ import annotations


def make(name: str, config: dict):
    from .daily import DailyGBM, DailyLSTM
    from .h1 import H1DailyEnsemble, H1GBM, TILSTM
    from .kronos import Kronos
    from .volatility import VolEnsemble
    if name.startswith('daily_gbm'):
        return DailyGBM(config, name=name)
    if name.startswith('daily_lstm'):
        return DailyLSTM(config, name=name)
    table = {'h1_daily_ensemble': H1DailyEnsemble, 'ti_lstm': TILSTM, 'h1_gbm': H1GBM,
             'vol_ensemble': VolEnsemble}
    if name in table:
        return table[name](config)
    if name == 'kronos_direction':
        return Kronos(config, kind='direction')
    if name == 'kronos_volatility':
        return Kronos(config, kind='volatility')
    raise KeyError(f'unknown challenger {name!r}')
