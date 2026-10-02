"""Challenger models, one module per model type (design D1). Common interface in base.py.

make(name, config) builds a challenger from its study-record name.
"""

from __future__ import annotations


def make(name: str, config: dict, price: str | None = None):
    """`price` ('mid' | 'bid') applies to the M15 session challengers only; it is
    an instance attribute rather than configuration, so the bid-versus-mid parity
    check runs one declared configuration on both definitions (design D17)."""
    from .daily import DailyGBM, DailyLSTM
    from .h1 import H1DailyEnsemble, H1GBM, TILSTM
    from .kronos import Kronos
    from .m15 import M15SessionGBM, M15SessionLSTM
    from .volatility import VolEnsemble
    if name.startswith('daily_gbm'):
        return DailyGBM(config, name=name)
    if name.startswith('daily_lstm'):
        return DailyLSTM(config, name=name)
    m15 = {'m15_session_gbm': M15SessionGBM, 'm15_session_lstm': M15SessionLSTM}
    if name in m15:
        return m15[name](config, **({'price': price} if price else {}))
    table = {'h1_daily_ensemble': H1DailyEnsemble, 'ti_lstm': TILSTM, 'h1_gbm': H1GBM,
             'vol_ensemble': VolEnsemble}
    if name in table:
        return table[name](config)
    if name == 'kronos_direction':
        return Kronos(config, kind='direction')
    if name == 'kronos_volatility':
        return Kronos(config, kind='volatility')
    raise KeyError(f'unknown challenger {name!r}')
