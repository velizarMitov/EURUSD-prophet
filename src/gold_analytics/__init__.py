"""gold_analytics -- the read-only, descriptive gold market page /gold-analytics
(openspec change gold-market-analytics, specs gold-analytics/market-data and
gold-analytics/dashboard).

How gold behaved: price in eight timeframes, sessions in their own zones, time
of day, the real bid-ask spread from ticks, news hours, volatility, history and
time benchmarks (TWAP, arrival). DESCRIPTIVE ONLY: no forecast, no signal, no
verdict, and no text that says buy or sell.

ISOLATION CONTRACT
  * A separate package, not src/gold: gold's contract uses spread nowhere and
    /gold shows none, while this page shows the spread by design. From src/gold
    it imports only src.gold.data.resolve_gold_symbol.
  * MT5 is read only: the live tick, bars and ticks in a range. No order,
    position or account call, and no order, broker, sizing or stop-loss code.
  * No volume of any kind. tick_volume and real_volume are dropped at the MT5
    boundary; tick counts exist only to judge spread coverage and are never
    shown as volume.
  * Runs no model. Writes only its own cache under the gitignored
    research_models/gold/analytics/. Never writes under models/ or results/.

Layout:
  data        MT5 source with cache fallback (stale / as_of)
  clock       Berlin label <-> instant, Sofia display
  sessions    session intervals per day, session statistics
  refresh     the per-day tick pull into the 5-minute spread cache
  events      NFP / CPI (FRED) and FOMC release instants, news-hour windows
  intraday    time-of-day statistics
  volatility  ATR, realised volatility, regime
  history     periods, drawdown, histogram
  benchmarks  TWAP and arrival
  service     the section payloads behind /api/gold-analytics/{section}
"""

from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analytics_config.json')


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def repo_path(rel: str, repo: str = REPO) -> str:
    """Absolute path for a repo-relative config path (absolute paths pass through)."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(repo, *rel.split('/'))
