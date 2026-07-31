"""Kronos (NeoQuasar/Kronos-base) evaluation harness and observational server.

EXTERNAL FOUNDATION MODEL, ZERO-SHOT. Not this project's model, not validated by
this project's methodology, and not part of any pre-registered hypothesis family.

Established before any result was read (see the Phase 0 report):
  * training cutoff June 2024; the only honest evaluation window is 2024-07-01
    onward, and the published weights (uploaded 2025-06-30) have not changed
  * the no-volume path is supported and trained for; MT5 tick_volume is NEVER
    substituted for volume
  * the tokenizer normalises PER WINDOW and reconstructs EURUSD at MAE/range
    0.206 (H1) against 0.411 for an in-corpus control, but occupies only 6.5% of
    level-1 codes against 18.1% for that control -- reduced resolution, carried
    as a caveat on every result
  * elapsed time between bars is NOT encoded, so the ~48h weekend gap is a single
    step to the model; `spans_weekend_gap` is reported on every prediction

Observational only. Nothing here places an order, sizes a position or sets a
stop.
"""

from .vendor import (KRONOS_COMMIT, KronosPinMismatch, KronosUnavailable,
                     assert_pinned, import_upstream, upstream_path)

__all__ = ['KRONOS_COMMIT', 'KronosPinMismatch', 'KronosUnavailable',
           'assert_pinned', 'import_upstream', 'upstream_path']
