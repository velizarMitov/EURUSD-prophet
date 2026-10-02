"""forecast_eval -- horizon study and forward evaluation (standalone research system).

A NEW system, separate from production. It trains its own challenger models,
imports the project's feature code read-only, and never writes under models/,
never touches serving, and never sends an order. See
openspec/changes/horizon-study-forward-evaluation/ for the specs and design.

Layout (design D1):
  splits      purged walk-forward, CPCV, label uniqueness
  targets     horizon targets and cost breakeven
  costs       spread / round-trip / swap cost model
  benchmarks  random-walk nulls
  stats       Pesaran-Timmermann, Clark-West, DM-HLN, block bootstrap, FWER
  overfit     PBO (CSCV), Deflated Sharpe, append-only trial log
  features    read-only adapters over the project's feature builders
  challengers one module per model type
  study       the horizon study (descriptive output only)
  mt5_reader  read-only MT5 access
  forward_logger, refit, prereg   the forward arbiter
"""
