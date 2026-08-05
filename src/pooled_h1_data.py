"""
Pooled multi-instrument H1 data acquisition (research-only hypothesis family).

WHY THIS EXISTS
---------------
The single-instrument EURUSD H1 sample (~60k bars, 2016-12 -> present) sits
below the sample size at which sequential deep learning has been shown to work
on intraday data. The literature's stated remedy for a fixed-history single
instrument is to POOL several correlated instruments and train one shared
architecture across all of them, raising the effective sample size. This module
acquires the raw material for testing whether that remedy actually buys anything
here -- it does NOT itself train, score, or decide anything.

HARD BOUNDARY -- research only, production untouched
----------------------------------------------------
Nothing here writes to, imports side-effects from, or otherwise mutates any
production serving artifact. In particular it NEVER touches
``results/eurusd_h1.csv`` (that file feeds the LIVE H1 ensemble via
``src/h1_features.py`` and must not be rewritten). Pooled data is written to a
SEPARATE tree, ``results/pooled_h1/<SYMBOL>_h1.csv``. The protected set
(``models/``, ``_train_pipeline.py``, ``src/inference.py``, ``src/features.py``,
``src/paper_trading.py``, ``config.json``, ``results/eurusd_h1.csv``) is asserted
untouched by a unit test.

STEP 0 CONTRACT (this module)
-----------------------------
* Source is a live MT5 terminal ONLY. There is deliberately NO yfinance
  fallback for this program (matches the M15 harmonic program's provenance
  stance in ``fetch_m15_market_data`` -- mixing a second vendor's intraday
  history into a would-be MT5-native pooled experiment is not worth the
  provenance risk). An on-disk cache is an acceptable OFFLINE fallback (same
  "never hard-fail a reader" convention as every other fetch chain), but no
  other LIVE source is ever tried.
* Bars are pulled with the BAR API ``copy_rates_from_pos(sym, TIMEFRAME_H1,...)``
  -- never a tick-level API (``copy_ticks_*``).
* NO tick volume. The stored frame carries OHLC only; ``tick_volume`` is dropped
  at acquisition. (It is a broker tick count, not traded volume, and the project
  already excludes it from every model -- see ARCHITECTURE_DOCS §2.1. Keeping it
  out at the source removes any chance a pooled feature accidentally consumes it.)
* Broker symbol names may carry suffixes (``EURUSD.a``, ``EURUSD_raw``, ...).
  Each base is resolved against ``mt5.symbols_get()`` and the EXACT resolved
  string is reported. If ANY of the four cannot be resolved unambiguously, the
  acquisition STOPS and raises ``SymbolResolutionError`` rather than silently
  fetching three of four or guessing a suffix.
"""

import os

import pandas as pd

# The four correlated majors FETCHED for the effective-sample-size test. Three
# are XXX/USD; USDCHF is USD/XXX and is INVERTED to CHFUSD before pooling (see
# invert_to_chfusd) so all four share one consistent "USD in the denominator"
# convention -- otherwise the same event ("USD strengthens") would be a
# down-move in three series and an up-move in the fourth, feeding the pooled
# model contradictory labels for one phenomenon.
POOLED_INSTRUMENTS = ('EURUSD', 'GBPUSD', 'AUDUSD', 'USDCHF')

# The instruments actually POOLED (post quote-convention alignment): USDCHF is
# replaced by its inverse CHFUSD. This is the set every downstream step uses.
POOLED_PAIRS = ('EURUSD', 'GBPUSD', 'AUDUSD', 'CHFUSD')

POOLED_DIR = os.path.join('results', 'pooled_h1')

# Retired instruments live here rather than being deleted, so every completed
# program that consumed them stays reproducible (see load_pooled_h1).
RETIRED_DIR = os.path.join(POOLED_DIR, 'retired')

# ---------------------------------------------------------------------------
# CHF RETIREMENT (owner decision, 2026-08-04)
# ---------------------------------------------------------------------------
# USDCHF/CHFUSD are retired from ACTIVE use. This is an OWNER DECISION, not a
# data-driven withdrawal of any result -- see results/pooled_h1/retired/README.md
# and results/DATA_STATUS.md.
#
# The historical POOLED_INSTRUMENTS/POOLED_PAIRS tuples above are deliberately
# LEFT INTACT: they describe the pooled experiment (H_pool.1/H_pool.2) as it was
# actually run, and src/pooled_h1_model.py -- a completed, protected program --
# imports POOLED_PAIRS. Rewriting them would silently redefine a finished result.
# Anything fetched GOING FORWARD uses the ACTIVE lists instead.
RETIRED_INSTRUMENTS = {
    'USDCHF': ('42-day hole 2026-06-15 06:00 -> 2026-07-28 00:00 from an MT5 '
               'Market Watch partial history sync; retired by owner decision'),
    'CHFUSD': ('derived by inversion from USDCHF and inherits the identical hole'),
}
ACTIVE_INSTRUMENTS = tuple(s for s in POOLED_INSTRUMENTS if s not in RETIRED_INSTRUMENTS)
ACTIVE_PAIRS = tuple(s for s in POOLED_PAIRS if s not in RETIRED_INSTRUMENTS)

# OHLC only -- tick_volume is intentionally never stored (see module docstring).
OHLC_COLUMNS = ['open', 'high', 'low', 'close']

# One pip in the price units of a 5-digit major (EURUSD, GBPUSD, AUDUSD, and --
# after inversion -- CHFUSD, whose ~1.14 level shares the same 4th-decimal pip).
PIP_SIZE = 0.0001


class SymbolResolutionError(RuntimeError):
    """Raised when a requested base symbol cannot be resolved to exactly one
    concrete broker symbol -- STEP 0 stops rather than fetching a partial set."""


def resolve_symbol(base: str, available_names) -> str:
    """
    Resolve one canonical base (e.g. ``EURUSD``) to the exact broker symbol
    string to request from MT5, tolerating vendor suffixes.

    Resolution order (deterministic, never a silent guess):
      1. An EXACT case-sensitive match (``EURUSD`` == ``EURUSD``) wins outright.
      2. Otherwise, symbols that start with the base up to a single separator
         (``EURUSD.a``, ``EURUSD_raw``, ``EURUSD-ECN``) are considered. If there
         is exactly ONE such candidate it is used; if there are several, that is
         ambiguous and raises (we will not pick one arbitrarily).
      3. No candidate at all raises.

    ``available_names`` is the iterable of ``s.name`` from ``mt5.symbols_get()``
    (injected so this is unit-testable without a live terminal).
    """
    names = list(available_names)

    exact = [n for n in names if n == base]
    if exact:
        return exact[0]

    # Suffix candidates: base followed immediately by a non-alphanumeric
    # separator (so GBPUSD never accidentally matches a hypothetical GBPUSDX).
    def _is_suffix_variant(n: str) -> bool:
        if not n.upper().startswith(base.upper()):
            return False
        tail = n[len(base):]
        return tail == '' or not tail[0].isalnum()

    candidates = [n for n in names if _is_suffix_variant(n) and n != base]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise SymbolResolutionError(
            f"{base!r} is ambiguous: multiple broker symbols match -> {candidates}. "
            "Refusing to pick one arbitrarily."
        )
    raise SymbolResolutionError(
        f"{base!r} could not be resolved to any broker symbol. STEP 0 stops "
        "rather than fetch a partial pooled set."
    )


def _fetch_h1_ohlc_from_mt5(symbol: str, bars: int):
    """
    Pull ``bars`` H1 OHLC rows for one already-resolved broker ``symbol`` from a
    live MT5 terminal. Returns a UTC-indexed, ascending OHLC DataFrame (NO
    tick_volume) or None if no terminal is reachable / the symbol returned
    nothing. Mirrors ``live_data._fetch_h1_from_mt5`` (UTC localisation,
    ``copy_rates_from_pos``) but drops every non-OHLC column at the source.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None

    from .mt5_coverage import assert_coverage, sync_symbol

    try:
        if not mt5.initialize():
            return None
        # PREVENTION. This is the exact path that produced the 42-day hole in
        # USDCHF_h1.csv: the symbol was not in Market Watch, so only a partial
        # history block existed, and copy_rates_from_pos reached further back to
        # satisfy `bars` -- 70,000 bars and a correct-looking last timestamp
        # around six missing weeks. Select and force the pull first.
        sync_symbol(mt5, symbol)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
        mt5.shutdown()
    except Exception:
        return None

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    # Same UTC convention as the live H1 stream so any downstream daily/session
    # bucketing is timezone-unambiguous and directly comparable to
    # results/eurusd_h1.csv (which is also UTC).
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    # DETECTION. Raises CoverageError naming the missing range rather than
    # writing a holed CSV; never repairs by filling.
    return assert_coverage(df[OHLC_COLUMNS], 'H1', label='MT5 %s H1' % symbol)


def _resolve_all_or_stop(instruments):
    """Resolve every requested base against the live terminal's symbol list.
    Returns an ordered {base: resolved_symbol} map, or raises
    SymbolResolutionError (STOP) if a live terminal is unreachable or ANY base
    fails to resolve -- never returns a partial map."""
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise SymbolResolutionError("MetaTrader5 not importable; cannot resolve symbols.") from e

    if not mt5.initialize():
        raise SymbolResolutionError("No live MT5 terminal reachable; cannot resolve symbols.")
    try:
        available = [s.name for s in (mt5.symbols_get() or [])]
    finally:
        mt5.shutdown()

    resolved = {}
    for base in instruments:
        resolved[base] = resolve_symbol(base, available)  # raises -> STOP
    return resolved


def fetch_pooled_h1(instruments=ACTIVE_INSTRUMENTS, bars: int = 70000,
                    out_dir: str = POOLED_DIR, write: bool = True):
    """
    STEP 0 acquisition: resolve and fetch H1 OHLC for every pooled instrument
    from a live MT5 terminal, writing each to ``<out_dir>/<BASE>_h1.csv``.

    Resolution is ALL-OR-NOTHING: every base is resolved up front and, if any
    fails, ``SymbolResolutionError`` is raised BEFORE a single bar is fetched or
    written -- there is never a half-acquired pooled set on disk from a bad run.

    ``bars`` defaults to 70,000 so each instrument comfortably covers the same
    ~60k-bar span as the live EURUSD H1 stream (2016-12 -> present); the shared
    training window is intersected downstream, not here.

    Returns ``(frames, resolved, meta)``:
      * ``frames``   -- {base: UTC-indexed OHLC DataFrame}
      * ``resolved`` -- {base: exact broker symbol string used}
      * ``meta``     -- {base: {'symbol','rows','start','end'}} for reporting
    Never writes to ``results/eurusd_h1.csv`` (a distinct ``out_dir`` tree).
    """
    resolved = _resolve_all_or_stop(instruments)

    frames, meta = {}, {}
    for base, symbol in resolved.items():
        df = _fetch_h1_ohlc_from_mt5(symbol, bars)
        if df is None or len(df) == 0:
            raise SymbolResolutionError(
                f"{base!r} resolved to {symbol!r} but returned no H1 bars. STEP 0 stops."
            )
        frames[base] = df
        meta[base] = {
            'symbol': symbol,
            'rows': int(len(df)),
            'start': df.index.min().isoformat(),
            'end': df.index.max().isoformat(),
        }
        if write:
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, f"{base}_h1.csv"))

    return frames, resolved, meta


def invert_to_chfusd(usdchf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Invert a USD/CHF OHLC frame into CHF/USD (quote-convention alignment,
    pre-registered). Every price becomes its reciprocal, and CRITICALLY the
    high/low SWAP: the reciprocal is monotonically DECREASING, so the bar's
    highest USDCHF price maps to the LOWEST CHFUSD price and vice-versa.

        close_new = 1 / close_old
        open_new  = 1 / open_old
        high_new  = 1 / low_old      <-- swap
        low_new   = 1 / high_old     <-- swap

    Forgetting the swap is a classic silent bug: 1/high_old < 1/low_old would
    otherwise produce bars with high < low. The output is guaranteed
    high >= low on every bar (since low_old <= high_old => 1/high_old <=
    1/low_old). tick_volume is not present (OHLC-only frames) and is never
    reintroduced.
    """
    inv = pd.DataFrame(index=usdchf_df.index.copy())
    inv['open'] = 1.0 / usdchf_df['open']
    inv['high'] = 1.0 / usdchf_df['low']    # swap
    inv['low'] = 1.0 / usdchf_df['high']    # swap
    inv['close'] = 1.0 / usdchf_df['close']
    return inv[OHLC_COLUMNS]


def build_pooled_pairs(out_dir: str = POOLED_DIR, write_chfusd: bool = True):
    """
    Load the four fetched majors and return the POOLED set (USDCHF replaced by
    its CHFUSD inverse), reading from the on-disk acquisition CSVs. The raw
    USDCHF cache is left exactly as fetched; the inverted series is (optionally)
    written to ``<out_dir>/CHFUSD_h1.csv`` so both conventions stay inspectable.

    Returns {pair: UTC-indexed OHLC DataFrame} keyed by POOLED_PAIRS.
    """
    raw = load_pooled_h1(instruments=POOLED_INSTRUMENTS, out_dir=out_dir)
    missing = [b for b in POOLED_INSTRUMENTS if b not in raw]
    if missing:
        raise FileNotFoundError(f"Missing fetched majors {missing} under {out_dir!r}.")

    chfusd = invert_to_chfusd(raw['USDCHF'])
    if write_chfusd:
        os.makedirs(out_dir, exist_ok=True)
        chfusd.to_csv(os.path.join(out_dir, 'CHFUSD_h1.csv'))

    return {
        'EURUSD': raw['EURUSD'], 'GBPUSD': raw['GBPUSD'],
        'AUDUSD': raw['AUDUSD'], 'CHFUSD': chfusd,
    }


def load_pooled_h1(instruments=POOLED_INSTRUMENTS, out_dir: str = POOLED_DIR):
    """
    OFFLINE reader for previously-acquired pooled H1 CSVs (the cache-fallback
    convention). Returns {base: UTC-indexed OHLC DataFrame} for every base whose
    ``<out_dir>/<BASE>_h1.csv`` exists; raises FileNotFoundError if none do (so a
    downstream step never silently trains on an empty pool).
    """
    frames = {}
    for base in instruments:
        path = os.path.join(out_dir, f"{base}_h1.csv")
        if not os.path.exists(path):
            # Retired instruments are MOVED, never deleted, so a completed
            # program that consumed one still reproduces byte-for-byte.
            retired = os.path.join(out_dir, 'retired', f"{base}_h1.csv")
            if not os.path.exists(retired):
                continue
            path = retired
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
        frames[base] = df[OHLC_COLUMNS]
    if not frames:
        raise FileNotFoundError(
            f"No pooled H1 CSVs found under {out_dir!r}; run fetch_pooled_h1() first."
        )
    return frames


if __name__ == '__main__':
    frames, resolved, meta = fetch_pooled_h1()
    print("Resolved broker symbols:")
    for base, sym in resolved.items():
        print(f"  {base:8s} -> {sym}")
    print("\nAcquired H1 OHLC (tick_volume intentionally dropped):")
    for base, m in meta.items():
        print(f"  {base:8s} {m['rows']:>7d} bars  {m['start']} -> {m['end']}")
