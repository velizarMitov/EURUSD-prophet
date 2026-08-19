"""
INPUT-DATA PROVENANCE GUARD.

WHY THIS FILE EXISTS
--------------------
The repository already byte-pins CODE and MODEL artifacts
(tests/fixtures/*_protected_sha256.json). It pinned no INPUT DATA at all, and
that gap hid a real defect: commit c638f8d, titled "Add unit tests for
Yildirim/Toroslu/Fiore (2021) replication study", also rewrote
results/eurusd_h1.csv (60,000 -> 60,056 rows). Commit f2645a0, titled "Refactor
code structure for improved readability and maintainability", rewrote the same
file AND 30 model artifacts. A data rewrite that rides inside a commit named for
something else is invisible in review, and every downstream number silently
changes underneath the committed results.

TWO CLASSES OF INPUT, TWO DIFFERENT GUARANTEES
----------------------------------------------
Byte-pinning every input CSV would be wrong, and the repository already worked
out why (see the PINNING POLICY docstring in tests/test_h1_production.py):

  FROZEN  research inputs that no running code rewrites -- results/pooled_h1/*,
          results/eurusd_m15.csv, results/eurusd_features.csv. Each has exactly
          one or a handful of revisions in git history, changing only when a
          human deliberately regenerated it. These are byte-pinned. A moved
          digest is a real event and fails loudly.

  ROLLING results/eurusd_h1.csv is an OPERATIONAL CACHE.
          src/live_data.py::fetch_h1_market_data writes cache_path on every
          successful MT5/yfinance pull, so the file legitimately changes
          whenever anyone runs a prediction or a retrain -- 25 revisions in git
          history, the MT5 window sliding forward each time. Byte-pinning it
          would assert only that nobody had run the app: a permanently red
          guard, which cannot detect the NEXT change. It is given a RECORDED
          PROVENANCE STAMP plus structural invariants instead, and DATA.md must
          agree with the file on disk -- so the documentation can no longer
          drift silently, which is the specific way this defect stayed
          invisible.

WHAT A FAILURE HERE MEANS
-------------------------
  * a frozen digest moved      -> an input was rewritten. Find out which commit
                                  and why BEFORE re-baselining. If the rewrite
                                  was intended, re-baseline in its own commit
                                  that says so.
  * a rolling stamp went stale -> the cache was refreshed. That is allowed, but
                                  the stamp and DATA.md must be re-stamped in a
                                  commit that declares it, so downstream results
                                  can be dated against the input they used.
"""
import hashlib
import json
import os
import re

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, 'tests', 'fixtures', 'input_data_protected_sha256.json')
DATA_MD = os.path.join(REPO, 'DATA.md')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope='module')
def manifest():
    if not os.path.exists(FIXTURE):
        pytest.skip('input-data fixture missing')
    with open(FIXTURE, encoding='utf-8') as fh:
        return json.load(fh)


# -- 1. the frozen inputs are byte-identical ----------------------------------

def test_frozen_inputs_are_sha256_identical(manifest):
    """The load-bearing research inputs must not move under the committed
    results. Three of the four H1 families (h1_direction, h1_multiday,
    pooled_h1) read results/pooled_h1/EURUSD_h1.csv, NOT the rolling cache --
    so pinning this tree is what actually protects their numbers."""
    frozen = manifest['frozen']
    assert len(frozen) >= 10, 'the frozen set shrank -- an input lost its guard'

    moved = []
    for rel, meta in frozen.items():
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f'protected input vanished: {rel}'
        got = _sha256(path)
        if got != meta['sha256']:
            moved.append(f'{rel}\n    expected {meta["sha256"]}\n    got      {got}')
    assert not moved, (
        'PROTECTED INPUT DATA MODIFIED:\n  ' + '\n  '.join(moved)
        + '\n\nAn input CSV changed. Identify the commit that rewrote it and why '
          'before re-baselining this fixture. Do not re-baseline as a side effect '
          'of unrelated work -- that is exactly how c638f8d and f2645a0 hid theirs.')


def test_frozen_pooled_inputs_still_carry_the_rows_the_h1_families_scored_on(manifest):
    """Row counts and spans, asserted independently of the digest so a failure
    says WHAT changed, not merely that something did."""
    for rel, meta in manifest['frozen'].items():
        if '/pooled_h1/' not in rel:
            continue
        df = pd.read_csv(os.path.join(REPO, rel))
        tcol = df.columns[0]
        assert len(df) == meta['n_rows'], \
            f'{rel}: {len(df)} rows, expected {meta["n_rows"]}'
        assert str(df[tcol].iloc[0]) == meta['first_timestamp'], f'{rel}: first row moved'
        assert str(df[tcol].iloc[-1]) == meta['last_timestamp'], f'{rel}: last row moved'


def test_the_three_replication_instruments_share_one_window(manifest):
    """EURUSD/GBPUSD/AUDUSD were pulled as one pooled snapshot; a divergent
    window would mean one of them was refetched on its own."""
    spans = {rel: (m['first_timestamp'], m['last_timestamp'])
             for rel, m in manifest['frozen'].items()
             if re.search(r'/pooled_h1/(EUR|GBP|AUD)USD_h1\.csv$', rel)}
    assert len(spans) == 3, f'expected 3 pooled instruments, found {sorted(spans)}'
    assert len(set(spans.values())) == 1, \
        f'pooled instruments no longer share a window: {spans}'


# -- 2. the rolling cache: stamped, not pinned --------------------------------

def test_rolling_cache_is_deliberately_not_byte_pinned(manifest):
    """The policy itself, asserted -- so a future change that byte-pins the
    operational cache trips here and has to justify itself rather than quietly
    creating a permanently red guard."""
    rolling = manifest['rolling']
    assert 'results/eurusd_h1.csv' in rolling
    assert 'results/eurusd_h1.csv' not in manifest['frozen'], (
        'results/eurusd_h1.csv was moved into the frozen set. It is rewritten by '
        'src/live_data.py::fetch_h1_market_data on every successful pull; pinning '
        'it asserts only that nobody ran the app. See this module docstring.')
    for meta in rolling.values():
        assert 'sha256_at_stamp' in meta and 'stamped_commit' in meta


def test_rolling_cache_structural_invariants_hold(manifest):
    """What must be true of the cache no matter how far the MT5 window slid:
    schema, UTC, strictly increasing unique timestamps, sane OHLC."""
    for rel in manifest['rolling']:
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f'{rel} missing'
        df = pd.read_csv(path)
        assert list(df.columns) == ['time', 'open', 'high', 'low', 'close', 'tick_volume'], \
            f'{rel}: schema changed -> {list(df.columns)}'
        ts = pd.to_datetime(df['time'], utc=True)
        assert ts.is_monotonic_increasing, f'{rel}: timestamps not sorted'
        assert ts.is_unique, f'{rel}: duplicate timestamps'
        assert df[['open', 'high', 'low', 'close']].notna().all().all(), \
            f'{rel}: NaN in OHLC'
        assert (df['high'] >= df['low']).all(), f'{rel}: high < low'
        assert len(df) > 50_000, f'{rel}: only {len(df)} rows -- the cache was truncated'


# -- 3. DATA.md may not drift away from the files it documents ----------------

def _data_md_rows():
    """Parse the '1. What ships with the repository' table into
    {relative path: documented row count}."""
    with open(DATA_MD, encoding='utf-8') as fh:
        text = fh.read()
    out = {}
    for m in re.finditer(r'^\|\s*`([^`]+)`\s*\|\s*([0-9,]+)\s*\|', text, re.M):
        out[m.group(1)] = int(m.group(2).replace(',', ''))
    return out


def test_data_md_row_counts_match_the_files_on_disk(manifest):
    """The defect that made this invisible: DATA.md claimed 60,032 rows for
    results/eurusd_h1.csv long after the file held 60,056. Documentation that
    cannot go stale silently is the actual guard for a file that is allowed to
    move."""
    documented = _data_md_rows()
    assert documented, 'could not parse the DATA.md inventory table'

    wrong = []
    for rel, claimed in documented.items():
        path = os.path.join(REPO, rel)
        if not rel.endswith('.csv') or not os.path.isfile(path):
            continue
        actual = len(pd.read_csv(path))
        if actual != claimed:
            wrong.append(f'{rel}: DATA.md says {claimed:,}, file has {actual:,}')
    assert not wrong, (
        'DATA.md is out of date:\n  ' + '\n  '.join(wrong)
        + '\n\nUpdate the inventory table in DATA.md to match the committed files.')


def test_data_md_documents_every_guarded_input(manifest):
    """Every file this guard protects has to appear in DATA.md, so the inventory
    and the guard cannot drift apart."""
    with open(DATA_MD, encoding='utf-8') as fh:
        text = fh.read()
    missing = [rel for rel in list(manifest['frozen']) + list(manifest['rolling'])
               if rel not in text and not rel.endswith('pooled_h1_run_log.csv')]
    assert not missing, f'guarded inputs absent from DATA.md: {missing}'
