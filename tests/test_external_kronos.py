"""
Tests for the Kronos external-model harness (src/external/kronos/).

The point of these is the DISCIPLINE, not the arithmetic: the commit pin must
BITE, the generation parameters must be frozen, the clean-window boundary must be
enforced, degradation must never touch an existing family, and no capability that
turns a forecast into an action may exist. Every guard is shown to bite, not
merely to pass.
"""

import ast
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA_FIXTURE = os.path.join(REPO, 'tests', 'fixtures', 'kronos_protected_sha256.json')

from src.external.kronos import loader, serving, vendor
from src.external.kronos.predict import (context_has_weekend_gap, p_up_from_paths)


# ── 1. the protected set ──────────────────────────────────────────────────────

def test_protected_set_is_sha256_identical():
    """models/, _train_pipeline.py, src/features.py, src/volatility.py,
    src/paper_trading.py, src/h1_direction_model.py, src/pooled_h1_model.py,
    config.json and EVERY hypothesis log.

    src/inference.py, src/live_data.py, api.py and static/index.html are
    deliberately NOT pinned here: the brief permits ADDITIVE modification of
    those four, so pinning them would assert the opposite of what was asked.
    Their invariant is asserted in test_only_additive_changes_to_the_four below.
    """
    with open(SHA_FIXTURE) as fh:
        expected = json.load(fh)
    assert len(expected) > 40
    for banned in ('src/inference.py', 'src/live_data.py', 'api.py', 'static/index.html'):
        assert banned not in expected, f'{banned} is additively modifiable, not pinned'
    for rel, digest in expected.items():
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f'protected file vanished: {rel}'
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        assert got == digest, f'PROTECTED FILE MODIFIED: {rel}'


# Last commit BEFORE any Kronos work (verified: no src/external in its tree).
# The additive check must diff against THIS, not HEAD: once the Kronos work is
# committed, a working-tree-vs-HEAD diff is empty and the test passes vacuously,
# silently ceasing to guard the thing it exists to guard.
PRE_KRONOS_REF = '6319df2'
ADDITIVE_FILES = ['src/inference.py', 'api.py', 'static/index.html',
                  'pyproject.toml', '.gitignore']

# src/live_data.py WAS on the additive list and is deliberately no longer.
# The later MT5 data-integrity program (2026-08-04, results/DATA_STATUS.md) was
# separately authorised to modify its fetch paths, and installing the coverage
# guard replaced three bare `return df[[...]]` statements with a two-line
# assign-then-`assert_coverage(...)` -- 3 deletions, in three functions. Those
# are the ONLY deletions permitted here, and the count is asserted exactly so
# the exemption cannot quietly widen.
#
# api.py joined for the same reason on 2026-08-08: the retrain-observability
# program was separately authorised to modify its logging and status paths after
# the 2026-08-07 run went silent for the last half of its work and was watched
# for four hours (child stdout was block-buffered and never flushed; nothing
# guaranteed a final log line; the run's identity lived only in an in-memory
# Popen handle). Rewriting start_retrain/retrain_status bodies is not expressible
# additively -- leaving the old bodies in place would mean two handlers on one
# route. 22 deletions: the old _retrain dict line and those two function bodies.
# Everything else in that program is additive, and the count is asserted exactly.
LATER_PROGRAM_FILES = {'src/live_data.py': 3, 'api.py': 22}


def test_only_additive_changes_outside_the_new_package():
    """Every file the Kronos program touched outside src/external/kronos/ and
    results/external_kronos/ must GAIN lines and lose none."""
    probe = subprocess.run(['git', '-C', REPO, 'cat-file', '-e', PRE_KRONOS_REF + '^{commit}'],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f'baseline commit {PRE_KRONOS_REF} not in this history')
    tree = subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', PRE_KRONOS_REF],
                          capture_output=True, text=True).stdout
    assert 'src/external' not in tree, f'{PRE_KRONOS_REF} is not a pre-Kronos baseline'

    out = subprocess.run(['git', '-C', REPO, 'diff', '--numstat', PRE_KRONOS_REF, '--']
                         + ADDITIVE_FILES + sorted(LATER_PROGRAM_FILES),
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip('git unavailable')
    seen = {}
    for line in out.stdout.strip().splitlines():
        added, removed, path = line.split('\t')
        if removed == '-':
            continue                                    # binary
        path = path.replace('\\', '/')
        seen[path] = (added, removed)
        allowed = LATER_PROGRAM_FILES.get(path, 0)
        assert int(removed) == allowed, (
            f'{path} DELETED {removed} lines; additive only'
            if not allowed else
            f'{path} DELETED {removed} lines but only {allowed} are accounted for '
            f'by the MT5 coverage guard -- a later program has widened its footprint')
    # the guard must actually be looking at something
    assert 'src/inference.py' in seen and int(seen['src/inference.py'][0]) > 0, \
        'additive check found no changes at all -- wrong baseline?'


def test_no_hypothesis_log_was_created_or_touched():
    """This is an external model, not a pre-registered family of ours."""
    with open(SHA_FIXTURE) as fh:
        expected = json.load(fh)
    logs = [k for k in expected if k.endswith('hypothesis_log.csv')]
    assert len(logs) >= 10, 'fixture should pin every existing hypothesis log'
    assert not os.path.exists(os.path.join(REPO, 'results', 'kronos_hypothesis_log.csv'))
    assert not os.path.exists(
        os.path.join(REPO, 'results', 'external_kronos', 'kronos_hypothesis_log.csv'))


# ── 2. the commit pin must BITE ───────────────────────────────────────────────

def test_pin_mismatch_raises_and_is_not_recoverable(tmp_path):
    """A wrong revision is a correctness failure, not a missing extra: it must
    raise KronosPinMismatch rather than degrade to unavailable."""
    fake = tmp_path / 'upstream'
    (fake / 'model').mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(fake)], check=True)
    (fake / 'model' / '__init__.py').write_text('')
    subprocess.run(['git', '-C', str(fake), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(fake), '-c', 'user.email=t@t', '-c', 'user.name=t',
                    'commit', '-qm', 'x'], check=True)
    with pytest.raises(vendor.KronosPinMismatch):
        vendor.assert_pinned(str(fake))


def test_missing_upstream_degrades_rather_than_raising_pin_error(tmp_path):
    with pytest.raises(vendor.KronosUnavailable):
        vendor.assert_pinned(str(tmp_path / 'nope'))


def test_pinned_commit_is_recorded_and_matches_if_present():
    assert vendor.KRONOS_COMMIT == '67b630e67f6a18c9e9be918d9b4337c960db1e9a'
    assert len(vendor.KRONOS_COMMIT) == 40
    path = vendor.upstream_path()
    if os.path.isdir(os.path.join(path, 'model')):
        assert vendor.assert_pinned(path) == vendor.KRONOS_COMMIT
    assert vendor.KRONOS_COMMIT[:10] in loader.model_version()


# ── 3. generation parameters are FROZEN ───────────────────────────────────────

def test_generation_parameters_are_frozen():
    """Declared in Phase A.4 before any accuracy was seen. With a sampler,
    quietly sweeping these against the evaluation data is unusually easy, and it
    is the same data-snooping the project's methodology exists to prevent."""
    assert loader.T == 1.0
    assert loader.TOP_P == 0.9
    # pred_len moved 1 -> 24 with the direction->volatility retirement: the
    # authors never use 1, and at 1 the sampled endpoints collapse onto ~7.6
    # distinct values out of 30. 24 is the demo's horizon and the horizon the
    # volatility numbers were measured at. See loader's module docstring.
    assert loader.PRED_LEN == 24
    assert loader.DEFAULT_MODEL == 'mini'
    assert loader.SAMPLE_COUNT == 30
    assert loader.CONTEXT_BARS == 512
    # sample_count was FIXED BY A.2, not chosen to make a number look better.
    assert loader.MC_NOISE_ESTIMATE == pytest.approx(0.0817)


def test_clean_window_boundary_is_declared_and_after_the_cutoff():
    assert loader.TRAINING_CUTOFF == '2024-06'
    assert loader.CLEAN_WINDOW_START == '2024-07-01'
    assert pd.Timestamp(loader.CLEAN_WINDOW_START) > pd.Timestamp('2024-06-30')


def test_evaluation_never_indexed_before_the_clean_window():
    """The generated clean-window file must not contain a single row whose
    CONTEXT could have started before 2024-07-01."""
    p = os.path.join(REPO, 'results', 'external_kronos', 'kronos_clean_window_pup.csv')
    if not os.path.exists(p):
        pytest.skip('clean-window generation not present')
    df = pd.read_csv(p, parse_dates=['forecast_time', 'last_context_time'])
    start = pd.Timestamp(loader.CLEAN_WINDOW_START, tz='UTC')
    assert df['forecast_time'].min() >= start
    h1 = pd.read_csv(os.path.join(REPO, 'results/eurusd_h1.csv'), parse_dates=['time'])
    h1 = h1.sort_values('time').reset_index(drop=True)
    earliest_ctx = h1['time'].iloc[int(df['idx'].min()) - loader.CONTEXT_BARS]
    assert earliest_ctx >= start, 'a context window began before the clean period'


# ── 4. the forecast object ────────────────────────────────────────────────────

def test_p_up_is_the_share_of_paths_above_the_last_close():
    paths = np.array([[1.10], [1.20], [1.30], [0.90], [0.95]])
    obj = p_up_from_paths(paths, last_close=1.00)
    assert obj['p_up'] == pytest.approx(3 / 5)
    assert obj['n_paths'] == 5
    assert obj['gen_distinct'] == 5
    # strictly above: a path landing exactly on the last close is not an up
    assert p_up_from_paths(np.array([[1.0], [1.0]]), 1.0)['p_up'] == 0.0
    # and it reads the FINAL element of a multi-step path, not the first
    multi = np.array([[1.5, 0.9], [0.5, 1.1]])
    assert p_up_from_paths(multi, 1.0)['p_up'] == pytest.approx(0.5)


def test_weekend_gap_detection_bites_only_on_a_real_gap():
    contiguous = pd.date_range('2025-01-06 00:00', periods=30, freq='h')
    assert context_has_weekend_gap(contiguous) is False
    gapped = contiguous.tolist()[:10] + [contiguous[9] + pd.Timedelta(hours=49)]
    assert context_has_weekend_gap(pd.DatetimeIndex(gapped)) is True


# ── 5. the ledger ─────────────────────────────────────────────────────────────

def _log_row(**kw):
    row = dict(called_at_utc='2026-07-31 10:00:00', model_version='v1', p_up=0.6,
               direction='UP', n_paths=30, gen_sd=0.0001, gen_distinct_closes=6,
               as_of_bar_close='2026-07-31 10:00:00',
               forecast_bar_start='2026-07-31 10:00:00',
               forecast_bar_end='2026-07-31 11:00:00', minutes_remaining_at_call=45,
               forecast_bar_status='open', spans_weekend_gap=False,
               as_of_close=1.1000, data_source='injected')
    row.update(kw)
    return row


def test_ledger_excludes_degraded_statuses_and_dedups_by_forecast_bar(tmp_path, monkeypatch):
    log = tmp_path / 'results' / 'external_kronos' / 'log.csv'
    log.parent.mkdir(parents=True)
    rows = [
        _log_row(),                                                    # settles
        _log_row(called_at_utc='2026-07-31 10:30:00', p_up=0.9),       # dup bar -> ignored
        _log_row(forecast_bar_start='2026-07-31 12:00:00', forecast_bar_status='market_closed'),
        _log_row(forecast_bar_start='2026-07-31 13:00:00', forecast_bar_status='clock_unconfirmed'),
        _log_row(forecast_bar_start='2026-07-31 14:00:00', forecast_bar_status='already_closed'),
    ]
    pd.DataFrame(rows)[serving.LOG_COLUMNS].to_csv(log, index=False)

    closes = pd.Series([1.1050, 1.2, 1.3, 1.4],
                       index=pd.to_datetime(['2026-07-31 10:00:00', '2026-07-31 12:00:00',
                                             '2026-07-31 13:00:00', '2026-07-31 14:00:00']))
    monkeypatch.setattr('src.h1_direction_serving.realised_h1_closes',
                        lambda **kw: closes)
    led = serving.build_ledger(log_path=str(log.relative_to(tmp_path)), base_dir=str(tmp_path))
    assert len(led) == 1, 'only the open, first-of-bar call may settle'
    assert led.iloc[0]['p_up'] == 0.6, 'the FIRST call settles, not the later one'
    assert bool(led.iloc[0]['correct']) is True


def test_reliability_table_reports_realised_rate_per_bin():
    led = pd.DataFrame({
        'p_up': [0.05, 0.15, 0.85, 0.95],
        'realised_direction': ['DOWN', 'UP', 'UP', 'UP'],
    })
    tab = serving.reliability_table(led, n_bins=5)
    assert sum(r['n'] for r in tab) == 4
    assert tab[0]['n'] == 2 and tab[0]['realised_up_rate'] == pytest.approx(0.5)
    assert tab[-1]['n'] == 2 and tab[-1]['realised_up_rate'] == pytest.approx(1.0)
    assert serving.reliability_table(pd.DataFrame({'p_up': [], 'realised_direction': []})) == []


# ── 6. NO capability that turns a forecast into an action ─────────────────────

def _strip_comments(src: str) -> str:
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)
    return ' '.join(out).lower()


def test_no_execution_capability_anywhere_in_the_harness():
    """Scans IDENTIFIERS, not prose: a module may DESCRIBE the boundary
    ('no order placement, no sizing, no stop-loss') without implementing it."""
    banned = ('order', 'position_size', 'sizing', 'stop_loss', 'stoploss',
              'take_profit', 'leverage', 'broker', 'execute_trade', 'place')
    root = os.path.join(REPO, 'src', 'external', 'kronos')
    for f in os.listdir(root):
        if not f.endswith('.py'):
            continue
        idents = _strip_comments(open(os.path.join(root, f), encoding='utf-8').read())
        for term in banned:
            assert term not in idents.split(), f'{f}: execution identifier {term!r}'


def test_no_imported_trading_frame_in_the_harness_or_the_view():
    """The metrics are accuracy, calibration and sharpness. 'pip' may appear only
    as a unit of price movement, never as profit."""
    banned = ('breakeven', 'break-even', 'sharpe', 'equity curve', 'transaction cost',
              'pips of profit', 'p&l', 'pnl', 'spread_pips')
    root = os.path.join(REPO, 'src', 'external', 'kronos')
    for f in os.listdir(root):
        if not f.endswith('.py'):
            continue
        src = open(os.path.join(root, f), encoding='utf-8').read().lower()
        for term in banned:
            assert term not in src, f'{f}: trading-frame term {term!r}'


def test_p_up_is_printed_not_drawn():
    """A gauge or bar would imply a precision p_up does not have."""
    html = open(os.path.join(REPO, 'static', 'index.html'), encoding='utf-8').read()
    assert 'kronosPup' in html
    # split on the FUNCTION, not the onclick attribute, or this slices the card
    # markup instead of the renderer and the assertion means nothing.
    block = html.split('async function fetchKronos')[1][:3500]
    assert 'toFixed(3)' in block, 'p_up must be printed to 3dp'
    # Strip // comments first. The renderer DOCUMENTS why there is no gauge
    # ("a gauge would imply a precision this number does not have"), and a naive
    # substring scan flags that prose as the very thing it forbids.
    code = '\n'.join(ln.split('//')[0] for ln in block.splitlines()).lower()
    assert 'gauge' not in code
    assert '<progress' not in code
    assert 'width:' not in code, 'no proportional bar for p_up'


# ── 7. the notes ride along on every result ───────────────────────────────────

def test_kronos_unavailable_never_affects_any_existing_family():
    """C.3, the non-negotiable one. With Kronos switched off, the endpoint must
    return a clear unavailable payload and EVERY existing surface must still
    work. Nothing here loads torch or touches the GPU."""
    from fastapi.testclient import TestClient
    import api as api_module

    original = getattr(api_module.service, 'kronos_ready', False)
    api_module.service.kronos_ready = False
    api_module.service.kronos_error = 'simulated: torch not installed'
    try:
        client = TestClient(api_module.app)
        r = client.get('/api/kronos-volatility')
        assert r.status_code == 200, 'must degrade, never 500'
        body = r.json()
        assert body['available'] is False
        assert 'torch not installed' in body['reason']
        # the retired direction endpoint still answers, and never with a call
        d = client.get('/api/kronos-direction').json()
        assert d['available'] is False and d['retired'] is True
        assert d.get('direction') is None
        assert 'not a trading instruction' in body['disclaimer']

        # every pre-existing surface is untouched
        assert client.get('/').status_code == 200
        assert client.get('/history').status_code == 200
        assert client.get('/paper-trading').status_code == 200
        assert client.get('/api/paper-trading').status_code == 200
        assert client.get('/api/h1-direction').status_code == 200
        assert client.get('/h1-direction').status_code == 200
    finally:
        api_module.service.kronos_ready = original


def test_kronos_page_renders_with_no_settled_observations():
    from fastapi.testclient import TestClient
    import api as api_module

    client = TestClient(api_module.app)
    r = client.get('/kronos-volatility')
    assert r.status_code == 200
    html = r.text
    assert 'Kronos' in html
    assert 'p_vol_amp_calibrated' in html
    assert 'p_up' not in html, 'the retired direction number must not reappear'
    assert 'External foundation model' in html or 'external' in html.lower()
    # the caveats must be VISIBLE PAGE TEXT, not just module constants
    assert 'June 2024' in html
    assert 'not a trading instruction' in html
    # and it must link back rather than sit orphaned
    for href in ('/', '/history', '/paper-trading', '/h1-direction'):
        assert 'href="%s"' % href in html


def test_kronos_is_linked_from_every_other_view():
    from fastapi.testclient import TestClient
    import api as api_module

    client = TestClient(api_module.app)
    for path in ('/history', '/paper-trading'):
        assert '/kronos-volatility' in client.get(path).text, f'{path} does not link to Kronos'
    assert '/kronos-volatility' in open(
        os.path.join(REPO, 'static', 'index.html'), encoding='utf-8').read()


def test_every_caveat_is_carried_and_says_the_right_thing():
    assert 'June 2024' in loader.CONTAMINATION_NOTE
    assert '2024-07-01' in loader.CONTAMINATION_NOTE
    assert '2025-06-30' in loader.CONTAMINATION_NOTE
    # The note was rewritten for pred_len=24: the single-step collapse it used
    # to describe is gone (29.9 distinct endpoints of 30), but the paths are
    # still under-dispersed, which is what the scale correction exists for.
    assert '0.206' in loader.TOKENIZER_NOTE and '0.411' in loader.TOKENIZER_NOTE
    assert '29.9' in loader.TOKENIZER_NOTE and '0.655' in loader.TOKENIZER_NOTE
    assert 'no information' in loader.DIRECTION_RETIRED_REASON
    assert 'weekend' in loader.WEEKEND_NOTE.lower()
    assert 'not a trading instruction' in loader.DISCLAIMER
    assert 'zero-shot' in loader.DISCLAIMER
