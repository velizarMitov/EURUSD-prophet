"""Served-vs-declared artifact monitor (src/artifact_provenance.py).

On 2026-09-11 a retrain rewrote 18 production artifacts and nothing anywhere
said so for eight days. The monitor compares every models/ digest pinned by a
tests/fixtures/*_protected_sha256.json fixture against the file on disk and
reports what differs and for how long. These tests pin that it BITES (a
one-byte change is caught, a missing file is caught, fixtures disagreeing is
caught), that it says "undeclared for N days" and never "error", that it is
cheap to poll, that it never raises, and that the real api.py serves it at
startup, after a hot-reload, and at GET /api/provenance.
"""
import hashlib
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import artifact_provenance as AP  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def sandbox(tmp_path):
    """A fake repo: two artifacts under models/, two fixtures pinning them
    (overlapping on one path, as the real fixtures do), and one non-fixture
    JSON that must be ignored."""
    models = tmp_path / 'models'
    (models / 'baseline').mkdir(parents=True)
    a = models / 'baseline' / 'global_scaler.pkl'
    b = models / 'h1_gbm.pkl'
    a.write_bytes(b'scaler-bytes')
    b.write_bytes(b'gbm-bytes')
    fx = tmp_path / 'tests' / 'fixtures'
    fx.mkdir(parents=True)
    (fx / 'alpha_protected_sha256.json').write_text(json.dumps({
        'models/baseline/global_scaler.pkl': _sha(b'scaler-bytes'),
        'models/h1_gbm.pkl': _sha(b'gbm-bytes'),
        'src/features.py': 'not-a-models-path-and-not-checked',
    }), encoding='utf-8')
    (fx / 'beta_protected_sha256.json').write_text(json.dumps({
        'models/h1_gbm.pkl': _sha(b'gbm-bytes'),
        '_policy': {'rolling': 'a dict value, skipped'},
    }), encoding='utf-8')
    (fx / 'PROTECTED_SET_REBASELINE_2026-09-19.json').write_text(json.dumps({
        'models/h1_gbm.pkl': 'a rebaseline RECORD, not a pin'}), encoding='utf-8')
    return tmp_path


# ── 1. declared set ────────────────────────────────────────────────────────

def test_declared_digests_take_the_union_of_models_paths_only(sandbox):
    digests, fixtures, conflicts = AP.declared_digests(str(sandbox))
    assert set(digests) == {'models/baseline/global_scaler.pkl', 'models/h1_gbm.pkl'}
    assert fixtures == ['alpha_protected_sha256.json', 'beta_protected_sha256.json']
    assert conflicts == {}


def test_fixtures_that_disagree_are_reported_not_resolved(sandbox):
    fx = sandbox / 'tests' / 'fixtures' / 'beta_protected_sha256.json'
    fx.write_text(json.dumps({'models/h1_gbm.pkl': _sha(b'something-else')}), encoding='utf-8')
    digests, _, conflicts = AP.declared_digests(str(sandbox))
    assert 'models/h1_gbm.pkl' not in digests
    assert set(conflicts) == {'models/h1_gbm.pkl'}
    s = AP.ProvenanceMonitor(str(sandbox)).status()
    assert s['undeclared'] and s['conflicting'] == ['models/h1_gbm.pkl']
    assert 'inconsistently' in s['summary']


# ── 2. the check bites ─────────────────────────────────────────────────────

def test_clean_tree_reports_nothing_undeclared(sandbox):
    s = AP.ProvenanceMonitor(str(sandbox)).status()
    assert s['available'] and s['n_pinned'] == 2
    assert s['undeclared'] is False
    assert s['diverged'] == [] and s['missing'] == [] and s['conflicting'] == []
    assert s['summary'] is None and s['undeclared_since'] is None


def test_one_changed_byte_is_reported_with_its_age(sandbox):
    path = sandbox / 'models' / 'h1_gbm.pkl'
    path.write_bytes(b'gbm-bytez')
    written = time.time() - 8 * 86400
    os.utime(path, (written, written))

    s = AP.ProvenanceMonitor(str(sandbox)).status(now=time.time())

    assert s['undeclared'] is True
    assert [d['path'] for d in s['diverged']] == ['models/h1_gbm.pkl']
    d = s['diverged'][0]
    assert d['declared'] == _sha(b'gbm-bytes')[:12] and d['on_disk'] == _sha(b'gbm-bytez')[:12]
    assert s['undeclared_days'] == pytest.approx(8.0, abs=0.01)
    assert s['undeclared_since'].endswith('UTC')
    assert '1 of 2 served model artifacts differ' in s['summary']
    assert 'undeclared for 8.0 days' in s['summary']


def test_a_missing_pinned_artifact_is_reported(sandbox):
    (sandbox / 'models' / 'baseline' / 'global_scaler.pkl').unlink()
    s = AP.ProvenanceMonitor(str(sandbox)).status()
    assert s['undeclared'] and s['missing'] == ['models/baseline/global_scaler.pkl']
    assert 'missing from disk' in s['summary']


def test_undeclared_since_is_the_earliest_diverged_write(sandbox):
    a = sandbox / 'models' / 'baseline' / 'global_scaler.pkl'
    b = sandbox / 'models' / 'h1_gbm.pkl'
    now = time.time()
    a.write_bytes(b'x'); os.utime(a, (now - 3 * 86400, now - 3 * 86400))
    b.write_bytes(b'y'); os.utime(b, (now - 1 * 86400, now - 1 * 86400))
    s = AP.ProvenanceMonitor(str(sandbox)).status(now=now)
    assert s['undeclared_days'] == pytest.approx(3.0, abs=0.01)


def test_wording_is_a_fact_not_an_alarm(sandbox):
    (sandbox / 'models' / 'h1_gbm.pkl').write_bytes(b'changed')
    s = AP.ProvenanceMonitor(str(sandbox)).status()
    assert 'undeclared for' in s['summary']
    assert 'error' not in s['summary'].lower()
    assert 'fail' not in s['summary'].lower()


# ── 3. cheap to poll, never raises ─────────────────────────────────────────

def test_rehashes_only_when_a_pinned_file_changes(sandbox):
    m = AP.ProvenanceMonitor(str(sandbox))
    m.status(); m.status(); m.status()
    assert m.hash_runs == 1
    p = sandbox / 'models' / 'h1_gbm.pkl'
    p.write_bytes(b'gbm-bytes!')          # size changes -> signature changes
    assert m.status()['undeclared'] is True
    assert m.hash_runs == 2
    m.invalidate()
    m.status()
    assert m.hash_runs == 3, 'invalidate() must force a re-hash after a hot-reload'


def test_no_fixtures_means_unavailable_not_an_exception(tmp_path):
    s = AP.ProvenanceMonitor(str(tmp_path)).status()
    assert s['available'] is False and s['undeclared'] is False
    assert 'no *_protected_sha256.json' in s['reason']


def test_malformed_fixture_is_skipped_not_fatal(sandbox):
    (sandbox / 'tests' / 'fixtures' / 'broken_protected_sha256.json').write_text('{not json', encoding='utf-8')
    s = AP.ProvenanceMonitor(str(sandbox)).status()
    assert s['available'] and 'broken_protected_sha256.json' not in s['fixtures']


# ── 4. the real repository, and the real serving path ──────────────────────

def test_real_fixtures_pin_the_served_set_and_it_is_clean_or_says_why():
    """Against THIS repository. A red result here is not a test bug: it means
    models/ on disk is not what the fixtures declare, which is the exact
    condition the banner exists to surface -- and the checksum tests in
    test_h1_production / test_external_kronos will be red for the same reason."""
    digests, fixtures, conflicts = AP.declared_digests(REPO)
    assert len(digests) >= 40 and conflicts == {}
    assert {'h1_production_protected_sha256.json', 'kronos_protected_sha256.json'} <= set(fixtures)
    s = AP.ProvenanceMonitor(REPO).status()
    assert s['available'] and s['n_pinned'] == len(digests)
    assert s['undeclared'] is False, s['summary']


def test_api_exposes_the_check_at_startup_hot_reload_and_a_route():
    import api
    assert isinstance(api.provenance, AP.ProvenanceMonitor)
    assert api._startup_provenance['available'] is True
    body = api.provenance_status()
    assert body['n_pinned'] >= 40 and 'undeclared' in body
    routes = {r.path for r in api.app.routes}
    assert '/api/provenance' in routes
    src = open(api.__file__, encoding='utf-8').read()
    reload_at = src.index('_retrain["reloaded"] = True')
    assert src.index('provenance.invalidate()', reload_at) - reload_at < 200, \
        'the hot-reload must invalidate the digest cache'
    assert 'payload["provenance"] = provenance.status()' in src


def test_dashboard_renders_the_banner_from_the_route():
    html = open(os.path.join(REPO, 'static', 'index.html'), encoding='utf-8').read()
    assert 'id="provenanceBanner"' in html
    assert "fetch('/api/provenance')" in html
    assert "window.addEventListener('load', refreshProvenance)" in html
    assert 'renderProvenance(s.provenance)' in html, 'must refresh after a completed retrain'
    banner_css = html[html.index('.provenance-banner {'):html.index('.provenance-banner.active')]
    assert '#e74c3c' not in banner_css, 'the banner must not reuse the error red'
