"""Pinned weights are verified before every load (openspec change
tsfm-second-opinion, spec external-forecasts/tsfm-challengers "Weights are
pinned and verified")."""
import io
import os

import pytest

from src.external.tsfm import vendor as V


def _artifact(tmp_path, name='chronos2'):
    d = tmp_path / 'research_models' / 'external_tsfm' / name
    d.mkdir(parents=True)
    (d / 'model.safetensors').write_bytes(b'weights-v1')
    (d / 'config.json').write_text('{"a": 1}', encoding='utf-8')
    manifest = str(tmp_path / 'results' / 'external_tsfm' / 'artifact_manifest.csv')
    V.record(name, [str(d / 'model.safetensors'), str(d / 'config.json')], 'test@rev', manifest, str(tmp_path))
    return d, manifest


def test_unchanged_files_pass(tmp_path):
    _, manifest = _artifact(tmp_path)
    m = V.read_manifest(manifest)
    digest = V.verify('chronos2', m, str(tmp_path))
    assert len(digest) == 16
    assert digest == V.verify('chronos2', m, str(tmp_path))


def test_altered_file_is_refused(tmp_path):
    d, manifest = _artifact(tmp_path)
    (d / 'model.safetensors').write_bytes(b'weights-v2')
    with pytest.raises(V.ArtifactMismatch, match='differs from the manifest digest'):
        V.verify('chronos2', V.read_manifest(manifest), str(tmp_path))


def test_missing_file_is_refused(tmp_path):
    d, manifest = _artifact(tmp_path)
    os.remove(d / 'config.json')
    with pytest.raises(V.ArtifactMismatch, match='missing'):
        V.verify('chronos2', V.read_manifest(manifest), str(tmp_path))


def test_unknown_artifact_is_refused(tmp_path):
    _, manifest = _artifact(tmp_path)
    with pytest.raises(V.ArtifactMismatch, match='not in the artifact manifest'):
        V.verify('timesfm25', V.read_manifest(manifest), str(tmp_path))


def test_record_keeps_other_artifacts(tmp_path):
    _, manifest = _artifact(tmp_path, 'chronos2')
    _artifact(tmp_path, 'timesfm25')
    m = V.read_manifest(manifest)
    assert set(m) == {'chronos2', 'timesfm25'}
    assert all(p.startswith('research_models/external_tsfm/') for p, _ in m['chronos2'])


def _fake_opener(payload: bytes):
    def opener(req, timeout=None):
        return io.BytesIO(payload)
    return opener


def test_fetch_refuses_a_download_that_is_not_the_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.setitem(V.PINS, 'fake', {'repo': 'x/y', 'revision': 'abc', 'files': {
        'model.safetensors': ('lfs', '0' * 64)}})
    with pytest.raises(V.ArtifactMismatch, match='pinned upstream digest'):
        V.fetch('fake', repo=str(tmp_path), opener=_fake_opener(b'tampered'),
                manifest=str(tmp_path / 'manifest.csv'))
    assert not (tmp_path / 'research_models' / 'external_tsfm' / 'fake' / 'model.safetensors').exists()


def test_fetch_records_a_matching_download(tmp_path, monkeypatch):
    payload = b'the pinned weights'
    import hashlib
    monkeypatch.setitem(V.PINS, 'fake', {'repo': 'x/y', 'revision': 'abc', 'files': {
        'model.safetensors': ('lfs', hashlib.sha256(payload).hexdigest()),
        'config.json': ('blob', hashlib.sha1(b'blob %d\0' % len(payload) + payload).hexdigest())}})
    rows = V.fetch('fake', repo=str(tmp_path), opener=_fake_opener(payload),
                   manifest=str(tmp_path / 'manifest.csv'))
    assert {r['path'] for r in rows} == {'research_models/external_tsfm/fake/config.json',
                                         'research_models/external_tsfm/fake/model.safetensors'}
    assert V.verify('fake', V.read_manifest(str(tmp_path / 'manifest.csv')), str(tmp_path))


def test_pins_are_full_commits_before_the_clean_window():
    from src.external.tsfm import CLEAN_START
    for model, pin in V.PINS.items():
        assert len(pin['revision']) == 40, model
        assert pin['released'] < CLEAN_START, model
        assert set(pin['files']) == {'config.json', 'model.safetensors'}, model
