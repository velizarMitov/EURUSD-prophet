"""rts-signals task 2.2: the read-only Gmail fetch with a fake client."""

import base64
import os
import subprocess

import pytest

from src.rts_signals import REPO, load_config
from src.rts_signals.gmail import NOT_AUTHORISED, fetch, query, token_path
from src.rts_signals.ingest import REPORTS, read_csv, rpath

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'rts_signals')


class _Call:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class FakeGmail:
    """Records every method name called on it; only list / get / attachments.get exist."""

    def __init__(self, messages):
        self.messages_ = messages
        self.calls = []

    def users(self):
        return self

    def messages(self):
        return self

    def attachments(self):
        self.calls.append('attachments')
        return _Att(self)

    def list(self, **kw):
        self.calls.append('list')
        return _Call({'messages': [{'id': m} for m in self.messages_]})

    def get(self, userId, id, format):          # noqa: A002
        self.calls.append('get')
        return _Call({'id': id, 'internalDate': '1764198000000',
                      'payload': {'parts': [{'filename': 'report.pdf', 'body': {'attachmentId': f'att-{id}'}},
                                            {'filename': 'logo.png', 'body': {'attachmentId': 'x'}}]}})

    def __getattr__(self, name):                # send / modify / trash / delete would land here
        raise AssertionError(f'forbidden Gmail call: {name}')


class _Att:
    def __init__(self, g):
        self.g = g

    def get(self, userId, messageId, id):       # noqa: A002
        self.g.calls.append('attachments.get')
        return _Call({'data': base64.urlsafe_b64encode(self.g.messages_[messageId]).decode()})


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c['paths'] = dict(c['paths'], results=str(tmp_path / 'results'), artifacts=str(tmp_path / 'art'))
    return c


def test_query_names_both_senders():
    q = query(load_config())
    assert 'ber@rts-forex.com' in q and ' OR ' in q and 'has:attachment' in q


def test_new_attachment_saved_once(cfg, monkeypatch):
    from src.rts_signals import ingest as ing
    with open(os.path.join(FIX, 'spot_112526.txt'), encoding='utf-8') as fh:
        text = fh.read()
    monkeypatch.setattr(ing, 'extract_text', lambda data: text)
    g = FakeGmail({'m1': b'%PDF-fake-1125'})
    r = fetch(cfg, svc=g)
    assert (r['messages'], r['pdfs'], r['stored']) == (1, 1, 1)
    reps = read_csv(rpath(cfg, REPORTS))
    assert len(reps) == 1 and reps[0]['source'] == 'gmail'
    assert reps[0]['received_utc'] == '2025-11-26T23:00:00+00:00'          # the message's own internalDate
    assert fetch(cfg, svc=g)['messages'] == 0                     # the message is remembered
    assert len(read_csv(rpath(cfg, REPORTS))) == 1
    assert set(g.calls) <= {'list', 'get', 'attachments', 'attachments.get'}


def test_missing_token_reports_not_authorised(cfg):
    r = fetch(cfg)
    assert r['status'] == NOT_AUTHORISED and r['stored'] == 0


def test_no_write_methods_in_source():
    with open(os.path.join(REPO, 'src', 'rts_signals', 'gmail.py'), encoding='utf-8') as fh:
        src = fh.read()
    for bad in ('.send(', '.modify(', '.trash(', '.delete(', '.insert(', '.batchModify(', 'gmail.modify',
                'gmail.send', 'mail.google.com/'):
        assert bad not in src, bad


def test_token_path_is_gitignored():
    rel = os.path.relpath(token_path(load_config()), REPO).replace('\\', '/')
    assert rel.startswith('research_models/rts_signals/secrets/')
    assert subprocess.run(['git', 'check-ignore', '-q', rel], cwd=REPO).returncode == 0
