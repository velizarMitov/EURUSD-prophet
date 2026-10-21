"""The /gold page and /api/gold route (task 10.2; spec gold-forecasting/gold-view)."""
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_WORDS = ('spread', 'pips', 'p&l', 'profit', 'pnl', 'sharpe')


@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    import api
    return TestClient(api.app)


def test_api_gold_returns_the_read_only_payload(client):
    r = client.get('/api/gold')
    assert r.status_code == 200
    body = r.json()
    assert body['label'] == 'research — no proven edge'
    assert {c['cell'] for c in body['cells']} >= {'h1_gbm', 'h1_lstm', 'd1_lstm'}
    for c in body['cells']:
        assert c['label'] in ('research — no proven edge', 'statistical skill, gross of cost; not a trading result')


def test_gold_page_renders_the_verdict_labels_and_no_cost_words(client):
    r = client.get('/gold')
    assert r.status_code == 200
    html = r.text
    assert "fetch('/api/gold'" in html
    assert 'research — no proven edge' in html and 'изследване — без доказано предимство' in html
    lower = html.lower()
    for w in COST_WORDS:
        assert w not in lower, w


def test_console_links_to_the_gold_page():
    with open(os.path.join(REPO, 'static', 'index.html'), encoding='utf-8') as fh:
        assert 'href="/gold"' in fh.read()
