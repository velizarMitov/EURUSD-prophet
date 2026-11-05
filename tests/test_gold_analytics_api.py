"""Routes and page of the gold analytics view (openspec change gold-market-analytics,
tasks 4.2 and 5.1)."""
import os
import re

import pytest

from src.gold_analytics import REPO, load_config
from src.gold_analytics import service as svc_mod
from src.gold_analytics.service import SECTIONS, Service
from tests.test_gold_analytics_service import NOW, FakeSource, _fred

PAGE = os.path.join(REPO, 'static', 'gold_analytics.html')


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import api
    cfg = load_config()
    cfg['paths'] = {**cfg['paths'], **{k: str(tmp_path / cfg['paths'][k].split('/')[-1])
                                       for k in ('cache_dir', 'bars_m1', 'bars_m15', 'spread_5m', 'events',
                                                 'last_tick')}}
    monkeypatch.setattr(svc_mod, '_SERVICE', Service(cfg, FakeSource(), clock=lambda: NOW, fred=_fred))
    return TestClient(api.app)


@pytest.mark.parametrize('section', SECTIONS)
def test_every_section_answers_200_with_fake_data(client, section):
    r = client.get(f'/api/gold-analytics/{section}')
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['section'] == section and 'stale' in body and 'as_of' in body


def test_bad_parameters_answer_400_with_the_allowed_values(client):
    r = client.get('/api/gold-analytics/chart?timeframe=2h')
    assert r.status_code == 400
    for tf in ('1m', '5m', '15m', '30m', '1h', '4h', 'D1', 'W1'):
        assert f"'{tf}'" in r.json()['detail']
    assert client.get('/api/gold-analytics/orderbook').status_code == 400
    assert client.get('/api/gold-analytics/history?start=2025-01-01&end=2025-02-31').status_code == 400
    assert client.get('/api/gold-analytics/overview?x=1').status_code == 400


def test_the_page_is_served_with_the_shell(client):
    r = client.get('/gold-analytics')
    assert r.status_code == 200 and 'data-ps-shell' in r.text
    assert re.search(r'href="/gold-analytics" aria-current="page"', r.text)


def _page():
    with open(PAGE, encoding='utf-8') as fh:
        return fh.read()


def test_the_page_loads_nothing_from_outside():
    html = _page()
    assert not re.search(r'https?://', html), 'an external URL'
    assert not re.search(r'''(?:src|href)\s*=\s*["']//''', html)
    assert '@import' not in html and '<link' not in html.lower()


def test_the_page_has_no_trade_or_volume_words():
    low = _page().lower()
    for word in ('buy', 'sell', 'купи', 'продай', 'volume', 'vwap', 'poc', 'profile'):
        assert word not in low, word
