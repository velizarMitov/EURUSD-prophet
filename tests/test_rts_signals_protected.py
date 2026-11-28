"""rts-signals task 1.2: the family never changes production, other families or their registries; it has
no order code; its secrets are gitignored; a second pass changes nothing.

The end-to-end pass runs on fixtures in a temporary tree (fake Gmail, fake MT5 feed, temporary MT5
Common Files folder). Files appended to by other families' scheduled loggers are excluded from the
digest, as in the gold protection tests; their registries and pre-registrations are covered."""
import glob
import hashlib
import os
import re
import subprocess

import pandas as pd

import src.rts_signals as RS
from src.rts_signals import REPO, load_config
from src.rts_signals.run import run
from tests.test_rts_signals_gmail import FakeGmail

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'rts_signals')
LIVE_WRITERS = re.compile(r'^results/(gold/(forward|macro|accel|window)/|gold/gold_[hd]1|gold/gvz_|'
                          r'fx_ny_orb/(forward|live)/|gold_nyfwd/forward/|forward_eval/|external_tsfm/forward/)')


def _digest(rel):
    p = os.path.join(REPO, *rel.split('/'))
    h = hashlib.sha256()
    if os.path.isdir(p):
        for d, _, fs in sorted(os.walk(p)):
            for f in sorted(fs):
                r = os.path.relpath(os.path.join(d, f), REPO).replace('\\', '/')
                if LIVE_WRITERS.match(r) or '__pycache__' in r:
                    continue
                h.update(r.encode())
                with open(os.path.join(d, f), 'rb') as fh:
                    h.update(fh.read())
    elif os.path.exists(p):
        with open(p, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


def registries():
    pats = ['results/*hypothesis_log.csv', 'results/*/*hypothesis_log.csv', 'results/*/*/*hypothesis_log.csv',
            'results/*/PRE_REGISTRATION.json']
    found = sorted({p for pat in pats for p in glob.glob(os.path.join(REPO, *pat.split('/')))})
    return [os.path.relpath(p, REPO).replace('\\', '/') for p in found if 'rts_signals' not in p]


def digests():
    return {rel: _digest(rel) for rel in ['models', 'config.json', 'src/gold_nyfwd', 'src/fx_ny_orb'] + registries()}


class Feed:
    """Flat M1 bars for every symbol from 2026-11-20 to 2026-11-28."""

    def bars(self, symbol, start, end):
        idx = pd.date_range('2026-11-20', '2026-11-28', freq='1min', tz='UTC', name='time')
        idx = idx[(idx >= start) & (idx < end) & (idx.dayofweek < 5)]
        lvl = {'USDJPY': 159.2, 'GBPUSD': 1.31, 'EURUSD': 1.0975, 'AUDUSD': 0.663, 'USDCHF': 0.831,
               'USDCAD': 1.42}[symbol]
        return pd.DataFrame({'open': lvl, 'high': lvl + 0.0002, 'low': lvl - 0.0002, 'close': lvl, 'spread': 10.0},
                            index=idx)


def cfg_in(tmp_path):
    c = load_config()
    c['paths'] = dict(c['paths'], results=str(tmp_path / 'results'), artifacts=str(tmp_path / 'art'))
    c['mt5']['history_start'] = '2026-11-20'
    return c


def snapshot(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.md') or f.endswith('.log'):
                continue
            with open(os.path.join(d, f), 'rb') as fh:
                out[os.path.relpath(os.path.join(d, f), root)] = fh.read()
    return out


def test_end_to_end_pass_changes_no_protected_path_and_is_idempotent(tmp_path, monkeypatch):
    before = digests()
    c = cfg_in(tmp_path)
    from src.rts_signals import ingest as ing
    texts = {}
    for name in ('spot_112326.txt', 'spot_112526.txt'):
        with open(os.path.join(FIX, name), encoding='utf-8') as fh:
            texts[name.encode()] = fh.read()
    monkeypatch.setattr(ing, 'extract_text', lambda data: texts[data])
    env = {'MT5_COMMON_FILES': str(tmp_path / 'common')}
    now = pd.Timestamp('2026-11-27 12:00', tz='UTC')
    g = FakeGmail({'m1': b'spot_112326.txt', 'm2': b'spot_112526.txt'})
    out = run(c, svc=g, feed=Feed(), env=env, now_label=now)
    assert out['gmail']['stored'] == 2 and out['export'] == 'exported'
    assert (tmp_path / 'common' / 'EURUSDProphet' / 'rts_signals.tsv').exists()
    first = snapshot(tmp_path / 'results')
    out2 = run(c, svc=g, feed=Feed(), env=env, now_label=now)
    assert out2['gmail']['messages'] == 0 and out2['history'] == 'unchanged' and out2['report'] == 'unchanged'
    assert snapshot(tmp_path / 'results') == first
    assert digests() == before
    assert len(registries()) >= 15


def test_no_source_names_an_order_or_position_call():
    bad = re.compile(r'order_send|order_check|positions_get|positions_total|orders_get|TRADE_ACTION|order_calc')
    for name in os.listdir(os.path.dirname(RS.__file__)):
        if name.endswith('.py'):
            with open(os.path.join(os.path.dirname(RS.__file__), name), encoding='utf-8') as fh:
                assert not bad.findall(fh.read()), name


def test_no_module_names_a_write_target_outside_the_family_trees():
    target = re.compile(r"""['"](models/|config\.json|results/(?!rts_signals)[a-z_]+|research_models/(?!rts_signals)[a-z_]+)""")
    for name in os.listdir(os.path.dirname(RS.__file__)):
        if name.endswith('.py') or name.endswith('.json'):
            with open(os.path.join(os.path.dirname(RS.__file__), name), encoding='utf-8') as fh:
                hits = target.findall(fh.read())
            assert not hits, (name, hits)
    p = load_config()['paths']
    assert p['results'] == 'results/rts_signals' and p['artifacts'] == 'research_models/rts_signals'


def test_secrets_and_raw_reports_are_gitignored():
    for rel in ('research_models/rts_signals/secrets/token.json', 'research_models/rts_signals/secrets/client_secret.json',
                'research_models/rts_signals/inbox/x.pdf', 'research_models/rts_signals/drop/x.pdf',
                'research_models/rts_signals/unparsed.csv'):
        assert subprocess.run(['git', 'check-ignore', '-q', rel], cwd=REPO).returncode == 0, rel


def test_fixtures_hold_no_pdf_and_no_full_report():
    for f in os.listdir(FIX):
        assert not f.lower().endswith('.pdf')
        with open(os.path.join(FIX, f), encoding='utf-8') as fh:
            assert len(fh.read().splitlines()) < 60, f            # trimmed excerpts only
