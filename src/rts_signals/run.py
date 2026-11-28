"""One pass of the rts_signals family (task EURUSDProphet-RTSInbox, hourly at :20; design D9).

    python -m src.rts_signals.run              fetch -> ingest -> bars -> simulate -> ledger -> reports -> chart
    python -m src.rts_signals.run --no-gmail   skip the Gmail fetch (drop folder only)

Every step is idempotent and none stops the pass: Gmail without a token reports `gmail not authorised`,
an unreachable MT5 reports `mt5 unreachable`, an unwritable chart folder reports `export failed`. A second
pass with nothing new changes no stored row; the markdown files are rewritten only when their content
(apart from the generated-at line) changed.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from . import load_config, results_dir


def _same_but_stamp(path: str, text: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as fh:
        old = fh.read()
    strip = lambda t: [l for l in t.splitlines() if not l.startswith('_Generated')]   # noqa: E731
    return strip(old) == strip(text)


def _write_md(path: str, text: str) -> str:
    if _same_but_stamp(path, text):
        return 'unchanged'
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    return 'written'


def run(cfg: dict | None = None, gmail: bool = True, feed=None, svc=None, env=None, now_label=None) -> dict:
    from . import export, history, ingest, ledger, reconcile, report, rules, simulate
    cfg = cfg or load_config()
    out = {'utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    if gmail:
        from .gmail import fetch
        try:
            out['gmail'] = fetch(cfg, svc=svc)
        except Exception as e:                   # network or API trouble: the drop folder still works
            out['gmail'] = {'status': f'gmail failed ({type(e).__name__})'}
    out['ingested'] = [(r.get('file'), r['status']) for r in ingest.ingest_all(cfg)]
    from .bars import top_up
    try:
        out['bars'] = top_up(cfg, feed)
    except Exception as e:
        out['bars'] = f'mt5 unreachable ({type(e).__name__})'
    res = results_dir(cfg)
    os.makedirs(res, exist_ok=True)
    recs = simulate.load_recs(cfg)
    if len(recs):
        df = simulate.simulate_all(cfg, now_label, recs=recs)
        p = os.path.join(res, simulate.BASE)
        new = df.to_csv(index=False, lineterminator='\n', float_format='%.6g')
        if not os.path.exists(p) or open(p, encoding='utf-8').read() != new:
            with open(p, 'w', encoding='utf-8', newline='') as fh:
                fh.write(new)
        reconcile.write(cfg)
    out['ledger'] = ledger.update_from_files(cfg, now_label)
    out['history'] = _write_md(os.path.join(res, 'history.md'), history.build(cfg))
    out['report'] = _write_md(os.path.join(res, 'report.md'), report.build(cfg))
    out['readme'] = _write_md(os.path.join(res, 'README.md'), rules.readme())
    out['export'] = export.safe_rebuild(cfg, env, now_label)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='one rts_signals pass')
    ap.add_argument('--no-gmail', action='store_true', help='skip the Gmail fetch')
    a = ap.parse_args(argv)
    print(run(gmail=not a.no_gmail))


if __name__ == '__main__':
    main()
