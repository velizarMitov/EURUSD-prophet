"""Gold model versions (openspec change gold-forecasting-v2, spec gold-forecasting/model-versions).

A monthly refit never touches the registered models. It writes a NEW version,

    research_models/gold/versions/<YYYYMMDD>/<cell>/...   + manifest.csv

fitted on every bar available at run time, and makes it the single active SHADOW
version (versions/active.json). The logger forecasts with it beside the frozen
registered models into predictions_v2.csv (shadow=1); shadow rows never reach a
registered forward verdict, only the leaderboard.

A version becomes the registered model only through

    python -m src.gold.versions promote <YYYYMMDD> --reason "..."

which refuses while a registered forward window is open, copies the version into
the registered layout, rewrites results/gold/artifact_manifest.csv, records old
and new digests and the reason in results/gold/RESULTS.md, and retires the
forward registration so a new window must be registered before scoring.

    python -m src.gold.versions refit      (the monthly task)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
from datetime import datetime, timezone

import pandas as pd

from . import REPO, load_config, repo_path


class VersionRefused(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc)


def versions_dir(cfg2) -> str:
    return repo_path(cfg2['paths']['versions'])


def version_cfg(cfg: dict, cfg2: dict, vid: str) -> dict:
    """A round-one config whose artifact, manifest and trial-log paths point into
    the version folder: the round-one code paths then read and write only there."""
    root = os.path.join(versions_dir(cfg2), vid)
    c = copy.deepcopy(cfg)
    c['paths']['artifacts'] = root
    c['paths']['manifest'] = os.path.join(root, 'manifest.csv')
    c['paths']['trial_log'] = os.path.join(root, 'trials.csv')
    return c


def active(cfg2) -> str | None:
    path = os.path.join(versions_dir(cfg2), 'active.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh).get('version')


def _set_active(cfg2, vid: str) -> None:
    path = os.path.join(versions_dir(cfg2), 'active.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'version': vid, 'activated_at': _now().isoformat(timespec='seconds'),
                   'note': 'the single active shadow version; older ones are retired'}, fh, indent=1)


def _record_health(cfg, outcome: str, note: str) -> None:
    from . import health as HL
    t = _now().isoformat(timespec='seconds')
    HL.record({'started_at': t, 'ended_at': t, 'outcome': outcome, 'mt5': '', 'note': f'refit: {note}'[:300]}, cfg)


def refit(cfg=None, cfg2=None, frames=None, today=None, saver=None) -> dict:
    """Fit every forward cell on all data available now into a new version and
    make it the active shadow. Exits cleanly if another refit holds the lock."""
    from . import forward_logger as FL
    from . import prereg2 as P2
    from . import study as ST
    cfg = cfg or load_config()
    cfg2 = cfg2 or P2.load_config_v2()
    vid = (today or _now()).strftime('%Y%m%d')
    root = os.path.join(versions_dir(cfg2), vid)
    os.makedirs(versions_dir(cfg2), exist_ok=True)
    try:
        with FL.RunLock(os.path.join(versions_dir(cfg2), 'retrain.lock')):
            if os.path.exists(os.path.join(root, 'manifest.csv')):
                raise VersionRefused(f'version {vid} already exists')
            if frames is None:
                from . import clock as K
                from . import data as D
                frames = {cad: D.load(cad, cfg=cfg)[0] for cad in ('D1', 'H1')}
                K.verify(frames['H1'].index, cfg['clock']['weekly_open_threshold'])
            os.makedirs(root, exist_ok=True)
            rows, failures = (saver or ST.save_artifacts)(frames, version_cfg(cfg, cfg2, vid))
            if failures:
                raise VersionRefused(f'version {vid} incomplete: {failures}')
            _set_active(cfg2, vid)
            _record_health(cfg, 'ok', f'version {vid}, {len(rows)} artifacts')
            return {'version': vid, 'artifacts': len(rows), 'active': True}
    except FL.Locked as e:
        _record_health(cfg, 'locked', f'skipped: {e}')
        return {'version': None, 'skipped': str(e)}


# ── promotion ───────────────────────────────────────────────────────────────

def forward_window(cfg, now=None) -> dict:
    """{open, ends}: a registered forward window is open while the forward
    registration exists, its cap has not passed and not every forward verdict
    is final."""
    from .forward_report import CAP_YEARS
    fwd = repo_path(cfg['paths']['forward'])
    reg_path = os.path.join(fwd, 'PRE_REGISTRATION.json')
    if not os.path.exists(reg_path):
        return {'open': False, 'ends': None, 'reason': 'no forward registration'}
    with open(reg_path, encoding='utf-8') as fh:
        reg = json.load(fh)
    start = pd.Timestamp(reg.get('created_at') or reg.get('registered_at'))
    start = start if start.tzinfo else start.tz_localize('UTC')
    ends = start + pd.Timedelta(days=365.25 * float(reg.get('cap_years', CAP_YEARS)))
    now = pd.Timestamp(now or _now())
    now = now if now.tzinfo else now.tz_localize('UTC')
    verdicts = os.path.join(fwd, 'forward_verdicts.json')
    final = False
    if os.path.exists(verdicts):
        with open(verdicts, encoding='utf-8') as fh:
            v = json.load(fh).get('verdicts', [])
        final = bool(v) and all(str(x.get('verdict', '')).startswith(('KEEP', 'DROP')) for x in v)
    return {'open': bool(now < ends and not final), 'ends': str(ends.date()),
            'reason': 'all forward verdicts final' if final else ''}


def promote(vid: str, reason: str, cfg=None, cfg2=None, now=None) -> dict:
    from . import forward_logger as FL
    from . import prereg2 as P2
    cfg = cfg or load_config()
    cfg2 = cfg2 or P2.load_config_v2()
    if not reason or not reason.strip():
        raise VersionRefused('a promotion needs a reason')
    w = forward_window(cfg, now)
    if w['open']:
        raise VersionRefused(f'the registered forward window is open until {w["ends"]} '
                             '(or until every forward verdict is final): promotion refused')
    vcfg = version_cfg(cfg, cfg2, vid)
    new_manifest = FL.read_rows(vcfg['paths']['manifest'])
    if not new_manifest:
        raise VersionRefused(f'version {vid} has no manifest')
    for cell in {r['cell'] for r in new_manifest}:
        FL.verify_cell(cell, FL.read_manifest(vcfg))
    old = {c: FL.cell_digest(e) for c, e in FL.read_manifest(cfg).items()}
    art = repo_path(cfg['paths']['artifacts'])
    rows = []
    for r in new_manifest:
        src = repo_path(r['path'])
        rel = os.path.relpath(src, vcfg['paths']['artifacts'])
        dst = os.path.join(art, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        rows.append({**r, 'path': os.path.relpath(dst, REPO).replace('\\', '/')})
    with open(repo_path(cfg['paths']['manifest']), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(new_manifest[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    new = {c: FL.cell_digest(e) for c, e in FL.read_manifest(cfg).items()}
    fwd = repo_path(cfg['paths']['forward'])
    retired = []
    for name in ('PRE_REGISTRATION.json', 'PRE_REGISTRATION.md'):
        p = os.path.join(fwd, name)
        if os.path.exists(p):
            dst = os.path.join(fwd, name.replace('PRE_REGISTRATION', f'PRE_REGISTRATION_superseded_{vid}'))
            os.replace(p, dst)
            retired.append(os.path.basename(dst))
    stamp = pd.Timestamp(now or _now()).strftime('%Y-%m-%d')
    lines = [f'\n## {stamp} — version {vid} promoted to the registered gold models\n',
             f'Reason (owner): {reason.strip()}\n',
             '| cell | old digest | new digest |', '|---|---|---|']
    lines += [f'| {c} | {old.get(c, "")} | {new.get(c, "")} |' for c in sorted(set(old) | set(new))]
    lines.append('\nA new forward window starts: the forward registration was retired'
                 + (f' ({", ".join(retired)})' if retired else '') + ', so a new one must be committed '
                 'before any forward scoring.\n')
    with open(repo_path(cfg['paths']['results_md']), 'a', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    return {'version': vid, 'old': old, 'new': new, 'retired': retired}


def main(argv=None):
    ap = argparse.ArgumentParser(description='gold model versions')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('refit', help='fit a new shadow version (the monthly task)')
    p = sub.add_parser('promote', help='make a version the registered model')
    p.add_argument('version')
    p.add_argument('--reason', required=True)
    sub.add_parser('status', help='active shadow version and forward window')
    args = ap.parse_args(argv)
    if args.cmd == 'refit':
        print(json.dumps(refit(), default=str))
    elif args.cmd == 'promote':
        print(json.dumps(promote(args.version, args.reason), default=str, indent=1))
    else:
        from .prereg2 import load_config_v2
        print(json.dumps({'active': active(load_config_v2()), 'window': forward_window(load_config())}, indent=1))


if __name__ == '__main__':
    main()
