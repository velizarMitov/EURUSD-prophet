"""Round-two gold models for live OBSERVATION (openspec change gold-forecasting-v2, task 6.2).

Every round-two cell was DROP on validation (results/gold/RESULTS.md, 2026-10-23). The
owner chose to log them live anyway, for the leaderboard only: their registered
verdicts stand, and forward data can confirm a cleared cell but never revive a
dropped one.

    d1_lstm_macro, d1_gbm_macro  the daily models with the macro block
    h1_selective                 the H1 GBM that abstains below a confidence tau
    h1_barrier                   the H1 GBM on triple-barrier labels (k hours)

Fitted once on every pinned row (the `final` positions) into
research_models/gold/v2/<cell>/, with digests in results/gold/artifact_manifest_v2.csv;
tau and the column lists are stored beside each model. The registered models,
their manifest and round one's files are never touched.

    python -m src.gold.models_v2 --fit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import REPO, load_config, repo_path

CELLS = {
    'd1_lstm_macro': {'base': 'd1_lstm', 'cadence': 'D1', 'macro': True},
    'd1_gbm_macro': {'base': 'd1_gbm', 'cadence': 'D1', 'macro': True},
    'h1_selective': {'base': 'h1_gbm', 'cadence': 'H1', 'macro': False},
    'h1_barrier': {'base': 'h1_gbm', 'cadence': 'H1', 'macro': False},
}
VERSION = 'v2'
MANIFEST_FIELDS = ['cell', 'cadence', 'path', 'sha256', 'bytes', 'fit_rows', 'last_bar', 'written_at']


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def art_dir(cfg2) -> str:
    return repo_path(cfg2['paths']['v2_artifacts'])


def manifest_path(cfg2) -> str:
    return repo_path(cfg2['paths']['v2_manifest'])


def overrides(cell: str) -> dict | None:
    from . import features as F
    from . import macro as M
    return {'columns': F.D1_FEATURE_COLUMNS + M.MACRO_COLUMNS} if CELLS[cell]['macro'] else None


def make(cell: str, cfg: dict):
    from . import challengers as C
    return C.make(CELLS[cell]['base'], cfg, overrides(cell))


# ── fitting (once) ──────────────────────────────────────────────────────────

def fit_all(cfg=None, cfg2=None, frames=None, maker=None) -> dict:
    """Fit the four cells on every pinned row and write them with digests."""
    from . import challengers as C
    from . import direction2 as D2
    from . import macro as M
    from . import prereg as P
    from . import prereg2 as P2
    from . import splits as S
    from . import study as ST
    from . import targets as G
    cfg = cfg or load_config()
    cfg2 = cfg2 or P2.load_config_v2()
    reg2 = P2.registration(cfg2)
    maker = maker or (lambda cell: make(cell, cfg))
    frames = frames or ST.load_pinned(cfg, P.registration(cfg))
    seed = cfg['seeds'][0]
    rows, failures = [], []
    d1 = frames['D1']
    mac = M.load_pinned_features(cfg2, reg2['families']['macro']['pinned']['features_sha256'], d1.index).to_numpy(float)
    inp_d1 = M.with_macro(C.build_inputs(d1, 'D1'), mac)[1]
    h1 = frames['H1']
    inp_h1 = C.build_inputs(h1, 'H1')
    st = cfg2['study']['direction2']
    for cell, spec in CELLS.items():
        df, inp = (d1, inp_d1) if spec['cadence'] == 'D1' else (h1, inp_h1)
        tg = G.build_targets(df['close'])
        pos = S.stage_positions(len(df), 'final', cfg)['fit']
        extra = {'base': spec['base'], 'columns': (overrides(cell) or {}).get('columns'), 'fit_rows': int(len(pos))}
        try:
            if cell == 'h1_barrier':
                b = st['barrier']
                lab = D2.barrier_labels(df, b['k_hours'], b['m'], b['ewma_span'])['label'].to_numpy()
                tg = dict(tg, dir=lab)
                pos = D2.barrier_purge(len(df), b['k_hours'], pos)
                extra.update(barrier=b)
            if cell == 'h1_selective':
                tau, _ = D2.inner_tau(lambda: maker(cell), inp, pos, tg, reg2['families']['direction2']['selective_floor'],
                                      st['selective']['inner_splits'], seed)
                extra.update(tau=tau, floor=reg2['families']['direction2']['selective_floor'])
            m = maker(cell)
            m.fit(inp, pos, tg, seed=seed)
            path = os.path.join(art_dir(cfg2), cell)
            files = list(m.save(path))
            meta = os.path.join(path, 'v2_meta.json')
            with open(meta, 'w', encoding='utf-8') as fh:
                json.dump(extra, fh, indent=1)
            for f in files + [meta]:
                blob = open(f, 'rb').read()
                rows.append({'cell': cell, 'cadence': spec['cadence'],
                             'path': os.path.relpath(f, REPO).replace('\\', '/'),
                             'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob),
                             'fit_rows': len(pos), 'last_bar': str(df.index[-1]), 'written_at': _now()})
        except Exception as e:  # noqa: BLE001 -- recorded, never hidden
            failures.append({'cell': cell, 'error': f'{type(e).__name__}: {e}'})
        finally:
            try:
                import keras
                keras.backend.clear_session()
            except Exception:
                pass
    os.makedirs(os.path.dirname(manifest_path(cfg2)), exist_ok=True)
    with open(manifest_path(cfg2), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {'artifacts': len(rows), 'failures': failures}


# ── live use (logger) ───────────────────────────────────────────────────────

def manifest(cfg2) -> dict:
    out = {}
    path = manifest_path(cfg2)
    if not os.path.exists(path):
        return out
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r['cell'], []).append((r['path'], r['sha256']))
    return out


def load(cell: str, cfg: dict, cfg2: dict):
    path = os.path.join(art_dir(cfg2), cell)
    with open(os.path.join(path, 'v2_meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    return make(cell, cfg).load(path), meta


def live_macro(cfg2, d1: pd.DataFrame) -> np.ndarray:
    """The macro block on the live D1 frame, fetched fresh (FRED and MT5); raises
    MacroUnavailable naming the series when a source cannot supply it."""
    from . import macro as M
    start = d1.index[0] - pd.Timedelta(days=30)
    feats = M.macro_features(d1, M.load_inputs(cfg2, start, d1.index[-1], allow_fetch=True),
                             cfg2['study']['macro']['zscore_window'])
    return feats.to_numpy(float)


def forecast(cell: str, model, meta: dict, inp, k: int = -1) -> dict:
    """One row's p_up / ret_pct / vol_pct; h1_selective blanks its direction when it abstains."""
    out = model.predict(inp)
    row = {name: (float(v[k]) if np.isfinite(v[k]) else '') for name, v in out.items()}
    if cell == 'h1_selective' and row.get('p_up', '') != '':
        if abs(row['p_up'] - 0.5) < float(meta['tau']):
            row['p_up'] = ''
            row['abstained'] = True
    if cell == 'h1_barrier':
        row['ret_pct'] = ''                       # its regressor head was not the barrier target
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description='round-two gold models for live observation')
    ap.add_argument('--fit', action='store_true')
    args = ap.parse_args(argv)
    if args.fit:
        print(json.dumps(fit_all(), indent=1))


if __name__ == '__main__':
    main()
