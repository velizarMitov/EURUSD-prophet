"""The frozen range models (spec fx-ny-orb-hot/range-forward; design D7).

    python -m src.fx_ny_orb.frozen      fit once and write the frozen files and their digests

Per pair: OLS HAR-X (intercept + fx_nyvol's 14 features) and OLS HAR (intercept + the three range
history features), fitted on fx_nyvol's eligible days up to `frozen.fit_end` (read-only fx_nyvol
dataset on the fx_session caches). Files: research_models/fx_ny_orb/frozen_<PAIR>.json; digests:
results/fx_ny_orb/forward/frozen_manifest.csv. Every live pass loads through `load`, which refuses a
file whose digest differs from the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np
import pandas as pd

from . import load_config, repo_path
from .registry import results_dir

MANIFEST = os.path.join('forward', 'frozen_manifest.csv')


class FrozenChanged(RuntimeError):
    pass


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return coef


def predict(coef, x) -> float:
    return float(coef[0] + np.asarray(x, dtype=float) @ np.asarray(coef[1:], dtype=float))


def nyvol_data() -> tuple:
    from ..fx_nyvol import load_config as nyvol_config
    from ..fx_nyvol.target import dataset, load_frames
    ny = nyvol_config()
    return dataset(*load_frames(ny), ny), ny


def frozen_path(cfg: dict, pair: str) -> str:
    return os.path.join(repo_path(cfg['paths']['artifacts']), f'frozen_{pair}.json')


def manifest_path(cfg: dict) -> str:
    return os.path.join(results_dir(cfg), MANIFEST)


def _sha(p) -> str:
    with open(p, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fit(cfg: dict | None = None, data: tuple | None = None) -> dict:
    cfg = cfg or load_config()
    data_by_pair, ny = data if data is not None else nyvol_data()
    cols, har = ny['features']['columns'], ny['features']['har_columns']
    end = pd.Timestamp(cfg['frozen']['fit_end'], tz='UTC')
    out = {}
    for pair in cfg['pairs']:
        d = data_by_pair[pair]
        d = d[(d['reason'] == '') & (d.index <= end)].sort_index()
        rec = {'pair': pair, 'columns': cols, 'har_columns': har,
               'coef_harx': [float(c) for c in ols(d[cols].to_numpy(float), d['y'].to_numpy(float))],
               'coef_har': [float(c) for c in ols(d[har].to_numpy(float), d['y'].to_numpy(float))],
               'n_days': int(len(d)), 'first_day': str(d.index.min().date()), 'last_day': str(d.index.max().date()),
               'fit_end': cfg['frozen']['fit_end']}
        p = frozen_path(cfg, pair)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8', newline='\n') as fh:
            json.dump(rec, fh, indent=1, sort_keys=True)
        out[pair] = rec
    os.makedirs(os.path.dirname(manifest_path(cfg)), exist_ok=True)
    with open(manifest_path(cfg), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['pair', 'path', 'sha256'])
        for pair in cfg['pairs']:
            w.writerow([pair, os.path.basename(frozen_path(cfg, pair)), _sha(frozen_path(cfg, pair))])
    return out


def manifest_sha(cfg: dict) -> str:
    return _sha(manifest_path(cfg)) if os.path.exists(manifest_path(cfg)) else ''


def load(cfg: dict, pair: str) -> dict:
    if not os.path.exists(manifest_path(cfg)):
        raise FrozenChanged('no frozen manifest: run python -m src.fx_ny_orb.frozen')
    with open(manifest_path(cfg), newline='', encoding='utf-8') as fh:
        want = {r['pair']: r['sha256'] for r in csv.DictReader(fh)}
    p = frozen_path(cfg, pair)
    if pair not in want or not os.path.exists(p) or _sha(p) != want[pair]:
        raise FrozenChanged(f'frozen model for {pair} is missing or differs from its recorded digest')
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def main(argv=None):
    argparse.ArgumentParser(description='fit and freeze the fx-ny-orb-hot range models').parse_args(argv)
    for pair, rec in fit().items():
        print(f"{pair}: {rec['n_days']} days {rec['first_day']}..{rec['last_day']}")


if __name__ == '__main__':
    main()
