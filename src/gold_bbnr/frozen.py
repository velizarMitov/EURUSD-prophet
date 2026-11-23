"""The frozen Part 2 models, refit on every usable day up to 2026-11-20 (spec gold-bb-nyrange/ny-range).

    python -m src.gold_bbnr.frozen      gated; writes research_models/gold_bbnr/ and results/gold_bbnr/frozen_manifest.csv

HAR and HAR-X are OLS on all usable days; LightGBM and the 5-seed MLP keep the walk-forward's last-15 %
early-stopping slice (1-day embargo). Each file's SHA-256 goes to the manifest, and `load` refuses a file
whose digest differs (FrozenChanged). Never refit silently.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from . import config_digest, load_config, repo_path, sha256_file
from .registry import guard, path

MANIFEST = 'frozen_manifest.csv'


class FrozenChanged(RuntimeError):
    pass


def art_dir(cfg: dict) -> str:
    return repo_path(cfg['paths']['artifacts'])


def split(n: int, cfg: dict) -> tuple:
    emb, frac = int(cfg['folds_ny']['embargo_days']), float(cfg['folds_ny']['val_fraction'])
    n_val = int(round(n * frac))
    return np.arange(0, n - n_val - emb), np.arange(n - n_val, n)


def fit(cfg: dict, rows, log=print) -> dict:
    """Fit and write the four models from the usable rows (features + y). Returns {file: sha256}."""
    import torch
    from ..fx_nyvol import models as M
    from ..fx_nyvol.splits import Standardiser
    cols, har_cols = cfg['features_ny']['columns'], cfg['features_ny']['har_columns']
    X, y, Xh = rows[cols].to_numpy(float), rows['y'].to_numpy(float), rows[har_cols].to_numpy(float)
    fit_i, val_i = split(len(rows), cfg)
    d = art_dir(cfg)
    os.makedirs(d, exist_ok=True)
    files = {}
    ols = {'har': M.ols_fit(Xh, y).tolist(), 'harx': M.ols_fit(X, y).tolist(), 'har_columns': har_cols,
           'columns': cols, 'first_day': str(rows.index.min().date()), 'last_day': str(rows.index.max().date()),
           'n_days': len(rows), 'config_digest': config_digest(cfg)}
    p = os.path.join(d, 'ols.json')
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(ols, fh, indent=1)
    files['ols.json'] = p
    lgbm = M.train_lightgbm(X[fit_i], y[fit_i], X[val_i], y[val_i], cfg['lightgbm_ny'])
    p = os.path.join(d, 'lightgbm.txt')
    lgbm.booster_.save_model(p)
    files['lightgbm.txt'] = p
    xs, ys = Standardiser(X[fit_i]), Standardiser(y[fit_i])
    states, epochs = {}, []
    for s in cfg['mlp']['seeds']:
        m, meta = M.train_mlp(xs.transform(X[fit_i]).astype(np.float32), ys.transform(y[fit_i]).astype(np.float32),
                              xs.transform(X[val_i]).astype(np.float32), ys.transform(y[val_i]).astype(np.float32),
                              cfg['mlp'], s)
        states[str(s)] = {k: v.cpu() for k, v in m.state_dict().items()}
        epochs.append(meta['best_epoch'])
    p = os.path.join(d, 'mlp.pt')
    torch.save({'states': states, 'x_mean': xs.mean, 'x_sd': xs.sd, 'y_mean': ys.mean, 'y_sd': ys.sd,
                'hidden': cfg['mlp']['hidden'], 'dropout': cfg['mlp']['dropout']}, p)
    files['mlp.pt'] = p
    log(f'frozen: {len(rows)} days {ols["first_day"]}..{ols["last_day"]} | trees {lgbm.best_iteration_} | '
        f'mlp epochs {epochs}')
    return write_manifest(cfg, files)


def write_manifest(cfg: dict, files: dict) -> dict:
    dig = {name: sha256_file(p) for name, p in files.items()}
    with open(path(cfg, MANIFEST), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['file', 'sha256'])
        for k, v in sorted(dig.items()):
            w.writerow([k, v])
    return dig


def verify(cfg: dict) -> dict:
    with open(path(cfg, MANIFEST), newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = os.path.join(art_dir(cfg), r['file'])
        if not os.path.exists(p) or sha256_file(p) != r['sha256']:
            raise FrozenChanged(f"{r['file']} differs from its recorded digest; restore it, never refit silently")
    return {r['file']: r['sha256'] for r in rows}


def load_ols(cfg: dict) -> dict:
    verify(cfg)
    with open(os.path.join(art_dir(cfg), 'ols.json'), encoding='utf-8') as fh:
        return json.load(fh)


def run(cfg: dict | None = None, gate=None, inp=None, log=print) -> dict:
    from . import ny_features as NF
    from .inputs import load
    cfg = cfg or load_config()
    guard(cfg)
    if gate is None:
        from .prereg import gate
    gate(cfg)
    inp = inp or load(cfg)
    ds = NF.dataset(inp.h1, inp.d1, inp.dxy_h1, cfg)
    return fit(cfg, ds[ds['reason'] == ''].sort_index(), log)


def main(argv=None):
    argparse.ArgumentParser(description='refit and freeze the gold-bb-nyrange Part 2 models').parse_args(argv)
    for k, v in run().items():
        print(k, v)


if __name__ == '__main__':
    main()
