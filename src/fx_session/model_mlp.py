"""The conditional binary MLP on CUDA (spec fx-session/evaluation; design D6).

    Linear(n,32) LayerNorm GELU Dropout(.2)  Linear(32,16) LayerNorm GELU Dropout(.2)  Linear(16,1)

BCEWithLogits WITHOUT class weights (the fx_fib lesson: weighting spread the probabilities and
ruined calibration), AdamW, ReduceLROnPlateau, early stopping on the validation log-loss with
the best epoch restored. CUDA only (no CPU fallback); the fold lives on the device and batches are
index slices from a seeded device generator. The ensemble is the mean probability of the five
seeds and is never served partially. Trained only for a pair whose LightGBM cell is KEEP.
"""

from __future__ import annotations

import copy

from ..fx_fib.model_mlp_cuda import CudaUnavailable, pick_device, run_metadata, set_determinism  # noqa: F401  (sets CUBLAS env first)

import numpy as np
import torch
from torch import nn

from . import artifacts as AR

EPS = 1e-6


def build_mlp(n_features: int, hidden, dropout) -> nn.Sequential:
    layers, prev = [], n_features
    for h, d in zip(hidden, dropout):
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(d)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp(X_fit, y_fit, X_val, y_val, params: dict, seed: int):
    """One member. Returns (model on CPU in eval mode, metadata). Restores torch's global
    deterministic-algorithms switch afterwards."""
    was = torch.are_deterministic_algorithms_enabled()
    try:
        return _train(X_fit, y_fit, X_val, y_val, params, seed)
    finally:
        torch.use_deterministic_algorithms(was)


def _train(X_fit, y_fit, X_val, y_val, params, seed):
    device = pick_device(params.get('allow_cpu', False))
    set_determinism(seed)
    xf = torch.as_tensor(X_fit, dtype=torch.float32, device=device)
    yf = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    model = build_mlp(xf.shape[1], params['hidden'], params['dropout']).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=params['plateau_factor'],
                                                       patience=params['plateau_patience'])
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    bs = int(params['batch_size'])
    best, best_state, best_epoch, bad, epoch = float('inf'), None, -1, 0, 0
    for epoch in range(int(params['max_epochs'])):
        model.train()
        perm = torch.randperm(len(xf), generator=gen, device=device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xf[b]).squeeze(1), yf[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xv).squeeze(1), yv))
        sched.step(vl)
        if vl < best - 1e-7:
            best, best_state, best_epoch, bad = vl, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= params['patience']:
                break
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    meta = run_metadata(device, seed)
    meta.update(best_epoch=best_epoch, epochs_trained=epoch + 1, best_val_logloss=best,
                hidden=list(params['hidden']), dropout=list(params['dropout']))
    return model, meta


def predict_up(model: nn.Module, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1)).numpy()
    return np.clip(p.astype(np.float64), EPS, 1 - EPS)


def ensemble_up(models: list, X, n_members: int) -> np.ndarray:
    if len(models) != n_members or any(m is None for m in models):
        raise ValueError(f'partial ensemble: {sum(m is not None for m in models)} of {n_members} members')
    return np.mean([predict_up(m, X) for m in models], axis=0)


def save_member(cfg, model, meta, pair, fold, seed, now_iso) -> str:
    blob = {'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'meta': meta}
    return AR.save(cfg, blob, pair, fold, f'mlp_seed{seed}', 'mlp', now_iso)


def load_ensemble(cfg, pair: str, fold: int, n_features: int) -> list:
    """All seed members of a fold, digest-verified, or ArtifactError -- never a partial ensemble."""
    manifest = AR.read_manifest(cfg)
    models = []
    for s in cfg['mlp']['seeds']:
        blob = AR.load(cfg, f'{AR.fold_dir(cfg, pair, fold)}/mlp_seed{s}.joblib', manifest)
        m = build_mlp(n_features, blob['meta']['hidden'], blob['meta']['dropout'])
        m.load_state_dict(blob['state_dict'])
        models.append(m.eval())
    return models
