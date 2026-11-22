"""The HAR reference, the LightGBM regressor and the conditional MSE MLP (spec fx-ny-volatility/evaluation;
design D6).

HAR       OLS of y on [1, lag1, ma5, ma20], fitted per pair per fold on all training days (fit + val).
          Closed form, deterministic. HAR-X (descriptive only) is the same OLS on all 14 features.
LightGBM  squared-error regressor, fx_session's fixed settings with objective l2, early stopping on the
          validation l2, fixed seed, deterministic.
MLP       32-16, LayerNorm, GELU, dropout 0.2, linear head (fx_session's build_mlp, read-only); MSE on the
          fit-standardised target; AdamW, plateau schedule, early stopping with the best epoch restored;
          the mean of 5 seeds, de-standardised. CUDA only. Trained only behind a LightGBM KEEP.
"""

from __future__ import annotations

import copy

from ..fx_fib.model_mlp_cuda import CudaUnavailable, pick_device, run_metadata, set_determinism  # noqa: F401

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

from ..fx_session.model_mlp import build_mlp


# -- OLS (HAR and HAR-X) -------------------------------------------------------------------------

def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return coef


def ols_predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + np.asarray(X, dtype=float) @ coef[1:]


# -- LightGBM ------------------------------------------------------------------------------------

def train_lightgbm(X_fit, y_fit, X_val, y_val, params: dict):
    p = {k: v for k, v in params.items() if k not in ('n_estimators', 'early_stopping_rounds')}
    model = lgb.LGBMRegressor(n_estimators=int(params['n_estimators']), **p)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric='l2',
              callbacks=[lgb.early_stopping(int(params['early_stopping_rounds']), verbose=False)])
    return model


# -- the MLP -------------------------------------------------------------------------------------

def train_mlp(X_fit, y_fit, X_val, y_val, params: dict, seed: int):
    """One member on standardised inputs and target. Returns (model on CPU in eval mode, metadata)."""
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
    loss_fn = nn.MSELoss()
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
    meta.update(best_epoch=best_epoch, epochs_trained=epoch + 1, best_val_mse=best,
                hidden=list(params['hidden']), dropout=list(params['dropout']))
    return model, meta


def predict_std(model: nn.Module, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy().astype(np.float64)


def ensemble_predict(models: list, X, n_members: int, y_scaler) -> np.ndarray:
    """Mean of the members' standardised predictions, de-standardised. Never a partial ensemble."""
    if len(models) != n_members or any(m is None for m in models):
        raise ValueError(f'partial ensemble: {sum(m is not None for m in models)} of {n_members} members')
    return y_scaler.inverse(np.mean([predict_std(m, X) for m in models], axis=0))
