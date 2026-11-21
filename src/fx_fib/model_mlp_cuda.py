"""The compact MLP on CUDA (spec fx-fibonacci/models; design D9).

Registered architecture (hidden [128, 64, 32], dropout [0.3, 0.2]):
    Linear(n,128) LayerNorm ReLU Dropout(.3)  Linear(128,64) LayerNorm ReLU Dropout(.2)
    Linear(64,32) ReLU  Linear(32,3)          -> softmax probabilities
The descriptive 64-32 variant uses the same builder with hidden [64, 32].

The whole fold lives on the device as float32 (a few MB); minibatches are index slices drawn
from a seeded torch.Generator ON the device, so no DataLoader and no host<->device copy per
batch. Class-weighted cross-entropy (weights from the fit rows only), AdamW with weight decay,
ReduceLROnPlateau and early stopping on the UNWEIGHTED validation log-loss; the best epoch's
weights are restored. Determinism: fixed seed, deterministic algorithms, cuBLAS workspace.
"""

from __future__ import annotations

import copy
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # must precede the first CUDA call

import numpy as np
import torch
from torch import nn

N_CLASSES = 3


class CudaUnavailable(RuntimeError):
    pass


def build_mlp(n_features: int, hidden, dropout) -> nn.Sequential:
    layers, prev = [], n_features
    for i, h in enumerate(hidden):
        layers.append(nn.Linear(prev, h))
        if i < len(hidden) - 1:
            layers += [nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout[i])]
        else:
            layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, N_CLASSES))
    return nn.Sequential(*layers)


def pick_device(allow_cpu: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if allow_cpu:
        return torch.device('cpu')
    raise CudaUnavailable('CUDA is not available and the config does not allow CPU training '
                          '(mlp.allow_cpu = false); refusing to train.')


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    return len(y) / (N_CLASSES * counts)


def run_metadata(device: torch.device, seed: int) -> dict:
    return {'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'seed': int(seed)}


def train_mlp(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              params: dict, seed: int, hidden=None, dropout=None):
    """Train one member. Returns (model on CPU in eval mode, metadata dict). The global
    deterministic-algorithms switch is restored afterwards so other torch code is unaffected."""
    was = torch.are_deterministic_algorithms_enabled()
    try:
        return _train(X_fit, y_fit, X_val, y_val, params, seed, hidden, dropout)
    finally:
        torch.use_deterministic_algorithms(was)


def _train(X_fit, y_fit, X_val, y_val, params, seed, hidden, dropout):
    hidden = hidden or params['hidden']
    dropout = dropout or params['dropout']
    device = pick_device(params.get('allow_cpu', False))
    set_determinism(seed)
    xf = torch.as_tensor(X_fit, dtype=torch.float32, device=device)
    yf = torch.as_tensor(y_fit, dtype=torch.long, device=device)
    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    model = build_mlp(xf.shape[1], hidden, dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights(y_fit), dtype=torch.float32, device=device))
    val_loss_fn = nn.CrossEntropyLoss()
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
            loss = loss_fn(model(xf[b]), yf[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(val_loss_fn(model(xv), yv))
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
                hidden=list(hidden), dropout=list(dropout))
    return model, meta


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy().astype(np.float64)


def ensemble_proba(models: list, X: np.ndarray, n_members: int) -> np.ndarray:
    """Mean class probabilities of exactly n_members models; refuses a partial ensemble."""
    if len(models) != n_members or any(m is None for m in models):
        raise ValueError(f'partial ensemble: {sum(m is not None for m in models)} of {n_members} members')
    return np.mean([predict_proba(m, X) for m in models], axis=0)
